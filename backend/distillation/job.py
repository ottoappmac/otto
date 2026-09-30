"""Singleton distillation train job — start, poll, cancel.

Mirrors ``backend.memory`` / ``ToolJob``: one in-process job the UI polls.
Training is exclusive GPU work.  The job refuses to start while a chat
session is generating, then unloads cached MLX weights before ``mlx_lm.lora``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

from backend.distillation.collector import collect_trajectories, persist_trajectories
from backend.distillation.formatter import to_sft_jsonl
from backend.distillation.model_pair import ModelPairError
from backend.distillation.paths import sft_path
from backend.distillation.train import (
    DEFAULT_ITERS,
    parse_iter_from_line,
    train_lora,
)
from backend.distillation.fuse import fused_ready, fuse_lora
from backend.session_manager import _sessions_dir
from backend.session_transcript import _transcripts_dir

logger = logging.getLogger(__name__)

LOG_TAIL = 400
ACTIVE_STATES = frozenset({"running"})

# Injected in tests to skip a real subprocess.
StreamLoraFn = Callable[[list[str], "DistillStatus", asyncio.Event], Coroutine[Any, Any, int]]


class DistillBusy(RuntimeError):
    """Job already running, or a live session is using the GPU."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "busy",
        blocking: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.blocking = blocking or []


@dataclass
class DistillStatus:
    state: str = "idle"  # idle | queued | running | success | error | cancelled
    phase: str = ""  # collecting | preparing | unloading | training | fusing
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    log_lines: list[str] = field(default_factory=list)
    n_trajectories: int = 0
    n_sft: int = 0
    adapter_path: str | None = None
    student_id: str = ""
    teacher_id: str = ""
    iters: int = DEFAULT_ITERS
    iter_current: int | None = None
    pid: int | None = None
    warnings: list[str] = field(default_factory=list)
    catalog_id: str = ""
    display_name: str = ""
    purpose: str = ""
    fused_path: str = ""
    omlx_model_id: str = ""

    def append(self, line: str) -> None:
        text = (line or "").rstrip("\n")
        if not text:
            return
        self.log_lines.append(text)
        if len(self.log_lines) > LOG_TAIL:
            self.log_lines = self.log_lines[-LOG_TAIL:]
        parsed = parse_iter_from_line(text)
        if parsed is not None:
            self.iter_current = parsed

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "phase": self.phase,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "log_lines": list(self.log_lines),
            "n_trajectories": self.n_trajectories,
            "n_sft": self.n_sft,
            "adapter_path": self.adapter_path,
            "student_id": self.student_id,
            "teacher_id": self.teacher_id,
            "iters": self.iters,
            "iter_current": self.iter_current,
            "warnings": list(self.warnings),
            "catalog_id": self.catalog_id,
            "display_name": self.display_name,
            "purpose": self.purpose,
            "fused_path": self.fused_path,
            "omlx_model_id": self.omlx_model_id,
            "blocking_sessions": blocking_sessions(),
        }


_status = DistillStatus()
_cancel = asyncio.Event()
_lock = asyncio.Lock()
_task: asyncio.Task[None] | None = None
_stream_lora: StreamLoraFn | None = None
_deferred: dict[str, Any] | None = None


def get_status() -> DistillStatus:
    return _status


def reset() -> None:
    """Test helper — drop in-flight state.  Does not kill a live subprocess."""
    global _task, _deferred
    _cancel.clear()
    _status.state = "idle"
    _status.phase = ""
    _status.started_at = None
    _status.finished_at = None
    _status.error = None
    _status.log_lines = []
    _status.n_trajectories = 0
    _status.n_sft = 0
    _status.adapter_path = None
    _status.student_id = ""
    _status.teacher_id = ""
    _status.iters = DEFAULT_ITERS
    _status.iter_current = None
    _status.pid = None
    _status.warnings = []
    _status.catalog_id = ""
    _status.display_name = ""
    _status.purpose = ""
    _status.fused_path = ""
    _status.omlx_model_id = ""
    _task = None
    _deferred = None


def set_stream_lora_for_tests(fn: StreamLoraFn | None) -> None:
    global _stream_lora
    _stream_lora = fn


def is_busy() -> bool:
    return _status.state in ACTIVE_STATES


def blocking_sessions() -> list[dict[str, Any]]:
    """In-process MLX sessions that currently own Metal / generation."""
    try:
        from backend.state import session_mgr
    except Exception:  # noqa: BLE001
        return []
    out: list[dict[str, Any]] = []
    try:
        for info in session_mgr.list_active():
            if getattr(info, "status", None) != "running":
                continue
            provider = (getattr(info, "llm_provider", None) or "").strip().lower()
            # Distilled binds run as in-process MLX even if Settings is frontier.
            if provider != "mlx" and not getattr(info, "distill_catalog_id", None):
                continue
            out.append({
                "id": getattr(info, "id", ""),
                "title": getattr(info, "title", "") or "",
                "provider": provider or "mlx",
            })
    except Exception:  # noqa: BLE001
        logger.debug("blocking_sessions: list_active failed", exc_info=True)
    return out


def request_cancel() -> str:
    global _deferred
    if _status.state == "queued" or _deferred is not None:
        _deferred = None
        _status.state = "cancelled"
        _status.append("Queued train cancelled")
        _finish("cancelled", "Queued train cancelled")
        return "cancel_requested"
    if not is_busy():
        return "not_running"
    _cancel.set()
    pid = _status.pid
    if pid:
        _kill_process_group(pid)
        _status.append(f"Cancel requested — sent SIGTERM to pid {pid}")
    else:
        _status.append("Cancel requested")
    return "cancel_requested"


def _kill_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _machine_ram() -> tuple[float, float]:
    try:
        from backend.setup_capabilities import _ram_bytes, _wired_limit_gb_macos

        ram_gb = (_ram_bytes() or 0) / (1024 ** 3)
        return float(ram_gb), float(_wired_limit_gb_macos(ram_gb))
    except Exception:  # noqa: BLE001
        return 0.0, 0.0


async def start_train_job(
    *,
    iters: int | None = None,
    output_name: str = "activity",
    purpose: str = "",
    teacher_id: str | None = None,
    student_id: str | None = None,
    defer_session_id: str | None = None,
    upload_id: str | None = None,
) -> DistillStatus:
    """Arm a background train.  Raises :class:`DistillBusy` if it cannot start.

    When *defer_session_id* is the only in-process MLX session currently
    generating, the request is queued and started once that turn ends.
    """
    global _task, _deferred
    async with _lock:
        if is_busy() or _status.state == "queued":
            raise DistillBusy("A distillation job is already running", code="busy")
        blocking = blocking_sessions()
        only_self = (
            bool(defer_session_id)
            and len(blocking) == 1
            and blocking[0].get("id") == defer_session_id
        )
        if blocking and not only_self:
            raise DistillBusy(
                "Stop running MLX chat sessions first — training needs exclusive GPU memory",
                code="sessions_running",
                blocking=blocking,
            )
        payload = {
            "iters": int(iters) if iters and iters > 0 else DEFAULT_ITERS,
            "output_name": output_name or ("upload" if upload_id else "activity"),
            "purpose": (purpose or "").strip(),
            "teacher_id": teacher_id,
            "student_id": student_id,
            "upload_id": (upload_id or "").strip() or None,
        }
        if only_self:
            _deferred = payload
            _status.state = "queued"
            _status.phase = ""
            _status.error = None
            _status.purpose = payload["purpose"]
            _status.iters = payload["iters"]
            _status.append(
                "Queued LoRA train — starts when this MLX session finishes generating"
            )
            return _status
        return _arm_train_locked(**payload)


async def start_fuse_job(*, catalog_id: str) -> DistillStatus:
    """Fuse an existing LoRA into a standalone MLX model for Turbo/oMLX."""
    global _task
    from backend.distillation.catalog import resolve_catalog_id

    cid = (catalog_id or "").strip()
    rec = resolve_catalog_id(cid)
    if rec is None:
        raise ValueError(f"Unknown distilled catalog id {cid!r}")
    async with _lock:
        if is_busy() or _status.state == "queued":
            raise DistillBusy("A distillation job is already running", code="busy")
        blocking = blocking_sessions()
        if blocking:
            raise DistillBusy(
                "Stop running MLX chat sessions first — fuse needs exclusive GPU memory",
                code="sessions_running",
                blocking=blocking,
            )
        if rec.get("omlx_model_id") and fused_ready(rec.get("fused_path") or ""):
            _status.state = "success"
            _status.phase = ""
            _status.error = None
            _status.adapter_path = rec["adapter_path"]
            _status.student_id = rec["base_repo_id"]
            _status.catalog_id = rec["catalog_id"]
            _status.display_name = rec["display_name"]
            _status.fused_path = rec["fused_path"]
            _status.omlx_model_id = rec["omlx_model_id"]
            _status.append(f"Turbo fused model already at {_status.fused_path}")
            return _status
        _cancel.clear()
        _status.state = "running"
        _status.phase = "unloading"
        _status.started_at = _now()
        _status.finished_at = None
        _status.error = None
        _status.log_lines = []
        _status.adapter_path = rec["adapter_path"]
        _status.student_id = rec["base_repo_id"]
        _status.teacher_id = rec.get("teacher_model_id") or ""
        _status.catalog_id = rec["catalog_id"]
        _status.display_name = rec["display_name"]
        _status.fused_path = ""
        _status.omlx_model_id = ""
        _status.warnings = []
        _status.iter_current = None
        _status.pid = None
        _status.append(f"Fusing {rec['display_name']} for Turbo/oMLX")
        _task = asyncio.create_task(_run_fuse_pipeline(), name="distill-fuse")
        return _status


async def _run_fuse_pipeline() -> None:
    try:
        _status.phase = "unloading"
        report = await _unload_mlx()
        _status.append(
            f"Unloaded {report.get('evicted_models', 0)} cached MLX model(s), "
            f"metal={report.get('metal_cleared', False)}"
        )
        if _cancelled():
            _finish("cancelled", "Cancelled after unload")
            return
        await _fuse_adapter_into_omlx_model()
        if _cancelled():
            _finish("cancelled", "Cancelled during fuse")
            return
        if not _status.omlx_model_id:
            raise RuntimeError(
                "Fuse did not produce a Turbo-loadable MLX model. "
                "Standard can still use the LoRA adapter."
            )
        _finish("success")
    except asyncio.CancelledError:
        _finish("cancelled", "Job cancelled")
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Distillation fuse failed")
        _status.append(f"Error: {exc}")
        _finish("error", str(exc))


def _arm_train_locked(
    *,
    iters: int,
    output_name: str,
    purpose: str,
    teacher_id: str | None,
    student_id: str | None,
    upload_id: str | None = None,
) -> DistillStatus:
    """Caller must hold ``_lock``.  Starts the pipeline task."""
    global _task
    _cancel.clear()
    _status.state = "running"
    _status.phase = "collecting"
    _status.started_at = _now()
    _status.finished_at = None
    _status.error = None
    _status.log_lines = []
    _status.n_trajectories = 0
    _status.n_sft = 0
    _status.adapter_path = None
    _status.iter_current = None
    _status.pid = None
    _status.warnings = []
    _status.catalog_id = ""
    _status.display_name = ""
    _status.purpose = purpose
    _status.iters = iters
    _status.fused_path = ""
    _status.omlx_model_id = ""
    _status.append("Queued LoRA train" if not purpose else f"Queued LoRA train ({purpose})")
    _task = asyncio.create_task(
        _run_pipeline(
            iters=iters,
            output_name=output_name,
            purpose=purpose,
            teacher_id=teacher_id,
            student_id=student_id,
            upload_id=upload_id,
        ),
        name="distill-train",
    )
    return _status


async def start_deferred_if_idle() -> DistillStatus | None:
    """Start a queued train once no in-process MLX session is generating."""
    global _deferred
    async with _lock:
        if _deferred is None:
            return None
        if is_busy():
            return None
        if blocking_sessions():
            return None
        payload = _deferred
        _deferred = None
        return _arm_train_locked(**payload)


async def _run_pipeline(
    *,
    iters: int,
    output_name: str,
    purpose: str = "",
    teacher_id: str | None,
    student_id: str | None,
    upload_id: str | None = None,
) -> None:
    try:
        await _pipeline(
            iters=iters,
            output_name=output_name,
            purpose=purpose,
            teacher_id=teacher_id,
            student_id=student_id,
            upload_id=upload_id,
        )
    except asyncio.CancelledError:
        _finish("cancelled", "Job cancelled")
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Distillation train failed")
        _status.append(f"Error: {exc}")
        _finish("error", str(exc))


def _finish(state: str, error: str | None = None) -> None:
    _status.state = state
    _status.phase = ""
    _status.finished_at = _now()
    _status.error = error
    _status.pid = None


def _cancelled() -> bool:
    return _cancel.is_set()


async def _pipeline(
    *,
    iters: int,
    output_name: str,
    purpose: str = "",
    teacher_id: str | None,
    student_id: str | None,
    upload_id: str | None = None,
) -> None:
    from backend.config import AppConfig

    cfg = await AppConfig.aload()
    dist = cfg.distillation
    teacher = (teacher_id or dist.teacher_model_id).strip()
    student = (student_id or dist.student_model_id).strip()
    _status.student_id = student
    _status.teacher_id = teacher
    _status.append(f"Teacher {teacher} → student {student}")

    if _cancelled():
        _finish("cancelled", "Cancelled before collect")
        return

    sft = sft_path(data_dir=dist.data_dir)
    if upload_id:
        from backend.distillation.uploads import write_sft_from_upload

        _status.phase = "preparing"
        _status.append(f"Loading uploaded dataset {upload_id}…")
        report = await asyncio.to_thread(
            write_sft_from_upload, upload_id, sft, data_dir=dist.data_dir,
        )
        n_written = int(report.get("n_written") or 0)
        _status.n_trajectories = n_written
        _status.n_sft = n_written
        _status.append(f"Using {n_written} valid rows from upload")
        if n_written <= 0:
            raise ModelPairError("Uploaded file has no valid training rows.")
    else:
        _status.phase = "collecting"
        _status.append("Collecting trajectories from session transcripts…")
        trajectories = await asyncio.to_thread(
            collect_trajectories,
            transcripts_dir=_transcripts_dir(),
            sessions_dir=_sessions_dir(),
            min_tool_calls=dist.min_tool_calls,
            teacher_model_id=teacher,
            filter_sessions_by_teacher=dist.filter_sessions_by_teacher,
        )
        _status.n_trajectories = len(trajectories)
        _status.append(f"Accepted {len(trajectories)} trajectories")
        if not trajectories:
            raise ModelPairError(
                "No high-quality trajectories yet. Run some teacher sessions first."
            )
        if _cancelled():
            _finish("cancelled", "Cancelled after collect")
            return

        await asyncio.to_thread(
            persist_trajectories, trajectories, data_dir=dist.data_dir,
        )

        _status.phase = "preparing"
        _status.append("Writing SFT JSONL…")
        await asyncio.to_thread(to_sft_jsonl, trajectories, sft)
        _status.n_sft = len(trajectories)
    if _cancelled():
        _finish("cancelled", "Cancelled after prepare")
        return

    _status.phase = "unloading"
    _status.append("Unloading cached MLX chat weights…")
    report = await _unload_mlx()
    _status.append(
        "Unloaded {evicted} cached model(s), metal_cleared={metal}".format(
            evicted=report.get("evicted_models", 0),
            metal=report.get("metal_cleared", False),
        )
    )
    if _cancelled():
        _finish("cancelled", "Cancelled after unload")
        return

    ram_gb, _wired = _machine_ram()
    prepared = await asyncio.to_thread(
        train_lora,
        sft_jsonl=sft,
        student_id=student,
        teacher_id=teacher,
        output_name=output_name,
        purpose=purpose,
        ram_gb=ram_gb,
        allow_family_mismatch=True,
        run=False,
        unique_name=True,
        iters=iters,
    )
    _status.adapter_path = str(prepared.get("adapter_path") or "")
    _status.catalog_id = str(prepared.get("catalog_id") or "")
    _status.display_name = str(prepared.get("display_name") or "")
    _status.warnings = list(prepared.get("warnings") or [])
    for warn in _status.warnings:
        _status.append(f"Warning: {warn}")
    argv = list(prepared["argv"])
    _status.append("$ " + " ".join(argv))

    _status.phase = "training"
    _status.append("Starting mlx_lm.lora…")
    rc = await _invoke_lora(argv)
    if _cancelled():
        _finish("cancelled", "Cancelled during training")
        return
    if rc != 0:
        raise ModelPairError(f"mlx_lm.lora exited {rc}. See the log.")
    _status.append(f"Adapter written to {_status.adapter_path}")
    if _status.catalog_id:
        _status.append(f"Catalogued as {_status.display_name} ({_status.catalog_id})")
    await _fuse_adapter_into_omlx_model()
    _finish("success")


async def _fuse_adapter_into_omlx_model() -> None:
    """Write a standalone MLX folder oMLX can load.  LoRA train still counts as success if fuse fails."""
    from pathlib import Path

    adapter = Path(_status.adapter_path or "")
    if not adapter.is_dir():
        return
    try:
        prepared = fuse_lora(
            adapter_path=adapter,
            student_id=_status.student_id,
            run=False,
        )
    except Exception as exc:  # noqa: BLE001
        _status.append(f"Fuse skipped: {exc}")
        return
    if prepared.get("skipped"):
        _status.fused_path = str(prepared["fused_path"])
        _status.omlx_model_id = str(prepared["omlx_model_id"])
        _status.append(f"Turbo fused model ready ({_status.omlx_model_id})")
        return
    if _cancelled():
        return
    _status.phase = "fusing"
    _status.append(
        "Fusing LoRA into a standalone MLX model so Turbo/oMLX can load it…"
    )
    argv = list(prepared["argv"])
    _status.append("$ " + " ".join(argv))
    rc = await _invoke_lora(argv)
    if _cancelled():
        return
    dest = Path(prepared["fused_path"])
    if rc != 0 or not fused_ready(dest):
        msg = (
            f"mlx_lm.fuse did not produce a loadable model (exit {rc}). "
            "Standard still has the LoRA; use Fuse on the Turbo Distilled tab to retry."
        )
        _status.warnings.append(msg)
        _status.append(msg)
        return
    from backend.distillation.adapter_meta import patch_adapter_meta

    patch_adapter_meta(adapter, {"fused_path": str(dest)})
    _status.fused_path = str(dest)
    _status.omlx_model_id = dest.name
    _status.append(f"Fused Turbo model written to {dest} (oMLX id {dest.name})")


async def _unload_mlx() -> dict[str, Any]:
    try:
        from backend.routes.mlx import mlx_unload

        return await mlx_unload()
    except Exception as exc:  # noqa: BLE001
        logger.debug("MLX unload failed (continuing to train)", exc_info=True)
        return {"status": "skipped", "error": str(exc), "evicted_models": 0, "metal_cleared": False}


async def _invoke_lora(argv: list[str]) -> int:
    fn = _stream_lora
    if fn is not None:
        return await fn(argv, _status, _cancel)
    return await stream_lora(argv, _status, _cancel)


async def stream_lora(
    argv: list[str],
    status: DistillStatus,
    cancel: asyncio.Event,
) -> int:
    """Run ``mlx_lm.lora`` and stream combined output into *status*."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    status.pid = proc.pid
    assert proc.stdout is not None

    while True:
        if cancel.is_set():
            if proc.returncode is None:
                _kill_process_group(proc.pid)
            try:
                await asyncio.wait_for(proc.wait(), timeout=8.0)
            except asyncio.TimeoutError:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                await proc.wait()
            return -15
        try:
            raw = await asyncio.wait_for(proc.stdout.readline(), timeout=0.4)
        except asyncio.TimeoutError:
            if proc.returncode is not None:
                break
            continue
        if not raw:
            break
        status.append(raw.decode("utf-8", errors="replace").rstrip("\n"))

    rc = await proc.wait()
    status.pid = None
    return int(rc)
