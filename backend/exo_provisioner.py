"""Async wrapper around :mod:`backend.exo_cli` for the FastAPI / agent layer.

``backend.exo_cli`` is intentionally stdlib-only and synchronous: the same
file is shipped to remote nodes via ``scp`` and executed with the bare
system ``python3``.  This wrapper bridges those sync entry points into:

* the FastAPI request / event loop world (``backend/routes/exo.py``)
* the LangChain ``@tool`` ecosystem the deep agent uses
  (``backend/exo_tools.py``)

It also persists structured progress for long-running operations
(``provision`` can take 2–5 minutes on first run) so the UI can poll a
job id instead of holding an HTTP connection open.

Design notes
------------

The CLI calls :func:`backend.exo_cli.fail`, which prints to stderr and
``sys.exit``s.  In a long-lived backend process that would kill the
worker, so every wrapper here installs a ``_capture_fail`` shim before
delegating to the sync helper and surfaces the error as an exception
instead.  All blocking work is dispatched via :func:`asyncio.to_thread`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from backend import exo_cli, exo_runtime
from backend.config import AppConfig, ExoConfig, ExoRemoteConfig
from backend.exo_cli import (
    ClusterStatus,
    NodeInfo,
    detect_prereqs,
    exo_repo_dir,
    exo_root,
    fetch_cluster_status,
    is_running,
    load_state,
    log_file,
    pid_file,
    provision_exo,
    run_remote,
    smoke_test,
    start_local,
    state_file,
    stop_local,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# fail() neutralisation
# ---------------------------------------------------------------------------


class ExoCliError(RuntimeError):
    """Raised in place of ``exo_cli.fail()``'s ``SystemExit`` so callers can
    handle provisioner errors without taking the FastAPI worker down."""


@contextlib.contextmanager
def _no_sys_exit() -> Any:
    """Temporarily replace :func:`exo_cli.fail` so a failing CLI helper
    raises :class:`ExoCliError` instead of calling ``sys.exit``."""

    original = exo_cli.fail

    def _capture(msg: str, code: int = 1) -> None:  # noqa: ARG001
        raise ExoCliError(msg)

    exo_cli.fail = _capture  # type: ignore[assignment]
    try:
        yield
    finally:
        exo_cli.fail = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Job tracking (provision and other long-running ops)
# ---------------------------------------------------------------------------


@dataclass
class ExoJobPhase:
    """One named step within an :class:`ExoJob`.

    Phases are ordered; a job may have any number of them.  The UI
    renders them as a horizontal step list with per-step status icons.
    """

    name: str
    status: str = "pending"   # pending | running | done | error
    message: str = ""         # short human-readable note (e.g. timing, pid)

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "message": self.message}


@dataclass
class ExoJob:
    """One provision / up / down request, surfaced to the UI for polling.

    ``log_lines`` is a tail-bounded list — old lines are dropped so the
    structure can't grow without bound during a long ``uv sync``.

    ``phases`` is an optional ordered list of :class:`ExoJobPhase` objects
    that give the UI a step-by-step progress view for longer operations
    (e.g. "Provision → Start → Verify").
    """

    id: str
    kind: str  # "provision" | "up" | "stop" | "smoke" | "remote"
    target: str  # "local" | ssh alias
    status: str = "pending"  # pending | running | done | error
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    error: Optional[str] = None
    log_lines: list[str] = field(default_factory=list)
    result: Optional[dict] = None
    phases: list[ExoJobPhase] = field(default_factory=list)

    _LOG_TAIL = 1000

    def append(self, line: str) -> None:
        line = line.rstrip("\n")
        if not line:
            return
        self.log_lines.append(line)
        if len(self.log_lines) > self._LOG_TAIL:
            self.log_lines = self.log_lines[-self._LOG_TAIL:]

    def set_phase(self, name: str, status: str, message: str = "") -> None:
        """Update the status (and optional message) of the named phase."""
        for ph in self.phases:
            if ph.name == name:
                ph.status = status
                if message:
                    ph.message = message
                return

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "target": self.target,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "log_lines": list(self.log_lines),
            "result": self.result,
            "phases": [ph.to_dict() for ph in self.phases],
        }


_jobs: dict[str, ExoJob] = {}
_jobs_lock = threading.Lock()
_JOB_RETENTION_S = 60 * 60  # keep last hour of completed jobs


def _new_job(kind: str, target: str) -> ExoJob:
    with _jobs_lock:
        # GC very old completed jobs so the dict can't grow forever.
        cutoff = time.time() - _JOB_RETENTION_S
        for jid in list(_jobs.keys()):
            j = _jobs[jid]
            if j.finished_at and j.finished_at < cutoff:
                _jobs.pop(jid, None)
        job = ExoJob(id=uuid.uuid4().hex, kind=kind, target=target)
        _jobs[job.id] = job
    return job


def _finalise(job: ExoJob, *, error: Optional[str] = None, result: Optional[dict] = None) -> None:
    job.finished_at = time.time()
    job.error = error
    job.result = result
    job.status = "error" if error else "done"


def get_job(job_id: str) -> Optional[ExoJob]:
    with _jobs_lock:
        return _jobs.get(job_id)


def list_jobs() -> list[ExoJob]:
    with _jobs_lock:
        return sorted(_jobs.values(), key=lambda j: j.started_at, reverse=True)


# ---------------------------------------------------------------------------
# Stdout capture so CLI ``progress(...)`` output ends up in the job log.
# ---------------------------------------------------------------------------


class _LineCapture:
    """File-like object that fans every write to a logger and a callback."""

    def __init__(self, sink: Callable[[str], None], echo: Optional[Any] = None) -> None:
        self._buf = ""
        self._sink = sink
        self._echo = echo

    def write(self, data: str) -> int:
        if not data:
            return 0
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._sink(line)
            if self._echo is not None:
                try:
                    self._echo.write(line + "\n")
                    self._echo.flush()
                except Exception:
                    pass
        return len(data)

    def flush(self) -> None:
        if self._buf:
            self._sink(self._buf)
            if self._echo is not None:
                try:
                    self._echo.write(self._buf)
                    self._echo.flush()
                except Exception:
                    pass
            self._buf = ""


@contextlib.contextmanager
def _capture_stdout(job: ExoJob) -> Any:
    import sys

    saved_out, saved_err = sys.stdout, sys.stderr
    sys.stdout = _LineCapture(job.append, echo=saved_out)
    sys.stderr = _LineCapture(job.append, echo=saved_err)
    try:
        yield
    finally:
        sys.stdout = saved_out
        sys.stderr = saved_err


# ---------------------------------------------------------------------------
# Resolved-info / status snapshots
# ---------------------------------------------------------------------------


def _node_to_dict(n: NodeInfo) -> dict:
    return {
        "node_id": n.node_id,
        "chip": n.chip,
        "friendly_name": n.friendly_name,
        "memory_total_gb": n.memory_total_gb,
        "memory_free_gb": n.memory_free_gb,
    }


def cluster_status_to_dict(s: ClusterStatus) -> dict:
    return {
        "reachable": s.reachable,
        "base_url": s.base_url,
        "master_node_id": s.master_node_id,
        "peer_count": s.peer_count,
        "rdma_connections": s.rdma_connections,
        "loaded_models": s.loaded_models,
        "instances": s.instances,
        "runners": s.runners,
        "nodes": [_node_to_dict(n) for n in s.nodes],
        "error": (s.raw or {}).get("error") if isinstance(s.raw, dict) else None,
    }


def model_cards_dir(cfg: ExoConfig) -> Path:
    """Directory holding the curated ``*.toml`` inference model cards.

    The prebuilt artifact packs the whole exo checkout (so the cards live
    under the extracted runtime dir), while the legacy source build keeps
    them in the cloned repo. Resolving this per-mode keeps the model
    catalog populated regardless of how exo was installed.
    """
    base = exo_runtime.runtime_dir() if is_prebuilt_mode(cfg) else exo_repo_dir()
    return base / "resources" / "inference_model_cards"


def info_snapshot(cfg: ExoConfig) -> dict:
    state = load_state()
    p = detect_prereqs()
    prebuilt = is_prebuilt_mode(cfg)
    runtime_state = exo_runtime.load_runtime_state()
    # ``installed`` is the single, mode-aware signal the UI should use to
    # decide whether a provision/download is still needed — replacing the
    # old habit of sniffing source-only fields like ``state.exo_repo_dir``.
    if prebuilt:
        installed = exo_runtime.is_installed(cfg.repo_ref)
    else:
        installed = bool(state.exo_repo_dir and state.git_commit)
    return {
        "platform": exo_cli.platform.system(),
        "python": exo_cli.sys.version.split()[0],
        "app_data_dir": str(exo_cli.get_app_data_dir()),
        "exo_root": str(exo_root()),
        "exo_repo_dir": str(exo_repo_dir()),
        "state_file": str(state_file()),
        "pid_file": str(pid_file()),
        "log_file": str(log_file()),
        "state": asdict(state),
        "prereqs": asdict(p),
        "config": cfg.model_dump(),
        "running": is_running(cfg.api_port),
        # Unified, mode-aware install signals.
        "mode_effective": "prebuilt" if prebuilt else "source",
        "installed": installed,
        "prebuilt": asdict(runtime_state) if prebuilt else None,
    }


# ---------------------------------------------------------------------------
# Local lifecycle
# ---------------------------------------------------------------------------


async def _run_in_thread(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(fn, *args, **kwargs)


async def aprovision(
    cfg: ExoConfig,
    *,
    force: bool = False,
    force_mismatch: bool = False,
) -> ExoJob:
    """Kick off ``provision_exo`` in a background thread, returning the job
    descriptor immediately so the caller can poll for progress.

    ``force_mismatch`` suppresses the MLX-version preflight warning when
    EXO's pinned MLX disagrees with Otto's bundled MLX.  See
    :func:`exo_cli.provision_exo` for the rationale (the warning is
    informational; we never block here).
    """

    job = _new_job("provision", "local")
    job.status = "running"
    _apply_config_env(cfg)
    prebuilt = is_prebuilt_mode(cfg)

    def _worker() -> None:
        try:
            if prebuilt:
                job.append("downloading prebuilt exo runtime…")
                state = exo_runtime.install(
                    exo_ref=cfg.repo_ref,
                    prebuilt_url=getattr(cfg, "prebuilt_url", ""),
                    progress=job.append,
                    force=force,
                )
            else:
                with _capture_stdout(job), _no_sys_exit():
                    state = provision_exo(
                        exo_ref=cfg.repo_ref,
                        repo_url=cfg.repo_url,
                        force=force,
                        auto_prereqs=cfg.auto_provision,
                        force_mismatch=force_mismatch,
                    )
            _finalise(job, result={"state": asdict(state)})
        except Exception as exc:
            msg = str(exc)
            if prebuilt and _is_no_prebuilt_error(msg):
                logger.warning("exo prebuilt runtime not available: %s", msg)
            else:
                logger.exception("exo provision failed")
            _finalise(job, error=msg)

    threading.Thread(target=_worker, name=f"exo-prov-{job.id[:8]}", daemon=True).start()
    return job


def is_prebuilt_mode(cfg: ExoConfig) -> bool:
    """Whether to use the downloaded prebuilt runtime instead of a source build.

    Prebuilt is the default but only applies on a supported host (Apple
    Silicon). Anything else transparently falls back to source mode.
    """
    return cfg.mode == "prebuilt" and exo_runtime.is_supported_host()


def _is_no_prebuilt_error(msg: str) -> bool:
    """True when the error means 'no artifact published yet' (expected before first release)."""
    lower = msg.lower()
    return "manifest" in lower or "404" in lower or "no prebuilt" in lower


def _single_node_default(cfg: ExoConfig) -> bool:
    """True for the single-Mac default (no multi-node placement requested)."""
    if int(cfg.min_nodes or 1) > 1:
        return False
    return not any(r.enabled for r in cfg.remotes)


def _prebuilt_launch_cmd(cfg: ExoConfig) -> list[str]:
    return exo_runtime.launch_command(
        api_port=cfg.api_port,
        libp2p_port=cfg.libp2p_port,
        force_master=_single_node_default(cfg),
    )


async def astart_local(cfg: ExoConfig) -> dict:
    """Start the local exo daemon. Returns ``{"pid": int, "running": bool}``.

    Does NOT block on ``provision`` — call :func:`aprovision` separately
    when the cluster is being set up for the first time.
    """

    _apply_config_env(cfg)
    prebuilt = is_prebuilt_mode(cfg)
    cmd_override = _prebuilt_launch_cmd(cfg) if prebuilt else None
    cwd_override = exo_runtime.runtime_dir() if prebuilt else None
    try:
        with _no_sys_exit():
            pid = await _run_in_thread(
                start_local,
                api_port=cfg.api_port,
                libp2p_port=cfg.libp2p_port,
                wait_seconds=120.0,
                cmd_override=cmd_override,
                cwd_override=cwd_override,
            )
    except ExoCliError as exc:
        logger.error("start_local failed: %s", exc)
        raise

    return {"pid": int(pid or 0), "running": is_running(cfg.api_port)}


async def astop_local(cfg: ExoConfig) -> dict:
    _apply_config_env(cfg)
    with _no_sys_exit():
        stopped = await _run_in_thread(stop_local, api_port=cfg.api_port)
    return {"stopped": bool(stopped), "running": is_running(cfg.api_port)}


async def afetch_status(cfg: ExoConfig) -> dict:
    s = await _run_in_thread(fetch_cluster_status, cfg.effective_base_url)
    return cluster_status_to_dict(s)


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------


def _instance_model_id(inst: dict) -> str | None:
    """Extract a model id from an exo ``/state.instances[*]`` value.

    exo wraps each instance in a single-key dict keyed by the instance
    *variant* (``MlxRingInstance`` / ``MlxJacclInstance`` / etc.); the
    actual ``modelId`` lives at
    ``<variant>.shardAssignments.modelId``. Older shapes also expose
    ``model_id`` / ``model`` directly on the inner dict — try those as
    fallbacks for forward/back compat.
    """
    if not isinstance(inst, dict):
        return None
    direct = inst.get("modelId") or inst.get("model_id") or inst.get("model")
    if isinstance(direct, str) and direct:
        return direct
    for v in inst.values():
        if not isinstance(v, dict):
            continue
        sa = v.get("shardAssignments") if isinstance(v, dict) else None
        if isinstance(sa, dict):
            mid = sa.get("modelId") or sa.get("model_id")
            if isinstance(mid, str) and mid:
                return mid
        mid = v.get("modelId") or v.get("model_id") or v.get("model")
        if isinstance(mid, str) and mid:
            return mid
    return None


def _list_models_sync(base_url: str, *, timeout: float = 10.0) -> dict:
    """Query exo's OpenAI-compatible model catalog and merge with state.

    Returns ``{"reachable": bool, "base_url": str, "models": [...]}`` where
    each model is::

        {
          "id":         "<modelId>",
          "name":       "<friendly name when known>",
          "downloaded": bool,   # files are on disk somewhere in the cluster
          "loaded":     bool,   # an instance is currently in memory
        }

    The cluster's ``/v1/models`` is the full catalog (~120 ids); the
    ``?status=downloaded`` variant is the subset that has weights on disk
    on at least one node, and ``/state``'s ``instances`` map gives us the
    "currently loaded" set. If the cluster is unreachable, returns
    ``{"reachable": False, "models": []}`` so the UI can render a clean
    "cluster offline" state.
    """
    base = exo_cli._strip_v1(base_url)
    out: dict = {"reachable": False, "base_url": base, "models": []}
    try:
        catalog_raw = exo_cli.http_get_json(f"{base}/v1/models", timeout=timeout)
    except Exception as exc:
        out["error"] = str(exc)
        return out
    out["reachable"] = True

    catalog: list[dict] = list(catalog_raw.get("data") or [])
    downloaded_ids: set[str] = set()
    try:
        dl_raw = exo_cli.http_get_json(
            f"{base}/v1/models?status=downloaded", timeout=timeout
        )
        for m in dl_raw.get("data") or []:
            mid = m.get("id")
            if mid:
                downloaded_ids.add(mid)
    except Exception:
        # Old exo versions might not support the filter — treat as zero
        # downloaded rather than failing the whole listing.
        pass

    loaded_ids: set[str] = set()
    try:
        state = exo_cli.http_get_json(f"{base}/state", timeout=timeout)
        for inst in (state.get("instances") or {}).values():
            mid = _instance_model_id(inst)
            if mid:
                loaded_ids.add(mid)
    except Exception:
        pass

    out["models"] = [
        {
            "id": m.get("id"),
            "name": m.get("name") or m.get("id"),
            "downloaded": m.get("id") in downloaded_ids,
            "loaded": m.get("id") in loaded_ids,
        }
        for m in catalog
        if m.get("id")
    ]
    return out


async def alist_models(cfg: ExoConfig, *, timeout: float = 10.0) -> dict:
    return await _run_in_thread(_list_models_sync, cfg.effective_base_url,
                                timeout=timeout)


def _delete_instances_for_model_sync(
    base: str, model_id: str, *, timeout: float = 5.0,
) -> int:
    """Delete any ``/state.instances`` entries belonging to ``model_id``.

    exo's ``POST /place_instance`` does **not** replace an existing
    instance — calling it twice for the same model with different
    ``min_nodes`` (e.g. user re-tuned placement after a peer joined)
    leaves both instances running, each holding a copy of the model in
    memory. To make the place call idempotent we first ``DELETE
    /instance/{instance_id}`` on every existing instance with a matching
    ``modelId``.

    Returns the number of instances actually deleted. Best-effort: any
    individual DELETE failure is swallowed (logged at debug) so a stale
    or already-removed entry doesn't block a fresh place call.
    """
    try:
        state = exo_cli.http_get_json(f"{base}/state", timeout=timeout)
    except Exception as exc:
        logger.debug("could not fetch /state to dedupe instances: %s", exc)
        return 0
    deleted = 0
    for inst_id, inst in (state.get("instances") or {}).items():
        if _instance_model_id(inst) != model_id:
            continue
        try:
            req = urllib.request.Request(
                f"{base}/instance/{inst_id}",
                method="DELETE",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout):
                deleted += 1
        except Exception as exc:
            logger.debug(
                "DELETE /instance/%s failed (continuing): %s", inst_id, exc,
            )
    return deleted


def _preload_model_sync(
    base_url: str,
    model_id: str,
    *,
    timeout: float = 30.0,
    min_nodes: int = 1,
    replace_existing: bool = True,
    sharding: Optional[str] = None,
    instance_meta: Optional[str] = None,
) -> dict:
    """Place a cluster instance for ``model_id`` and wait for it to load.

    exo doesn't auto-load on the first chat completion — that returns
    ``404 No instance found``. The dedicated path is
    ``POST /place_instance`` which schedules sharding across reachable
    nodes; we then poll ``/state.instances`` until the model appears
    (or ``timeout`` runs out).

    ``min_nodes`` forces exo's scheduler to spread the model across at
    least that many cluster nodes (default 1 lets it pick the cheapest
    single-node placement when the model fits).

    ``replace_existing`` (default True): when there's already a loaded
    instance for ``model_id`` (e.g. a previous placement with a
    different ``min_nodes``), DELETE it before placing the new one so
    we don't end up with multiple stacked instances of the same model.

    Returns ``{ok, model, elapsed_seconds, first_choice?, error?, detail?,
    replaced?}``.
    """
    base = exo_cli._strip_v1(base_url)
    started = time.monotonic()

    replaced = 0
    if replace_existing:
        replaced = _delete_instances_for_model_sync(base, model_id)

    payload: dict = {"model_id": model_id}
    if min_nodes and min_nodes > 1:
        payload["min_nodes"] = int(min_nodes)
    if sharding:
        payload["sharding"] = sharding
    if instance_meta:
        payload["instance_meta"] = instance_meta
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/place_instance",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=min(timeout, 30.0)) as resp:
            place_payload = json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
        except Exception:
            detail = ""
        return {
            "ok": False,
            "model": model_id,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "error": f"HTTP {exc.code}: {exc.reason}",
            "detail": detail,
            "replaced": replaced,
        }
    except Exception as exc:
        return {
            "ok": False,
            "model": model_id,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "error": str(exc),
            "replaced": replaced,
        }

    # Poll /state.instances every 2s until the model shows up. exo can
    # be downloading multi-GB shards across nodes here, so we keep
    # going until the caller's overall timeout is exhausted.
    deadline = started + max(5.0, timeout)
    last_err = ""
    while time.monotonic() < deadline:
        try:
            state = exo_cli.http_get_json(f"{base}/state", timeout=5)
            for inst in (state.get("instances") or {}).values():
                if _instance_model_id(inst) == model_id:
                    return {
                        "ok": True,
                        "model": model_id,
                        "elapsed_seconds": round(
                            time.monotonic() - started, 2
                        ),
                        "first_choice": "",
                        "place_response": place_payload,
                        "replaced": replaced,
                    }
        except Exception as exc:
            last_err = str(exc)
        time.sleep(2.0)

    return {
        "ok": False,
        "model": model_id,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "error": (
            f"timed out waiting for {model_id!r} to load"
            + (f" (last poll error: {last_err})" if last_err else "")
        ),
        "place_response": place_payload,
        "replaced": replaced,
    }


async def apreload_model(
    cfg: ExoConfig,
    model_id: str,
    *,
    timeout: float = 30.0,
    min_nodes: Optional[int] = None,
) -> dict:
    """Async wrapper around :func:`_preload_model_sync`.

    ``min_nodes`` defaults to ``cfg.min_nodes`` when not overridden by
    the caller, so the UI's value flows through automatically.  ``sharding``
    and ``instance_meta`` always come from ``cfg`` so a placement matches
    the user's configured strategy / transport.
    """
    effective_min_nodes = (
        int(min_nodes) if min_nodes is not None else int(cfg.min_nodes or 1)
    )
    return await _run_in_thread(
        _preload_model_sync,
        cfg.effective_base_url,
        model_id,
        timeout=timeout,
        min_nodes=effective_min_nodes,
        sharding=cfg.sharding,
        instance_meta=cfg.instance_meta,
    )


async def asmoke(cfg: ExoConfig) -> ExoJob:
    job = _new_job("smoke", "local")
    job.status = "running"
    _apply_config_env(cfg)

    def _worker() -> None:
        try:
            with _capture_stdout(job), _no_sys_exit():
                rc = smoke_test(cfg.effective_base_url)
            _finalise(job, result={"return_code": rc})
        except Exception as exc:
            logger.exception("exo smoke failed")
            _finalise(job, error=str(exc))

    threading.Thread(target=_worker, name=f"exo-smoke-{job.id[:8]}", daemon=True).start()
    return job


# ---------------------------------------------------------------------------
# Remote lifecycle (SSH)
# ---------------------------------------------------------------------------


def _remote_env(cfg: ExoConfig, remote: ExoRemoteConfig) -> dict[str, str]:
    """Forward env vars to the remote node when invoking ``exo_cli``.

    The script's :func:`run_remote` reads from ``os.environ`` of THIS
    process for the values to forward.  We temporarily set them around
    each call so config-driven launches don't depend on the user shell.
    """

    extras = {
        "EXO_REPO_URL": cfg.repo_url,
        "EXO_REF": cfg.repo_ref,
        "EXO_API_PORT": str(cfg.api_port),
        "EXO_LIBP2P_PORT": str(cfg.libp2p_port),
        "EXO_BASE_URL": cfg.effective_base_url,
    }
    if cfg.no_terminal_wrap:
        extras["EXO_NO_TERMINAL_WRAP"] = "1"
    if remote.app_data_dir:
        extras["OTTO_APP_DATA_DIR"] = remote.app_data_dir
    return extras


@contextlib.contextmanager
def _patched_env(extras: dict[str, str]) -> Any:
    saved: dict[str, Optional[str]] = {}
    try:
        for k, v in extras.items():
            saved[k] = os.environ.get(k)
            os.environ[k] = v
        yield
    finally:
        for k, prev in saved.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


async def arun_remote(
    cfg: ExoConfig,
    remote: ExoRemoteConfig,
    subcommand: list[str],
    *,
    job_kind: str = "remote",
) -> ExoJob:
    """Run ``exo_cli <subcommand>`` on a remote SSH host as a tracked job."""

    job = _new_job(job_kind, remote.ssh_alias)
    job.status = "running"
    extras = _remote_env(cfg, remote)

    def _worker() -> None:
        try:
            with _capture_stdout(job), _no_sys_exit(), _patched_env(extras):
                rc = run_remote(remote.ssh_alias, subcommand)
            _finalise(job, result={"return_code": rc})
        except Exception as exc:
            logger.exception("exo remote %s failed", remote.ssh_alias)
            _finalise(job, error=str(exc))

    threading.Thread(target=_worker, name=f"exo-rem-{job.id[:8]}", daemon=True).start()
    return job


def _remote_subcmd_for(action: str, cfg: ExoConfig) -> list[str]:
    """Build the remote argv for ``up/down/status/smoke/provision``.

    Only flags supported by the remote subcommand are included; missing
    flags fall back to the script's own defaults (which read from the
    forwarded env vars).
    """
    base = ["--api-port", str(cfg.api_port)]
    if action == "up":
        return [
            "up",
            "--ref", cfg.repo_ref,
            "--repo-url", cfg.repo_url,
            *base,
            "--libp2p-port", str(cfg.libp2p_port),
        ]
    if action == "down":
        return ["down", *base]
    if action == "status":
        return ["status", "--base-url", cfg.effective_base_url, "--json"]
    if action == "smoke":
        return ["smoke", "--base-url", cfg.effective_base_url]
    if action == "provision":
        return ["provision", "--ref", cfg.repo_ref, "--repo-url", cfg.repo_url]
    if action == "info":
        return ["info"]
    raise ValueError(f"Unknown exo remote action: {action}")


# ---------------------------------------------------------------------------
# Public high-level helpers (used by the FastAPI route + agent tools)
# ---------------------------------------------------------------------------


async def aup_local(cfg: ExoConfig, *, force: bool = False) -> dict:
    """Provision the local clone if needed, then start the daemon.

    Returns ``{"provision_job": <id|None>, "pid": int, "running": bool}``.
    The provision job runs in the background; the start step blocks
    on it via the same thread to keep the call chain linear.
    """

    provision_job_id: Optional[str] = None

    if is_prebuilt_mode(cfg):
        # Download + verify the prebuilt runtime on demand (no git/uv/npm).
        if force or not exo_runtime.is_installed(cfg.repo_ref):
            await _run_in_thread(
                exo_runtime.install,
                exo_ref=cfg.repo_ref,
                prebuilt_url=getattr(cfg, "prebuilt_url", ""),
                force=force,
            )
    else:
        state = load_state()
        needs_provision = (
            force
            or not state.deps_installed_for_commit
            or not (Path(state.exo_repo_dir or exo_repo_dir()) / ".venv").exists()
        )
        if needs_provision:
            job = await aprovision(cfg, force=force)
            provision_job_id = job.id
            # Block on the provisioner thread so start_local sees the .venv.
            while job.finished_at is None:
                await asyncio.sleep(0.5)
            if job.error:
                raise ExoCliError(f"provision failed: {job.error}")

    res = await astart_local(cfg)
    await _maybe_preload(cfg)
    return {"provision_job": provision_job_id, **res}


async def _maybe_preload(cfg: ExoConfig) -> None:
    """Auto-place the configured model so the first chat doesn't 404.

    exo does not auto-load on the first completion — it needs an explicit
    ``POST /place_instance``. When the user has picked a default model we do
    that for them right after the API comes up. Best-effort: a failure here
    is logged, not raised, so a started-but-not-yet-loaded cluster is still
    usable once the user (or a later request) places a model.
    """
    if not cfg.model_name:
        return
    if not is_running(cfg.api_port):
        return
    try:
        result = await apreload_model(cfg, cfg.model_name, timeout=600.0)
        if not result.get("ok"):
            logger.warning(
                "exo auto-preload of %s did not complete: %s",
                cfg.model_name,
                result.get("error"),
            )
    except Exception:
        logger.exception("exo auto-preload failed for %s", cfg.model_name)


async def adown_local(cfg: ExoConfig) -> dict:
    return await astop_local(cfg)


_UP_PHASES = ["provision", "start", "verify"]


async def aup_job(
    cfg: ExoConfig,
    *,
    force: bool = False,
    force_mismatch: bool = False,
) -> ExoJob:
    """Non-blocking variant of :func:`aup_local` that returns an :class:`ExoJob`
    immediately.  The caller polls ``/api/exo/jobs/<id>`` for live progress.

    Phases emitted (in order):
    - **provision** — skipped (instant "done") when deps are already installed.
    - **start**     — launches the local daemon.
    - **verify**    — confirms the API port is responding.

    ``force_mismatch`` is forwarded to :func:`provision_exo` to suppress
    the MLX-version preflight warning; see that function for the
    rationale.
    """

    job = _new_job("up", "local")
    job.phases = [ExoJobPhase(name=ph) for ph in _UP_PHASES]
    job.status = "running"
    _apply_config_env(cfg)

    def _run() -> None:
        import time as _time

        prebuilt = is_prebuilt_mode(cfg)

        try:
            # ── Phase 1: provision (prebuilt download | source build) ──────
            job.set_phase("provision", "running")
            if prebuilt:
                if force or not exo_runtime.is_installed(cfg.repo_ref):
                    job.append("downloading prebuilt exo runtime…")
                    try:
                        exo_runtime.install(
                            exo_ref=cfg.repo_ref,
                            prebuilt_url=getattr(cfg, "prebuilt_url", ""),
                            progress=job.append,
                            force=force,
                        )
                        job.set_phase("provision", "done")
                    except Exception as exc:
                        job.set_phase("provision", "error", str(exc))
                        job.set_phase("start", "error")
                        job.set_phase("verify", "error")
                        _finalise(job, error=f"runtime download failed: {exc}")
                        return
                else:
                    job.set_phase("provision", "done", "runtime already installed")
            else:
                state = load_state()
                needs_provision = (
                    force
                    or not state.deps_installed_for_commit
                    or not (Path(state.exo_repo_dir or exo_repo_dir()) / ".venv").exists()
                )
                if needs_provision:
                    job.append("provisioning dependencies…")
                    try:
                        with _capture_stdout(job), _no_sys_exit():
                            provision_exo(
                                exo_ref=cfg.repo_ref,
                                repo_url=cfg.repo_url,
                                force=force,
                                auto_prereqs=cfg.auto_provision,
                                force_mismatch=force_mismatch,
                            )
                        job.set_phase("provision", "done")
                    except Exception as exc:
                        job.set_phase("provision", "error", str(exc))
                        job.set_phase("start", "error")
                        job.set_phase("verify", "error")
                        _finalise(job, error=f"provision failed: {exc}")
                        return
                else:
                    job.set_phase("provision", "done", "already up-to-date")

            # ── Phase 2: start daemon ──────────────────────────────────────
            job.set_phase("start", "running")
            job.append("starting exo daemon…")
            t0 = _time.monotonic()
            try:
                with _no_sys_exit():
                    pid = start_local(
                        api_port=cfg.api_port,
                        libp2p_port=cfg.libp2p_port,
                        wait_seconds=120.0,
                        cmd_override=_prebuilt_launch_cmd(cfg) if prebuilt else None,
                        cwd_override=(
                            exo_runtime.runtime_dir() if prebuilt else None
                        ),
                    )
                elapsed = f"{_time.monotonic() - t0:.1f}s"
                job.set_phase("start", "done", f"pid {pid} · {elapsed}")
                job.append(f"daemon started (pid={pid})")
            except ExoCliError as exc:
                job.set_phase("start", "error", str(exc))
                job.set_phase("verify", "error")
                _finalise(job, error=str(exc))
                return

            # ── Phase 3: verify ────────────────────────────────────────────
            job.set_phase("verify", "running")
            running = is_running(cfg.api_port)
            if running:
                job.set_phase("verify", "done")
                job.append("API is responding ✓")
            else:
                job.set_phase("verify", "error", "API port not responding")
                _finalise(job, error="daemon started but API port is not responding")
                return

            # Queue the configured model as a non-blocking preload job so the
            # Up job completes the moment the daemon is verified running.
            # The preload is visible as a separate job in the Model section —
            # no need to block this job thread for up to 10 minutes.
            if cfg.model_name:
                try:
                    start_preload_job(cfg, cfg.model_name)
                    job.append(f"model preload queued for {cfg.model_name}")
                except Exception as exc:
                    job.append(f"auto-preload skipped: {exc}")

            _finalise(job, result={"pid": int(pid or 0), "running": True})

        except Exception as exc:
            logger.exception("aup_job worker failed")
            _finalise(job, error=str(exc))

    threading.Thread(target=_run, name=f"exo-up-{job.id[:8]}", daemon=True).start()
    return job


def _apply_config_env(cfg: ExoConfig) -> None:
    """Push ``ExoConfig`` values into ``os.environ`` so the CLI helpers
    pick up the user's settings even when called directly."""

    os.environ["EXO_REPO_URL"] = cfg.repo_url
    os.environ["EXO_REF"] = cfg.repo_ref
    os.environ["EXO_API_PORT"] = str(cfg.api_port)
    os.environ["EXO_LIBP2P_PORT"] = str(cfg.libp2p_port)
    os.environ["EXO_BASE_URL"] = cfg.effective_base_url
    if cfg.no_terminal_wrap:
        os.environ["EXO_NO_TERMINAL_WRAP"] = "1"
    else:
        os.environ.pop("EXO_NO_TERMINAL_WRAP", None)


# ---------------------------------------------------------------------------
# Convenience used by lifespan / settings UI
# ---------------------------------------------------------------------------


async def auto_start_if_enabled(app_cfg: AppConfig) -> None:
    """Optional auto-start hook for the backend lifespan.

    No-op when ``cfg.exo.enabled`` or ``cfg.exo.auto_start`` is False.
    Skips silently — never raises — so the backend always boots.
    """

    cfg = app_cfg.exo
    if not (cfg.enabled and cfg.auto_start):
        return

    try:
        if is_running(cfg.api_port):
            logger.info("exo: already running on :%d, skipping auto-start", cfg.api_port)
            return
        logger.info("exo: auto-start enabled, bringing local cluster up")
        await aup_local(cfg)
    except Exception as exc:
        if is_prebuilt_mode(cfg) and _is_no_prebuilt_error(str(exc)):
            logger.warning(
                "exo auto-start skipped: no prebuilt runtime published yet for ref=%s. "
                "Switch to source mode in Settings → Cluster → Setup, or wait for a release.",
                cfg.repo_ref,
            )
        else:
            logger.exception("exo auto-start failed — continuing without exo")
        return

    for remote in cfg.remotes:
        if not remote.enabled:
            continue
        try:
            logger.info("exo: auto-start remote %s", remote.ssh_alias)
            await arun_remote(cfg, remote, _remote_subcmd_for("up", cfg), job_kind="up")
        except Exception:
            logger.exception("exo auto-start failed for remote %s", remote.ssh_alias)


def tail_log(max_lines: int = 200) -> list[str]:
    """Return the last ``max_lines`` of the local exo log file."""
    path = log_file()
    if not path.exists():
        return []
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            block = 65536
            data = b""
            while size > 0 and data.count(b"\n") <= max_lines:
                step = min(block, size)
                size -= step
                f.seek(size)
                data = f.read(step) + data
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        return lines[-max_lines:]
    except OSError:
        return []


__all__ = [
    "ExoCliError",
    "ExoJob",
    "aprovision",
    "astart_local",
    "astop_local",
    "afetch_status",
    "asmoke",
    "arun_remote",
    "aup_local",
    "adown_local",
    "auto_start_if_enabled",
    "cluster_status_to_dict",
    "get_job",
    "info_snapshot",
    "list_jobs",
    "tail_log",
    "_remote_subcmd_for",
    # Async preload with progress.
    "start_preload_job",
    "get_preload_job",
    "list_preload_jobs",
    "cancel_preload_job",
    "preload_job_view",
]


# ---------------------------------------------------------------------------
# Async preload jobs — same UX shape as the MLX downloads
# ---------------------------------------------------------------------------
#
# The existing ``_preload_model_sync`` is consumed by the deep agent
# (``backend/exo_tools.py``) and the legacy ``POST /api/exo/models/preload``
# endpoint, so we leave it untouched.  This block layers an async,
# progress-tracking flavour on top — same in-memory ``_jobs`` pattern
# the MLX route uses, with bytes/files/eta/rate fields driven by exo's
# own ``state.downloads`` event stream.
#
# Why poll ``/state`` instead of subscribing to events?  exo doesn't
# expose its append-only event log over HTTP yet; the assembled
# ``State.downloads`` map (per-node, per-model :class:`DownloadOngoing`
# / :class:`DownloadCompleted` records) is the consolidated view a
# polling consumer is meant to use.

_preload_jobs: dict[str, dict[str, Any]] = {}
_preload_jobs_lock = threading.Lock()
_preload_cancels: dict[str, threading.Event] = {}


def bytesToHuman(n: int | float) -> str:
    """Compact, human-readable byte count for log/UI strings."""
    if not n or n < 0:
        return "0 B"
    units = ("B", "KB", "MB", "GB", "TB")
    v = float(n)
    i = 0
    while v >= 1024 and i < len(units) - 1:
        v /= 1024.0
        i += 1
    return f"{v:.1f} {units[i]}" if i > 0 else f"{int(v)} {units[i]}"


def _bytes_value(v: Any) -> int:
    """Pull a byte count out of an exo ``Memory`` object or scalar.

    exo emits ``{"inBytes": N}`` (camelCase) on the wire; pydantic
    snake_case (``in_bytes``) is the python-only shape.  Accept both
    plus a bare int for safety.
    """
    if isinstance(v, dict):
        for k in ("inBytes", "in_bytes"):
            inner = v.get(k)
            if isinstance(inner, (int, float)) and inner >= 0:
                return int(inner)
        return 0
    if isinstance(v, (int, float)) and v >= 0:
        return int(v)
    return 0


def _download_record_model_id(rec_inner: dict[str, Any]) -> str | None:
    """Walk a download record (already unwrapped from its tagged variant)
    and return the model id buried under ``shardMetadata.<variant>.modelCard``.
    """
    sm = rec_inner.get("shardMetadata") or rec_inner.get("shard_metadata") or {}
    if not isinstance(sm, dict):
        return None
    # ``shardMetadata`` is itself tagged: PipelineShardMetadata / TensorShardMetadata.
    for variant_body in sm.values():
        if not isinstance(variant_body, dict):
            continue
        mc = variant_body.get("modelCard") or variant_body.get("model_card") or {}
        if isinstance(mc, dict):
            mid = mc.get("modelId") or mc.get("model_id")
            if isinstance(mid, str) and mid:
                return mid
    # Some older builds carry the id directly on the record.
    direct = rec_inner.get("modelId") or rec_inner.get("model_id")
    if isinstance(direct, str) and direct:
        return direct
    return None


def _walk_node_downloads(
    state: dict[str, Any], model_id: str
) -> dict[str, tuple[str, dict[str, Any]]]:
    """Return ``{node_id: (variant_name, record_body)}`` for ``model_id``.

    Each entry in ``state.downloads[node]`` is a single-key dict where
    the key is the variant tag (``DownloadPending`` / ``DownloadOngoing``
    / ``DownloadCompleted`` / ``DownloadFailed``) and the value is the
    record body.  We unwrap that here so the caller can branch on the
    variant cheaply.

    When a node carries multiple records for the same model (shouldn't
    happen — the ``apply.py`` loop dedupes — but we're defensive), the
    last one wins because that matches event-source semantics.
    """
    out: dict[str, tuple[str, dict[str, Any]]] = {}
    downloads = state.get("downloads") or {}
    if not isinstance(downloads, dict):
        return out
    for node_id, entries in downloads.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict) or not entry:
                continue
            # Tagged variant — single key.
            variant = next(iter(entry.keys()))
            body = entry.get(variant)
            if not isinstance(body, dict):
                continue
            if _download_record_model_id(body) == model_id:
                out[node_id] = (variant, body)
    return out


def _aggregate_progress(
    node_records: dict[str, tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Sum per-node bytes / files / speeds into a single cluster-wide view.

    * ``DownloadOngoing`` exposes ``downloadProgress`` (bytes + files
      + speed + eta).  We sum those across nodes.
    * ``DownloadPending`` / ``DownloadCompleted`` / ``DownloadFailed``
      only have summary ``downloaded`` / ``total`` Memory fields; we
      use those when available.
    """
    bytes_done = 0
    bytes_total = 0
    files_done = 0
    files_total = 0
    speed = 0.0
    eta_ms_max = 0
    variants: list[str] = []

    for variant, rec in node_records.values():
        variants.append(variant)
        dp = rec.get("downloadProgress") or rec.get("download_progress")
        if isinstance(dp, dict):
            bytes_done += _bytes_value(dp.get("downloaded"))
            bytes_total += _bytes_value(dp.get("total"))
            files_done += int(
                dp.get("completedFiles") or dp.get("completed_files") or 0
            )
            files_total += int(dp.get("totalFiles") or dp.get("total_files") or 0)
            try:
                speed += float(dp.get("speed") or 0.0)
            except (TypeError, ValueError):
                pass
            try:
                eta_ms = int(dp.get("etaMs") or dp.get("eta_ms") or 0)
                eta_ms_max = max(eta_ms_max, eta_ms)
            except (TypeError, ValueError):
                pass
        else:
            bytes_done += _bytes_value(rec.get("downloaded"))
            bytes_total += _bytes_value(rec.get("total"))

    return {
        "bytes_done": bytes_done,
        "bytes_total": bytes_total,
        "files_done": files_done,
        "files_total": files_total,
        "rate_bps": float(speed),
        "eta_seconds": int(eta_ms_max / 1000) if eta_ms_max > 0 else None,
        "variants": variants,
    }


def _instance_loaded_for(state: dict[str, Any], model_id: str) -> bool:
    for inst in (state.get("instances") or {}).values():
        if _instance_model_id(inst) == model_id:
            return True
    return False


def _new_preload_job(model_id: str, base_url: str, min_nodes: int) -> str:
    job_id = uuid.uuid4().hex
    cancel = threading.Event()
    with _preload_jobs_lock:
        _preload_jobs[job_id] = {
            "job_id": job_id,
            "model_id": model_id,
            "base_url": base_url,
            "min_nodes": int(min_nodes),
            "stage": "placing",        # placing | downloading | loading | done | error | cancelled
            "status": "running",       # mirrors stage but in MLX-job-style enum
            "message": "",
            "started_at": time.time(),
            "elapsed_seconds": 0.0,
            "bytes_done": 0,
            "bytes_total": 0,
            "files_done": 0,
            "files_total": 0,
            "rate_bps": 0.0,
            "eta_seconds": None,
            "nodes_active": 0,
            "replaced": 0,
        }
        _preload_cancels[job_id] = cancel
    return job_id


def _preload_worker(
    job_id: str,
    base_url: str,
    model_id: str,
    *,
    min_nodes: int,
    timeout: float,
    poll_interval: float = 1.5,
    sharding: Optional[str] = None,
    instance_meta: Optional[str] = None,
) -> None:
    """Run the preload synchronously while writing progress into ``_preload_jobs``.

    Spawned by :func:`start_preload_job`; this function is never called
    directly from request handlers.
    """
    base = exo_cli._strip_v1(base_url)
    started = time.monotonic()
    cancel = _preload_cancels.get(job_id)

    def _set(**kw: Any) -> None:
        with _preload_jobs_lock:
            j = _preload_jobs.get(job_id)
            if j is not None:
                j.update(kw)
                j["elapsed_seconds"] = round(time.monotonic() - started, 2)

    # ─── Step 1 — make the placement idempotent ────────────────────
    replaced = 0
    try:
        replaced = _delete_instances_for_model_sync(base, model_id)
    except Exception:  # noqa: BLE001
        pass
    _set(replaced=replaced)
    if cancel and cancel.is_set():
        _set(status="cancelled", stage="cancelled", message="Cancelled before placement")
        return

    # ─── Step 2 — POST /place_instance ─────────────────────────────
    payload: dict = {"model_id": model_id}
    if min_nodes and min_nodes > 1:
        payload["min_nodes"] = int(min_nodes)
    if sharding:
        payload["sharding"] = sharding
    if instance_meta:
        payload["instance_meta"] = instance_meta
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/place_instance",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=min(timeout, 30.0)) as resp:
            _ = json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
        except Exception:
            detail = ""
        _set(
            status="error",
            stage="error",
            message=f"HTTP {exc.code}: {exc.reason}{(' — ' + detail) if detail else ''}",
        )
        return
    except Exception as exc:
        _set(status="error", stage="error", message=str(exc))
        return

    _set(stage="downloading", message="Placement accepted; cluster is downloading shards")

    # ─── Step 3 — poll /state for download + load progress ─────────
    deadline = started + max(60.0, timeout)
    # Track when we first entered the "loading" stage so we can detect a
    # runner that started and then shut down (most commonly: OOM).
    loading_stage_entered_at: float | None = None
    # How long to wait after all runners show ShuttingDown before declaring
    # a load failure. 8 s lets a fast in-memory load complete without a
    # spurious error while still surfacing real OOM failures quickly.
    _RUNNER_SHUTDOWN_GRACE = 8.0

    while time.monotonic() < deadline:
        if cancel and cancel.is_set():
            try:
                _delete_instances_for_model_sync(base, model_id)
            except Exception:  # noqa: BLE001
                pass
            _set(status="cancelled", stage="cancelled", message="Cancelled by user")
            return

        try:
            state = exo_cli.http_get_json(f"{base}/state", timeout=5)
        except Exception as exc:  # noqa: BLE001
            # Transient — keep going and report the last error if we
            # eventually time out.
            _set(message=f"polling /state: {exc}")
            time.sleep(poll_interval)
            continue

        node_records = _walk_node_downloads(state, model_id)
        agg = _aggregate_progress(node_records)
        nodes_active = len(node_records)
        variants = {v for v, _ in node_records.values()}

        # If any node reports DownloadFailed we're done — bubble the
        # error up immediately rather than waiting for the timeout.
        if "DownloadFailed" in variants:
            failure_msg = ""
            for v, body in node_records.values():
                if v == "DownloadFailed":
                    failure_msg = (
                        body.get("errorMessage")
                        or body.get("error_message")
                        or "DownloadFailed"
                    )
                    break
            _set(
                status="error",
                stage="error",
                message=f"Cluster reported download failure: {failure_msg}",
                nodes_active=nodes_active,
            )
            return

        # ``Completed`` for every node means bytes are on disk → loading.
        all_completed = bool(node_records) and variants <= {
            "DownloadCompleted", "Completed",
        }
        # ``Ongoing`` on at least one node is "actively downloading".
        any_ongoing = any(
            v in {"DownloadOngoing", "Ongoing"} for v in variants
        )

        if _instance_loaded_for(state, model_id):
            _set(
                stage="done",
                status="done",
                message="Loaded",
                nodes_active=nodes_active,
                bytes_done=agg["bytes_total"] or agg["bytes_done"],
                bytes_total=agg["bytes_total"] or agg["bytes_done"],
                files_done=agg["files_total"] or agg["files_done"],
                files_total=agg["files_total"] or agg["files_done"],
                rate_bps=0.0,
                eta_seconds=0,
            )
            return

        # ── Runner-shutdown detection ────────────────────────────────
        # When all downloads are completed (or the model was already on disk),
        # EXO creates a runner to load the model into GPU memory.  If that
        # runner shuts down (RunnerShuttingDown) before an instance appears,
        # it almost always means the Mac ran out of free RAM.  Detect this
        # fast so the user gets an actionable error instead of a 30-min hang.
        runners: dict = state.get("runners") or {}
        if all_completed or loading_stage_entered_at is not None:
            if loading_stage_entered_at is None:
                loading_stage_entered_at = time.monotonic()
            if runners:
                all_runners_dying = all(
                    isinstance(v, dict) and all(
                        "ShuttingDown" in k or "Error" in k or "Stopped" in k
                        for k in v
                    )
                    for v in runners.values()
                )
                if all_runners_dying:
                    elapsed_loading = time.monotonic() - loading_stage_entered_at
                    if elapsed_loading >= _RUNNER_SHUTDOWN_GRACE:
                        _set(
                            status="error",
                            stage="error",
                            message=(
                                f"EXO runner shut down while loading "
                                f"{model_id!r} — the node likely ran out "
                                f"of free RAM.  Close other apps to free "
                                f"memory and try again."
                            ),
                            nodes_active=nodes_active,
                        )
                        return

        if all_completed:
            stage = "loading"
            message = "All shards downloaded; loading model into memory"
        elif any_ongoing:
            stage = "downloading"
            message = (
                f"Downloading {bytesToHuman(agg['bytes_done'])} / "
                f"{bytesToHuman(agg['bytes_total'])} across "
                f"{nodes_active} node{'' if nodes_active == 1 else 's'}"
            )
        elif node_records:
            stage = "downloading"  # still "downloading" stage UX-wise
            message = (
                f"Queued on {nodes_active} node"
                f"{'' if nodes_active == 1 else 's'} — waiting for "
                f"scheduler"
            )
        else:
            stage = "downloading"
            message = "Waiting for the cluster to pick up the placement…"

        _set(
            stage=stage,
            nodes_active=nodes_active,
            bytes_done=agg["bytes_done"],
            bytes_total=agg["bytes_total"],
            files_done=agg["files_done"],
            files_total=agg["files_total"],
            rate_bps=agg["rate_bps"],
            eta_seconds=agg["eta_seconds"],
            message=message,
        )
        time.sleep(poll_interval)

    _set(
        status="error",
        stage="error",
        message=f"Timed out waiting for {model_id!r} to finish loading",
    )


def start_preload_job(
    cfg: "ExoConfig",
    model_id: str,
    *,
    timeout: float = 1800.0,
    min_nodes: Optional[int] = None,
) -> dict[str, Any]:
    """Spawn a background thread that places + waits for a preload.

    ``timeout`` defaults to 30 minutes — frontier-class models can
    legitimately take that long over a 100 Mb home connection.  The
    user-visible cancel button calls :func:`cancel_preload_job` which
    sets the ``threading.Event`` checked at every poll iteration.

    Returns the public job view (same shape as ``GET /api/exo/preload/{id}``).
    """
    effective_min_nodes = (
        int(min_nodes) if min_nodes is not None else int(cfg.min_nodes or 1)
    )
    job_id = _new_preload_job(model_id, cfg.effective_base_url, effective_min_nodes)
    threading.Thread(
        target=_preload_worker,
        args=(job_id, cfg.effective_base_url, model_id),
        kwargs={
            "min_nodes": effective_min_nodes,
            "timeout": timeout,
            "sharding": cfg.sharding,
            "instance_meta": cfg.instance_meta,
        },
        name=f"exo-preload-{job_id[:8]}",
        daemon=True,
    ).start()
    return preload_job_view(job_id) or {"job_id": job_id, "model_id": model_id}


def get_preload_job(job_id: str) -> Optional[dict[str, Any]]:
    return preload_job_view(job_id)


def list_preload_jobs() -> list[dict[str, Any]]:
    with _preload_jobs_lock:
        items = [(jid, dict(j)) for jid, j in _preload_jobs.items()]
    items.sort(key=lambda kv: -float(kv[1].get("started_at") or 0))
    return [preload_job_view(jid) or {} for jid, _ in items]


def cancel_preload_job(job_id: str) -> bool:
    with _preload_jobs_lock:
        if job_id not in _preload_jobs:
            return False
        ev = _preload_cancels.get(job_id)
    if ev is not None:
        ev.set()
    return True


def preload_job_view(job_id: str) -> Optional[dict[str, Any]]:
    """Public, JSON-safe snapshot — same field naming as the MLX route
    so the frontend can reuse its progress-bar component shape.
    """
    with _preload_jobs_lock:
        j = _preload_jobs.get(job_id)
        if j is None:
            return None
        snap = dict(j)
    bytes_done = int(snap.get("bytes_done", 0) or 0)
    bytes_total = int(snap.get("bytes_total", 0) or 0)
    if bytes_total > 0 and bytes_done > bytes_total:
        bytes_total = bytes_done
    return {
        "job_id": snap.get("job_id", job_id),
        "model_id": snap.get("model_id", ""),
        "base_url": snap.get("base_url", ""),
        "min_nodes": int(snap.get("min_nodes", 1) or 1),
        "stage": snap.get("stage", "placing"),
        "status": snap.get("status", "running"),
        "message": snap.get("message", ""),
        "started_at": snap.get("started_at"),
        "elapsed_seconds": float(snap.get("elapsed_seconds", 0.0) or 0.0),
        "bytes_done": bytes_done,
        "bytes_total": bytes_total,
        "files_done": int(snap.get("files_done", 0) or 0),
        "files_total": int(snap.get("files_total", 0) or 0),
        "rate_bps": float(snap.get("rate_bps", 0.0) or 0.0),
        "eta_seconds": snap.get("eta_seconds"),
        "nodes_active": int(snap.get("nodes_active", 0) or 0),
        "replaced": int(snap.get("replaced", 0) or 0),
    }
