"""MLX / Hugging Face Hub helpers: cache path, local model listing, downloads.

Also surfaces the **Setup Hub** endpoints used by the On-Device tab's
empty state and (future) first-run overlay:

* ``GET /api/mlx/capabilities`` — local hardware probe.
* ``GET /api/mlx/catalog``      — curated MLX models scored against
  the probe so the UI can label each one comfortable / tight / over.
* ``POST /api/mlx/download``    — Hub snapshot download, upgraded with
  ``allow_patterns`` filtering, configurable concurrency, optional
  ``hf_transfer`` acceleration, and **real per-file progress** wired
  through ``tqdm_class``.  Status endpoint surfaces bytes done /
  total / rate / ETA / current file.
* ``POST /api/mlx/download/{job_id}/cancel`` — cooperative cancel.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.config import AppConfig
from backend.mlx_hub_paths import default_hub_cache_relative_suffix, resolve_hf_hub_cache_dir

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mlx", tags=["mlx"])


class _DownloadCancelledError(Exception):
    """Raised from tqdm callbacks when the user requests a cancel.

    Using a custom exception (not KeyboardInterrupt) keeps it out of
    Python's signal-handling path while still propagating cleanly
    through snapshot_download's internal thread-pool machinery.
    """


def _resolve_hub_cache(cache_dir: str | None, cfg: AppConfig) -> str:
    if cache_dir and cache_dir.strip():
        return resolve_hf_hub_cache_dir(cache_dir.strip())
    return resolve_hf_hub_cache_dir(cfg.llm.mlx.hf_hub_cache)


class MlxDownloadRequest(BaseModel):
    repo_id: str = Field(..., min_length=1, description="Hugging Face model repo id, e.g. mlx-community/quantized-gemma-2b-it")
    label: str = ""
    hf_token: str = ""
    cache_dir: str | None = None
    # When True, restrict the download to safetensors/tokenizer/config files
    # (typical MLX repos ship everything else as duplicates of the same
    # weights in different formats).  Defaults to True; turn off only for
    # repos where the user explicitly wants the extra artefacts.
    safetensors_only: bool = True
    # Optional hint passed straight to ``snapshot_download``.  16 is a
    # reasonable default on a wired connection; the Hub default is 8.
    max_workers: int = 16
    # Opt-in ``hf_transfer`` accelerator for non-Xet repos.  Wrapped in a
    # try/except + retry-without on failure inside the worker.
    use_hf_transfer: bool = True


# ``_jobs`` keys are job ids; values are the structured progress
# record that ``GET /api/mlx/download/{job_id}`` returns.  Updated
# under ``_jobs_lock`` from both the worker thread and (rarely) the
# request thread when a cancel is issued.
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
# Per-job cancel flags.  Checked between files in the worker.
_cancels: dict[str, threading.Event] = {}


@router.get("/hub-default")
async def mlx_hub_default():
    """Return resolved default Hub cache path and ~/.cache-relative hints for the UI."""
    cache_root = str(Path.home() / ".cache")
    suffix = default_hub_cache_relative_suffix()
    return {
        "path": resolve_hf_hub_cache_dir(None),
        "cache_root": cache_root,
        "default_suffix": suffix,
        "default_relative": f"{suffix}",  # under cache_root
    }


@router.get("/local-models")
async def mlx_local_models(
    cache_dir: str | None = Query(None, description="Override hub cache path for this request only"),
):
    """List models present in the local Hub cache (``scan_cache_dir``).

    ``scan_cache_dir`` walks the HF filesystem and can be slow when a download
    is in progress (directory actively modified).  Offloaded to a thread so the
    event loop is never blocked.
    """
    cfg = await AppConfig.aload()
    hub = _resolve_hub_cache(cache_dir, cfg)
    path = Path(hub)
    if not path.is_dir():
        return {"hub_cache": hub, "models": [], "error": None}

    bookmark_labels = {b.repo_id: b.label for b in cfg.llm.mlx.mlx_bookmarks if b.label}

    def _scan() -> dict:
        try:
            from huggingface_hub import scan_cache_dir as _scan_cache_dir

            info = _scan_cache_dir(path)
            rows: list[dict[str, Any]] = []
            for repo in sorted(info.repos, key=lambda r: r.repo_id.lower()):
                label = bookmark_labels.get(repo.repo_id, "")
                name = f"{label} — {repo.repo_id}" if label else repo.repo_id
                rows.append(
                    {
                        "repo_id": repo.repo_id,
                        "name": name,
                        "size_mb": round(repo.size_on_disk / (1024 * 1024), 1) if repo.size_on_disk else 0,
                    },
                )
            return {"hub_cache": hub, "models": rows, "error": None}
        except Exception as exc:
            logger.exception("mlx_local_models failed")
            return {"hub_cache": hub, "models": [], "error": str(exc)}

    return await asyncio.to_thread(_scan)


# ---------------------------------------------------------------------------
# Setup hub: capabilities probe + scored MLX catalog
# ---------------------------------------------------------------------------


@router.get("/capabilities")
async def mlx_capabilities(
    cache_dir: str | None = Query(None, description="Override hub cache for this probe."),
):
    """Probe the local machine for the On-Device empty state.

    ``probe_capabilities`` does sysctl calls and a Hub-cache scan — offloaded
    to a thread so the event loop is never blocked.
    """
    from backend.setup_capabilities import probe_capabilities

    cfg = await AppConfig.aload()
    hub = _resolve_hub_cache(cache_dir, cfg)
    return await asyncio.to_thread(probe_capabilities, hub)


@router.get("/catalog")
async def mlx_catalog(
    ctx_len: int = Query(8192, ge=512, le=131072, description="Context length used to estimate KV cache."),
    kv_bits: int | None = Query(None, description="KV cache bits: 4, 8, or null for full precision."),
    comfortable_fraction: float = Query(0.5, ge=0.1, le=0.95),
    refresh: bool = Query(False, description="Force-refresh the live HF Hub overlay."),
    cache_dir: str | None = Query(None, description="Override hub cache for the probe."),
):
    """Return the curated MLX catalog scored against the local machine.

    Each row carries a ``fits`` field (``comfortable`` / ``tight`` /
    ``over``) plus the breakdown the UI shows in a tooltip.  Sorting
    pushes the comfortable rows to the top, then featured / popular /
    bigger.
    """
    from backend.mlx_catalog import fetch_catalog, is_enriching, score_catalog
    from backend.setup_capabilities import probe_capabilities

    cfg = await AppConfig.aload()
    hub = _resolve_hub_cache(cache_dir, cfg)

    # Both probe_capabilities (sysctl + disk) and scan_cache_dir (filesystem
    # walk) are blocking — run them concurrently in the thread pool so neither
    # stalls the event loop.
    def _probe() -> dict:
        return probe_capabilities(hub)

    def _is_snapshot_complete(hub_root: Path, repo_id: str) -> bool:
        """Return True when a repo has at least one snapshot with a valid
        config.json AND all .safetensors symlinks resolve to existing blobs.

        A partially-downloaded repo will have blobs missing or symlinks
        pointing to non-existent blob paths, so this returns False.
        """
        safe = "models--" + repo_id.replace("/", "--")
        snapshots_dir = hub_root / safe / "snapshots"
        if not snapshots_dir.is_dir():
            return False
        for snapshot in snapshots_dir.iterdir():
            if not snapshot.is_dir():
                continue
            config = snapshot / "config.json"
            # resolve() follows symlinks — fails if target missing
            try:
                if not config.resolve().exists():
                    continue
            except OSError:
                continue
            weights = list(snapshot.glob("*.safetensors"))
            if not weights:
                continue
            try:
                if all(w.resolve().exists() for w in weights):
                    return True
            except OSError:
                continue
        return False

    def _scan_cached() -> dict[str, bool]:
        """Return {repo_id: is_complete} for every repo in the Hub cache."""
        hub_path = Path(hub)
        try:
            from huggingface_hub import scan_cache_dir as _scan_cache_dir

            info = _scan_cache_dir(hub_path)
            return {
                r.repo_id: _is_snapshot_complete(hub_path, r.repo_id)
                for r in info.repos
            }
        except Exception:  # noqa: BLE001
            return {}

    token = (cfg.llm.mlx.hf_token or "").strip() or None
    caps, rows, cached_map = await asyncio.gather(
        asyncio.to_thread(_probe),
        fetch_catalog(token=token, force=refresh),
        asyncio.to_thread(_scan_cached),
    )

    scored = score_catalog(
        rows,
        ram_gb=float(caps.get("ram_gb") or 0.0),
        wired_limit_gb=float(caps.get("wired_limit_gb") or 0.0),
        free_disk_gb=float(caps.get("free_disk_gb") or 0.0),
        ctx_len=int(ctx_len),
        kv_bits=kv_bits,
        comfortable_fraction=float(comfortable_fraction),
        cached_map=cached_map,
    )

    counts = {"comfortable": 0, "tight": 0, "over": 0, "unknown": 0}
    for row in scored:
        counts[str(row.get("fits", "unknown"))] += 1

    return {
        "capabilities": caps,
        "ctx_len": ctx_len,
        "kv_bits": kv_bits,
        "comfortable_fraction": comfortable_fraction,
        "counts": counts,
        "models": scored,
        "enriching": is_enriching(),
    }


_DEFAULT_ALLOW_PATTERNS = [
    "*.safetensors",
    "*.json",
    "tokenizer*",
    "*.txt",
    "*.model",
    "*.tiktoken",
    "*.py",
    "README*",
]
_DEFAULT_IGNORE_PATTERNS = [
    "original/*",
    "consolidated.*",
    "*.bin",
    "*.gguf",
    "*.pth",
    "*.pt",
    "*.msgpack",
    "*.h5",
    "onnx/*",
]


def _make_progress_tqdm(job_id: str):
    """Custom ``tqdm`` subclass that pipes progress into ``_jobs[job_id]``.

    In ``huggingface_hub`` 1.x, ``snapshot_download`` instantiates the
    custom ``tqdm_class`` exactly twice:

    1. The shared **bytes** progress bar (``desc="Downloading (incomplete
       total...)"``).  Its ``total`` grows as each file's metadata is
       fetched and its ``n`` increments as bytes land on disk.  This is
       the source of truth for ``bytes_done`` / ``bytes_total``.
    2. The outer ``thread_map`` **files** bar (``desc="Fetching N files"``).
       Counts files completed.

    Per-file ``hf_hub_download`` progress is fed through an internal
    ``_AggregatedTqdm`` (not a ``tqdm`` subclass) so we don't see the
    file names — but we don't need to: the shared bytes bar gives us a
    smooth overall view.  ``set_description`` is hooked because the
    snapshot module switches the bytes bar to ``"Download complete"``
    when done.
    """
    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:  # huggingface_hub depends on tqdm; can't really happen
        return None

    class _ProgressTqdm(_tqdm):  # type: ignore[misc]
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            kwargs.setdefault("disable", False)
            # huggingface_hub ≥ 1.11 (Xet backend) passes a `name` kwarg that
            # tqdm.__init__ does not accept — strip it before calling super.
            kwargs.pop("name", None)
            self._desc_low = str(kwargs.get("desc", "") or "").lower()
            self._is_files_bar = self._desc_low.startswith("fetching")
            self._is_bytes_bar = "downloading" in self._desc_low or kwargs.get("unit") == "B"
            super().__init__(*args, **kwargs)
            self._job_id = job_id

        def _publish(self) -> None:
            try:
                with _jobs_lock:
                    job = _jobs.get(self._job_id)
                    if job is None:
                        return
                    if self._is_files_bar:
                        job["files_done"] = int(self.n)
                        job["files_total"] = int(self.total or 0)
                    elif self._is_bytes_bar:
                        job["bytes_done"] = int(self.n)
                        job["bytes_total"] = int(self.total or 0)
                        job["current_file"] = ""  # snapshot doesn't expose names
                        # Sliding-window rate / ETA so a slow chunk
                        # doesn't dominate the estimate.
                        now = time.time()
                        samples = job.setdefault("_samples", [])
                        samples.append((now, int(self.n)))
                        if len(samples) > 12:
                            del samples[0]
                        if len(samples) >= 2:
                            t0, b0 = samples[0]
                            t1, b1 = samples[-1]
                            dt = max(0.001, t1 - t0)
                            rate = max(0.0, (b1 - b0) / dt)
                            job["rate_bps"] = rate
                            total = int(self.total or job.get("bytes_total", 0) or 0)
                            if total > 0 and rate > 0:
                                remaining = max(0, total - int(self.n))
                                job["eta_seconds"] = int(remaining / rate)
            except Exception:  # noqa: BLE001
                pass

        def update(self, n: int | float = 1) -> bool:  # type: ignore[override]
            ret = super().update(n)
            self._publish()
            # Check cancel on every progress tick so the download stops
            # as soon as the next chunk lands rather than waiting for the
            # entire snapshot_download call to complete.
            ev = _cancels.get(self._job_id)
            if ev is not None and ev.is_set():
                raise _DownloadCancelledError("Cancelled by user")
            return ret

        def set_description(self, desc: str | None = None, refresh: bool = True) -> None:  # type: ignore[override]
            super().set_description(desc, refresh=refresh)
            if desc:
                self._desc_low = desc.lower()

        def refresh(self, *args, **kwargs):  # type: ignore[override]
            ret = super().refresh(*args, **kwargs)
            # The shared bar's ``total`` grows incrementally as files are
            # discovered; refresh() is called at each growth, so this is
            # how we keep ``bytes_total`` honest.
            self._publish()
            return ret

    return _ProgressTqdm


def _preflight_total_bytes(repo_id: str, token: str | None, allow_patterns: list[str] | None,
                           ignore_patterns: list[str] | None) -> int:
    """Best-effort total expected bytes for the download.

    Calls ``model_info(files_metadata=True)`` once and sums the
    ``size`` of siblings matching the allow/ignore filter.  Returns 0
    on any failure — the UI then shows an indeterminate bar.
    """
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import filter_repo_objects
    except Exception:  # noqa: BLE001
        return 0
    try:
        api = HfApi(token=token) if token else HfApi()
        info = api.model_info(repo_id, files_metadata=True)
        siblings = list(getattr(info, "siblings", []) or [])
        names = [s.rfilename for s in siblings]
        kept_names = set(
            filter_repo_objects(
                items=names,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
        )
        total = 0
        for s in siblings:
            if s.rfilename in kept_names and getattr(s, "size", None):
                total += int(s.size or 0)
        return total
    except Exception as exc:  # noqa: BLE001
        logger.debug("preflight model_info failed for %s: %s", repo_id, exc)
        return 0


@router.post("/download")
async def mlx_download(req: MlxDownloadRequest):
    """Start a Hub ``snapshot_download`` in a background thread.

    Wired up with:

    * ``allow_patterns`` / ``ignore_patterns`` — drops ``.bin`` /
      ``.gguf`` / ``original/*`` duplicates that most ``mlx-community``
      repos publish alongside the safetensors shards.
    * ``max_workers`` — defaults to 16 (Hub default is 8).
    * Optional ``hf_transfer`` acceleration (env var scoped to this
      call, retried without it on first failure).
    * Real progress via a ``tqdm_class`` that writes bytes / files /
      rate / ETA into ``_jobs[job_id]`` every update.
    """
    cfg = await AppConfig.aload()
    hub = _resolve_hub_cache(req.cache_dir, cfg)
    repo = req.repo_id.strip()
    if not repo:
        raise HTTPException(status_code=400, detail="repo_id is required")

    allow_patterns = _DEFAULT_ALLOW_PATTERNS if req.safetensors_only else None
    ignore_patterns = _DEFAULT_IGNORE_PATTERNS if req.safetensors_only else None

    job_id = uuid.uuid4().hex
    cancel_event = threading.Event()
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "pending",
            "message": "",
            "repo_id": repo,
            "hub_cache": hub,
            "started_at": time.time(),
            "bytes_done": 0,
            "bytes_total": 0,
            "files_done": 0,
            "files_total": 0,
            "current_file": "",
            "current_file_total": 0,
            "rate_bps": 0.0,
            "eta_seconds": None,
            "use_hf_transfer": bool(req.use_hf_transfer),
            "max_workers": int(req.max_workers),
            "_samples": [],
        }
        _cancels[job_id] = cancel_event

    token = (req.hf_token or cfg.llm.mlx.hf_token or "").strip() or None

    def run() -> None:
        with _jobs_lock:
            j = _jobs.get(job_id)
            if j is None:
                return
            j["status"] = "running"

        # Best-effort pre-flight so the UI bar has a denominator from
        # the very first paint.
        total_bytes = _preflight_total_bytes(repo, token, allow_patterns, ignore_patterns)
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["bytes_total"] = int(total_bytes)

        progress_tqdm = _make_progress_tqdm(job_id)

        def _do_download(use_hf_transfer: bool) -> None:
            from huggingface_hub import snapshot_download

            saved_env = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER")
            if use_hf_transfer:
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
            try:
                Path(hub).mkdir(parents=True, exist_ok=True)
                snapshot_download(
                    repo,
                    cache_dir=hub,
                    token=token,
                    local_files_only=False,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                    max_workers=max(1, int(req.max_workers)),
                    tqdm_class=progress_tqdm,
                )
            finally:
                if saved_env is None:
                    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
                else:
                    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = saved_env

        try:
            try:
                _do_download(req.use_hf_transfer)
            except _DownloadCancelledError:
                # User cancelled — don't retry the hf_transfer fallback path.
                raise
            except Exception as exc:  # noqa: BLE001
                # ``hf_transfer`` is intentionally feature-light (no
                # resume, terse errors) — retry once without it before
                # surfacing a hard error.
                if req.use_hf_transfer:
                    logger.warning(
                        "mlx download %s: hf_transfer path failed (%s); retrying without",
                        job_id, exc,
                    )
                    _do_download(False)
                else:
                    raise

            with _jobs_lock:
                if job_id in _jobs:
                    j = _jobs[job_id]
                    j["status"] = "done"
                    j["message"] = "Download complete"
                    # Reconcile counts: snapshot_download grows the
                    # ``total`` incrementally as files register, so it
                    # can briefly trail ``done``.  At completion both
                    # should equal the actual transferred bytes.
                    final_bytes = max(int(j.get("bytes_done", 0) or 0), int(j.get("bytes_total", 0) or 0))
                    j["bytes_done"] = final_bytes
                    j["bytes_total"] = final_bytes
                    if j.get("files_total"):
                        j["files_done"] = int(j["files_total"])
                    j["eta_seconds"] = 0
                    j["rate_bps"] = 0.0
                    j.pop("_samples", None)
        except _DownloadCancelledError:
            logger.info("mlx download %s cancelled by user", job_id)
            with _jobs_lock:
                if job_id in _jobs:
                    _jobs[job_id]["status"] = "cancelled"
                    _jobs[job_id]["message"] = "Cancelled by user"
        except Exception as exc:
            logger.exception("mlx download %s failed", job_id)
            with _jobs_lock:
                if job_id in _jobs:
                    _jobs[job_id]["status"] = "error"
                    _jobs[job_id]["message"] = str(exc)
        finally:
            _cancels.pop(job_id, None)

    threading.Thread(target=run, name=f"mlx-dl-{job_id[:8]}", daemon=True).start()
    return {"job_id": job_id, "repo_id": repo, "hub_cache": hub}


def _public_job_view(job_id: str, job: dict[str, Any]) -> dict[str, Any]:
    """Strip private ``_*`` keys before returning a job over the API.

    Also clamps ``bytes_done`` to ``bytes_total`` so the UI never sees
    a >100% bar — snapshot_download's shared total grows over a few
    seconds as files register, which can transiently make
    done > total.
    """
    bytes_done = int(job.get("bytes_done", 0) or 0)
    bytes_total = int(job.get("bytes_total", 0) or 0)
    if bytes_total > 0 and bytes_done > bytes_total:
        bytes_total = bytes_done
    return {
        "job_id": job_id,
        "status": job.get("status", "pending"),
        "message": job.get("message", ""),
        "repo_id": job.get("repo_id", ""),
        "hub_cache": job.get("hub_cache", ""),
        "started_at": job.get("started_at"),
        "bytes_done": bytes_done,
        "bytes_total": bytes_total,
        "files_done": int(job.get("files_done", 0) or 0),
        "files_total": int(job.get("files_total", 0) or 0),
        "current_file": job.get("current_file", ""),
        "current_file_total": int(job.get("current_file_total", 0) or 0),
        "rate_bps": float(job.get("rate_bps", 0.0) or 0.0),
        "eta_seconds": job.get("eta_seconds"),
        "use_hf_transfer": bool(job.get("use_hf_transfer", False)),
        "max_workers": int(job.get("max_workers", 0) or 0),
    }


@router.get("/download/{job_id}")
async def mlx_download_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return _public_job_view(job_id, job)


@router.get("/downloads")
async def mlx_download_list():
    """List all known jobs (running + recent terminal states).

    The threading lock is acquired in a worker thread so any transient
    contention with the download thread never stalls the event loop.
    """
    def _snapshot() -> list:
        with _jobs_lock:
            return [(jid, dict(j)) for jid, j in _jobs.items()]

    items = await asyncio.to_thread(_snapshot)
    items.sort(key=lambda kv: -float(kv[1].get("started_at") or 0))
    return {"jobs": [_public_job_view(jid, j) for jid, j in items]}


@router.post("/download/{job_id}/cancel")
async def mlx_download_cancel(job_id: str):
    """Cooperative cancel.

    Sets the cancel flag immediately and marks the job as "cancelling"
    so the UI can reflect this state at once.  The tqdm interrupt will
    stop the download at the next progress tick (end of current Xet
    chunk / HTTP chunk); files already on disk stay — a re-run resumes
    seamlessly because the Hub cache is content-addressed.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        if job.get("status") not in ("pending", "running"):
            # Already terminal or already cancelling — no-op.
            return {"status": job.get("status", "unknown"), "job_id": job_id}
        job["status"] = "cancelling"
        job["message"] = "Cancelling…"
        ev = _cancels.get(job_id)
    if ev is not None:
        ev.set()
    return {"status": "cancelling", "job_id": job_id}


# ── Turbo SSD cache admin ───────────────────────────────────────────────
#
# These endpoints let the Settings page surface how much disk the SSD
# cold-tier cache is using and clear it (globally or per model).  We do
# not require the turbo engine to currently be loaded — we walk the
# on-disk directory structure directly so the UI keeps working even
# when ``turbo_level != "ssd"`` but old entries still exist on disk.


def _ssd_root_for(cfg: AppConfig) -> Path:
    """Resolve the configured SSD cache root without creating it.

    Mirrors :func:`chat_models.mlx_turbo._ssd_cache.resolve_root`; kept
    duplicated to avoid importing the MLX module on non-Apple hosts.
    """
    override = (cfg.llm.mlx.turbo_ssd_dir or "").strip()
    if override:
        return Path(override).expanduser()
    from backend.config import get_app_data_dir
    return get_app_data_dir() / "kv_cache"


def _read_index_entries(root: Path) -> list[dict[str, Any]]:
    """Return the index.json entries at *root*, or [] if unreadable."""
    import json
    index_path = root / "index.json"
    if not index_path.exists():
        return []
    try:
        data = json.loads(index_path.read_text("utf-8"))
        return list(data.get("entries", []))
    except (OSError, json.JSONDecodeError):
        return []


def _disk_scan(root: Path) -> list[dict[str, Any]]:
    """Fallback: scan the tree when no usable index.json is available.

    Still buckets by model directory so the UI can show per-model
    rows.  Used after a clean install (no index yet) or when the
    index file is corrupted.
    """
    results: list[dict[str, Any]] = []
    if not root.is_dir():
        return results
    for path in root.rglob("*.safetensors"):
        try:
            rel = path.relative_to(root).parts
        except ValueError:
            continue
        # Expected layout: <model_slug>/v=.../kv=.../<shard>/<file>
        if not rel:
            continue
        model_slug = rel[0]
        try:
            size = path.stat().st_size
        except OSError:
            continue
        results.append({
            "model_slug": model_slug,
            "path": str(path),
            "size": size,
        })
    return results


@router.get("/turbo/ssd-cache")
async def mlx_turbo_ssd_cache_info():
    """Summarise on-disk KV-cache usage for the Settings UI."""
    cfg = await AppConfig.aload()
    root = _ssd_root_for(cfg)
    entries = _read_index_entries(root)

    per_model: dict[str, dict[str, Any]] = {}
    total = 0

    if entries:
        # Index-driven path — matches the live runtime's view of the cache.
        for e in entries:
            size = int(e.get("size", 0))
            total += size
            # ``model_dir`` stored in the index is an absolute path;
            # strip the root prefix so the UI can show friendly slugs.
            model_dir = e.get("model_dir", "")
            try:
                slug = Path(model_dir).relative_to(root).parts[0]
            except (ValueError, IndexError):
                slug = Path(model_dir).name or "unknown"
            bucket = per_model.setdefault(slug, {"entries": 0, "size_bytes": 0})
            bucket["entries"] += 1
            bucket["size_bytes"] += size
    else:
        # No index yet — fall back to a direct disk scan.
        for row in _disk_scan(root):
            slug = row["model_slug"]
            bucket = per_model.setdefault(slug, {"entries": 0, "size_bytes": 0})
            bucket["entries"] += 1
            bucket["size_bytes"] += row["size"]
            total += row["size"]

    models = [
        {
            "model_slug": slug,
            "entries": info["entries"],
            "size_bytes": info["size_bytes"],
            "size_gb": round(info["size_bytes"] / (1024 ** 3), 3),
        }
        for slug, info in sorted(per_model.items())
    ]

    return {
        "root": str(root),
        "exists": root.exists(),
        "entries": sum(m["entries"] for m in models),
        "total_bytes": total,
        "total_gb": round(total / (1024 ** 3), 3),
        "max_gb": int(cfg.llm.mlx.turbo_ssd_max_gb),
        "models": models,
    }


@router.delete("/turbo/ssd-cache")
async def mlx_turbo_ssd_cache_clear(
    model: str | None = Query(
        None,
        description=(
            "Optional model slug (directory name under the cache root). "
            "When omitted the entire cache is wiped."
        ),
    ),
):
    """Delete SSD cache files (global or per model).

    We clear the live index in memory *only if* the SSD store for the
    active chat is currently instantiated — otherwise we just wipe the
    on-disk files and let the next chat rebuild its index from scratch
    on init.  This avoids importing the MLX stack eagerly just to clear
    the cache on a non-Apple host.
    """
    import shutil

    cfg = await AppConfig.aload()
    root = _ssd_root_for(cfg)
    if not root.exists():
        return {"root": str(root), "removed_files": 0, "scope": "missing"}

    target = root if model in (None, "") else (root / model)
    if not target.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No SSD cache entries found for model slug {model!r}.",
        )

    removed = 0
    if target == root:
        # Global wipe: drop the index and all model subdirectories but
        # leave the root itself so subsequent writes don't race on
        # directory creation.
        index_path = root / "index.json"
        if index_path.exists():
            try:
                index_path.unlink()
            except OSError:
                pass
        for child in list(root.iterdir()):
            if child.is_dir():
                removed += sum(1 for _ in child.rglob("*.safetensors"))
                shutil.rmtree(child, ignore_errors=True)
        scope = "global"
    else:
        removed = sum(1 for _ in target.rglob("*.safetensors"))
        shutil.rmtree(target, ignore_errors=True)
        # Global index.json still references the removed files; drop the
        # stale entries so the next stats call doesn't report ghosts.
        import json
        index_path = root / "index.json"
        if index_path.exists():
            try:
                data = json.loads(index_path.read_text("utf-8"))
                target_str = str(target)
                data["entries"] = [
                    e for e in data.get("entries", [])
                    if not str(e.get("model_dir", "")).startswith(target_str)
                ]
                index_path.write_text(json.dumps(data), encoding="utf-8")
            except (OSError, json.JSONDecodeError):
                pass
        scope = f"model:{model}"

    logger.info(
        "SSD cache clear (%s) at %s — removed %d files",
        scope, target, removed,
    )
    return {
        "root": str(root),
        "scope": scope,
        "removed_files": removed,
    }


def _ram_stats_macos() -> dict[str, object]:
    """Return Activity Monitor-compatible memory stats via sysctl.

    Activity Monitor defines "Used" as App Memory + Wired + Compressed,
    NOT as total − free.  macOS aggressively fills the remaining pages with
    disk caches (speculative), so ``total − free`` is almost always near
    100% and meaningless.

    Pressure levels from ``vm.memory_pressure``:
        4 = Normal (green in Activity Monitor)
        2 = Warning (yellow)
        1 = Critical (red)
    """
    from backend.setup_capabilities import _sysctl  # noqa: PLC0415

    try:
        page_size = int(_sysctl("hw.pagesize") or "4096")
    except (ValueError, TypeError):
        page_size = 4096

    def _pages_gb(key: str) -> float:
        try:
            return round(int(_sysctl(key) or "0") * page_size / (1024 ** 3), 2)
        except (ValueError, TypeError):
            return 0.0

    app_gb = _pages_gb("vm.page_active_count")          # "App Memory"
    wired_gb = _pages_gb("vm.page_wire_count")           # "Wired Memory"
    compressed_gb = _pages_gb("vm.compressor_page_count")  # "Compressed"
    free_gb = _pages_gb("vm.page_free_count")            # truly free
    # Speculative pages are prefetched file caches — macOS treats them as
    # available, so Activity Monitor counts them in the "free" bucket.
    speculative_gb = _pages_gb("vm.page_speculative_count")

    used_gb = round(app_gb + wired_gb + compressed_gb, 2)

    try:
        pressure_val = int(_sysctl("vm.memory_pressure") or "4")
    except (ValueError, TypeError):
        pressure_val = 4

    if pressure_val == 1:
        pressure = "critical"
    elif pressure_val == 2:
        pressure = "warning"
    else:
        pressure = "normal"

    return {
        "used_gb": used_gb,
        "app_gb": app_gb,
        "wired_gb": wired_gb,
        "compressed_gb": compressed_gb,
        "free_gb": free_gb,
        "speculative_gb": speculative_gb,
        "memory_pressure": pressure,
    }


@router.get("/live-stats")
async def mlx_live_stats():
    """Return a lightweight point-in-time snapshot of GPU and RAM usage.

    Designed for polling (every ~8 s) — all reads are cheap in-memory or
    sysctl kernel calls with no filesystem or GPU work.

    Memory fields mirror Activity Monitor:
    * ``ram_used_gb``        — App Memory + Wired + Compressed (the
      definition Activity Monitor uses; ignores disk-cache speculative pages)
    * ``ram_app_gb``         — active app memory
    * ``ram_wired_gb``       — wired (kernel + GPU locked) memory
    * ``ram_compressed_gb``  — compressed memory
    * ``ram_free_gb``        — truly free pages (usually small on macOS)
    * ``memory_pressure``    — "normal" | "warning" | "critical"

    GPU fields:
    * ``active_gpu_mem_gb``  — bytes held by the MLX Metal allocator
    * ``gpu_limit_gb``       — Apple-Silicon wired-memory ceiling
    """
    import platform as _platform

    from backend.setup_capabilities import (  # noqa: PLC0415
        _ram_bytes,
        _wired_limit_gb_macos,
    )

    sysname = _platform.system()
    arch = _platform.machine()
    apple = sysname == "Darwin" and arch == "arm64"

    ram_bytes = _ram_bytes()
    ram_gb = round(ram_bytes / (1024 ** 3), 2) if ram_bytes else 0.0
    gpu_limit_gb = _wired_limit_gb_macos(ram_gb) if apple else ram_gb

    # Activity Monitor-style breakdown (macOS only; zeroed on other platforms).
    mem = _ram_stats_macos() if sysname == "Darwin" else {
        "used_gb": 0.0, "app_gb": 0.0, "wired_gb": 0.0,
        "compressed_gb": 0.0, "free_gb": 0.0, "speculative_gb": 0.0,
        "memory_pressure": "normal",
    }

    # Active MLX GPU memory — zero if MLX is not loaded.
    active_gpu_mem_gb = 0.0
    try:
        import mlx.core as mx  # type: ignore[import-untyped]  # noqa: PLC0415

        active_gpu_mem_gb = round(mx.get_active_memory() / (1024 ** 3), 3)
    except Exception:  # noqa: BLE001
        pass

    return {
        "apple_silicon": apple,
        "active_gpu_mem_gb": active_gpu_mem_gb,
        "gpu_limit_gb": gpu_limit_gb,
        "ram_gb": ram_gb,
        "ram_used_gb": mem["used_gb"],
        "ram_app_gb": mem["app_gb"],
        "ram_wired_gb": mem["wired_gb"],
        "ram_compressed_gb": mem["compressed_gb"],
        "ram_free_gb": mem["free_gb"],
        "memory_pressure": mem["memory_pressure"],
    }


@router.post("/unload")
async def mlx_unload() -> dict:
    """Evict every cached in-process MLX model and release Metal GPU memory.

    The dominant cost of an MLX provider is the model weights resident in
    unified Metal memory (a 4-bit 30B model is ~17 GB).  Those weights are
    kept alive by *several* independent strong references, and unless **all**
    of them are dropped the memory is never returned to the OS:

    1. ``backend.setup_chat._setup_model`` — the lazily-built chat model.
    2. ``chat_models.mlx._shared._LOADED_MODELS`` — the process-wide
       ``(model, tokenizer, draft) `` weight cache shared by every
       ``ChatMLXText`` instance.  **This is the cache the previous version
       of this handler forgot to clear, which is why memory was never
       released on a provider switch.**
    3. ``chat_models.mlx_turbo._registry._SINGLETONS`` — cached turbo chat
       instances (each holds its own ``self._model`` reference).
    4. Live ``Session.graph`` objects that embed a ``ChatMLXText`` — these
       are dropped separately when ``refresh_tools`` rebuilds the session
       graphs onto the new (non-MLX) provider after the settings save.

    Order matters: drop the Python references *first*, then ``gc.collect()``
    so the large tensors are finalised, then ``mx.clear_cache()`` so the
    Metal allocator actually returns the freed pool to the OS.

    Returns a small report so the caller/UI can confirm what was evicted.
    Idempotent — safe to call when nothing is loaded.
    """
    import gc

    # 1. Clear setup_chat module reference.
    try:
        import backend.setup_chat as _sc  # noqa: PLC0415

        _sc._setup_model = None  # type: ignore[attr-defined]
        _sc._setup_model_load_attempted = False  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    # 2. Evict the process-wide MLX weight cache (the real holder of the
    #    GPU weights).  Without this, every other step is futile.
    evicted_models = 0
    try:
        from chat_models.mlx._shared import evict_all_mlx_models  # noqa: PLC0415

        evicted_models = evict_all_mlx_models()
    except Exception:  # noqa: BLE001
        logger.debug("mlx_unload: could not evict _LOADED_MODELS", exc_info=True)

    # 3. Evict cached turbo singletons (each holds its own model ref).
    evicted_turbo = 0
    try:
        from chat_models.mlx_turbo import _registry as _turbo_reg  # noqa: PLC0415

        evicted_turbo = _turbo_reg.size()
        _turbo_reg.evict_all()
    except Exception:  # noqa: BLE001
        logger.debug("mlx_unload: could not evict turbo singletons", exc_info=True)

    # 4. Force GC before releasing Metal so Python finalises tensor refs.
    gc.collect()

    # 5. Release Metal allocator cache (prefer mx.clear_cache; fall back to
    #    deprecated mx.metal.clear_cache for older mlx versions).
    metal_cleared = False
    active_gb_before = active_gb_after = None
    try:
        import mlx.core as mx  # type: ignore[import-untyped]  # noqa: PLC0415

        try:
            active_gb_before = mx.get_active_memory() / (1024**3)
        except Exception:  # noqa: BLE001
            pass
        # Serialise against any in-flight generation/warmup — clearing the
        # Metal allocator pool concurrently with a stream_generate can abort
        # the whole process with a command-buffer OOM.  MLX_GEN_LOCK is the
        # same lock every generation path holds.
        from chat_models.mlx._shared import MLX_GEN_LOCK  # noqa: PLC0415

        with MLX_GEN_LOCK:
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            else:
                mx.metal.clear_cache()  # type: ignore[attr-defined]
        metal_cleared = True
        try:
            active_gb_after = mx.get_active_memory() / (1024**3)
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass

    # 6. Second GC pass after Metal clear.
    gc.collect()

    logger.info(
        "MLX unloaded: evicted %d cached model(s), %d turbo singleton(s); "
        "metal_cleared=%s active_gpu_mem %.2f→%.2f GB",
        evicted_models,
        evicted_turbo,
        metal_cleared,
        active_gb_before if active_gb_before is not None else float("nan"),
        active_gb_after if active_gb_after is not None else float("nan"),
    )
    return {
        "status": "unloaded",
        "metal_cleared": metal_cleared,
        "evicted_models": evicted_models,
        "evicted_turbo_singletons": evicted_turbo,
        "active_gpu_mem_gb_before": active_gb_before,
        "active_gpu_mem_gb_after": active_gb_after,
    }
