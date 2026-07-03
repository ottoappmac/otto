"""REST API for the local ``oMLX`` inference server.

Wraps :mod:`backend.omlx_provisioner`.  The setup wizard's oMLX install
screen and the Settings → LLM → oMLX section consume these endpoints;
the deep agent uses :mod:`backend.omlx_tools` which talks to the same
provisioner directly.

Endpoints
---------

``GET    /api/omlx``                   full info snapshot (detection, config)
``GET    /api/omlx/status``            live HTTP probe of /v1/models
``GET    /api/omlx/log``               tail of the spawn-fallback log
``POST   /api/omlx/install``           kick off ``brew tap`` + ``brew install``
``POST   /api/omlx/uninstall``         run ``brew uninstall`` (and stop service)
``POST   /api/omlx/start``             start the local server (brew services / spawn)
``POST   /api/omlx/stop``              stop the local server
``GET    /api/omlx/jobs``              list recent install/start/stop jobs
``GET    /api/omlx/jobs/{id}``         poll one job
``GET    /api/omlx/models/local``      scan HF hub cache, return cached MLX repos
``GET    /api/omlx/models/catalog``    hardware-scored curated model catalog
``POST   /api/omlx/models/load``       load a model into the running server
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend import omlx_provisioner as op
from backend.config import AppConfig, OmlxConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/omlx", tags=["omlx"])


async def _load_cfg() -> tuple[AppConfig, OmlxConfig]:
    app_cfg = await AppConfig.aload()
    return app_cfg, app_cfg.omlx


@router.get("")
async def omlx_info() -> dict:
    _, cfg = await _load_cfg()
    return op.info_snapshot(cfg)


@router.get("/status")
async def omlx_status() -> dict:
    _, cfg = await _load_cfg()
    return await op.afetch_status(cfg)


@router.get("/log")
async def omlx_log(lines: int = 200) -> dict:
    return {"lines": op.tail_log(max_lines=max(1, min(2000, lines)))}


@router.get("/version")
async def omlx_version() -> dict:
    """Return installed and latest-available oMLX versions.

    ``installed_version`` comes from ``omlx --version`` at detection time.
    ``latest_version`` is fetched from the GitHub releases API (cached 1 h).
    ``upgrade_available`` is True when both are known and latest > installed.
    ``homebrew`` indicates whether the brew CLI is present (required for
    in-app upgrade).
    """
    import re as _re

    _, cfg = await _load_cfg()

    det = await asyncio.to_thread(op.detect, cfg)
    latest = await op.aget_latest_release_version()

    installed = det.cli_version
    # Normalise "omlx 0.3.10" → "0.3.10" and "v0.4.2" → "0.4.2"
    if installed:
        m = _re.search(r"(\d+\.\d+[\w.]*)", installed)
        installed = m.group(1) if m else installed.strip().lstrip("v")

    upgrade_available = False
    if installed and latest:
        try:
            from packaging.version import Version
            upgrade_available = Version(latest) > Version(installed)
        except Exception:
            upgrade_available = latest != installed

    return {
        "installed_version": installed,
        "latest_version": latest,
        "upgrade_available": upgrade_available,
        "homebrew": det.homebrew,
    }


@router.post("/upgrade")
async def omlx_upgrade() -> dict:
    """Run ``brew upgrade omlx`` in a background job.

    Returns immediately with a ``job_id`` the caller polls via
    ``GET /api/omlx/jobs/{id}``.
    """
    _, cfg = await _load_cfg()
    job = await op.aupgrade(cfg)
    return {"job_id": job.id, **job.to_dict()}


@router.post("/install")
async def omlx_install() -> dict:
    _, cfg = await _load_cfg()
    job = await op.ainstall(cfg)
    return {"job_id": job.id, **job.to_dict()}


@router.post("/uninstall")
async def omlx_uninstall() -> dict:
    _, cfg = await _load_cfg()
    job = await op.auninstall(cfg)
    return {"job_id": job.id, **job.to_dict()}


@router.post("/start")
async def omlx_start() -> dict:
    _, cfg = await _load_cfg()
    job = await op.astart(cfg)
    return {"job_id": job.id, **job.to_dict()}


@router.post("/stop")
async def omlx_stop() -> dict:
    _, cfg = await _load_cfg()
    job = await op.astop(cfg)
    return {"job_id": job.id, **job.to_dict()}


@router.get("/jobs")
async def list_omlx_jobs() -> dict:
    return {"jobs": [j.to_dict() for j in op.list_jobs()]}


@router.get("/jobs/{job_id}")
async def get_omlx_job(job_id: str) -> dict:
    job = op.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job.to_dict()


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------


@router.get("/models/local")
async def omlx_local_models() -> dict:
    """List MLX models already present in the Hugging Face hub cache.

    Returns a list of repos ordered by size descending, each with:
    ``repo_id``, ``size_gb``, ``is_mlx`` (whether it looks like an
    MLX-format model from ``mlx-community``).

    Never raises — if the cache can't be scanned we return an empty list
    plus an ``error`` string so the UI can degrade gracefully.
    """
    app_cfg = await AppConfig.aload()
    mlx_cfg = app_cfg.llm.mlx

    def _scan() -> dict:
        try:
            from backend.mlx_hub_paths import resolve_hf_hub_cache_dir
            from huggingface_hub import scan_cache_dir

            hub = Path(resolve_hf_hub_cache_dir(mlx_cfg.hf_hub_cache))
            if not hub.is_dir():
                return {"models": [], "hub_cache_dir": str(hub), "error": None}

            info = scan_cache_dir(hub)
            models = []
            for repo in info.repos:
                rid = repo.repo_id
                size_gb = round((repo.size_on_disk or 0) / (1024 ** 3), 2)
                is_mlx = rid.startswith("mlx-community/")
                if not is_mlx:
                    # Heuristic: any repo whose cached snapshot contains a
                    # model.safetensors or *.safetensors (not .bin/.gguf)
                    # is likely MLX-compatible.
                    try:
                        snaps = list(repo.refs) or list(repo.revisions)
                        snap_dir = None
                        for snap in snaps:
                            p = getattr(snap, "snapshot_path", None)
                            if p and Path(str(p)).is_dir():
                                snap_dir = Path(str(p))
                                break
                        if snap_dir:
                            has_safetensors = any(snap_dir.glob("*.safetensors"))
                            has_bin = any(snap_dir.glob("*.bin"))
                            has_gguf = any(snap_dir.glob("*.gguf"))
                            is_mlx = has_safetensors and not has_bin and not has_gguf
                    except Exception:  # noqa: BLE001
                        pass
                models.append({"repo_id": rid, "size_gb": size_gb, "is_mlx": is_mlx})

            models.sort(key=lambda m: (0 if m["is_mlx"] else 1, -m["size_gb"]))
            return {"models": models, "hub_cache_dir": str(hub), "error": None}
        except Exception as exc:  # noqa: BLE001
            return {"models": [], "hub_cache_dir": "", "error": str(exc)}

    return await asyncio.to_thread(_scan)


@router.get("/models/catalog")
async def omlx_model_catalog(
    ctx_len: int = 8192,
    refresh: bool = False,
) -> dict:
    """Return the curated MLX catalog scored against the local hardware.

    Identical in spirit to ``GET /api/mlx/catalog`` but dedicated to the
    oMLX setup flow.  Each row carries ``fits`` (comfortable / tight /
    over), ``already_cached``, and the memory breakdown.
    """
    from backend.mlx_catalog import fetch_catalog, score_catalog
    from backend.setup_capabilities import probe_capabilities

    app_cfg = await AppConfig.aload()
    mlx_cfg = app_cfg.llm.mlx

    def _resolve_hub() -> str:
        from backend.mlx_hub_paths import resolve_hf_hub_cache_dir
        return resolve_hf_hub_cache_dir(mlx_cfg.hf_hub_cache)

    def _probe(hub: str) -> dict:
        return probe_capabilities(hub)

    def _scan_cached(hub: str) -> set:
        try:
            from huggingface_hub import scan_cache_dir as _scan
            return {r.repo_id for r in _scan(Path(hub)).repos}
        except Exception:  # noqa: BLE001
            return set()

    hub = await asyncio.to_thread(_resolve_hub)
    token = (mlx_cfg.hf_token or "").strip() or None

    caps, rows, cached_ids = await asyncio.gather(
        asyncio.to_thread(_probe, hub),
        fetch_catalog(token=token, force=refresh),
        asyncio.to_thread(_scan_cached, hub),
    )

    scored = score_catalog(
        rows,
        ram_gb=float(caps.get("ram_gb") or 0.0),
        wired_limit_gb=float(caps.get("wired_limit_gb") or 0.0),
        free_disk_gb=float(caps.get("free_disk_gb") or 0.0),
        ctx_len=ctx_len,
        cached_repo_ids=cached_ids,
    )
    return {"capabilities": caps, "models": scored}


@router.get("/models/search")
async def omlx_search_models(
    q: str,
    ctx_len: int = 8192,
    limit: int = 40,
) -> dict:
    """Live-search the HF Hub for ``mlx-community`` models matching ``q``.

    Complements ``GET /api/omlx/models/catalog`` (curated + top-downloads):
    lets the picker surface any model that falls outside the cached listing
    window.  Results are inferred rows scored against the local hardware, so
    they carry the same ``fits`` / ``already_cached`` fields as the catalog.

    Returns an empty list for a blank query — never raises on HF failures.
    """
    from backend.mlx_catalog import score_catalog, search_catalog
    from backend.setup_capabilities import probe_capabilities

    query = (q or "").strip()
    if not query:
        return {"models": [], "query": ""}

    app_cfg = await AppConfig.aload()
    mlx_cfg = app_cfg.llm.mlx

    def _resolve_hub() -> str:
        from backend.mlx_hub_paths import resolve_hf_hub_cache_dir
        return resolve_hf_hub_cache_dir(mlx_cfg.hf_hub_cache)

    def _probe(hub: str) -> dict:
        return probe_capabilities(hub)

    def _scan_cached(hub: str) -> set:
        try:
            from huggingface_hub import scan_cache_dir as _scan
            return {r.repo_id for r in _scan(Path(hub)).repos}
        except Exception:  # noqa: BLE001
            return set()

    hub = await asyncio.to_thread(_resolve_hub)
    token = (mlx_cfg.hf_token or "").strip() or None

    caps, rows, cached_ids = await asyncio.gather(
        asyncio.to_thread(_probe, hub),
        search_catalog(query, token=token, limit=max(1, min(100, limit))),
        asyncio.to_thread(_scan_cached, hub),
    )

    scored = score_catalog(
        rows,
        ram_gb=float(caps.get("ram_gb") or 0.0),
        wired_limit_gb=float(caps.get("wired_limit_gb") or 0.0),
        free_disk_gb=float(caps.get("free_disk_gb") or 0.0),
        ctx_len=ctx_len,
        cached_repo_ids=cached_ids,
    )
    return {"models": scored, "query": query}


# ---------------------------------------------------------------------------
# Cache / turbo-mode settings
# ---------------------------------------------------------------------------


@router.get("/cache")
async def omlx_get_cache() -> dict:
    """Return oMLX's current KV-cache / turbo-mode settings."""
    _, cfg = await _load_cfg()
    try:
        return await op.aget_cache_settings(cfg)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc


class _CacheRequest(BaseModel):
    cache_enabled: bool | None = None
    hot_cache_only: bool | None = None
    hot_cache_max_size: str | None = None
    ssd_cache_dir: str | None = None
    ssd_cache_max_size: str | None = None
    initial_cache_blocks: int | None = None
    max_context_window: int | None = None
    max_concurrent_requests: int | None = None


@router.put("/cache")
async def omlx_set_cache(req: _CacheRequest) -> dict:
    """Update oMLX's KV-cache / turbo-mode settings.

    Only supplied (non-null) fields are changed; omitted fields are left as-is.
    The running server is updated live via the admin API and settings are
    persisted to ``~/.omlx/settings.json``.
    """
    _, cfg = await _load_cfg()
    patch = {k: v for k, v in req.model_dump().items() if v is not None}
    try:
        return await op.aset_cache_settings(cfg, patch)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Live cache statistics + clear
# ---------------------------------------------------------------------------


@router.get("/cache-stats")
async def omlx_cache_stats() -> dict:
    """Return live oMLX cache performance metrics and on-disk footprint."""
    _, cfg = await _load_cfg()
    try:
        return await op.aget_cache_stats(cfg)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/cache-stats/clear-hot")
async def omlx_clear_hot_cache() -> dict:
    """Flush the in-memory (hot) KV page cache on the running oMLX server."""
    _, cfg = await _load_cfg()
    try:
        return await op.aclear_omlx_hot_cache(cfg)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/cache-stats/clear-ssd")
async def omlx_clear_ssd_cache() -> dict:
    """Flush the SSD cold-tier KV cache on the running oMLX server."""
    _, cfg = await _load_cfg()
    try:
        return await op.aclear_omlx_ssd_cache(cfg)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Model config (context window auto-fill)
# ---------------------------------------------------------------------------


@router.get("/model-config")
async def omlx_model_config(repo_id: str | None = None) -> dict:
    """Return key fields from the active (or specified) model's ``config.json``.

    Reads the local HuggingFace Hub cache — no network request is made.
    Returns ``max_context_window`` (the largest of the recognised fields) and
    the raw values so the UI can show the source field name.

    Query params
    ------------
    repo_id : str, optional
        HuggingFace repo id (e.g. ``mlx-community/Llama-3.2-3B-Instruct-4bit``).
        Defaults to the currently active model in settings.
    """
    import json as _json

    app_cfg = await AppConfig.aload()
    mlx_cfg = app_cfg.llm.mlx

    effective_repo = (repo_id or "").strip() or (app_cfg.omlx.model_name or "").strip() or mlx_cfg.hf_llm_model_id.strip()
    if not effective_repo:
        raise HTTPException(status_code=422, detail="No model selected")

    from backend.mlx_hub_paths import resolve_hf_hub_cache_dir
    from pathlib import Path as _Path

    hub = _Path(resolve_hf_hub_cache_dir(mlx_cfg.hf_hub_cache))

    # HF hub layout: models--{org}--{name}/snapshots/<hash>/config.json
    # We pick the most-recently-modified snapshot.
    folder_name = "models--" + effective_repo.replace("/", "--")
    snapshots_dir = hub / folder_name / "snapshots"

    config_path: _Path | None = None
    if snapshots_dir.is_dir():
        candidates = sorted(
            snapshots_dir.iterdir(),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for snap in candidates:
            candidate = snap / "config.json"
            if candidate.is_file():
                config_path = candidate
                break

    if config_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"config.json not found for '{effective_repo}' in {hub}",
        )

    try:
        data = _json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to read config.json: {exc}") from exc

    # Collect every field that could encode the context length, in priority order.
    ctx_fields = [
        "max_position_embeddings",
        "max_sequence_length",
        "context_length",
        "max_seq_len",
        "model_max_length",
    ]
    found: dict[str, int] = {}
    for field in ctx_fields:
        val = data.get(field)
        if isinstance(val, int) and val > 0:
            found[field] = val

    # rope_scaling may extend the base max_position_embeddings
    rope_factor: float = 1.0
    rope_scaling = data.get("rope_scaling")
    if isinstance(rope_scaling, dict):
        rope_factor = float(rope_scaling.get("factor") or 1.0) or 1.0

    max_context_window: int | None = None
    source_field: str | None = None
    if found:
        source_field = next(iter(found))
        raw = found[source_field]
        max_context_window = int(raw * rope_factor) if rope_factor > 1.0 else raw

    return {
        "repo_id": effective_repo,
        "max_context_window": max_context_window,
        "source_field": source_field,
        "rope_factor": rope_factor if rope_factor != 1.0 else None,
        "all_context_fields": found,
        "config_path": str(config_path),
    }


class _ModelRequest(BaseModel):
    model_id: str


@router.post("/models/unload")
async def omlx_unload_model(req: _ModelRequest) -> dict:
    """Unload a specific model from the running oMLX server.

    Calls ``DELETE /v1/models/<id>`` on the live server so weights are
    evicted from GPU RAM.  The model stays on disk.

    Returns the kicked-off ``OmlxJob`` so the caller can poll
    ``GET /api/omlx/jobs/{id}``.
    """
    if not req.model_id.strip():
        raise HTTPException(status_code=422, detail="model_id must not be empty")
    _, cfg = await _load_cfg()
    job = await op.aunload_model(cfg, req.model_id.strip())
    return {"job_id": job.id, **job.to_dict()}


@router.post("/models/load")
async def omlx_load_model(req: _ModelRequest) -> dict:
    """Load (or switch to) a specific model in the running oMLX server.

    Tries ``omlx load <model_id>`` first.  If that doesn't confirm
    reachability, stops the server and restarts it with
    ``omlx serve --model <model_id> --port <port>``.

    Returns the kicked-off ``OmlxJob`` so the caller can poll
    ``GET /api/omlx/jobs/{id}``.
    """
    if not req.model_id.strip():
        raise HTTPException(status_code=422, detail="model_id must not be empty")
    _, cfg = await _load_cfg()
    job = await op.aload_model(cfg, req.model_id.strip())
    return {"job_id": job.id, **job.to_dict()}
