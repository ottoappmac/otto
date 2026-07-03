"""API routes for the semantic embedding index."""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

from backend.config import AppConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/embeddings", tags=["embeddings"])


# ---------------------------------------------------------------------------
# Model download state — one background download at a time
# ---------------------------------------------------------------------------

@dataclass
class _DownloadState:
    downloading: bool = False
    bytes_downloaded: int = 0
    total_bytes: int = 0       # 0 until we can query the repo metadata
    error: str | None = None
    done: bool = False


_dl_state = _DownloadState()
_dl_lock = threading.Lock()


def _model_cache_dir(model_name: str) -> Path:
    """Return the huggingface_hub cache directory for a model repo."""
    slug = model_name.replace("/", "--")
    return Path.home() / ".cache" / "huggingface" / "hub" / f"models--{slug}"


def _is_model_cached(model_name: str) -> bool:
    """True when at least one snapshot is present in the HF cache."""
    cache_dir = _model_cache_dir(model_name)
    snapshots = cache_dir / "snapshots"
    if not snapshots.exists():
        return False
    # A complete snapshot has at least one subdirectory with actual model files
    for snap in snapshots.iterdir():
        if snap.is_dir() and any(snap.iterdir()):
            return True
    return False


def _cache_bytes(model_name: str) -> int:
    """Sum of all bytes currently in the model's cache directory."""
    cache_dir = _model_cache_dir(model_name)
    if not cache_dir.exists():
        return 0
    return sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())


def _repo_total_bytes(model_name: str) -> int:
    """Return an approximate download size for known models.

    We avoid calling the HF Hub API (which can 401 for gated repos or simply
    require a token) and instead keep a small lookup table.  Unknown models
    return 0 and the UI renders an indeterminate progress bar.
    """
    _KNOWN_SIZES: dict[str, int] = {
        # model_name                                  approximate bytes
        "sentence-transformers/all-MiniLM-L6-v2":     91_000_000,   # ~87 MB
        "sentence-transformers/all-mpnet-base-v2":    420_000_000,  # ~400 MB
        "BAAI/bge-small-en-v1.5":                    135_000_000,  # ~129 MB
        "BAAI/bge-base-en-v1.5":                     440_000_000,  # ~420 MB
        "BAAI/bge-large-en-v1.5":                   1_340_000_000, # ~1.3 GB
        "mlx-community/nomic-embed-text-v1.5-mlx":   280_000_000,  # ~267 MB
    }
    return _KNOWN_SIZES.get(model_name, 0)


def _run_download(model_name: str) -> None:
    """Background thread: download model + update _dl_state."""
    global _dl_state
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import]

        # Fetch total size before starting so the UI can show a denominator
        total = _repo_total_bytes(model_name)
        with _dl_lock:
            _dl_state.total_bytes = total

        snapshot_download(model_name)

        with _dl_lock:
            _dl_state.bytes_downloaded = _cache_bytes(model_name)
            _dl_state.done = True
            _dl_state.downloading = False
    except Exception as exc:
        with _dl_lock:
            _dl_state.error = str(exc)
            _dl_state.done = True
            _dl_state.downloading = False


@router.get("/model-status")
async def get_model_status() -> dict[str, Any]:
    """Return embedding model install status and live download progress.

    Fields:
        installed (bool): True when the model snapshot is fully cached.
        model_name (str): HuggingFace repo ID of the configured model.
        downloading (bool): True when a download is in progress.
        bytes_downloaded (int): Bytes written to the HF cache so far.
        total_bytes (int): Expected total bytes (0 if unknown).
        error (str | None): Last download error, if any.
    """
    cfg = await AppConfig.aload()
    model_name = cfg.memory.embedding.model_name

    installed = _is_model_cached(model_name)

    with _dl_lock:
        downloading = _dl_state.downloading
        # While downloading, refresh byte count from filesystem
        if downloading:
            _dl_state.bytes_downloaded = _cache_bytes(model_name)
        return {
            "installed": installed,
            "model_name": model_name,
            "downloading": downloading,
            "bytes_downloaded": _dl_state.bytes_downloaded if not installed else _cache_bytes(model_name),
            "total_bytes": _dl_state.total_bytes,
            "error": _dl_state.error,
        }


@router.post("/model-download")
async def start_model_download() -> JSONResponse:
    """Trigger background download of the embedding model.

    Safe to call multiple times — a no-op if already installed or downloading.
    """
    global _dl_state
    cfg = await AppConfig.aload()
    model_name = cfg.memory.embedding.model_name

    if _is_model_cached(model_name):
        return JSONResponse({"status": "already_installed", "model_name": model_name})

    with _dl_lock:
        if _dl_state.downloading:
            return JSONResponse({"status": "already_downloading", "model_name": model_name})
        _dl_state = _DownloadState(downloading=True)

    thread = threading.Thread(target=_run_download, args=(model_name,), daemon=True)
    thread.start()

    return JSONResponse({"status": "started", "model_name": model_name})


@router.get("/status")
async def get_embedding_status() -> dict[str, Any]:
    """Return index status: total chunks and per-source breakdown."""
    cfg = await AppConfig.aload()
    if not cfg.memory.embedding_enabled:
        return {"enabled": False, "total_chunks": 0, "sources": []}

    try:
        from backend.embedding_index import get_embedding_index
        idx = await get_embedding_index()
        status = await idx.get_status()
        return {"enabled": True, **status}
    except Exception as exc:
        logger.debug("Embedding status error: %s", exc, exc_info=True)
        return {"enabled": True, "total_chunks": 0, "sources": [], "error": str(exc)}


@router.post("/index")
async def index_path(body: dict = Body(...)) -> JSONResponse:
    """Index a user-pinned file or directory.

    Body: ``{"path": "/absolute/path/to/file-or-dir"}``

    The path must be absolute and must exist on the local filesystem.
    Indexing runs in the background; the endpoint returns immediately.
    """
    cfg = await AppConfig.aload()
    if not cfg.memory.embedding_enabled:
        return JSONResponse(
            status_code=400,
            content={"error": "Semantic search is disabled. Enable Memory first."},
        )

    raw_path = body.get("path", "").strip()
    if not raw_path:
        return JSONResponse(status_code=400, content={"error": "path is required"})

    p = Path(raw_path).expanduser().resolve()
    if not p.exists():
        return JSONResponse(status_code=404, content={"error": f"Path not found: {raw_path}"})

    try:
        from backend.embedding_index import get_embedding_index
        idx = await get_embedding_index()

        if p.is_dir():
            asyncio.create_task(idx.index_directory(p))
            return JSONResponse(
                status_code=202,
                content={"status": "indexing_started", "path": str(p), "type": "directory"},
            )
        else:
            asyncio.create_task(idx.index_file(p))
            return JSONResponse(
                status_code=202,
                content={"status": "indexing_started", "path": str(p), "type": "file"},
            )
    except Exception as exc:
        logger.warning("Embedding index trigger failed for %s: %s", raw_path, exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})


@router.delete("/source")
async def remove_source(body: dict = Body(...)) -> JSONResponse:
    """Remove all chunks for a given source path from the index."""
    cfg = await AppConfig.aload()
    if not cfg.memory.embedding_enabled:
        return JSONResponse(status_code=400, content={"error": "Semantic search is disabled."})

    source_path = body.get("path", "").strip()
    if not source_path:
        return JSONResponse(status_code=400, content={"error": "path is required"})

    try:
        from backend.embedding_index import get_embedding_index
        idx = await get_embedding_index()
        removed = await idx.remove_source(source_path)
        return JSONResponse(
            status_code=200,
            content={"status": "removed", "chunks_removed": removed},
        )
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@router.post("/reindex/memory")
async def reindex_memory() -> JSONResponse:
    """Manually trigger re-indexing of all memory topic files."""
    cfg = await AppConfig.aload()
    if not cfg.memory.embedding_enabled:
        return JSONResponse(status_code=400, content={"error": "Semantic search is disabled."})

    try:
        from backend.embedding_index import get_embedding_index
        idx = await get_embedding_index()
        asyncio.create_task(idx.index_memory())
        return JSONResponse(status_code=202, content={"status": "reindex_started"})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
