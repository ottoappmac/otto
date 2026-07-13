"""Whisper MLX speech-to-text.

Uses mlx-whisper to transcribe 16 kHz mono int16 PCM audio to text,
entirely on-device.  Exposes two module-level callables used by VoiceManager:

  configure(model_id)                        — hot-swap the model
  transcribe(pcm_bytes, *, language) → str   — async, serialised by a lock
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000

# Strings the Whisper model commonly emits for near-silence segments.
_HALLUCINATIONS: frozenset[str] = frozenset({
    "", " ", ".", "you", "bye.", "bye!",
    "thank you.", "thanks for watching.", "thank you for watching.",
    "thanks for watching!", "thank you for watching!",
    "thanks so much.", "thank you so much.",
    "see you.", "see you soon.", "goodbye.", "goodbye!",
})


def _is_hallucination(text: str) -> bool:
    return text.strip().lower() in _HALLUCINATIONS


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_model_id: str = "mlx-community/whisper-large-v3-turbo"
_lock: Optional[asyncio.Lock] = None


def configure(model_id: str) -> None:
    """Update the model repo used for transcription."""
    global _model_id
    _model_id = model_id


def _get_lock() -> asyncio.Lock:
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


# ---------------------------------------------------------------------------
# Model availability / download (used to gate Live Capture on the model being
# present, and to surface download progress instead of a silent first-run hang)
# ---------------------------------------------------------------------------


def _hf_cache_dir() -> Path:
    """The HuggingFace hub cache dir mlx-whisper downloads models into."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE)
    except Exception:  # noqa: BLE001
        return Path.home() / ".cache" / "huggingface" / "hub"


def _repo_cache_dirname(model_id: str) -> str:
    """HF's on-disk repo dir name, e.g. ``models--mlx-community--whisper-...``."""
    return "models--" + model_id.replace("/", "--")


def is_model_ready(model_id: str) -> bool:
    """True if the whole model repo is already present in the local HF cache.

    Uses ``snapshot_download(local_files_only=True)`` which resolves the cached
    revision and raises if any file for it is missing — so a partial/interrupted
    download correctly reports *not* ready.
    """
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model_id, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001
        return False


def expected_total_bytes(model_id: str) -> Optional[int]:
    """Best-effort total download size (bytes) for a model repo, or None.

    Queries the Hub for per-file sizes so download progress can be shown as a
    percentage. Returns None when offline / the API call fails, in which case
    callers should fall back to an indeterminate progress indicator.
    """
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_id, files_metadata=True)
        total = sum(int(s.size) for s in (info.siblings or []) if getattr(s, "size", None))
        return total or None
    except Exception:  # noqa: BLE001
        return None


def cached_bytes(model_id: str) -> int:
    """Bytes currently on disk for a model (incl. in-flight ``*.incomplete``)."""
    blobs = _hf_cache_dir() / _repo_cache_dirname(model_id) / "blobs"
    try:
        return sum(f.stat().st_size for f in blobs.glob("*") if f.is_file())
    except Exception:  # noqa: BLE001
        return 0


def download_model(model_id: str) -> None:
    """Download a model repo into the local HF cache (blocking — run off-thread).

    Opts into the ``hf_transfer`` accelerator for a faster first-run download,
    scoping ``HF_HUB_ENABLE_HF_TRANSFER`` to this call so we don't change the
    global process env.  ``hf_transfer`` is feature-light (no graceful resume,
    terse errors), so on failure we retry once with it disabled — that path
    resumes from whatever ``*.incomplete`` blobs are already on disk.
    """
    import os

    from huggingface_hub import snapshot_download

    def _snapshot(use_hf_transfer: bool) -> None:
        saved_env = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER")
        if use_hf_transfer:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        try:
            snapshot_download(model_id)
        finally:
            if saved_env is None:
                os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
            else:
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = saved_env

    try:
        _snapshot(use_hf_transfer=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "stt download %s: hf_transfer path failed (%s); retrying without",
            model_id,
            exc,
        )
        _snapshot(use_hf_transfer=False)


# ---------------------------------------------------------------------------
# Sync inference (runs in a thread via asyncio.to_thread)
# ---------------------------------------------------------------------------


def _transcribe_sync(pcm_bytes: bytes, model_id: str, language: Optional[str]) -> str:
    """Convert raw 16 kHz int16 PCM bytes → text via mlx-whisper."""
    try:
        import numpy as np
        import mlx_whisper  # type: ignore[import]
    except ImportError as exc:
        # ``import mlx_whisper`` transitively imports numba, scipy.signal and
        # tiktoken (via its ``transcribe``/``timing`` modules). A failure here
        # is usually one of those transitive deps missing from a packaged
        # build, not mlx-whisper itself — surface the real module so the log
        # isn't misleading.
        missing = getattr(exc, "name", None) or "mlx-whisper"
        raise RuntimeError(
            f"speech-to-text unavailable — failed to import '{missing}' "
            f"({exc}). In a source checkout run: pip install mlx-whisper"
        ) from exc

    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32_768.0

    if audio.size == 0:
        return ""

    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=model_id,
        language=language or None,
        fp16=False,
        verbose=False,
    )
    return (result.get("text") or "").strip()


# ---------------------------------------------------------------------------
# Public async interface
# ---------------------------------------------------------------------------


async def transcribe(pcm_bytes: bytes, *, language: Optional[str] = None) -> str:
    """Transcribe raw 16 kHz int16 PCM bytes → text.

    Acquires a global async lock so concurrent calls are serialised
    (mlx-whisper is not thread-safe).
    """
    lock = _get_lock()
    async with lock:
        text = await asyncio.to_thread(_transcribe_sync, pcm_bytes, _model_id, language)
    if _is_hallucination(text):
        return ""
    return text
