"""Whisper MLX speech-to-text.

Uses mlx-whisper to transcribe 16 kHz mono int16 PCM audio to text,
entirely on-device.  Exposes two module-level callables used by VoiceManager:

  configure(model_id)                        — hot-swap the model
  transcribe(pcm_bytes, *, language) → str   — async, serialised by a lock
"""

from __future__ import annotations

import asyncio
import logging
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
