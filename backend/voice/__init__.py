"""Voice subsystem: TTS (Kokoro MLX).

All heavy imports are deferred to the first use so the backend starts up
in < 100 ms regardless of whether voice models are downloaded.
"""
from __future__ import annotations

from backend.voice.voice_manager import VoiceManager

__all__ = ["VoiceManager"]
