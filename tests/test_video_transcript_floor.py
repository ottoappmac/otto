"""Tests for the video-only minimum-length floor on Whisper transcripts.

``stt._is_hallucination`` deliberately lets a lone word through, because it is
shared with the voice pipeline where "Stop" and "Yes" are real commands.  That
leaves single-word inventions ("All" on a clip of traffic noise) reaching the
model as the clip's audio transcript, so the video path applies its own floor.
"""

from __future__ import annotations

import pytest

from backend.video import ingest


@pytest.fixture
def fake_audio(monkeypatch):
    """Stub out ffmpeg extraction so only the transcript logic is exercised."""
    monkeypatch.setattr(ingest, "extract_audio_pcm16", lambda *a, **k: b"\x00\x01" * 100)

    def _set(text: str):
        class _Stt:
            @staticmethod
            async def transcribe(pcm, **kwargs):
                return text

        import backend.voice
        monkeypatch.setattr(backend.voice, "stt", _Stt, raising=False)

    return _set


@pytest.mark.parametrize("text", ["All", "you", "Okay", "Sm Sm"])
async def test_short_transcripts_are_dropped(fake_audio, text: str):
    fake_audio(text)
    assert await ingest.transcribe_video_audio("clip.mp4") == ""


@pytest.mark.parametrize("text", [
    "Welcome back everyone",
    "So today we are going to look at the deployment pipeline",
])
async def test_real_transcripts_survive(fake_audio, text: str):
    fake_audio(text)
    assert await ingest.transcribe_video_audio("clip.mp4") == text


async def test_no_audio_track_returns_empty(monkeypatch):
    monkeypatch.setattr(ingest, "extract_audio_pcm16", lambda *a, **k: b"")
    assert await ingest.transcribe_video_audio("clip.mp4") == ""


async def test_stt_failure_degrades_to_empty(monkeypatch):
    monkeypatch.setattr(ingest, "extract_audio_pcm16", lambda *a, **k: b"\x00\x01" * 100)

    class _Stt:
        @staticmethod
        async def transcribe(pcm, **kwargs):
            raise RuntimeError("model unavailable")

    import backend.voice
    monkeypatch.setattr(backend.voice, "stt", _Stt, raising=False)
    assert await ingest.transcribe_video_audio("clip.mp4") == ""
