"""The frame path must not download Whisper in the middle of an analysis.

``mlx_whisper`` fetches its weights on first use — well over a gigabyte — and
``watch_video`` has no channel to report that progress on, so a user who has
never touched Voice would just watch the request hang.  ``_run_frames``
therefore checks the cache first and settles for describing the picture.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from backend import video_tools
from backend.video import ingest


@pytest.fixture
def frame_path(monkeypatch, tmp_path: Path):
    """Run ``_run_frames`` against stub frames, a stub model, and real config."""
    monkeypatch.setattr(
        ingest, "sample_frames", lambda *a, **k: [(0.0, b"\xff\xd8frame")],
    )
    monkeypatch.setattr(ingest, "ffprobe_duration_secs", lambda *a, **k: 5.0)
    monkeypatch.setattr(
        video_tools, "_ask_one_pass",
        _async_return("A cat knocks a mug off a table."),
    )

    config = SimpleNamespace(video=SimpleNamespace(
        max_duration_secs=1800, max_frames=60, frames_per_request=24,
        frame_max_side=1024, frame_rate=1.0, include_audio=True,
        cache_transcripts=True, debug_save_frames=False,
    ))
    monkeypatch.setattr(video_tools, "_clamp_fps", lambda _c: 1.0)

    async def _run() -> str:
        return await video_tools._run_frames(
            config, object(), "What happens?", tmp_path / "clip.mp4",
            None, None, tmp_path,
        )

    return SimpleNamespace(run=_run, config=config, files_dir=tmp_path)


def _async_return(value):
    async def _fn(*a, **k):
        return value
    return _fn


def _transcripts(files_dir: Path) -> list[Path]:
    return [p for p in files_dir.rglob("*") if p.is_file() and p.suffix == ".txt"]


async def test_missing_model_skips_audio_and_says_so(frame_path, monkeypatch):
    monkeypatch.setattr(ingest, "speech_model_ready", lambda: False)
    monkeypatch.setattr(ingest, "speech_model_id", lambda: "mlx-community/whisper")

    def _boom(*a, **k):
        raise AssertionError("transcription attempted without the model")

    monkeypatch.setattr(ingest, "transcribe_video_audio", _boom)

    answer = await frame_path.run()
    assert "A cat knocks a mug off a table." in answer
    assert "mlx-community/whisper" in answer
    assert "not transcribed" in answer


async def test_skipped_audio_is_not_cached(frame_path, monkeypatch):
    """An empty cache entry would mark the clip transcribed for good."""
    monkeypatch.setattr(ingest, "speech_model_ready", lambda: False)
    monkeypatch.setattr(ingest, "speech_model_id", lambda: "mlx-community/whisper")
    monkeypatch.setattr(ingest, "transcribe_video_audio", _async_return(""))

    await frame_path.run()
    assert _transcripts(frame_path.files_dir) == []


async def test_present_model_transcribes_as_before(frame_path, monkeypatch):
    monkeypatch.setattr(ingest, "speech_model_ready", lambda: True)
    monkeypatch.setattr(
        ingest, "transcribe_video_audio", _async_return("hello there"),
    )

    answer = await frame_path.run()
    assert answer == "A cat knocks a mug off a table."
    cached = _transcripts(frame_path.files_dir)
    assert len(cached) == 1
    assert cached[0].read_text() == "hello there"


async def test_audio_disabled_never_probes_the_model(frame_path, monkeypatch):
    frame_path.config.video.include_audio = False

    def _boom(*a, **k):
        raise AssertionError("probed the speech model with audio off")

    monkeypatch.setattr(ingest, "speech_model_ready", _boom)

    assert await frame_path.run() == "A cat knocks a mug off a table."
