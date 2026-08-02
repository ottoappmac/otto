"""Video understanding — shared pipeline + the agent-facing ``watch_video`` tool.

``watch_video`` accepts three kinds of input:

- an uploaded/recorded video file in the session sandbox
  (e.g. ``/uploads/clip.mp4`` or a screen recording), or an absolute path,
- a public **YouTube URL**,
- (indirectly) a screen recording produced by ``backend.video.recorder``.

Two provider paths, chosen from ``VideoConfig.provider_preference`` and the
active chat provider:

- **Gemini (native)** — the clip / URL is handed to Gemini, which samples
  frames + audio itself and can refer to timestamps.  Best quality.
- **Frame-based** — Otto samples frames with ffmpeg and transcribes the
  audio locally, then sends the frames + transcript to any vision-capable
  model (the main model or the MLX VLM fallback).

The pipeline degrades gracefully: if Gemini is preferred but unconfigured
it falls back to the frame path; if no vision model is available at all it
returns an explanatory message rather than failing the run.  The same
:func:`analyze_video` coroutine backs both the agent tool and the
``/api/video/analyze`` REST route (see ``backend.routes.video``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from backend.utils import is_resolved_path_allowed, remap_to_virtual_path
from backend.video import ingest

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = (
    "Watch this video and give a detailed, structured summary: what happens "
    "(with approximate timestamps), any on-screen text, key steps or actions, "
    "and anything notable. If the user asked a specific question, answer it."
)

_MISSING_SPEECH_MODEL_NOTE = (
    "_Note: this is a description of the video only — the audio was not "
    "transcribed because the speech model ({model}) has not been downloaded "
    "yet. Download it under Settings → Voice → Speech to Text → Model to "
    "include speech next time._"
)


def _resolve_local_path(source: str, files_dir: Path) -> Optional[Path]:
    """Resolve a session-relative or absolute video path safely, or None."""
    source = remap_to_virtual_path(source, files_dir)
    vpath = source if source.startswith("/") else "/" + source
    if ".." in vpath or vpath.startswith("~"):
        return None
    full = files_dir / vpath.lstrip("/")
    if not is_resolved_path_allowed(full, files_dir, vpath):
        return None
    resolved = full.resolve()
    return resolved if resolved.is_file() else None


def _use_gemini(config: Any) -> bool:
    """True when the Gemini native path should handle video.

    Never returns True while the privacy lock is engaged — Gemini is a cloud
    provider, so the frame path (which can run fully on-device) is used
    instead when the user has locked cloud access.
    """
    video_cfg = config.video
    prefer = video_cfg.provider_preference == "google" or config.llm.provider == "google"
    if not (prefer and bool((config.llm.google.api_key or "").strip())):
        return False
    try:
        from backend import privacy_lock

        if privacy_lock.is_engaged(config):
            return False
    except Exception:  # noqa: BLE001
        pass
    return True


def _clamp_fps(config: Any) -> float:
    return max(0.1, min(30.0, float(config.video.frame_rate or 1.0)))


def _window_suffix(
    start: Optional[float], end: Optional[float], duration: Optional[float],
) -> str:
    """A filename suffix identifying a partial clip window.

    Empty when the window covers the whole clip, so the common case gets a
    clean ``<video>.txt`` rather than a spurious range.  ``max_duration_secs``
    always supplies an *end*, which for a short clip is well past its end.
    """
    starts_at_zero = not start
    ends_at_clip_end = end is None or (duration is not None and end >= duration)
    if starts_at_zero and ends_at_clip_end:
        return ""
    return f"_{int(start or 0)}-{int(end) if end else 'end'}"


def _transcript_cache_path(
    files_dir: Path, video: Path, start: Optional[float], end: Optional[float],
    duration: Optional[float],
) -> Path:
    """Where the cached Whisper transcript for *video* + window lives.

    Named after the video's session-relative path (``/`` → ``__``) rather than
    its bare stem so ``uploads/demo.mp4`` and ``recordings/demo.mp4`` cannot
    collide.
    """
    try:
        rel = video.resolve().relative_to(files_dir.resolve())
        key = "__".join(rel.with_suffix("").parts)
    except Exception:  # noqa: BLE001
        key = video.stem
    return files_dir / "transcripts" / f"{key}{_window_suffix(start, end, duration)}.txt"


def _read_cached_transcript(cache: Path, video: Path) -> Optional[str]:
    """Return the cached transcript, or None when absent or stale.

    An *existing but empty* file is a valid result (silence / filtered
    hallucination) and is returned as ``""`` so we don't re-run Whisper on
    every silent clip.
    """
    try:
        if not cache.is_file():
            return None
        if cache.stat().st_mtime < video.stat().st_mtime:
            return None  # video replaced since the transcript was written
        return cache.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return None


def _save_debug_frames(
    files_dir: Path,
    video: Path,
    frames: list[tuple[float, bytes]],
    *,
    fps: float,
    prompt: str,
    transcript: str,
) -> None:
    """Dump the sampled frames + a run manifest for inspection.  Best effort."""
    import json
    import time

    try:
        out = files_dir / "video-frames" / f"{video.stem}_{int(time.time())}"
        out.mkdir(parents=True, exist_ok=True)
        for ts, jpeg in frames:
            (out / f"frame_{ingest._fmt_ts(ts).replace(':', '-')}.jpg").write_bytes(jpeg)
        (out / "meta.json").write_text(
            json.dumps({
                "source": video.name,
                "fps": fps,
                "frame_count": len(frames),
                "timestamps": [ingest._fmt_ts(ts) for ts, _ in frames],
                "prompt": prompt,
                "transcript": transcript,
            }, indent=2),
            encoding="utf-8",
        )
        logger.info("video debug frames written to %s", out)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not write debug frames: %s", exc)


def parse_time(value: str) -> Optional[float]:
    """Parse a time string ("90", "MM:SS", "H:MM:SS") into seconds, or None."""
    v = (value or "").strip()
    if not v:
        return None
    try:
        if ":" in v:
            secs = 0.0
            for p in v.split(":"):
                secs = secs * 60 + float(p)
            return secs
        return float(v)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Provider paths
# ---------------------------------------------------------------------------


async def _run_gemini(config: Any, prompt: str, source: str, is_youtube: bool,
                      start: Optional[float], end: Optional[float]) -> str:
    import asyncio

    from backend.video import gemini

    g = config.llm.google
    text = await asyncio.to_thread(
        gemini.understand_video_sync,
        api_key=g.api_key,
        model=g.model_name or "gemini-2.5-flash",
        prompt=prompt,
        source=source,
        is_youtube=is_youtube,
        fps=_clamp_fps(config),
        media_resolution=config.video.media_resolution,
        start_secs=start,
        end_secs=end,
        max_output_tokens=g.max_tokens,
        temperature=g.temperature,
    )
    return text or "[Gemini returned no text for this video.]"


async def _run_frames(config: Any, frame_vision_llm: Optional[BaseChatModel],
                      prompt: str, path: Path,
                      start: Optional[float], end: Optional[float],
                      files_dir: Path) -> str:
    import asyncio

    if frame_vision_llm is None:
        return (
            "Error: no vision-capable model is available to watch this video. "
            "Configure a Gemini API key (best — native video), use a "
            "vision-capable cloud model, or set an MLX VLM fallback under "
            "Settings \u2192 LLM."
        )

    video_cfg = config.video
    max_dur = int(video_cfg.max_duration_secs or 0)
    if max_dur > 0:
        eff_start = start or 0.0
        if end is None or (end - eff_start) > max_dur:
            end = eff_start + max_dur

    fps = _clamp_fps(config)
    frames = await asyncio.to_thread(
        ingest.sample_frames,
        path,
        fps=fps,
        max_frames=int(video_cfg.max_frames or 60),
        max_side=int(video_cfg.frame_max_side or 1024),
        start_secs=start,
        end_secs=end,
    )
    if not frames:
        return (
            "Error: could not sample any frames from the video (is ffmpeg "
            "installed and the file a valid video?)."
        )

    transcript = ""
    audio_note = ""
    if video_cfg.include_audio:
        duration = await asyncio.to_thread(ingest.ffprobe_duration_secs, path)
        cache = _transcript_cache_path(files_dir, path, start, end, duration)
        cached = _read_cached_transcript(cache, path) if video_cfg.cache_transcripts else None
        if cached is not None:
            transcript = cached
        elif not await asyncio.to_thread(ingest.speech_model_ready):
            # Transcribing here would fetch the model mid-analysis with no way
            # to report progress. Watch the visuals now and let the user decide
            # whether to spend the download. Deliberately not cached — there is
            # nothing to remember, and an empty cache entry would mark the clip
            # as transcribed and suppress audio on every later run.
            audio_note = _MISSING_SPEECH_MODEL_NOTE.format(
                model=ingest.speech_model_id() or "the speech model",
            )
            logger.info("skipping video transcription — speech model not downloaded")
        else:
            transcript = await ingest.transcribe_video_audio(
                path, start_secs=start, end_secs=end,
            )
            if video_cfg.cache_transcripts:
                try:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    # Written even when empty: the file's presence is what marks
                    # the clip as already transcribed.
                    cache.write_text(transcript, encoding="utf-8")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("could not cache transcript: %s", exc)

    if video_cfg.debug_save_frames:
        _save_debug_frames(
            files_dir, path, frames, fps=fps, prompt=prompt, transcript=transcript,
        )

    batch = max(1, int(video_cfg.frames_per_request or 24))
    try:
        if len(frames) <= batch:
            answer = await _ask_one_pass(frame_vision_llm, prompt, frames, fps, transcript)
        else:
            answer = await _ask_in_batches(
                frame_vision_llm, prompt, frames, fps, transcript, batch,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("frame-based video understanding failed: %s", exc)
        return f"[Video understanding failed: {exc}]"
    return f"{answer}\n\n{audio_note}" if audio_note else answer


async def _invoke_text(llm: BaseChatModel, content: list[dict]) -> str:
    """Send one multimodal message and return the model's text reply."""
    resp = await llm.ainvoke([HumanMessage(content=content)])
    out = resp.content
    if isinstance(out, list):
        parts = [b.get("text", "") for b in out if isinstance(b, dict) and b.get("type") == "text"]
        return ("\n".join(parts).strip() or str(out))
    return str(out).strip()


async def _ask_one_pass(llm: BaseChatModel, prompt: str,
                        frames: list[tuple[float, bytes]], fps: float,
                        transcript: str) -> str:
    """Every frame in a single request — best quality when it fits."""
    content: list[dict] = [{
        "type": "text",
        "text": (
            f"{prompt}\n\nBelow are {len(frames)} frames sampled from the video "
            f"at {fps} fps, each labelled with its timestamp."
        ),
    }]
    content.extend(ingest.frames_to_image_blocks(frames))
    if transcript:
        content.append({"type": "text", "text": f"\nAudio transcript of the clip:\n{transcript}"})
    content.append({"type": "text", "text": f"\nTask: {prompt}"})
    return await _invoke_text(llm, content)


async def _ask_in_batches(llm: BaseChatModel, prompt: str,
                          frames: list[tuple[float, bytes]], fps: float,
                          transcript: str, batch: int) -> str:
    """Walk the video in order, carrying a running summary between batches.

    A single request carrying every frame is what trips oMLX's prefill guard
    (and any provider's context limit) on a long clip.  Sending the frames in
    order, each batch seeing the notes from the ones before it, keeps every
    request small while preserving the timeline the model needs to describe
    what happens over time.

    The trade-off is that the model can only compare frames *within* a batch;
    across batches it reasons over its own notes rather than the images.
    """
    notes = ""
    total = (len(frames) + batch - 1) // batch
    for i in range(0, len(frames), batch):
        chunk = frames[i:i + batch]
        n = i // batch + 1
        span = f"{ingest._fmt_ts(chunk[0][0])}–{ingest._fmt_ts(chunk[-1][0])}"
        head = (
            f"You are watching a video in order, a segment at a time. "
            f"This is segment {n} of {total}, covering {span}.\n\n"
        )
        if notes:
            head += f"Notes from the earlier segments:\n{notes}\n\n"
        head += (
            f"Below are {len(chunk)} frames from this segment at {fps} fps, "
            f"each labelled with its timestamp."
        )

        content: list[dict] = [{"type": "text", "text": head}]
        content.extend(ingest.frames_to_image_blocks(chunk))
        content.append({"type": "text", "text": (
            "\nRewrite the notes so they cover the video up to this point. "
            "Keep earlier detail that still matters, add what is new in this "
            "segment, and keep the timestamps. Be factual and concise — these "
            "notes are all you will have of the earlier frames.\n\n"
            f"The notes will be used to answer: {prompt}"
        )})
        notes = await _invoke_text(llm, content)
        logger.info("video batch %d/%d (%s) summarised", n, total, span)

    final: list[dict] = [{"type": "text", "text": (
        f"These are ordered notes taken while watching a video "
        f"({len(frames)} frames at {fps} fps):\n\n{notes}"
    )}]
    if transcript:
        final.append({"type": "text", "text": f"\nAudio transcript:\n{transcript}"})
    final.append({"type": "text", "text": f"\nTask: {prompt}"})
    return await _invoke_text(llm, final)


# ---------------------------------------------------------------------------
# Shared entry point (used by both the tool and the REST route)
# ---------------------------------------------------------------------------


async def analyze_video(
    *,
    source: str,
    question: str,
    start_time: str,
    end_time: str,
    config: Any,
    files_dir: Path,
    frame_vision_llm: Optional[BaseChatModel] = None,
) -> str:
    """Watch *source* and return the model's textual answer.

    Handles YouTube URLs, uploaded/recorded files, provider routing, the
    max-duration guard, and graceful fallbacks.  Never raises for the common
    failure modes — it returns an ``Error: ...`` string the caller can show.
    """
    prompt = (question or "").strip() or _DEFAULT_PROMPT
    start = parse_time(start_time)
    end = parse_time(end_time)
    src = (source or "").strip()
    if not src:
        return "Error: no video source provided."

    # ── YouTube ────────────────────────────────────────────────────────
    if ingest.is_youtube_url(src):
        if not config.video.youtube_enabled:
            return (
                "Error: YouTube input is disabled. Enable it under Settings "
                "\u2192 Video, or download the video and upload it."
            )
        if _use_gemini(config):
            return await _run_gemini(config, prompt, src, True, start, end)
        try:
            import asyncio

            dl_dir = files_dir / "video_cache"
            path = await asyncio.to_thread(ingest.download_youtube, src, dl_dir)
        except Exception as exc:  # noqa: BLE001
            return (
                f"Error: could not download the YouTube video ({exc}). Configure "
                "a Gemini API key to watch YouTube URLs natively without downloading."
            )
        return await _run_frames(config, frame_vision_llm, prompt, path, start, end, files_dir)

    if ingest.is_url(src):
        return (
            "Error: only YouTube URLs and local/uploaded video files are "
            "supported. Download the video and upload it to the session first."
        )

    # ── Local / uploaded / recorded file ────────────────────────────────
    path = _resolve_local_path(src, files_dir)
    if path is None:
        return f"Error: video not found or outside the session directory: {source}"

    if _use_gemini(config):
        return await _run_gemini(config, prompt, str(path), False, start, end)
    return await _run_frames(config, frame_vision_llm, prompt, path, start, end, files_dir)


# ---------------------------------------------------------------------------
# Agent tool
# ---------------------------------------------------------------------------


def build_video_tools(
    files_dir: Path,
    *,
    config: Any,
    frame_vision_llm: Optional[BaseChatModel] = None,
) -> list:
    """Build the ``watch_video`` tool bound to this session's config + models."""

    @tool
    async def watch_video(source: str, question: str = "",
                          start_time: str = "", end_time: str = "") -> str:
        """Watch a video and answer questions about it (visuals + audio).

        Use this whenever the user wants you to understand a video: an
        uploaded file, a screen recording, or an online (YouTube) video.
        Prefer this over view_image for anything with motion or audio.

        Args:
            source: Either a session path to a video file (e.g.
                "/uploads/demo.mp4" or a screen recording path) OR a public
                YouTube URL (e.g. "https://youtube.com/watch?v=...").
            question: What to find out about the video. Omit for a general
                structured summary with timestamps.
            start_time: Optional clip start as seconds ("90") or "MM:SS" /
                "H:MM:SS". Analyse only from here.
            end_time: Optional clip end (same formats). Analyse only up to here.
        """
        return await analyze_video(
            source=source,
            question=question,
            start_time=start_time,
            end_time=end_time,
            config=config,
            files_dir=files_dir,
            frame_vision_llm=frame_vision_llm,
        )

    return [watch_video]
