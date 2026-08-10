"""FastAPI routes for the video understanding ("watch video") feature.

Endpoints (all under ``/api/video``)
------------------------------------
GET  /api/video/permission     — Screen Recording permission (for recording).
GET  /api/video/status         — current recording status.
POST /api/video/record/start   — begin recording the screen into a session.
POST /api/video/record/stop    — stop and finalise the recording file.
POST /api/video/analyze        — watch a video (file / recording / YouTube)
                                 and return the model's answer.

The WebSocket for realtime "live watching" (``/ws/watch``) lives in
``backend.routes.video_ws``.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Response

from backend.capture import screen_capture as sc
from backend.config import AppConfig
from backend.session_manager import _session_files_dir
from backend.video import ingest, recorder

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/video", tags=["video"])


def _recordings_dir(session_id: str) -> Path:
    d = _session_files_dir(session_id) / "recordings"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _to_virtual(session_id: str, abs_path: str) -> str:
    """Map an absolute recording path back to its session-relative form."""
    try:
        rel = Path(abs_path).resolve().relative_to(_session_files_dir(session_id).resolve())
        return "/" + str(rel)
    except Exception:  # noqa: BLE001
        return abs_path


@router.get("/permission")
async def video_permission() -> dict[str, Any]:
    """Report Screen Recording capability + permission (needed to record)."""
    granted = await asyncio.to_thread(sc.screen_recording_granted)
    return {
        "supported": recorder.supported(),
        "granted": granted,  # True / False / None (unknown)
        "can_prompt": sc.supported(),
    }


@router.post("/permission/prompt")
async def video_permission_prompt() -> dict[str, Any]:
    """Trigger the macOS Screen Recording permission prompt."""
    await asyncio.to_thread(sc.request_screen_recording)
    granted = await asyncio.to_thread(sc.screen_recording_granted)
    return {"supported": recorder.supported(), "granted": granted}


@router.get("/status")
async def video_status() -> dict[str, Any]:
    return recorder.manager.status()


@router.get("/preview")
async def video_preview() -> Response:
    """A single JPEG of the screen right now, for the panel's live preview.

    ffmpeg owns the capture device while recording, so rather than tapping its
    output this grabs its own frame — the same call the live watchers use, so
    the preview matches what a model would be shown. Polled by the client, so
    it must never be cached.
    """
    cfg = await AppConfig.aload()
    if not cfg.video.live_preview:
        return Response(status_code=204)
    frame = await asyncio.to_thread(
        ingest.grab_screen_jpeg, int(cfg.video.frame_max_side or 1024),
    )
    if not frame:
        return Response(status_code=204)
    return Response(
        content=frame,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/audio-devices")
async def video_audio_devices() -> dict[str, Any]:
    """avfoundation audio inputs available to record alongside the screen."""
    devices = await asyncio.to_thread(recorder.list_devices)
    return {"devices": devices["audio"]}


@router.get("/speech-model")
async def video_speech_model() -> dict[str, Any]:
    """Whether the Whisper model the audio track needs is already downloaded.

    Analysis skips the transcript rather than blocking on a multi-gigabyte
    fetch, so Settings uses this to warn before the user relies on audio.
    """
    model = await asyncio.to_thread(ingest.speech_model_id)
    ready = await asyncio.to_thread(ingest.speech_model_ready)
    return {"model": model, "ready": ready}


@router.post("/record/start")
async def video_record_start(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Begin recording the screen into the given session's sandbox.

    Body: ``{session_id, fps?, audio?}``.  ``audio`` overrides the configured
    default for this one recording.
    """
    session_id = str(body.get("session_id") or "").strip()
    if not session_id:
        return {"error": "session_id is required."}
    cfg = await AppConfig.aload()
    fps = float(body.get("fps") or 5.0)
    max_side = int(cfg.video.frame_max_side or 1280)

    want_audio = bool(body.get("audio", cfg.video.record_audio))
    audio_device = None
    if want_audio:
        audio_device = await asyncio.to_thread(
            recorder.resolve_audio_device, cfg.video.record_audio_device or "",
        )
        if audio_device is None:
            return {"error": "No audio input device is available to record."}

    dest = _recordings_dir(session_id)
    result = await asyncio.to_thread(
        recorder.manager.start, dest, fps=fps, max_side=max_side,
        audio_device=audio_device,
    )
    if result.get("path"):
        result["virtual_path"] = _to_virtual(session_id, result["path"])
    return result


@router.post("/record/stop")
async def video_record_stop(body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """Stop the active recording and return the finished file path."""
    session_id = str(body.get("session_id") or "").strip()
    result = await asyncio.to_thread(recorder.manager.stop)
    if result.get("path") and session_id:
        result["virtual_path"] = _to_virtual(session_id, result["path"])
    return result


@router.post("/analyze")
async def video_analyze(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Watch a video and return the model's answer.

    Body: ``{session_id, source, question?, start_time?, end_time?}``.
    ``source`` is a session path (e.g. "/recordings/screen_x.mp4",
    "/uploads/clip.mp4") or a public YouTube URL.
    """
    session_id = str(body.get("session_id") or "").strip()
    source = str(body.get("source") or "").strip()
    if not session_id:
        return {"error": "session_id is required."}
    if not source:
        return {"error": "source is required."}

    question = str(body.get("question") or "")
    start_time = str(body.get("start_time") or "")
    end_time = str(body.get("end_time") or "")

    cfg = await AppConfig.aload()
    files_dir = _session_files_dir(session_id)

    from backend.video_tools import _use_gemini, analyze_video

    frame_llm = None
    if not _use_gemini(cfg):
        frame_llm = await _build_frame_vision_llm(cfg)

    try:
        text = await analyze_video(
            source=source,
            question=question,
            start_time=start_time,
            end_time=end_time,
            config=cfg,
            files_dir=files_dir,
            frame_vision_llm=frame_llm,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("video analyze failed")
        return {"error": str(exc)}
    return {"result": text}


async def _build_frame_vision_llm(cfg: AppConfig):
    """Build a vision-capable model for the frame path (standalone route use).

    Mirrors ``session_manager``'s fallback logic: use the main model when it
    natively sees images, otherwise the MLX VLM fallback (or None).
    """
    try:
        from deep_agent.model_factory import create_llm, create_mlx_vlm, supports_vision
        from utilities.environment import Environment
    except Exception as exc:  # noqa: BLE001
        logger.warning("cannot build frame vision model: %s", exc)
        return None

    provider = cfg.llm.provider
    if provider == "mlx":
        model_id = Environment.get_hf_llm_model_id()
    elif provider == "omlx":
        model_id = getattr(cfg.omlx, "model_name", "") or ""
    elif provider == "exo":
        model_id = getattr(cfg.exo, "model_name", "") or ""
    else:
        model_id = ""

    try:
        llm = await asyncio.to_thread(create_llm, provider)
    except Exception as exc:  # noqa: BLE001
        logger.warning("frame vision model construction failed: %s", exc)
        return None

    try:
        if supports_vision(provider, model_id):
            return llm
        vlm = await asyncio.to_thread(create_mlx_vlm, "mlx", llm)
        return vlm if vlm is not llm else None
    except Exception:  # noqa: BLE001
        return llm
