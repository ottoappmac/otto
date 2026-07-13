"""FastAPI routes for the voice subsystem (STT + wake word; no TTS).

WebSocket /ws/voice
-------------------
Bidirectional control channel between the frontend and the VoiceManager.

Inbound messages (client → server)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{"type": "start"}                   — open mic (wakeword mode auto-start)
{"type": "stop"}                    — close mic, return to idle
{"type": "ptt_start"}              — push-to-talk pressed
{"type": "ptt_stop"}               — push-to-talk released
{"type": "configure", "config": {}} — apply partial config update

Outbound events (server → client)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{"type": "state",       "state": "idle|listening|capturing|transcribing"}
{"type": "transcript",  "text": "..."}
{"type": "partial",     "text": "..."}
{"type": "wake"}
{"type": "error",       "message": "..."}

REST endpoints
--------------
GET  /api/voice/catalog          — scored VoiceCatalogRow list
GET  /api/voice/status           — current manager state + device lists
POST /api/voice/stt-test         — one-shot STT transcription (base64 PCM in → text out)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.config import AppConfig
from backend.setup_capabilities import probe_capabilities
from backend.mlx_hub_paths import resolve_hf_hub_cache_dir

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/voice", tags=["voice"])
ws_router = APIRouter(tags=["voice-ws"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_voice_manager():
    from backend.voice.voice_manager import get_manager
    return get_manager()


async def _get_loopback_manager():
    from backend.voice.loopback_manager import get_loopback_manager
    return get_loopback_manager()


async def _apply_config_to_loopback(cfg: AppConfig) -> None:
    mgr = await _get_loopback_manager()
    vc = cfg.voice
    mgr.configure({
        "stt_model": vc.stt_model,
        "stt_language": vc.stt_language,
        "mic_device": vc.mic_device,
        "loopback_enabled": vc.loopback_enabled,
        "loopback_vad_silence_secs": vc.loopback_vad_silence_secs,
        "loopback_max_segment_secs": vc.loopback_max_segment_secs,
        "loopback_live_partials": vc.loopback_live_partials,
        "loopback_partial_interval_secs": vc.loopback_partial_interval_secs,
    })


async def _apply_config_to_manager(cfg: AppConfig) -> None:
    mgr = await _get_voice_manager()
    vc = cfg.voice
    mgr.configure({
        "enabled": vc.enabled,
        "activation_mode": vc.activation_mode,
        "stt_model": vc.stt_model,
        "stt_language": vc.stt_language,
        "stt_enabled": vc.stt_enabled,
        "wake_model": vc.wake_model,
        "wake_enabled": vc.wake_enabled,
        "vad_silence_secs": vc.vad_silence_secs,
        "mic_device": vc.mic_device,
    })


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


@router.get("/catalog")
async def voice_catalog():
    """Return scored VoiceCatalogRow list for the model chooser."""
    from backend.voice_catalog import fetch_voice_catalog, score_voice_catalog, is_enriching

    cfg = await AppConfig.aload()
    hub = resolve_hf_hub_cache_dir(cfg.llm.mlx.hf_hub_cache)
    caps = probe_capabilities(hub)

    cached_map: dict[str, bool] = {}
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir(hub)
        cached_map = {repo.repo_id: True for repo in info.repos}
    except Exception:  # noqa: BLE001
        pass

    rows = await fetch_voice_catalog(token=cfg.llm.mlx.hf_token or None)
    scored = score_voice_catalog(
        rows,
        ram_gb=caps["ram_gb"],
        wired_limit_gb=caps["wired_limit_gb"],
        free_disk_gb=caps["free_disk_gb"],
        cached_map=cached_map,
    )
    return {"rows": scored, "enriching": is_enriching()}


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


@router.get("/status")
async def voice_status():
    """Return manager state, active config, and available audio devices."""
    from backend.voice.audio_io import list_input_devices
    from backend.voice.voice_manager import get_manager

    mgr = get_manager()
    cfg = await AppConfig.aload()

    return {
        "state": mgr.state.value,
        "config": cfg.voice.model_dump(),
        "input_devices": list_input_devices(),
    }


# ---------------------------------------------------------------------------
# Loopback (system audio) capability status
# ---------------------------------------------------------------------------


@router.get("/loopback-status")
async def voice_loopback_status():
    """Report system-audio capture capability + current state."""
    from backend.voice.loopback_manager import (
        get_loopback_manager,
        helper_available,
        macos_supported,
        mic_available,
    )

    mgr = get_loopback_manager()
    cfg = await AppConfig.aload()
    return {
        "state": mgr.state.value,
        "supported": macos_supported(),
        "helper_available": helper_available(),
        "mic_available": mic_available(),
        "config": {
            "loopback_enabled": cfg.voice.loopback_enabled,
            "loopback_vad_silence_secs": cfg.voice.loopback_vad_silence_secs,
            "loopback_max_segment_secs": cfg.voice.loopback_max_segment_secs,
            "loopback_live_partials": cfg.voice.loopback_live_partials,
            "loopback_partial_interval_secs": cfg.voice.loopback_partial_interval_secs,
            "loopback_auto_send_silence_secs": cfg.voice.loopback_auto_send_silence_secs,
        },
    }


# ---------------------------------------------------------------------------
# Model cache check
# ---------------------------------------------------------------------------


@router.get("/model-cached")
async def voice_model_cached(repo_id: str):
    """Return whether a HuggingFace model repo is already in the local cache."""
    cfg = await AppConfig.aload()
    hub = resolve_hf_hub_cache_dir(cfg.llm.mlx.hf_hub_cache)
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir(hub)
        cached = any(r.repo_id == repo_id for r in info.repos)
    except Exception:  # noqa: BLE001
        cached = False
    return {"repo_id": repo_id, "cached": cached}


# ---------------------------------------------------------------------------
# One-shot STT test
# ---------------------------------------------------------------------------


@router.post("/stt-test")
async def voice_stt_test(body: dict[str, Any]):
    """Transcribe base64-encoded 32-bit float PCM audio (16 kHz mono) → text.

    The frontend records via MediaRecorder, decodes to float32 via
    OfflineAudioContext, and base64-encodes the raw float32 buffer.
    We convert it to int16 PCM before handing it to Whisper.
    """
    import base64
    import numpy as np

    b64 = str(body.get("audio_b64", "")).strip()
    language = str(body.get("language", "")) or None
    if not b64:
        return {"error": "audio_b64 is required"}

    cfg = await AppConfig.aload()
    from backend.voice import stt as _stt
    _stt.configure(cfg.voice.stt_model)

    try:
        raw = base64.b64decode(b64)
        # Frontend sends float32 samples; convert to int16 for Whisper
        float_audio = np.frombuffer(raw, dtype=np.float32)
        pcm_bytes = (float_audio * 32_767).clip(-32_768, 32_767).astype(np.int16).tobytes()
        text = await _stt.transcribe(pcm_bytes, language=language)
        return {"text": text}
    except Exception as exc:  # noqa: BLE001
        logger.warning("STT test error: %s", exc)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# WebSocket control channel
# ---------------------------------------------------------------------------


@ws_router.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """Bidirectional voice control channel."""
    await websocket.accept()

    mgr = await _get_voice_manager()

    event_q: asyncio.Queue = asyncio.Queue(maxsize=256)
    mgr.add_client(event_q)

    cfg = await AppConfig.aload()
    await _apply_config_to_manager(cfg)

    await websocket.send_json({"type": "state", "state": mgr.state.value})

    async def _send_loop() -> None:
        while True:
            event = await event_q.get()
            try:
                await websocket.send_json(event)
            except Exception:  # noqa: BLE001
                return

    send_task = asyncio.create_task(_send_loop(), name="voice_ws_send")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                import json
                msg = json.loads(raw)
            except Exception:  # noqa: BLE001
                continue

            msg_type = msg.get("type", "")

            if msg_type == "start":
                asyncio.create_task(mgr.start())

            elif msg_type == "stop":
                asyncio.create_task(mgr.stop())

            elif msg_type == "ptt_start":
                asyncio.create_task(mgr.ptt_start())

            elif msg_type == "ptt_stop":
                asyncio.create_task(mgr.ptt_stop())

            elif msg_type == "configure":
                partial = msg.get("config", {})
                if isinstance(partial, dict):
                    cfg = await AppConfig.aload()
                    updated = cfg.voice.model_dump()
                    updated.update(partial)
                    cfg.voice = cfg.voice.model_validate(updated)
                    cfg.save()
                    await _apply_config_to_manager(cfg)
                    await websocket.send_json({"type": "state", "state": mgr.state.value})

    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.debug("voice WS error: %s", exc)
    finally:
        send_task.cancel()
        mgr.remove_client(event_q)


@ws_router.websocket("/ws/transcribe")
async def transcribe_websocket(websocket: WebSocket):
    """System-audio (loopback) transcription control channel.

    Inbound:  {"type": "start"|"stop"|"configure", "config": {...}}
    Outbound: {"type": "state"|"partial"|"segment"|"level"|"error", ...}
    """
    await websocket.accept()

    mgr = await _get_loopback_manager()

    event_q: asyncio.Queue = asyncio.Queue(maxsize=512)
    mgr.add_client(event_q)

    cfg = await AppConfig.aload()
    await _apply_config_to_loopback(cfg)

    await websocket.send_json({"type": "state", "state": mgr.state.value})

    async def _send_loop() -> None:
        while True:
            event = await event_q.get()
            try:
                await websocket.send_json(event)
            except Exception:  # noqa: BLE001
                return

    send_task = asyncio.create_task(_send_loop(), name="transcribe_ws_send")

    # Report speech-model readiness (and start downloading it if absent) so the
    # UI can gate Record and show progress instead of a silent first-run hang.
    asyncio.create_task(mgr.ensure_model_ready())

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                import json
                msg = json.loads(raw)
            except Exception:  # noqa: BLE001
                continue

            msg_type = msg.get("type", "")

            if msg_type == "start":
                sources = msg.get("sources")
                if not isinstance(sources, list):
                    sources = None
                asyncio.create_task(mgr.start(sources))

            elif msg_type == "prepare_model":
                asyncio.create_task(mgr.ensure_model_ready())

            elif msg_type == "stop":
                asyncio.create_task(mgr.stop())

            elif msg_type == "configure":
                partial = msg.get("config", {})
                if isinstance(partial, dict):
                    cfg = await AppConfig.aload()
                    updated = cfg.voice.model_dump()
                    updated.update(partial)
                    cfg.voice = cfg.voice.model_validate(updated)
                    cfg.save()
                    await _apply_config_to_loopback(cfg)
                    await websocket.send_json({"type": "state", "state": mgr.state.value})

    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.debug("transcribe WS error: %s", exc)
    finally:
        send_task.cancel()
        mgr.remove_client(event_q)
