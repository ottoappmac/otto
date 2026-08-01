"""WebSocket route for realtime "live watching" of the screen.

``/ws/watch`` streams screen frames (+ optional mic audio) to a Gemini
Live session and relays the model's running commentary back to the client.

Protocol
--------
Inbound (client → server):
  {"type": "start", "prompt"?, "fps"?, "audio"?}
  {"type": "stop"}

Outbound (server → client):
  {"type": "state", "state": "watching" | "idle", "mode": "gemini" | "local"}
  {"type": "commentary", "text": ...}
  {"type": "frame", "jpeg_b64": ...}
  {"type": "error", "message": ...}

With a Gemini key this streams to the Gemini Live API.  Without one it falls
back to ``LocalLiveWatchSession``, which describes a batch of recent frames
every few seconds using whatever vision model is configured — slower to
react, but it keeps working on-device and under the privacy lock.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.config import AppConfig
from backend.video import recorder
from backend.video.live_local import LocalLiveWatchSession, build_live_llm
from backend.video.live_watcher import LiveWatchSession

logger = logging.getLogger(__name__)

ws_router = APIRouter(tags=["video"])


@ws_router.websocket("/ws/watch")
async def watch_websocket(websocket: WebSocket) -> None:
    await websocket.accept()

    session: LiveWatchSession | LocalLiveWatchSession | None = None
    run_task: asyncio.Task | None = None

    async def _emit(event: dict) -> None:
        try:
            await websocket.send_json(event)
        except Exception:  # noqa: BLE001
            pass

    try:
        while True:
            msg = await websocket.receive_json()
            mtype = msg.get("type")

            if mtype == "start":
                if run_task and not run_task.done():
                    await _emit({"type": "error", "message": "Already watching."})
                    continue
                cfg = await AppConfig.aload()
                if not recorder.supported():
                    await _emit({
                        "type": "error",
                        "message": "Screen capture for live watching is macOS-only.",
                    })
                    continue

                # Gemini is the cloud path, so the privacy lock rules it out
                # and we fall through to the on-device batcher.
                locked = False
                try:
                    from backend import privacy_lock

                    locked = privacy_lock.is_engaged(cfg)
                except Exception:  # noqa: BLE001
                    pass
                api_key = "" if locked else (cfg.llm.google.api_key or "").strip()

                fps = float(msg.get("fps") or cfg.video.realtime_fps or 1.0)
                include_audio = bool(msg.get("audio", cfg.video.include_audio))
                prompt = str(msg.get("prompt") or "").strip()
                max_side = int(cfg.video.frame_max_side or 1024)
                preview = bool(cfg.video.live_preview)

                if api_key:
                    session = LiveWatchSession(
                        api_key=api_key,
                        prompt=prompt,
                        fps=fps,
                        max_side=max_side,
                        include_audio=include_audio,
                        emit_frames=preview,
                        on_event=_emit,
                    )
                    await _emit({"type": "mode", "mode": "gemini"})
                else:
                    # create_llm refuses cloud providers while the lock is on,
                    # so this can only ever hand back a local model.
                    llm = await build_live_llm(cfg)
                    if llm is None:
                        await _emit({
                            "type": "error",
                            "message": (
                                "Live watching needs a vision-capable model, and the "
                                "privacy lock rules out Gemini. Configure a local "
                                "vision model, or disengage the lock in Settings "
                                "\u2192 Privacy & Security."
                            ) if locked else (
                                "Live watching needs a vision-capable model. Add a "
                                "Gemini API key for realtime watching, or configure a "
                                "local vision model to watch in short batches."
                            ),
                        })
                        continue
                    session = LocalLiveWatchSession(
                        llm=llm,
                        prompt=prompt,
                        fps=fps,
                        max_side=max_side,
                        batch_secs=int(cfg.video.live_batch_secs or 12),
                        batch_frames=int(cfg.video.live_batch_frames or 4),
                        emit_frames=preview,
                        on_event=_emit,
                    )
                    await _emit({"type": "mode", "mode": "local"})
                run_task = asyncio.create_task(session.run())

            elif mtype == "stop":
                if session:
                    session.stop()
                if run_task:
                    try:
                        await asyncio.wait_for(run_task, timeout=10.0)
                    except Exception:  # noqa: BLE001
                        run_task.cancel()
                session = None
                run_task = None

            elif mtype == "ping":
                await _emit({"type": "pong"})

    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("watch websocket error: %s", exc)
    finally:
        if session:
            session.stop()
        if run_task and not run_task.done():
            run_task.cancel()
            try:
                await run_task
            except Exception:  # noqa: BLE001
                pass
