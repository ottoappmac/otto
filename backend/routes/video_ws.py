"""WebSocket route for realtime "live watching" of the screen.

``/ws/watch`` streams screen frames (+ optional mic audio) to a Gemini
Live session and relays the model's running commentary back to the client.

Protocol
--------
Inbound (client → server):
  {"type": "start", "prompt"?, "fps"?, "audio"?}
  {"type": "stop"}

Outbound (server → client):
  {"type": "state", "state": "watching" | "idle"}
  {"type": "commentary", "text": ...}
  {"type": "error", "message": ...}

Realtime watching is Gemini-only; the route rejects a ``start`` when no
Gemini API key is configured.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.config import AppConfig
from backend.video import recorder
from backend.video.live_watcher import LiveWatchSession

logger = logging.getLogger(__name__)

ws_router = APIRouter(tags=["video"])


@ws_router.websocket("/ws/watch")
async def watch_websocket(websocket: WebSocket) -> None:
    await websocket.accept()

    session: LiveWatchSession | None = None
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
                try:
                    from backend import privacy_lock

                    if privacy_lock.is_engaged(cfg):
                        await _emit({
                            "type": "error",
                            "message": "Privacy lock is engaged — realtime watching uses "
                                       "Gemini (cloud) and is disabled. Disengage in "
                                       "Settings \u2192 Privacy & Security.",
                        })
                        continue
                except Exception:  # noqa: BLE001
                    pass
                api_key = (cfg.llm.google.api_key or "").strip()
                if not api_key:
                    await _emit({
                        "type": "error",
                        "message": "Realtime watching requires a Gemini API key "
                                   "(Settings \u2192 LLM \u2192 Frontier \u2192 Google Gemini).",
                    })
                    continue
                if not recorder.supported():
                    await _emit({
                        "type": "error",
                        "message": "Screen capture for live watching is macOS-only.",
                    })
                    continue

                fps = float(msg.get("fps") or cfg.video.realtime_fps or 1.0)
                include_audio = bool(msg.get("audio", cfg.video.include_audio))
                prompt = str(msg.get("prompt") or "").strip()

                session = LiveWatchSession(
                    api_key=api_key,
                    prompt=prompt,
                    fps=fps,
                    max_side=int(cfg.video.frame_max_side or 1024),
                    include_audio=include_audio,
                    on_event=_emit,
                )
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
