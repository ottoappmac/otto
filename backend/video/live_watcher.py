"""Realtime "live watching" of the screen via the Gemini Live API.

Streams screen frames (JPEG, <= 1 FPS as the Live API requires) and,
optionally, microphone audio (16 kHz PCM) to a Gemini Live session and
relays the model's running text commentary back through an async callback.

This is Gemini-only: no other provider Otto supports offers realtime
native video streaming.  The caller (``backend.routes.video_ws``) is
responsible for gating on a configured Gemini API key.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

# Gemini Live model.  Distinct from the chat model — must be a *-live-*
# variant.  Kept as a module constant so it's easy to bump.
DEFAULT_LIVE_MODEL = "gemini-2.0-flash-live-001"

EventCb = Callable[[dict], Awaitable[None]]


def _grab_screen_jpeg(max_side: int) -> Optional[bytes]:
    """Capture the desktop and return a downscaled JPEG, or None."""
    from backend.capture import screen_capture as sc
    from backend.video import ingest

    result = sc.capture("desktop")
    b64 = result.get("image_b64") if isinstance(result, dict) else None
    if not b64:
        return None
    try:
        from PIL import Image

        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return ingest.downscale_jpeg(buf.getvalue(), max_side)
    except Exception as exc:  # noqa: BLE001
        logger.debug("live frame encode failed: %s", exc)
        return None


class LiveWatchSession:
    """Drive one Gemini Live screen-watching session."""

    def __init__(
        self,
        *,
        api_key: str,
        prompt: str,
        model: str = DEFAULT_LIVE_MODEL,
        fps: float = 1.0,
        max_side: int = 1024,
        include_audio: bool = False,
        on_event: EventCb,
    ) -> None:
        self.api_key = api_key
        self.prompt = prompt or "Watch my screen and describe what I'm doing. Point out anything notable, and warn me about errors."
        self.model = model or DEFAULT_LIVE_MODEL
        # Gemini Live accepts at most 1 frame per second.
        self.fps = max(0.1, min(1.0, float(fps)))
        self.max_side = int(max_side)
        self.include_audio = include_audio
        self.on_event = on_event
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        try:
            from google import genai
            from google.genai import types
        except Exception as exc:  # noqa: BLE001
            await self.on_event({"type": "error", "message": f"Gemini SDK unavailable: {exc}"})
            return

        client = genai.Client(api_key=self.api_key)
        config = {
            "response_modalities": ["TEXT"],
            "system_instruction": self.prompt,
        }

        try:
            async with client.aio.live.connect(model=self.model, config=config) as session:
                await self.on_event({"type": "state", "state": "watching"})
                # Seed the session with the task so it starts commenting.
                try:
                    await session.send_realtime_input(text=self.prompt)
                except Exception:  # noqa: BLE001
                    pass

                tasks = [
                    asyncio.create_task(self._send_frames(session, types)),
                    asyncio.create_task(self._receive(session)),
                ]
                if self.include_audio:
                    tasks.append(asyncio.create_task(self._send_audio(session, types)))

                await self._stop.wait()
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("live watch session error: %s", exc)
            await self.on_event({"type": "error", "message": str(exc)})
        finally:
            await self.on_event({"type": "state", "state": "idle"})

    async def _send_frames(self, session, types) -> None:
        interval = 1.0 / self.fps
        while not self._stop.is_set():
            frame = await asyncio.to_thread(_grab_screen_jpeg, self.max_side)
            if frame:
                try:
                    await session.send_realtime_input(
                        video=types.Blob(data=frame, mime_type="image/jpeg"),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("send frame failed: %s", exc)
            await asyncio.sleep(interval)

    async def _receive(self, session) -> None:
        # ``receive()`` yields until the end of a model turn; loop to keep
        # receiving subsequent turns for the life of the session.
        while not self._stop.is_set():
            try:
                async for response in session.receive():
                    text = getattr(response, "text", None)
                    if text:
                        await self.on_event({"type": "commentary", "text": text})
            except Exception as exc:  # noqa: BLE001
                if not self._stop.is_set():
                    logger.debug("live receive loop ended: %s", exc)
                return

    async def _send_audio(self, session, types) -> None:
        """Best-effort microphone capture → 16 kHz PCM → Live session."""
        try:
            import numpy as np
            import sounddevice as sd
        except Exception as exc:  # noqa: BLE001
            logger.info("realtime audio unavailable (%s) — continuing video-only", exc)
            return

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=50)

        def _cb(indata, frames, time_info, status):  # noqa: ANN001
            try:
                pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
                loop.call_soon_threadsafe(queue.put_nowait, pcm)
            except Exception:  # noqa: BLE001
                pass

        try:
            with sd.InputStream(samplerate=16000, channels=1, dtype="float32", callback=_cb):
                while not self._stop.is_set():
                    try:
                        pcm = await asyncio.wait_for(queue.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    try:
                        await session.send_realtime_input(
                            audio=types.Blob(data=pcm, mime_type="audio/pcm;rate=16000"),
                        )
                    except Exception:  # noqa: BLE001
                        pass
        except Exception as exc:  # noqa: BLE001
            logger.info("microphone capture stopped: %s", exc)
