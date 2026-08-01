"""Live screen watching for providers without a realtime video API.

Gemini Live streams frames and talks back continuously; nothing else Otto
supports does.  The next best thing, and what this module implements, is to
watch in short cycles: collect a handful of frames over a few seconds, hand
them to the vision model along with the notes from earlier cycles, and emit
the reply as commentary.

That means commentary lags the screen by roughly one cycle, and the model is
busy for the duration of each call — the honest cost of running locally.  The
frame-batch-plus-running-notes shape is deliberately the same as
``video_tools._ask_in_batches``: it keeps each request small enough to clear
oMLX's prefill guard and any provider's context limit.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import Awaitable, Callable, Optional

from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

EventCb = Callable[[dict], Awaitable[None]]

DEFAULT_PROMPT = (
    "Watch my screen and describe what I'm doing. Point out anything notable, "
    "and warn me about errors."
)


class LocalLiveWatchSession:
    """Watch the screen in repeating batches using any vision-capable model."""

    def __init__(
        self,
        *,
        llm: BaseChatModel,
        prompt: str,
        fps: float = 1.0,
        max_side: int = 1024,
        batch_secs: int = 12,
        batch_frames: int = 4,
        emit_frames: bool = False,
        on_event: EventCb,
    ) -> None:
        self.llm = llm
        self.prompt = (prompt or "").strip() or DEFAULT_PROMPT
        self.fps = max(0.1, min(4.0, float(fps)))
        self.max_side = int(max_side)
        self.batch_secs = max(3, int(batch_secs))
        self.batch_frames = max(1, int(batch_frames))
        self.emit_frames = emit_frames
        self.on_event = on_event
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        await self.on_event({"type": "state", "state": "watching"})
        notes = ""
        started = time.monotonic()
        try:
            while not self._stop.is_set():
                frames = await self._collect_batch(started)
                if self._stop.is_set() or not frames:
                    continue
                try:
                    notes = await self._describe(frames, notes)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("local live batch failed: %s", exc)
                    await self.on_event({
                        "type": "error",
                        "message": f"Could not describe the last few seconds: {exc}",
                    })
                    continue
                if notes:
                    await self.on_event({"type": "commentary", "text": notes})
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("local live watch session error: %s", exc)
            await self.on_event({"type": "error", "message": str(exc)})
        finally:
            await self.on_event({"type": "state", "state": "idle"})

    async def _collect_batch(self, started: float) -> list[tuple[float, bytes]]:
        """Grab ``batch_frames`` frames spread across one cycle."""
        from backend.video import ingest

        gap = self.batch_secs / self.batch_frames
        frames: list[tuple[float, bytes]] = []
        for _ in range(self.batch_frames):
            if self._stop.is_set():
                break
            frame = await asyncio.to_thread(ingest.grab_screen_jpeg, self.max_side)
            if frame:
                frames.append((time.monotonic() - started, frame))
                if self.emit_frames:
                    await self.on_event({
                        "type": "frame",
                        "jpeg_b64": base64.standard_b64encode(frame).decode(),
                    })
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=gap)
            except asyncio.TimeoutError:
                pass
        return frames

    async def _describe(
        self, frames: list[tuple[float, bytes]], notes: str,
    ) -> str:
        from backend.video import ingest
        from backend.video_tools import _invoke_text

        head = (
            "You are watching a screen live, a few seconds at a time. Below are "
            f"{len(frames)} frames from the last {self.batch_secs} seconds, each "
            "labelled with how long the session has been running."
        )
        if notes:
            head = (
                f"{head}\n\nWhat you have observed so far:\n{notes}"
            )

        content: list[dict] = [{"type": "text", "text": head}]
        content.extend(ingest.frames_to_image_blocks(frames))
        content.append({"type": "text", "text": (
            "\nRewrite your running commentary so it covers the session up to "
            "now. Keep what still matters, add what just happened, and lead "
            "with anything the user should act on. Be brief and factual — this "
            "is all you will remember of the earlier frames.\n\n"
            f"What the user asked you to watch for: {self.prompt}"
        )})
        return await _invoke_text(self.llm, content)


async def build_live_llm(cfg) -> Optional[BaseChatModel]:
    """Vision model for local live watching, or None when none is available.

    Mirrors ``backend.routes.video._build_frame_vision_llm`` — the same
    "main model if it sees images, else the MLX VLM" fallback the
    non-realtime frame path uses.
    """
    from backend.routes.video import _build_frame_vision_llm

    return await _build_frame_vision_llm(cfg)
