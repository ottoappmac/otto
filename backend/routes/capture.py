"""FastAPI routes for screen capture (Transcribe screenshot feature).

All endpoints are under ``/api/capture`` and are macOS-only in practice; on
other hosts they report ``supported: false`` / ``unsupported: true`` instead of
failing.

Endpoints
---------
GET  /api/capture/permission          — Screen Recording permission status.
POST /api/capture/permission/prompt   — trigger the macOS permission prompt.
GET  /api/capture/windows             — list capturable on-screen windows.
POST /api/capture/screen              — capture desktop or a window (with dedupe).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from fastapi import APIRouter, Body

from backend.capture import screen_capture as sc

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/capture", tags=["capture"])


# ---------------------------------------------------------------------------
# Permission
# ---------------------------------------------------------------------------


@router.get("/permission")
async def capture_permission() -> dict[str, Any]:
    """Report Screen Recording capability + permission state."""
    granted = await asyncio.to_thread(sc.screen_recording_granted)
    return {
        "supported": sc.supported(),
        "granted": granted,  # True / False / None (unknown)
        "can_prompt": sc.supported(),
    }


@router.post("/permission/prompt")
async def capture_permission_prompt() -> dict[str, Any]:
    """Trigger the macOS Screen Recording permission prompt (best effort)."""
    await asyncio.to_thread(sc.request_screen_recording)
    granted = await asyncio.to_thread(sc.screen_recording_granted)
    return {"supported": sc.supported(), "granted": granted}


# ---------------------------------------------------------------------------
# Window enumeration
# ---------------------------------------------------------------------------


@router.get("/windows")
async def capture_windows(thumbnails: bool = False) -> dict[str, Any]:
    """List on-screen normal windows (optionally with small thumbnails)."""
    if not sc.supported():
        return {"supported": False, "windows": []}
    windows = await asyncio.to_thread(sc.list_windows, thumbnails)
    return {"supported": True, "windows": windows}


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


@router.post("/screen")
async def capture_screen(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Capture the desktop or a single window.

    Body: ``{mode: "desktop"|"window", window_id?, last_hash?}``.
    When *last_hash* is supplied and the frame is visually unchanged, returns
    ``{unchanged: true, hash}`` and no image (used by auto-capture to dedupe).
    """
    mode = str(body.get("mode", "desktop"))
    window_id: Optional[int] = body.get("window_id")
    last_hash: Optional[str] = body.get("last_hash")
    if window_id is not None:
        try:
            window_id = int(window_id)
        except (TypeError, ValueError):
            window_id = None

    result = await asyncio.to_thread(sc.capture, mode, window_id, last_hash)
    return result
