"""OpenClaw session watcher status and control endpoints.

Exposes a ``/hooks/openclaw/status`` endpoint so the frontend can
display watcher health alongside the existing Claude Code hook status.

Unlike the Claude hook router these endpoints do not *receive* external
webhook POSTs — the push mechanism is the filesystem watcher running
inside the backend process (see :mod:`backend.openclaw_watcher`).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hooks/openclaw", tags=["hooks"])


@router.get("/status")
async def watcher_status() -> dict[str, Any]:
    """Return watcher health: running state, tracked counts, auto-monitor."""
    try:
        from backend.openclaw_watcher import oc_watcher
        return {"enabled": True, **oc_watcher.status()}
    except Exception as exc:
        logger.debug("Watcher status error: %s", exc)
        return {"enabled": False, "running": False, "error": str(exc)}
