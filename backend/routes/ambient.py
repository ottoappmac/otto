"""API routes for the ambient assistant.

All endpoints are under ``/api/ambient``.

Endpoints
---------
GET  /api/ambient/status          — pending count + last-sweep metadata.
GET  /api/ambient/hints           — list actionable hints.
POST /api/ambient/run             — trigger an immediate sweep (manual).
POST /api/ambient/hints/{id}/accept — accept a hint (mode: chat | run).
POST /api/ambient/hints/{id}/dismiss — dismiss a hint.
POST /api/ambient/hints/{id}/snooze  — snooze a hint (default 4 hours).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException

from backend.ambient_store import get_store
from backend.config import AppConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ambient", tags=["ambient"])

# Debounce guard: prevent piling up manual sweeps.
_sweep_task: Optional[asyncio.Task[Any]] = None


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


@router.get("/status")
async def ambient_status() -> dict[str, Any]:
    """Return the current ambient assistant status."""
    from backend.ambient_agent import is_sweep_running  # type: ignore[import]

    cfg = await AppConfig.aload()
    store = await get_store()

    pending = await store.pending_count()

    return {
        "enabled": cfg.ambient.enabled,
        "pending_hints": pending,
        "allow_auto_run": cfg.ambient.allow_auto_run,
        "interval_mins": cfg.ambient.interval_mins,
        "llm_family": cfg.ambient.llm_family,
        "mlx_model": cfg.ambient.mlx_model,
        "sweep_running": is_sweep_running(),
    }


# ---------------------------------------------------------------------------
# Hints list
# ---------------------------------------------------------------------------


@router.get("/hints")
async def list_hints() -> dict[str, Any]:
    """Return all pending / actionable hints."""
    from backend.ambient_agent import _in_quiet_hours  # type: ignore[import]
    cfg = await AppConfig.aload()
    store = await get_store()

    quiet = _in_quiet_hours(cfg.ambient)
    hints = await store.list_pending(quiet=quiet)

    # Mark any "pending" ones as "shown" now that the frontend has fetched them.
    for h in hints:
        if h.status == "pending":
            await store.mark_shown(h.id)

    return {"hints": [h.to_dict() for h in hints], "quiet_hours": quiet}


# ---------------------------------------------------------------------------
# Manual sweep trigger
# ---------------------------------------------------------------------------


@router.post("/run")
async def run_sweep() -> dict[str, Any]:
    """Trigger an immediate ambient sweep and wait for its result.

    Returns the sweep outcome so the frontend can surface skip reasons
    (e.g. no model configured, no context available) rather than showing
    a silent empty list.
    """
    global _sweep_task  # noqa: PLW0603

    from backend.ambient_agent import run_sweep as _run_sweep

    # If a sweep is already in flight (e.g. triggered by session-end hook),
    # wait for it to finish rather than spawning a second one.
    if _sweep_task is not None and not _sweep_task.done():
        result = await _sweep_task
        return {"status": "complete", **result}

    _sweep_task = asyncio.create_task(_run_sweep(is_manual=True))
    result = await _sweep_task
    return {"status": "complete", **result}


# ---------------------------------------------------------------------------
# Accept
# ---------------------------------------------------------------------------


@router.post("/hints/{hint_id}/accept")
async def accept_hint(
    hint_id: str,
    mode: str = Body("chat", embed=True),
    agent_name: Optional[str] = Body(None, embed=True),
) -> dict[str, Any]:
    """Accept a hint.

    ``mode`` is one of:
    - ``"chat"``  — frontend will open a pre-filled chat draft; no session is
      spawned here.  We just mark the hint accepted.
    - ``"run"``   — spawn a background session immediately and return
      ``session_id``.
    - ``"apply"`` — update the stored prompt of the schedule / trigger that
      produced an evaluation-triggered suggestion.
    """
    store = await get_store()
    hint = await store.get(hint_id)
    if hint is None:
        raise HTTPException(status_code=404, detail="Hint not found")

    if mode == "apply":
        target_kind = getattr(hint, "target_kind", None)
        target_id = getattr(hint, "target_id", None)
        if target_kind not in ("schedule", "trigger") or not target_id:
            raise HTTPException(
                status_code=400,
                detail="This suggestion cannot be applied to a schedule or trigger.",
            )

        if target_kind == "schedule":
            from backend.scheduler import load_schedule, save_schedule

            spec = await asyncio.to_thread(load_schedule, target_id)
            if spec is None:
                raise HTTPException(status_code=404, detail="Schedule not found")
            spec.prompt = hint.proposed_prompt
            await asyncio.to_thread(save_schedule, spec)
        else:
            from backend.trigger_manager import load_trigger, save_trigger

            spec = await asyncio.to_thread(load_trigger, target_id)
            if spec is None:
                raise HTTPException(status_code=404, detail="Trigger not found")
            spec.prompt = hint.proposed_prompt
            await asyncio.to_thread(save_trigger, spec)

        await store.accept(hint_id)
        return {"status": "applied", "target_kind": target_kind, "target_id": target_id}

    if mode == "run":
        cfg = await AppConfig.aload()
        if not cfg.ambient.allow_auto_run:
            raise HTTPException(
                status_code=403,
                detail="allow_auto_run is disabled — use mode=chat",
            )

        # Spawn a session via the session manager, same pattern as scheduler.
        from backend.state import session_mgr  # type: ignore[import]
        from backend.session_dispatch import kick_off_message  # type: ignore[import]

        session = await session_mgr.create_session(
            cfg,
            agent_name=agent_name or hint.suggested_agent,
            trigger_source="ambient",
            trigger_id=hint_id,
        )
        await store.accept(hint_id, session_id=session.id)

        asyncio.create_task(
            kick_off_message(session.id, hint.proposed_prompt),
        )

        return {"status": "spawned", "session_id": session.id}

    # mode == "chat": just mark accepted; frontend handles the navigation.
    await store.accept(hint_id)
    return {"status": "accepted", "session_id": None}


# ---------------------------------------------------------------------------
# Dismiss
# ---------------------------------------------------------------------------


@router.post("/hints/{hint_id}/dismiss")
async def dismiss_hint(hint_id: str) -> dict[str, Any]:
    store = await get_store()
    ok = await store.dismiss(hint_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Hint not found or already actioned")
    return {"status": "dismissed"}


# ---------------------------------------------------------------------------
# Snooze
# ---------------------------------------------------------------------------


@router.post("/hints/{hint_id}/snooze")
async def snooze_hint(
    hint_id: str,
    hours: int = Body(4, embed=True),
) -> dict[str, Any]:
    if not 1 <= hours <= 168:
        raise HTTPException(status_code=422, detail="hours must be between 1 and 168")
    store = await get_store()
    ok = await store.snooze(hint_id, hours=hours)
    if not ok:
        raise HTTPException(status_code=404, detail="Hint not found or already actioned")
    return {"status": "snoozed", "hours": hours}
