"""REST endpoints for the activity timeline UI."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from backend.activity_tracker import (
    clear_all,
    count_activity,
    daily_summary,
    db_size_bytes,
    list_apps,
    search_activity,
    tracker,
)
from backend.config import AppConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/activity", tags=["activity"])


@router.get("/status")
async def api_activity_status():
    cfg = await AppConfig.aload()
    return {
        "enabled": cfg.activity.enabled,
        "interval_secs": cfg.activity.interval_secs,
        "retain_days": cfg.activity.retain_days,
        "exclude_apps": list(cfg.activity.exclude_apps),
        "idle_threshold_secs": cfg.activity.idle_threshold_secs,
        "min_span_secs": cfg.activity.min_span_secs,
        "max_span_secs": cfg.activity.max_span_secs,
        "context_max_chars": cfg.activity.context_max_chars,
        "field_val_max_chars": cfg.activity.field_val_max_chars,
        "browser_text_max_chars": cfg.activity.browser_text_max_chars,
        "ax_walk_max_chars": cfg.activity.ax_walk_max_chars,
        "ax_walk_max_depth": cfg.activity.ax_walk_max_depth,
        "max_db_mb": cfg.activity.max_db_mb,
        "running": tracker._task is not None and not tracker._task.done(),
        "db_size_bytes": db_size_bytes(),
    }


@router.get("/search")
async def api_activity_search(
    q: str | None = Query(None, description="FTS5 match expression"),
    date_from: int | None = Query(None, description="Unix timestamp lower bound"),
    date_to: int | None = Query(None, description="Unix timestamp upper bound"),
    app: str | None = Query(None, description="Restrict to one app"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    order_by: str = Query(
        "ts",
        pattern="^(ts|rank)$",
        description="'ts' = chronological (newest first); 'rank' = FTS5 BM25 relevance (requires q)",
    ),
):
    """Search the activity log with pagination + ranking.

    Returns ``rows`` (the slice for this page), ``count`` (rows in
    this slice), and ``total`` (total matches across all pages — used
    by the UI to render "X of N" and decide whether to show a
    "Load more" button).
    """
    rows = await asyncio.to_thread(
        search_activity,
        q,
        date_from=date_from,
        date_to=date_to,
        app=app,
        limit=limit,
        offset=offset,
        order_by=order_by,
    )
    total = await asyncio.to_thread(
        count_activity,
        q,
        date_from=date_from,
        date_to=date_to,
        app=app,
    )
    return {"rows": rows, "count": len(rows), "total": total}


@router.get("/timeline")
async def api_activity_timeline(date: str | None = None):
    """Return all activity for a given local date (YYYY-MM-DD).

    If *date* is omitted, returns today's timeline.
    """
    from datetime import datetime
    if date:
        try:
            day = datetime.fromisoformat(date)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid date — use YYYY-MM-DD"},
            )
    else:
        day = datetime.now()
    start = day.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start.replace(hour=23, minute=59, second=59)
    rows = await asyncio.to_thread(
        search_activity,
        None,
        date_from=int(start.timestamp()),
        date_to=int(end.timestamp()),
        limit=2000,
    )
    summary = await asyncio.to_thread(
        daily_summary,
        int(start.timestamp()),
        int(end.timestamp()),
    )
    return {
        "date": start.date().isoformat(),
        "rows": rows,
        "summary": summary,
    }


@router.get("/apps")
async def api_activity_apps(days: int = Query(30, ge=1, le=365)):
    rows = await asyncio.to_thread(list_apps, days=days)
    return {"days": days, "apps": rows}


@router.delete("/all")
async def api_activity_clear_all():
    """Wipe the entire activity DB.  Used by the privacy controls UI."""
    deleted = await asyncio.to_thread(clear_all)
    return {"deleted": deleted}


@router.get("/_debug/sample")
async def api_activity_debug_sample():
    """Return what the tracker sees *right now* without writing to the DB.

    Useful for diagnosing why the timeline isn't picking up app switches —
    runs the same code path the background loop uses, plus reports the
    tracker's in-memory dedup state.
    """
    import time
    from backend.activity_tracker import _get_active_macos, _seconds_since_last_input
    t0 = time.time()
    sample = await asyncio.to_thread(_get_active_macos)
    elapsed = time.time() - t0
    idle = await asyncio.to_thread(_seconds_since_last_input)
    return {
        "sample": sample,
        "sample_took_ms": round(elapsed * 1000, 1),
        "idle_secs": idle,
        "tracker_last": tracker._last,
        "tracker_last_id": tracker._last_id,
    }
