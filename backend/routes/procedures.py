"""API for procedural memory (``backend/procedural_memory``).

App: status, list and inspect saved runs, label a run (thumbs up / down), exclude or delete one.
Benchmark backend only (``OTTO_PROCEDURE_MODE`` set): point the write store at a run's folder, save a finished task
with its evaluator score, and record the outcome against the items the task used.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from backend.procedural_memory import service

router = APIRouter(prefix="/api/procedures", tags=["procedures"])


def _app_store():
    """The store the app's Procedures view reads and edits. With memory on, the live store; with it off, the app's
    own ``procedures.db`` if it exists, so what was saved stays visible (and deletable) after turning memory off.
    None when nothing has been saved."""
    st = service.write_store()
    if st is not None:
        return st
    if os.environ.get(service.ENV_MODE):
        return None
    from backend.config import get_app_data_dir
    path = get_app_data_dir() / "procedures.db"
    return service.store(path) if path.exists() else None


def _app_store_or_404():
    st = _app_store()
    if st is None:
        raise HTTPException(404, "nothing saved")
    return st


@router.get("/status")
async def status() -> dict[str, Any]:
    try:
        return await asyncio.to_thread(service.status)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/episodes")
async def episodes(status: str | None = None, limit: int = 200) -> dict[str, Any]:
    """Saved runs, newest first, with the store's counts. Readable with memory off (see ``_app_store``)."""
    st = _app_store()
    if st is None:
        return {"episodes": [], "counts": {}}
    rows = await asyncio.to_thread(st.list_episodes, status=status, limit=min(max(limit, 1), 1000))
    for r in rows:
        r.pop("data", None)
    return {"episodes": rows, "counts": await asyncio.to_thread(st.counts)}


@router.get("/episodes/{episode_id}")
async def episode(episode_id: str) -> dict[str, Any]:
    st = _app_store_or_404()
    ep = await asyncio.to_thread(st.episode, episode_id)
    if ep is None:
        raise HTTPException(404, "no such episode")
    items = await asyncio.to_thread(st.items_for_episode, episode_id)
    return {"episode": ep, "labels": st.labels(episode_id),
            "items": [{"id": i.id, "kind": i.kind, "intent": i.intent, "body": i.body, "label_source": i.label_source,
                       "from_outcome": i.from_outcome} for i in items]}


@router.post("/label")
async def label(body: dict = Body(...)) -> dict[str, Any]:
    """User feedback on a run. ``label``: success | failure. ``source`` defaults to ``user``."""
    session_id, value = str(body.get("session_id") or ""), str(body.get("label") or "")
    source = str(body.get("source") or "user")
    if source not in ("user", "verified"):
        raise HTTPException(400, "source must be 'user' or 'verified'")
    try:
        ids = await service.label_session(session_id, value, source, note=body.get("note"),
                                          episode_id=body.get("episode_id"))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"episodes": ids}


@router.post("/episodes/{episode_id}/feedback")
async def episode_feedback(episode_id: str, body: dict = Body(...)) -> dict[str, Any]:
    """Worked / Didn't work for one run from the Procedures page. Didn't work excludes it, as in the chat."""
    worked = body.get("worked")
    if not isinstance(worked, bool):
        raise HTTPException(400, "worked must be true or false")
    rt = service.runtime()
    st = service.write_store(rt) if rt.mode == "app" else None
    if st is None:
        raise HTTPException(409, "procedural memory is off")
    if st.episode(episode_id) is None:
        raise HTTPException(404, "no such episode")
    return {"status": await service.apply_user_answer(episode_id, worked, st=st, rt=rt)}


@router.post("/episodes/{episode_id}/exclude")
async def exclude(episode_id: str) -> dict[str, Any]:
    """"Don't learn from this run": its pieces are removed and it is never admitted again."""
    st = _app_store_or_404()
    if st.episode(episode_id) is None:
        raise HTTPException(404, "no such episode")
    st.drop_items(episode_id)
    st.set_status(episode_id, "excluded")
    return {"ok": True}


@router.delete("/episodes/{episode_id}")
async def delete(episode_id: str) -> dict[str, Any]:
    st = _app_store_or_404()
    await asyncio.to_thread(st.delete_episode, episode_id)
    return {"ok": True}


@router.get("/sessions/{session_id}/feedback")
async def session_feedback(session_id: str) -> dict[str, Any]:
    """Whether the chat should ask "did it work?" for this session's macOS desktop agent runs."""
    try:
        return await asyncio.to_thread(service.session_feedback_state, session_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/sessions/{session_id}/feedback")
async def post_session_feedback(session_id: str, body: dict = Body(...)) -> dict[str, Any]:
    """Thumbs up (``worked: true``) saves the runs to procedural memory; thumbs down excludes them."""
    worked = body.get("worked")
    if not isinstance(worked, bool):
        raise HTTPException(400, "worked must be true or false")
    try:
        return await service.record_session_feedback(session_id, worked)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


# ── benchmark backend only ──────────────────────────────────────────────────────────────────────────────────

def _benchmark_only() -> None:
    if not os.environ.get(service.ENV_MODE):
        raise HTTPException(403, "benchmark endpoints are only available in a benchmark backend")


@router.post("/benchmark/write-db")
async def benchmark_write_db(body: dict = Body(...)) -> dict[str, Any]:
    _benchmark_only()
    try:
        path = service.set_write_path(str(body.get("path") or ""))
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    service.write_store()
    return {"write_path": str(path)}


@router.post("/benchmark/ingest")
async def benchmark_ingest(body: dict = Body(...)) -> dict[str, Any]:
    """Save a finished benchmark task's run, labelled by the evaluator (score 1.0 = success, anything else =
    failure), into the write store. Never into the snapshot being served."""
    _benchmark_only()
    session_id = str(body.get("session_id") or "")
    score = body.get("score")
    if not session_id or not isinstance(score, (int, float)):
        raise HTTPException(400, "session_id and a numeric score are required")
    if service.write_store() is None:
        raise HTTPException(409, "no write store; call /benchmark/write-db first")
    label = "success" if float(score) >= 1.0 else "failure"
    ids = await service.save_session(session_id, track="benchmark", task_id=body.get("task_id"), label=label,
                                     label_source="benchmark", confidence=1.0)
    return {"episodes": ids, "label": label}


@router.post("/benchmark/outcome")
async def benchmark_outcome(body: dict = Body(...)) -> dict[str, Any]:
    _benchmark_only()
    score = body.get("score")
    outcome = "success" if isinstance(score, (int, float)) and float(score) >= 1.0 else "failure"
    n = service.record_outcome(str(body.get("session_id") or ""), score=score, outcome=outcome)
    return {"updated": n}
