"""Schedule management API routes."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from backend.scheduler import (
    MAX_SCHEDULES,
    attachments_dir,
    delete_schedule_files,
    get_scheduler,
    list_attachments,
    load_all_schedules,
    load_schedule,
    register_job,
    remove_job,
    runs_dir,
    save_schedule,
    schedule_dir,
    validate_schedule_id,
)
from backend.schemas import (
    ScheduleCreateRequest,
    ScheduleSpec,
    ScheduleUpdateRequest,
)
from backend.utils import is_safe_path_segment

router = APIRouter(prefix="/api/schedules", tags=["schedules"])

_EVAL_FIELDS = ("eval_status", "eval_overall_score", "eval_pass_count", "eval_total")


def _attach_eval_fields(runs: list) -> None:
    """Populate eval_* fields on each run from its linked session's meta JSON.

    Eval scores are stamped onto ``{session_id}.json`` by the evaluator, so we
    read them lazily here rather than duplicating them into each run.json.
    """
    import json

    from backend.session_manager import _sessions_dir

    sessions_dir = _sessions_dir()
    cache: dict[str, dict] = {}
    for run in runs:
        sid = getattr(run, "session_id", None)
        if not sid:
            continue
        if sid not in cache:
            meta_path = sessions_dir / f"{sid}.json"
            data: dict = {}
            if meta_path.is_file():
                try:
                    data = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
            cache[sid] = data
        meta = cache[sid]
        for field in _EVAL_FIELDS:
            if meta.get(field) is not None:
                setattr(run, field, meta[field])


def _reject_unsafe_ids(*ids: str) -> JSONResponse | None:
    """404 early when any URL path segment could traverse the filesystem."""
    if all(is_safe_path_segment(i) for i in ids):
        return None
    return JSONResponse(status_code=404, content={"error": "Not found"})


def _validate_cron(expression: str) -> str | None:
    """Return an error message if *expression* is not a valid cron string."""
    try:
        from apscheduler.triggers.cron import CronTrigger
        CronTrigger.from_crontab(expression)
        return None
    except (ValueError, KeyError) as exc:
        return f"Invalid cron expression: {exc}"


# ---------------------------------------------------------------------------
# Status polling (for frontend notifications) — must precede /{schedule_id}
# ---------------------------------------------------------------------------

@router.get("/status/poll")
async def api_schedule_status():
    """Lightweight endpoint for the frontend to poll schedule activity."""
    from backend.scheduler import is_schedule_running, _fix_orphaned_runs

    schedules = await asyncio.to_thread(load_all_schedules)
    running: list[str] = []
    recently_completed: list[dict[str, Any]] = []

    for s in schedules:
        if s.last_status == "running":
            if is_schedule_running(s.id):
                running.append(s.id)
            else:
                s.last_status = "error"
                s.last_error = "Run lost — no active task found"
                await asyncio.to_thread(save_schedule, s)
                await asyncio.to_thread(_fix_orphaned_runs, s.id)
                recently_completed.append({
                    "id": s.id,
                    "status": s.last_status,
                    "last_run": s.last_run.isoformat() if s.last_run else None,
                    "error": s.last_error,
                })
        elif s.last_status in ("success", "error", "cancelled"):
            recently_completed.append({
                "id": s.id,
                "status": s.last_status,
                "last_run": s.last_run.isoformat() if s.last_run else None,
                "error": s.last_error,
            })

    return {"running": running, "recently_completed": recently_completed}


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

@router.get("")
async def api_list_schedules():
    schedules = await asyncio.to_thread(load_all_schedules)
    return [s.model_dump(mode="json") for s in schedules]


@router.post("")
async def api_create_schedule(req: ScheduleCreateRequest):
    existing = await asyncio.to_thread(load_all_schedules)
    if len(existing) >= MAX_SCHEDULES:
        return JSONResponse(
            status_code=400,
            content={"error": f"Maximum of {MAX_SCHEDULES} schedules reached."},
        )

    id_err = validate_schedule_id(req.id)
    if id_err:
        return JSONResponse(status_code=400, content={"error": id_err})

    if await asyncio.to_thread(load_schedule, req.id):
        return JSONResponse(
            status_code=409,
            content={"error": f"Schedule '{req.id}' already exists."},
        )

    if req.agent_name:
        from backend.agent_library import get_agent
        if not await asyncio.to_thread(get_agent, req.agent_name):
            return JSONResponse(
                status_code=400,
                content={"error": f"Agent '{req.agent_name}' not found."},
            )

    cron_err = _validate_cron(req.cron_expression)
    if cron_err:
        return JSONResponse(status_code=400, content={"error": cron_err})

    spec = ScheduleSpec(
        id=req.id,
        agent_name=req.agent_name,
        prompt=req.prompt,
        cron_expression=req.cron_expression,
    )
    await asyncio.to_thread(save_schedule, spec)

    scheduler = get_scheduler()
    register_job(scheduler, spec.id, spec.cron_expression)

    return spec.model_dump(mode="json")


@router.get("/{schedule_id}")
async def api_get_schedule(schedule_id: str):
    spec = await asyncio.to_thread(load_schedule, schedule_id)
    if not spec:
        return JSONResponse(status_code=404, content={"error": "Schedule not found"})
    return spec.model_dump(mode="json")


@router.put("/{schedule_id}")
async def api_update_schedule(schedule_id: str, req: ScheduleUpdateRequest):
    spec = await asyncio.to_thread(load_schedule, schedule_id)
    if not spec:
        return JSONResponse(status_code=404, content={"error": "Schedule not found"})

    if req.prompt is not None:
        spec.prompt = req.prompt
    if req.agent_name is not None:
        if req.agent_name:
            from backend.agent_library import get_agent
            if not await asyncio.to_thread(get_agent, req.agent_name):
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Agent '{req.agent_name}' not found."},
                )
        spec.agent_name = req.agent_name or None
    if req.keep_last_n_runs is not None:
        spec.keep_last_n_runs = max(1, req.keep_last_n_runs)
    if req.cron_expression is not None:
        cron_err = _validate_cron(req.cron_expression)
        if cron_err:
            return JSONResponse(status_code=400, content={"error": cron_err})
        spec.cron_expression = req.cron_expression
    if req.enabled is not None:
        spec.enabled = req.enabled

    await asyncio.to_thread(save_schedule, spec)

    scheduler = get_scheduler()
    if spec.enabled:
        register_job(scheduler, spec.id, spec.cron_expression)
    else:
        remove_job(scheduler, spec.id)

    return spec.model_dump(mode="json")


@router.delete("/{schedule_id}")
async def api_delete_schedule(schedule_id: str):
    if not await asyncio.to_thread(load_schedule, schedule_id):
        return JSONResponse(status_code=404, content={"error": "Schedule not found"})

    scheduler = get_scheduler()
    remove_job(scheduler, schedule_id)
    await asyncio.to_thread(delete_schedule_files, schedule_id)

    return {"deleted": True}


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@router.post("/{schedule_id}/toggle")
async def api_toggle_schedule(schedule_id: str):
    spec = await asyncio.to_thread(load_schedule, schedule_id)
    if not spec:
        return JSONResponse(status_code=404, content={"error": "Schedule not found"})

    spec.enabled = not spec.enabled
    await asyncio.to_thread(save_schedule, spec)

    scheduler = get_scheduler()
    if spec.enabled:
        register_job(scheduler, spec.id, spec.cron_expression)
    else:
        remove_job(scheduler, spec.id)

    return spec.model_dump(mode="json")


@router.post("/{schedule_id}/run-now")
async def api_run_schedule_now(schedule_id: str):
    spec = await asyncio.to_thread(load_schedule, schedule_id)
    if not spec:
        return JSONResponse(status_code=404, content={"error": "Schedule not found"})

    from backend.scheduler import run_schedule_immediately

    run_schedule_immediately(schedule_id)

    return {"status": "triggered"}


@router.post("/{schedule_id}/stop")
async def api_stop_schedule(schedule_id: str):
    spec = await asyncio.to_thread(load_schedule, schedule_id)
    if not spec:
        return JSONResponse(status_code=404, content={"error": "Schedule not found"})

    from backend.scheduler import _running_schedule_tasks, stop_schedule_run

    cancelled = stop_schedule_run(schedule_id)
    if not cancelled:
        if spec.last_status == "running":
            spec.last_status = "error"
            spec.last_error = "Run lost — no active task found"
            await asyncio.to_thread(save_schedule, spec)
        return JSONResponse(status_code=409, content={"error": "No active run to stop"})

    task = _running_schedule_tasks.get(schedule_id)
    if task and not task.done():
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    return {"status": "cancelled"}


@router.post("/{schedule_id}/open-folder")
async def api_open_schedule_folder(schedule_id: str):
    from backend.utils import open_in_file_manager

    if (resp := _reject_unsafe_ids(schedule_id)) is not None:
        return resp
    d = schedule_dir(schedule_id)
    if not d.exists():
        return JSONResponse(status_code=404, content={"error": "Schedule not found"})

    try:
        open_in_file_manager(d)
        return {"status": "opened", "path": str(d)}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


# ---------------------------------------------------------------------------
# Run history
# ---------------------------------------------------------------------------

@router.get("/{schedule_id}/runs")
async def api_list_runs(
    schedule_id: str,
    limit: int = 20,
    offset: int = 0,
    after: Optional[str] = None,
    before: Optional[str] = None,
    status: Optional[str] = None,
    search: Optional[str] = None,
):
    from backend.scheduler import is_schedule_running, load_runs_paginated, _save_run

    if not await asyncio.to_thread(load_schedule, schedule_id):
        return JSONResponse(status_code=404, content={"error": "Schedule not found"})

    after_dt = datetime.fromisoformat(after) if after else None
    before_dt = datetime.fromisoformat(before) if before else None

    runs, total = await asyncio.to_thread(
        load_runs_paginated, schedule_id, limit, offset, after_dt, before_dt, status or None, search or None
    )
    actually_running = is_schedule_running(schedule_id)
    for run in runs:
        if run.status == "running" and not actually_running:
            run.status = "cancelled"
            run.error = "Interrupted — server restarted while running"
            run.finished_at = datetime.now(timezone.utc)
            await asyncio.to_thread(_save_run, run)
    await asyncio.to_thread(_attach_eval_fields, runs)
    return {"runs": [r.model_dump(mode="json") for r in runs], "total": total}


@router.get("/{schedule_id}/runs/stats")
async def api_run_stats(
    schedule_id: str,
    after: Optional[str] = None,
    before: Optional[str] = None,
    status: Optional[str] = None,
    search: Optional[str] = None,
):
    from backend.scheduler import compute_run_stats

    if not await asyncio.to_thread(load_schedule, schedule_id):
        return JSONResponse(status_code=404, content={"error": "Schedule not found"})

    after_dt = datetime.fromisoformat(after) if after else None
    before_dt = datetime.fromisoformat(before) if before else None

    return await asyncio.to_thread(
        compute_run_stats, schedule_id, after_dt, before_dt, status or None, search or None
    )


# ---------------------------------------------------------------------------
# File access (latest + per-run)
# ---------------------------------------------------------------------------

def _safe_resolve(base: Path, user_path: str) -> Path | None:
    """Resolve a user-provided path within a base directory safely."""
    resolved = (base / user_path).resolve()
    if not resolved.is_relative_to(base.resolve()):
        return None
    return resolved


@router.get("/{schedule_id}/latest")
async def api_list_latest_files(schedule_id: str):
    if (resp := _reject_unsafe_ids(schedule_id)) is not None:
        return resp
    latest = schedule_dir(schedule_id) / "latest"

    def _collect() -> list[dict[str, Any]]:
        if not latest.exists():
            return []
        return [
            {"path": fp.relative_to(latest).as_posix(), "size": fp.stat().st_size}
            for fp in sorted(latest.rglob("*")) if fp.is_file()
        ]

    return await asyncio.to_thread(_collect)


@router.get("/{schedule_id}/latest/{file_path:path}")
async def api_download_latest_file(schedule_id: str, file_path: str):
    if (resp := _reject_unsafe_ids(schedule_id)) is not None:
        return resp
    latest = schedule_dir(schedule_id) / "latest"
    resolved = _safe_resolve(latest, file_path)
    if not resolved or not resolved.is_file():
        return JSONResponse(status_code=404, content={"error": "File not found"})
    return FileResponse(path=resolved, filename=resolved.name, media_type="application/octet-stream")


@router.get("/{schedule_id}/runs/{run_id}/files")
async def api_list_run_files(schedule_id: str, run_id: str):
    if (resp := _reject_unsafe_ids(schedule_id, run_id)) is not None:
        return resp
    files_dir = runs_dir(schedule_id) / run_id / "files"

    def _collect() -> list[dict[str, Any]]:
        if not files_dir.exists():
            return []
        return [
            {"path": fp.relative_to(files_dir).as_posix(), "size": fp.stat().st_size}
            for fp in sorted(files_dir.rglob("*")) if fp.is_file()
        ]

    return await asyncio.to_thread(_collect)


@router.get("/{schedule_id}/runs/{run_id}/files/{file_path:path}")
async def api_download_run_file(schedule_id: str, run_id: str, file_path: str):
    if (resp := _reject_unsafe_ids(schedule_id, run_id)) is not None:
        return resp
    files_dir = runs_dir(schedule_id) / run_id / "files"
    resolved = _safe_resolve(files_dir, file_path)
    if not resolved or not resolved.is_file():
        return JSONResponse(status_code=404, content={"error": "File not found"})
    return FileResponse(path=resolved, filename=resolved.name, media_type="application/octet-stream")


# ---------------------------------------------------------------------------
# Attachments (input files shared across runs of a schedule)
# ---------------------------------------------------------------------------

@router.get("/{schedule_id}/attachments")
async def api_list_attachments(schedule_id: str):
    if (resp := _reject_unsafe_ids(schedule_id)) is not None:
        return resp
    if not await asyncio.to_thread(load_schedule, schedule_id):
        return JSONResponse(status_code=404, content={"error": "Schedule not found"})
    return await asyncio.to_thread(list_attachments, schedule_id)


@router.post("/{schedule_id}/attachments/{file_path:path}")
async def api_upload_attachment(
    schedule_id: str,
    file_path: str,
    file: UploadFile = File(...),
):
    """Upload a file attachment shared by every run of the schedule."""
    if (resp := _reject_unsafe_ids(schedule_id)) is not None:
        return resp
    if not await asyncio.to_thread(load_schedule, schedule_id):
        return JSONResponse(status_code=404, content={"error": "Schedule not found"})

    base = attachments_dir(schedule_id)
    resolved = _safe_resolve(base, file_path)
    if not resolved:
        return JSONResponse(status_code=400, content={"error": "Invalid path"})

    content = await file.read()

    def _write() -> None:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_bytes(content)

    await asyncio.to_thread(_write)
    return {"status": "uploaded", "path": file_path, "size": len(content)}


@router.delete("/{schedule_id}/attachments/{file_path:path}")
async def api_delete_attachment(schedule_id: str, file_path: str):
    if (resp := _reject_unsafe_ids(schedule_id)) is not None:
        return resp
    base = attachments_dir(schedule_id)
    resolved = _safe_resolve(base, file_path)
    if not resolved or not resolved.is_file():
        return JSONResponse(status_code=404, content={"error": "File not found"})
    await asyncio.to_thread(resolved.unlink)
    return {"status": "deleted", "path": file_path}


@router.post("/{schedule_id}/runs/{run_id}/open-folder")
async def api_open_run_folder(schedule_id: str, run_id: str):
    from backend.utils import open_in_file_manager

    if (resp := _reject_unsafe_ids(schedule_id, run_id)) is not None:
        return resp
    run_dir = runs_dir(schedule_id) / run_id
    if not run_dir.exists():
        return JSONResponse(status_code=404, content={"error": "Run not found"})

    files_dir = run_dir / "files"
    target = files_dir if files_dir.exists() else run_dir

    try:
        open_in_file_manager(target)
        return {"status": "opened", "path": str(target)}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
