"""Trigger management API routes."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from backend.schemas import (
    TriggerCreateRequest,
    TriggerSpec,
    TriggerUpdateRequest,
)
from backend.trigger_manager import (
    MAX_TRIGGERS,
    _fix_orphaned_runs,
    _running_trigger_tasks,
    _save_run,
    delete_trigger_files,
    is_trigger_running,
    load_all_triggers,
    load_runs_paginated,
    load_trigger,
    register_job,
    remove_job,
    run_trigger_immediately,
    runs_dir,
    save_trigger,
    stop_trigger_run,
    trigger_dir,
    validate_poll_seconds,
    validate_spec,
    validate_trigger_id,
)
from backend.utils import is_safe_path_segment

router = APIRouter(prefix="/api/triggers", tags=["triggers"])


def _reject_unsafe_ids(*ids: str) -> JSONResponse | None:
    """404 early when any URL path segment could traverse the filesystem."""
    if all(is_safe_path_segment(i) for i in ids):
        return None
    return JSONResponse(status_code=404, content={"error": "Not found"})


# ---------------------------------------------------------------------------
# Status polling — must precede /{trigger_id}
# ---------------------------------------------------------------------------

@router.get("/status/poll")
async def api_trigger_status() -> dict[str, Any]:
    triggers = await asyncio.to_thread(load_all_triggers)
    running: list[str] = []
    recently_completed: list[dict[str, Any]] = []

    for s in triggers:
        if s.last_status == "running":
            if is_trigger_running(s.id):
                running.append(s.id)
            else:
                s.last_status = "error"
                s.last_error = "Run lost — no active task found"
                await asyncio.to_thread(save_trigger, s)
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
async def api_list_triggers() -> list[dict[str, Any]]:
    triggers = await asyncio.to_thread(load_all_triggers)
    return [s.model_dump(mode="json") for s in triggers]


@router.post("")
async def api_create_trigger(req: TriggerCreateRequest):
    existing = await asyncio.to_thread(load_all_triggers)
    custom_count = sum(1 for t in existing if not t.builtin)
    if custom_count >= MAX_TRIGGERS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Maximum of {MAX_TRIGGERS} triggers reached."},
        )

    id_err = validate_trigger_id(req.id)
    if id_err:
        return JSONResponse(status_code=400, content={"error": id_err})

    if await asyncio.to_thread(load_trigger, req.id):
        return JSONResponse(
            status_code=409,
            content={"error": f"Trigger '{req.id}' already exists."},
        )

    if req.agent_name:
        from backend.agent_library import get_agent
        if not await asyncio.to_thread(get_agent, req.agent_name):
            return JSONResponse(
                status_code=400,
                content={"error": f"Agent '{req.agent_name}' not found."},
            )

    spec = TriggerSpec(
        id=req.id,
        type=req.type,
        prompt=req.prompt,
        poll_seconds=req.poll_seconds,
        agent_name=req.agent_name,
        # fileos
        path=req.path,
        watch=req.watch,
        glob=req.glob,
        # macostool
        script=req.script,
        language=req.language,
        match=req.match,
        # http
        url=req.url,
        http_mode=req.http_mode,
        method=req.method,
        headers=req.headers or {},
        body=req.body,
        json_path=req.json_path,
        # git
        repo_path=req.repo_path,
        branch=req.branch or "HEAD",
        author_filter=req.author_filter,
        path_filter=req.path_filter,
        # shell
        command=req.command,
        shell_mode=req.shell_mode,
        cwd=req.cwd,
        env=req.env or {},
    )
    spec_err = validate_spec(spec)
    if spec_err:
        return JSONResponse(status_code=400, content={"error": spec_err})

    await asyncio.to_thread(save_trigger, spec)
    register_job(spec.id, spec.poll_seconds)

    return spec.model_dump(mode="json")


@router.get("/{trigger_id}")
async def api_get_trigger(trigger_id: str):
    spec = await asyncio.to_thread(load_trigger, trigger_id)
    if not spec:
        return JSONResponse(status_code=404, content={"error": "Trigger not found"})
    return spec.model_dump(mode="json")


@router.put("/{trigger_id}")
async def api_update_trigger(trigger_id: str, req: TriggerUpdateRequest):
    spec = await asyncio.to_thread(load_trigger, trigger_id)
    if not spec:
        return JSONResponse(status_code=404, content={"error": "Trigger not found"})

    if req.prompt is not None:
        spec.prompt = req.prompt
    if req.poll_seconds is not None:
        err = validate_poll_seconds(req.poll_seconds)
        if err:
            return JSONResponse(status_code=400, content={"error": err})
        spec.poll_seconds = req.poll_seconds
    if req.enabled is not None:
        spec.enabled = req.enabled
    if req.agent_name is not None:
        if req.agent_name:
            from backend.agent_library import get_agent
            if not await asyncio.to_thread(get_agent, req.agent_name):
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Agent '{req.agent_name}' not found."},
                )
        spec.agent_name = req.agent_name or None
    if req.path is not None:
        spec.path = req.path
    if req.watch is not None:
        spec.watch = req.watch
    if req.glob is not None:
        spec.glob = req.glob
    if req.script is not None:
        spec.script = req.script
    if req.language is not None:
        spec.language = req.language
    if req.match is not None:
        spec.match = req.match or None
    # http
    if req.url is not None:
        spec.url = req.url or None
    if req.http_mode is not None:
        spec.http_mode = req.http_mode
    if req.method is not None:
        spec.method = req.method
    if req.headers is not None:
        spec.headers = req.headers
    if req.body is not None:
        spec.body = req.body or None
    if req.json_path is not None:
        spec.json_path = req.json_path or None
    # git
    if req.repo_path is not None:
        spec.repo_path = req.repo_path or None
    if req.branch is not None:
        spec.branch = req.branch or "HEAD"
    if req.author_filter is not None:
        spec.author_filter = req.author_filter or None
    if req.path_filter is not None:
        spec.path_filter = req.path_filter or None
    # shell
    if req.command is not None:
        spec.command = req.command or None
    if req.shell_mode is not None:
        spec.shell_mode = req.shell_mode
    if req.cwd is not None:
        spec.cwd = req.cwd or None
    if req.env is not None:
        spec.env = req.env
    if req.keep_last_n_runs is not None:
        spec.keep_last_n_runs = max(1, req.keep_last_n_runs)

    spec_err = validate_spec(spec)
    if spec_err:
        return JSONResponse(status_code=400, content={"error": spec_err})

    await asyncio.to_thread(save_trigger, spec)
    if spec.enabled:
        register_job(spec.id, spec.poll_seconds)
    else:
        remove_job(spec.id)

    return spec.model_dump(mode="json")


@router.delete("/{trigger_id}")
async def api_delete_trigger(trigger_id: str):
    spec = await asyncio.to_thread(load_trigger, trigger_id)
    if not spec:
        return JSONResponse(status_code=404, content={"error": "Trigger not found"})
    if spec.builtin:
        return JSONResponse(
            status_code=403,
            content={"error": f"Trigger '{trigger_id}' is a managed built-in and cannot be deleted."},
        )
    remove_job(trigger_id)
    await asyncio.to_thread(delete_trigger_files, trigger_id)
    return {"deleted": True}


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@router.post("/{trigger_id}/toggle")
async def api_toggle_trigger(trigger_id: str):
    spec = await asyncio.to_thread(load_trigger, trigger_id)
    if not spec:
        return JSONResponse(status_code=404, content={"error": "Trigger not found"})

    spec.enabled = not spec.enabled
    await asyncio.to_thread(save_trigger, spec)
    if spec.enabled:
        register_job(spec.id, spec.poll_seconds)
    else:
        remove_job(spec.id)

    return spec.model_dump(mode="json")


@router.post("/{trigger_id}/run-now")
async def api_run_trigger_now(trigger_id: str):
    spec = await asyncio.to_thread(load_trigger, trigger_id)
    if not spec:
        return JSONResponse(status_code=404, content={"error": "Trigger not found"})
    run_trigger_immediately(trigger_id)
    return {"status": "triggered"}


@router.post("/{trigger_id}/stop")
async def api_stop_trigger(trigger_id: str):
    spec = await asyncio.to_thread(load_trigger, trigger_id)
    if not spec:
        return JSONResponse(status_code=404, content={"error": "Trigger not found"})

    cancelled = stop_trigger_run(trigger_id)
    if not cancelled:
        if spec.last_status == "running":
            spec.last_status = "error"
            spec.last_error = "Run lost — no active task found"
            await asyncio.to_thread(save_trigger, spec)
        return JSONResponse(status_code=409, content={"error": "No active run to stop"})

    task = _running_trigger_tasks.get(trigger_id)
    if task and not task.done():
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    return {"status": "cancelled"}


@router.post("/{trigger_id}/open-folder")
async def api_open_trigger_folder(trigger_id: str):
    from backend.utils import open_in_file_manager

    if (resp := _reject_unsafe_ids(trigger_id)) is not None:
        return resp
    d = trigger_dir(trigger_id)
    if not d.exists():
        return JSONResponse(status_code=404, content={"error": "Trigger not found"})
    try:
        open_in_file_manager(d)
        return {"status": "opened", "path": str(d)}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


# ---------------------------------------------------------------------------
# Run history
# ---------------------------------------------------------------------------

@router.get("/{trigger_id}/runs")
async def api_list_trigger_runs(
    trigger_id: str,
    limit: int = 20,
    offset: int = 0,
    after: Optional[str] = None,
    before: Optional[str] = None,
    status: Optional[str] = None,
):
    if not await asyncio.to_thread(load_trigger, trigger_id):
        return JSONResponse(status_code=404, content={"error": "Trigger not found"})

    after_dt = datetime.fromisoformat(after) if after else None
    before_dt = datetime.fromisoformat(before) if before else None

    runs, total = await asyncio.to_thread(
        load_runs_paginated, trigger_id, limit, offset, after_dt, before_dt, status or None
    )
    actually_running = is_trigger_running(trigger_id)
    for run in runs:
        if run.status == "running" and not actually_running:
            run.status = "cancelled"
            run.error = "Interrupted — server restarted while running"
            run.finished_at = datetime.now(timezone.utc)
            await asyncio.to_thread(_save_run, run)
    return {"runs": [r.model_dump(mode="json") for r in runs], "total": total}


# ---------------------------------------------------------------------------
# File access
# ---------------------------------------------------------------------------

def _safe_resolve(base: Path, user_path: str) -> Path | None:
    resolved = (base / user_path).resolve()
    if not resolved.is_relative_to(base.resolve()):
        return None
    return resolved


@router.get("/{trigger_id}/runs/{run_id}/files")
async def api_list_trigger_run_files(trigger_id: str, run_id: str):
    if (resp := _reject_unsafe_ids(trigger_id, run_id)) is not None:
        return resp
    files_dir = runs_dir(trigger_id) / run_id / "files"

    def _collect() -> list[dict[str, Any]]:
        if not files_dir.exists():
            return []
        return [
            {"path": fp.relative_to(files_dir).as_posix(), "size": fp.stat().st_size}
            for fp in sorted(files_dir.rglob("*")) if fp.is_file()
        ]

    return await asyncio.to_thread(_collect)


@router.get("/{trigger_id}/runs/{run_id}/files/{file_path:path}")
async def api_download_trigger_run_file(
    trigger_id: str, run_id: str, file_path: str,
):
    if (resp := _reject_unsafe_ids(trigger_id, run_id)) is not None:
        return resp
    files_dir = runs_dir(trigger_id) / run_id / "files"
    resolved = _safe_resolve(files_dir, file_path)
    if not resolved or not resolved.is_file():
        return JSONResponse(status_code=404, content={"error": "File not found"})
    return FileResponse(
        path=resolved, filename=resolved.name,
        media_type="application/octet-stream",
    )


@router.post("/{trigger_id}/runs/{run_id}/open-folder")
async def api_open_trigger_run_folder(trigger_id: str, run_id: str):
    from backend.utils import open_in_file_manager

    if (resp := _reject_unsafe_ids(trigger_id, run_id)) is not None:
        return resp
    run_dir = runs_dir(trigger_id) / run_id
    if not run_dir.exists():
        return JSONResponse(status_code=404, content={"error": "Run not found"})

    files_dir = run_dir / "files"
    target = files_dir if files_dir.exists() else run_dir

    try:
        open_in_file_manager(target)
        return {"status": "opened", "path": str(target)}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
