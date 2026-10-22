"""Allowed-folder grants.

A grant is the user's consent for a real directory. Sessions bind to a
grant; they do not accept an arbitrary host path on their own.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

from backend.workspace_grants import (
    get_grant,
    list_grants,
    revoke_grant,
    update_grant,
    upsert_grant,
)
from backend.workspace_policy import GrantRejected, validate_grantable

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/workspace-grants", tags=["workspaces"])


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


@router.get("")
async def api_list_grants():
    grants = await asyncio.to_thread(list_grants)
    return [g.model_dump(mode="json") for g in grants]


@router.post("/preview")
async def api_preview_grant(body: dict = Body(...)):
    """Resolve a path and say whether it can be allowed. Does not store it."""
    raw = body.get("path") or body.get("source") or ""
    if not isinstance(raw, str) or not raw.strip():
        return _error(400, "path is required")
    try:
        real = await asyncio.to_thread(validate_grantable, raw)
    except FileNotFoundError as exc:
        return _error(404, str(exc))
    except NotADirectoryError as exc:
        return _error(400, str(exc))
    except GrantRejected as exc:
        return _error(400, str(exc))
    from backend.workspace_grants import find_by_realpath
    existing = await asyncio.to_thread(find_by_realpath, str(real))
    return {
        "ok": True,
        "realpath": str(real),
        "label": real.name or "project",
        "existing_id": existing.id if existing else None,
        "existing_mode": existing.mode if existing else None,
        "coverage": (
            "Otto can read and edit files in this folder, and run commands with "
            "this folder as the working directory. A read-only grant blocks edits "
            "and shell writes here. Shell commands can still read most of the disk "
            "outside this folder; the sandbox blocks SSH keys, keychains, and Otto's own data."
        ),
    }


@router.post("/kinds")
async def api_path_kinds(body: dict = Body(...)):
    """Say whether dropped paths are files or folders. Does not read them."""
    raw_paths = body.get("paths") or []
    if not isinstance(raw_paths, list) or len(raw_paths) > 50:
        return _error(400, "paths must be a list of at most 50 paths")
    rows = []
    for raw in raw_paths:
        if not isinstance(raw, str) or not raw.strip():
            continue
        path = Path(raw).expanduser()
        rows.append({"path": raw, "is_dir": path.is_dir(), "is_file": path.is_file()})
    return {"paths": rows}


@router.post("")
async def api_create_grant(body: dict = Body(...)):
    raw = body.get("path") or body.get("source") or ""
    mode = body.get("mode") or "write"
    if mode not in ("read", "write"):
        return _error(400, "mode must be read or write")
    if not isinstance(raw, str) or not raw.strip():
        return _error(400, "path is required")
    try:
        real = await asyncio.to_thread(validate_grantable, raw)
    except FileNotFoundError as exc:
        return _error(404, str(exc))
    except NotADirectoryError as exc:
        return _error(400, str(exc))
    except GrantRejected as exc:
        return _error(400, str(exc))
    grant = await asyncio.to_thread(
        upsert_grant, real, mode=mode, source="user", label=body.get("label") or None,
    )
    return grant.model_dump(mode="json")


@router.patch("/{grant_id}")
async def api_update_grant(grant_id: str, body: dict = Body(...)):
    mode = body.get("mode")
    label = body.get("label")
    network = body.get("network") if "network" in body else None
    network_hosts = body.get("network_hosts") if "network_hosts" in body else None
    shell_allow = body.get("shell_allow") if "shell_allow" in body else None
    dev_command = body.get("dev_command") if "dev_command" in body else None
    dev_autostart = body.get("dev_autostart") if "dev_autostart" in body else None
    if dev_command is not None and not isinstance(dev_command, str):
        return _error(400, "dev_command must be text")
    if dev_autostart is not None and not isinstance(dev_autostart, bool):
        return _error(400, "dev_autostart must be true or false")
    if mode is not None and mode not in ("read", "write"):
        return _error(400, "mode must be read or write")
    if network is not None and not isinstance(network, bool):
        return _error(400, "network must be true or false")
    if network_hosts is not None and not isinstance(network_hosts, list):
        return _error(400, "network_hosts must be a list")
    if shell_allow is not None and not isinstance(shell_allow, list):
        return _error(400, "shell_allow must be a list")
    try:
        grant = await asyncio.to_thread(
            update_grant,
            grant_id,
            mode=mode,
            label=label,
            network=network,
            network_hosts=network_hosts,
            shell_allow=shell_allow,
            dev_command=dev_command,
            dev_autostart=dev_autostart,
        )
    except Exception as exc:
        return _error(400, str(exc))
    if grant is None:
        return _error(404, "Allowed folder not found")
    if mode is not None:
        await _refresh_bound_sessions(grant_id, mode)
    return grant.model_dump(mode="json")


@router.delete("/{grant_id}")
async def api_revoke_grant(grant_id: str):
    grant = get_grant(grant_id)
    if grant is None:
        return _error(404, "Allowed folder not found")
    await _unbind_sessions(grant_id)
    removed = await asyncio.to_thread(revoke_grant, grant_id)
    if removed is None:
        return _error(404, "Allowed folder not found")
    return {"status": "revoked", "id": grant_id}


async def _refresh_bound_sessions(grant_id: str, mode: str) -> None:
    from backend.state import session_mgr

    for session_id, workspace in await asyncio.to_thread(_bound, grant_id):
        workspace.mode = mode
        try:
            await session_mgr.set_workspace(session_id, workspace)
        except Exception:
            logger.debug("could not refresh workspace mode for %s", session_id, exc_info=True)


def _bound(grant_id: str) -> list[tuple[str, object]]:
    import json

    from backend.schemas import SessionWorkspace
    from backend.session_manager import _sessions_dir

    found: list[tuple[str, SessionWorkspace]] = []
    root = _sessions_dir()
    if not root.is_dir():
        return found
    for path in root.glob("*.json"):
        if path.name.endswith(".eval.json") or path.name.endswith(".workspace.json"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        raw = data.get("workspace") if isinstance(data, dict) else None
        if not isinstance(raw, dict) or raw.get("grant_id") != grant_id:
            continue
        try:
            found.append((path.stem, SessionWorkspace.model_validate(raw)))
        except Exception:
            continue
    return found


async def _unbind_sessions(grant_id: str) -> None:
    from backend.state import running_tasks, stop_requested
    from backend.session_manager import _session_files_dir
    from backend.workspace import remove_workspace_link

    from backend.state import session_mgr

    for session_id, workspace in await asyncio.to_thread(_bound, grant_id):
        stop_requested.add(session_id)
        task = running_tasks.get(session_id)
        if task and not task.done():
            task.cancel()
        try:
            files_dir = _session_files_dir(session_id)
            await asyncio.to_thread(remove_workspace_link, files_dir, workspace)
        except Exception:
            logger.debug("unlink workspace failed for %s", session_id, exc_info=True)
        try:
            await session_mgr.set_workspace(session_id, None)
        except Exception:
            logger.debug("clear workspace failed for %s", session_id, exc_info=True)
