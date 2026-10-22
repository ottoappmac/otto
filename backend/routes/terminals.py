"""Terminals for a Code session: list, open, close, and a live WebSocket.

The WebSocket carries raw terminal bytes out (binary frames) and JSON in
(``input``, ``resize``, ``interrupt``). A new connection first receives the
retained scrollback, so a tab that was closed and reopened shows what it missed.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Body, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from backend import terminal
from backend.terminal_context import build_context, open_terminal, session_mode

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["terminals"])
ws_router = APIRouter(tags=["terminals"])


def _info(term: terminal.Terminal) -> dict:
    info = term.info()
    info["ports"] = terminal.listening_ports(term) if term.kind == "process" and term.alive else []
    return info


@router.get("/{session_id}/terminals")
async def api_list_terminals(session_id: str):
    terms = terminal.list_terminals(session_id)
    return {"terminals": await asyncio.to_thread(lambda: [_info(t) for t in terms])}


@router.post("/{session_id}/terminals")
async def api_open_terminal(session_id: str, body: dict = Body(default={})):
    rows = int(body.get("rows") or 24)
    cols = int(body.get("cols") or 100)
    result = await asyncio.to_thread(open_terminal, session_id, kind="shell", rows=rows, cols=cols)
    if not result.get("ok"):
        return JSONResponse(status_code=409, content={"error": result.get("error")})
    return _info(result["terminal"])


@router.delete("/{session_id}/terminals/{terminal_id}")
async def api_close_terminal(session_id: str, terminal_id: str):
    if not terminal.close(session_id, terminal_id):
        return JSONResponse(status_code=404, content={"error": "No such terminal."})
    return {"status": "closed"}


@ws_router.websocket("/ws/terminal/{session_id}/{terminal_id}")
async def websocket_terminal(websocket: WebSocket, session_id: str, terminal_id: str):
    await websocket.accept()
    term = terminal.get(session_id, terminal_id)
    if term is None:
        await websocket.send_json({"type": "error", "content": "No such terminal."})
        await websocket.close()
        return

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    def on_data(data: bytes) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, data)

    def on_exit(_code: int | None) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, None)

    backlog = term.attach(on_data)
    term.exit_listeners.append(on_exit)
    if not term.alive:
        queue.put_nowait(None)

    async def pump() -> None:
        if backlog:
            await websocket.send_bytes(backlog)
        while True:
            chunk = await queue.get()
            if chunk is None:
                # Flush anything the reader queued before the exit signal.
                while not queue.empty():
                    rest = queue.get_nowait()
                    if rest:
                        await websocket.send_bytes(rest)
                await websocket.send_json({"type": "exit", "code": term.exit_code})
                return
            await websocket.send_bytes(chunk)

    pump_task = asyncio.create_task(pump())
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            kind = msg.get("type")
            if kind == "input" and isinstance(msg.get("data"), str):
                term.write(msg["data"].encode("utf-8"))
            elif kind == "resize":
                term.resize(int(msg.get("rows") or 0), int(msg.get("cols") or 0))
            elif kind == "interrupt":
                term.interrupt()
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        logger.debug("terminal websocket ended", exc_info=True)
    finally:
        term.detach(on_data)
        if on_exit in term.exit_listeners:
            term.exit_listeners.remove(on_exit)
        pump_task.cancel()


def _dev_state(session_id: str) -> tuple[dict | None, JSONResponse | None]:
    """The dev-server settings of the session's folder."""
    from backend.output_analysis import detect_dev_command
    from backend.session_manager import _read_workspace_meta
    from backend.workspace_grants import get_grant

    workspace = _read_workspace_meta(session_id)
    grant = get_grant(workspace.grant_id) if workspace is not None and workspace.grant_id else None
    if workspace is None or grant is None:
        return None, JSONResponse(status_code=404, content={"error": "No allowed folder is mapped."})
    from pathlib import Path
    detected = detect_dev_command(Path(workspace.host_path))
    command = (grant.dev_command or "").strip() or detected
    running = next(
        (t for t in terminal.list_terminals(session_id) if t.kind == "process" and t.alive and (t.command or "").strip() == command),
        None,
    )
    return {
        "command": command,
        "configured": bool((grant.dev_command or "").strip()),
        "detected": detected,
        "autostart": bool(grant.dev_autostart),
        "running_id": running.id if running else None,
    }, None


@router.get("/{session_id}/dev")
async def api_dev_state(session_id: str):
    state, err = await asyncio.to_thread(_dev_state, session_id)
    return err if err is not None else state


@router.post("/{session_id}/dev/start")
async def api_dev_start(session_id: str):
    """Start the folder's dev server. Does nothing when it is already running."""
    state, err = await asyncio.to_thread(_dev_state, session_id)
    if err is not None:
        return err
    if not state["command"]:
        return JSONResponse(status_code=409, content={"error": "No dev command is set, and none was detected."})
    if state["running_id"]:
        return {**state, "started": False}
    mode = session_mode(session_id)
    if mode in ("plan", "ask"):
        return JSONResponse(status_code=409, content={"error": f"{mode.capitalize()} mode cannot start a dev server."})
    ctx, reason = await asyncio.to_thread(build_context, session_id)
    if ctx is None:
        return JSONResponse(status_code=409, content={"error": reason})
    if ctx.readonly:
        return JSONResponse(status_code=409, content={"error": "This folder is read-only, so a dev server cannot start."})
    result = await asyncio.to_thread(open_terminal, session_id, kind="process", command=state["command"])
    if not result.get("ok"):
        return JSONResponse(status_code=409, content={"error": result.get("error")})
    return {**state, "running_id": result["terminal"].id, "started": True}
