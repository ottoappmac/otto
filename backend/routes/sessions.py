"""Session management and WebSocket API routes."""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from backend.config import AppConfig
from backend.schemas import SessionCreateRequest
from backend.session_manager import (
    _append_message_async,
    _load_messages_async,
    _session_files_dir,
    _sessions_dir,
)
from backend.state import (
    context_queues,
    message_queues,
    running_tasks,
    session_mgr,
    stop_requested,
    subagent_tasks,
)
from backend.streaming_subagent import reset_subagent_queue, set_subagent_queue

logger = logging.getLogger(__name__)


async def _maybe_index_file(path: Path) -> None:
    """Background-index a newly uploaded or symlinked file.

    Guarded by ``memory.embedding_enabled`` — soft-fails on any error.
    """
    try:
        cfg = await AppConfig.aload()
        if not cfg.memory.embedding_enabled:
            return
        from backend.embedding_index import get_embedding_index
        idx = await get_embedding_index()
        asyncio.create_task(idx.index_file(path))
    except Exception:
        logger.debug("File index trigger failed for %s", path, exc_info=True)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])
ws_router = APIRouter(tags=["websocket"])


@router.get("")
async def api_list_sessions():
    active = session_mgr.list_active()
    history = await asyncio.to_thread(session_mgr.list_history)
    seen = {s.id for s in active}
    combined = [s.model_dump() for s in active]
    for s in history:
        if s.id in seen:
            continue
        d = s.model_dump()
        # Sessions saved before status-tracking was added remain "idle".
        # Infer "completed" for any non-active session that received messages.
        if d.get("status") == "idle" and (d.get("message_count") or 0) > 0:
            d["status"] = "completed"
        combined.append(d)
    return combined


@router.post("")
async def api_create_session(req: SessionCreateRequest):
    try:
        cfg = await AppConfig.aload()
        session = await session_mgr.create_session(
            config=cfg,
            agent_name=req.agent_name,
            trigger_source=req.trigger_source,
        )
        return session.to_info().model_dump()
    except Exception as exc:
        # Surface privacy-lock rejections as a 403 with a readable message
        # rather than the generic 500 — the provider is blocked before any
        # LLM client is constructed.
        try:
            from backend.privacy_lock import PrivacyLockActive
            if isinstance(exc, PrivacyLockActive):
                provider = cfg.llm.provider
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "privacy_lock",
                        "detail": (
                            f"Privacy Lock is engaged. Provider '{provider}' sends data "
                            "off-device and is blocked. Start a new session with a local "
                            "provider (afm, mlx, omlx, exo) or disengage the lock in "
                            "Settings → Privacy & Security."
                        ),
                        "llm_provider": provider,
                    },
                )
        except ImportError:
            pass
        logger.error("Session creation failed: %s\n%s", exc, traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": "Session creation failed. Check backend logs for details."},
        )


@router.get("/{session_id}")
async def api_get_session(session_id: str):
    """Return the SessionInfo for a single session.

    Reads from the in-memory active map first, falling back to the
    on-disk meta file so the endpoint works for closed sessions too.
    Used by the chat page to render parent/child link badges without
    fetching the entire session list.
    """
    active = session_mgr.get_session(session_id)
    if active is not None:
        return active.to_info().model_dump()

    meta_path = _sessions_dir() / f"{session_id}.json"
    if not await asyncio.to_thread(meta_path.exists):
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    try:
        raw = await asyncio.to_thread(meta_path.read_text, "utf-8")
        return json.loads(raw)
    except Exception as exc:
        logger.warning("Failed to read session meta for %s: %s", session_id, exc)
        return JSONResponse(status_code=500, content={"error": "Failed to read session metadata"})


@router.get("/{session_id}/messages")
async def api_get_session_messages(session_id: str):
    return await _load_messages_async(session_id)


@router.get("/{session_id}/graph")
async def api_session_graph(session_id: str):
    """Reconstruct the agent delegation graph from persisted messages.

    Returns nodes and edges representing the orchestrator + subagent + tool-call
    tree so the frontend can render a visual canvas without re-implementing the
    reconstruction logic.
    """
    messages = await _load_messages_async(session_id)
    return _build_graph_from_messages(messages)


def _build_graph_from_messages(messages: list[dict]) -> dict:
    """Build a graph (nodes + edges) from a flat message list.

    Node kinds: ``orchestrator`` | ``subagent`` | ``tool``
    Node statuses: ``done`` | ``running``
    """
    nodes: dict[str, dict] = {
        "orch": {"id": "orch", "label": "Orchestrator", "kind": "orchestrator",
                 "status": "running", "toolCount": 0}
    }
    edges: list[dict] = []
    edge_set: set[str] = set()

    def add_edge(f: str, t: str) -> None:
        k = f"{f}\u2192{t}"
        if k not in edge_set:
            edge_set.add(k)
            edges.append({"from": f, "to": t})

    # task tool_call_id → {type, nodeId}
    task_calls: dict[str, dict] = {}
    # subagent display name → node id
    display_to_node: dict[str, str] = {}
    # tool_call_id → tool node id
    tc_to_tool: dict[str, str] = {}
    auto_id = 0

    session_done = any(m.get("type") == "done" for m in messages)

    for msg in messages:
        meta = msg.get("metadata") or {}
        sub_display: str | None = meta.get("subagent")
        tc_id: str | None = meta.get("tool_call_id")
        mtype = msg.get("type", "")
        content = msg.get("content", "")

        if mtype == "tool_call":
            if not sub_display:
                if content == "task":
                    args = meta.get("args") or {}
                    sa_type = str(args.get("subagent_type", "subagent"))
                    auto_id += 1
                    node_id = f"sa-{tc_id or auto_id}"
                    nodes[node_id] = {"id": node_id, "label": sa_type, "kind": "subagent",
                                      "status": "running", "toolCount": 0}
                    add_edge("orch", node_id)
                    if tc_id:
                        task_calls[tc_id] = {"type": sa_type, "nodeId": node_id}
                else:
                    nodes["orch"]["toolCount"] += 1
            else:
                sa_node_id = display_to_node.get(sub_display)
                if not sa_node_id:
                    base = sub_display.rsplit(" #", 1)[0] if " #" in sub_display else sub_display
                    for _tcid, info in task_calls.items():
                        if info["type"] == base:
                            sa_node_id = info["nodeId"]
                            display_to_node[sub_display] = sa_node_id
                            break
                if not sa_node_id:
                    auto_id += 1
                    sa_node_id = f"sa-disp-{auto_id}"
                    nodes[sa_node_id] = {"id": sa_node_id, "label": sub_display, "kind": "subagent",
                                         "status": "running", "toolCount": 0}
                    add_edge("orch", sa_node_id)
                    display_to_node[sub_display] = sa_node_id
                nodes[sa_node_id]["toolCount"] += 1
                auto_id += 1
                tool_id = f"tool-{sa_node_id}-{tc_id or auto_id}"
                nodes[tool_id] = {
                    "id": tool_id, "label": content, "kind": "tool",
                    "status": "running", "toolCount": 0,
                }
                add_edge(sa_node_id, tool_id)
                if tc_id:
                    tc_to_tool[tc_id] = tool_id

        elif mtype == "tool_result":
            if not sub_display and tc_id and tc_id in task_calls:
                sa_node_id = task_calls[tc_id]["nodeId"]
                if sa_node_id in nodes:
                    nodes[sa_node_id]["status"] = "done"
            elif tc_id and tc_id in tc_to_tool:
                tool_id = tc_to_tool[tc_id]
                if tool_id in nodes:
                    nodes[tool_id]["status"] = "done"

    if session_done:
        for n in nodes.values():
            n["status"] = "done"

    return {"nodes": list(nodes.values()), "edges": edges}


@router.get("/{session_id}/timeline")
async def api_session_timeline(session_id: str):
    """Return a normalized, time-stamped event timeline for a session.

    Events are read from the transcript ``.jsonl`` (full content + ``ts``
    timestamps) merged with UI message metadata (tool args, subagent names,
    images, etc.) from ``.messages.json``.  Per-tool durations are derived
    from consecutive ``ts`` deltas between ``tool_call`` and ``tool_result``
    events for matching ``tool_call_id``.
    """
    import json as _json
    from backend.config import get_app_data_dir

    transcripts_dir = get_app_data_dir() / "transcripts"
    transcript_path = transcripts_dir / f"{session_id}.jsonl"

    # Fall back gracefully: if no transcript exists load from messages
    events: list[dict] = []
    if await asyncio.to_thread(transcript_path.exists):
        def _read_transcript() -> list[dict]:
            out = []
            with transcript_path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            out.append(_json.loads(line))
                        except Exception:
                            pass
            return out
        events = await asyncio.to_thread(_read_transcript)

    # Load UI messages for metadata (args, subagent, images, stats)
    ui_msgs = await _load_messages_async(session_id)

    # Build a lookup from tool_call_id → ui message metadata
    ui_meta_by_tcid: dict[str, dict] = {}
    for m in ui_msgs:
        tc_id = (m.get("metadata") or {}).get("tool_call_id")
        if tc_id:
            ui_meta_by_tcid[tc_id] = m.get("metadata") or {}

    # Compute per-tool durations from transcript timestamps
    # pending: tool_call_id → (tool_name, ts_str)
    pending_calls: dict[str, tuple[str, str]] = {}
    call_durations: dict[str, int] = {}  # tool_call_id → duration_ms

    for ev in events:
        tc_id = ev.get("tool_call_id")
        ts = ev.get("ts")
        if ev.get("type") == "tool_call" and tc_id and ts:
            pending_calls[tc_id] = (ev.get("tool") or "", ts)
        elif ev.get("type") == "tool_result" and tc_id and ts and tc_id in pending_calls:
            try:
                from datetime import datetime
                t_start = datetime.fromisoformat(pending_calls[tc_id][1].replace("Z", "+00:00"))
                t_end = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                call_durations[tc_id] = int((t_end - t_start).total_seconds() * 1000)
            except Exception:
                pass
            del pending_calls[tc_id]

    # Normalise events into frontend-friendly shape
    normalized: list[dict] = []
    for ev in events:
        tc_id = ev.get("tool_call_id")
        ui_m = ui_meta_by_tcid.get(tc_id or "", {}) if tc_id else {}
        entry: dict = {
            "ts": ev.get("ts"),
            "type": ev.get("type"),
            "role": ev.get("role"),
            "tool": ev.get("tool"),
            "tool_call_id": tc_id,
            "content": ev.get("content"),
            "meta": ev.get("meta") or {},
            # Augment from UI message metadata
            "args": (ev.get("meta") or {}).get("args") or ui_m.get("args"),
            "subagent": (ev.get("meta") or {}).get("subagent") or ui_m.get("subagent"),
            "images": ui_m.get("images"),
            "stats": (ev.get("meta") or {}).get("stats") or ui_m.get("stats"),
            "duration_ms": call_durations.get(tc_id or ""),
        }
        # Strip None values to keep payload lean
        entry = {k: v for k, v in entry.items() if v is not None}
        normalized.append(entry)

    return {"session_id": session_id, "events": normalized, "total": len(normalized)}


@router.get("/{session_id}/status")
async def api_session_status(session_id: str):
    active = session_mgr.get_session(session_id) is not None
    meta_path = _sessions_dir() / f"{session_id}.json"
    has_meta = await asyncio.to_thread(meta_path.exists)
    running = session_id in running_tasks and not running_tasks[session_id].done()
    return {"active": active or has_meta, "running": running}


@router.post("/{session_id}/stop")
async def api_stop_session(session_id: str):
    # Flag the session as stopping *before* cancelling anything so that any
    # blocking, non-cancellable subagent work (the macOS desktop agent in
    # particular) sees the request at its next step boundary and unwinds
    # cooperatively rather than running to completion.
    stop_requested.add(session_id)

    task = running_tasks.pop(session_id, None)
    if task and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    # Cancel any parallel subagent invocations that were scheduled as their
    # own tasks — cancelling the top-level run task above does not propagate
    # to these orphans, which is how a "stopped" desktop agent kept running.
    for sub_task in list(subagent_tasks.get(session_id, ())):
        if not sub_task.done():
            sub_task.cancel()
    orphans = [t for t in subagent_tasks.get(session_id, ()) if not t.done()]
    if orphans:
        await asyncio.gather(*orphans, return_exceptions=True)
    subagent_tasks.pop(session_id, None)

    queue = message_queues.get(session_id)
    if queue:
        await queue.put({"type": "stopped", "content": "Agent stopped by user"})
    # Stamp terminal status on the session
    session = session_mgr.get_session(session_id)
    if session:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        session.status = "stopped"
        session.finished_at = now
        session.duration_ms = int((now - session.created_at).total_seconds() * 1000)
        await session.save_meta_async()
    return {"status": "stopped"}


@router.delete("/{session_id}")
async def api_delete_session(session_id: str):
    session = session_mgr.get_session(session_id)
    schedule_id = getattr(session, "schedule_id", None) if session else None

    stop_requested.add(session_id)
    task = running_tasks.pop(session_id, None)
    if task and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    for sub_task in list(subagent_tasks.get(session_id, ())):
        if not sub_task.done():
            sub_task.cancel()
    orphans = [t for t in subagent_tasks.get(session_id, ()) if not t.done()]
    if orphans:
        await asyncio.gather(*orphans, return_exceptions=True)
    subagent_tasks.pop(session_id, None)

    if schedule_id:
        from backend.scheduler import _running_schedule_tasks
        sched_task = _running_schedule_tasks.get(schedule_id)
        if sched_task and not sched_task.done():
            sched_task.cancel()
            try:
                await sched_task
            except (asyncio.CancelledError, Exception):
                pass

    message_queues.pop(session_id, None)
    stop_requested.discard(session_id)
    await session_mgr.delete_session(session_id)
    return {"status": "deleted"}


@router.delete("")
async def api_delete_all_sessions():
    """Delete every session (history clear-all)."""
    # Collect IDs from in-memory active sessions first, then scan the
    # sessions directory so we catch persisted sessions that exceed the
    # paginated list_history cap.
    all_ids: list[str] = [s.id for s in session_mgr.list_active()]
    seen: set[str] = set(all_ids)
    for p in _sessions_dir().glob("*.json"):
        if p.name.endswith((".messages.json", ".eval.json")):
            continue
        sid = p.stem
        if sid not in seen:
            all_ids.append(sid)
            seen.add(sid)

    deleted = 0
    for sid in all_ids:
        stop_requested.add(sid)
        task = running_tasks.pop(sid, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        for sub_task in list(subagent_tasks.get(sid, ())):
            if not sub_task.done():
                sub_task.cancel()
        orphans = [t for t in subagent_tasks.get(sid, ()) if not t.done()]
        if orphans:
            await asyncio.gather(*orphans, return_exceptions=True)
        subagent_tasks.pop(sid, None)
        message_queues.pop(sid, None)
        stop_requested.discard(sid)
        try:
            await session_mgr.delete_session(sid)
            deleted += 1
        except Exception:  # noqa: BLE001
            pass
    return {"status": "deleted", "count": deleted}


@router.get("/{session_id}/files")
async def api_list_session_files(session_id: str):
    """List files created by the agent in this session."""
    files_dir = _session_files_dir(session_id)

    def _collect() -> list[dict]:
        if not files_dir.exists():
            return []
        results = []
        for fp in sorted(files_dir.rglob("*")):
            if not fp.is_file():
                continue
            rel = fp.relative_to(files_dir).as_posix()
            try:
                stat = fp.stat()
                results.append({
                    "path": rel,
                    "size": stat.st_size,
                    "modified_at": stat.st_mtime,
                })
            except OSError:
                results.append({"path": rel, "size": 0, "modified_at": 0})
        return results

    return await asyncio.to_thread(_collect)


@router.post("/{session_id}/files/open-folder")
async def api_open_session_files_folder(session_id: str, path: str | None = None):
    """Open a session file's folder in the OS file manager.

    If *path* is given (relative to the session files dir), reveals that
    file's parent folder (selecting the file on macOS).  Otherwise opens
    the root session files directory.
    """
    import platform
    import subprocess

    files_dir = _session_files_dir(session_id)
    if not files_dir.exists():
        return JSONResponse(status_code=404, content={"error": "Files directory not found"})

    if path:
        target = (files_dir / path).resolve()
        if not target.is_relative_to(files_dir.resolve()):
            return JSONResponse(status_code=400, content={"error": "Invalid path"})
        folder = target.parent if target.is_file() else target
    else:
        target = None
        folder = files_dir

    if not folder.exists():
        folder = files_dir

    system = platform.system()
    try:
        if system == "Darwin":
            if target and target.is_file():
                subprocess.Popen(["open", "-R", str(target)])
            else:
                subprocess.Popen(["open", str(folder)])
        elif system == "Windows":
            if target and target.is_file():
                subprocess.Popen(["explorer", "/select,", str(target)])
            else:
                subprocess.Popen(["explorer", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])
        return {"status": "opened", "path": str(folder)}
    except Exception as exc:
        logger.warning("Failed to open folder: %s", exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})


@router.get("/{session_id}/files/{file_path:path}")
async def api_download_session_file(session_id: str, file_path: str):
    """Download a specific file created by the agent."""
    import mimetypes
    files_dir = _session_files_dir(session_id)
    resolved = (files_dir / file_path).resolve()
    if not resolved.is_relative_to(files_dir.resolve()):
        return JSONResponse(status_code=400, content={"error": "Invalid path"})
    if not resolved.is_file():
        return JSONResponse(status_code=404, content={"error": "File not found"})
    media_type, _ = mimetypes.guess_type(str(resolved))
    if not media_type:
        media_type = "application/octet-stream"
    return FileResponse(
        path=resolved,
        filename=Path(file_path).name,
        media_type=media_type,
    )


@router.post("/{session_id}/files/{file_path:path}")
async def api_upload_session_file(
    session_id: str,
    file_path: str,
    file: UploadFile = File(...),
):
    """Upload a file to the session's filesystem so the agent can read it."""
    files_dir = _session_files_dir(session_id)
    resolved = (files_dir / file_path).resolve()
    if not resolved.is_relative_to(files_dir.resolve()):
        return JSONResponse(status_code=400, content={"error": "Invalid path"})
    resolved.parent.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    await asyncio.to_thread(resolved.write_bytes, content)
    asyncio.create_task(_maybe_index_file(resolved))
    return {"status": "uploaded", "path": file_path, "size": len(content)}


@router.post("/{session_id}/links")
async def api_create_session_link(
    session_id: str,
    body: dict = Body(...),
):
    """Create a symlink inside the session's files dir pointing at an
    absolute OS path on the host filesystem.

    This lets the user drag-and-drop a local file or folder into the chat
    without copying bytes — the agent sees it via the standard virtual
    path tools (ls, read_file, view_image, grep, glob).  Read-only by
    convention; writes through the link follow the symlink to the real
    file, which is usually undesirable but allowed for now.

    Returns the session-relative virtual path (e.g. "/links/photo.png")
    that can be embedded in the next user message.
    """
    src_raw = body.get("source", "")
    if not isinstance(src_raw, str) or not src_raw:
        return JSONResponse(status_code=400, content={"error": "source is required"})

    src = Path(src_raw).expanduser().resolve()
    if not src.exists():
        return JSONResponse(status_code=404, content={"error": f"source not found: {src_raw}"})

    files_dir = _session_files_dir(session_id)
    links_dir = files_dir / "links"
    await asyncio.to_thread(links_dir.mkdir, parents=True, exist_ok=True)

    base = src.name or "link"
    target = links_dir / base
    n = 1
    while await asyncio.to_thread(target.exists):
        stem, suffix = src.stem or "link", src.suffix
        target = links_dir / f"{stem}-{n}{suffix}"
        n += 1

    await asyncio.to_thread(target.symlink_to, src)
    if src.is_file():
        asyncio.create_task(_maybe_index_file(src))
    return {
        "path": f"/links/{target.name}",
        "is_dir": src.is_dir(),
        "source": str(src),
    }


@router.delete("/{session_id}/files/{file_path:path}")
async def api_delete_session_file(session_id: str, file_path: str):
    """Delete a file from the session's filesystem."""
    files_dir = _session_files_dir(session_id)
    resolved = (files_dir / file_path).resolve()
    if not resolved.is_relative_to(files_dir.resolve()):
        return JSONResponse(status_code=400, content={"error": "Invalid path"})
    if not resolved.is_file():
        return JSONResponse(status_code=404, content={"error": "File not found"})
    await asyncio.to_thread(resolved.unlink)
    return {"status": "deleted", "path": file_path}


# ---- Agent streaming helpers ----

def _exc_chain(exc: BaseException) -> list[BaseException]:
    """Return the full exception chain (exc + all __cause__ / __context__ links)."""
    seen: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in seen:
        seen.append(current)
        current = current.__cause__ or (
            current.__context__ if not current.__suppress_context__ else None
        )
    return seen


def _friendly_error(exc: Exception) -> tuple[str, str]:
    """Turn common backend exceptions into (user_message, error_code)."""
    msg = str(exc)
    exc_type = type(exc).__name__

    if "ClosedResourceError" in exc_type or "ClosedResourceError" in msg:
        return (
            "The MCP server connection was lost. "
            "Please check that the required MCP servers are running "
            "(Tools page) and start a new chat session.",
            "mcp_connection",
        )

    if any(k in exc_type for k in ("APIConnectionError", "APITimeoutError")):
        # Walk the exception chain looking for the specific RemoteProtocolError
        # that oMLX (and other local servers) produce when the inference worker
        # crashes or is OOM-killed mid-stream.  The pattern is distinctive and
        # warrants a more actionable message than the generic "Connection error."
        chain_msg = " ".join(
            str(e) for e in _exc_chain(exc)
        ).lower()
        if "incomplete chunked read" in chain_msg or "peer closed connection" in chain_msg:
            # oMLX returns HTTP 200 then drops the connection when inference
            # fails server-side, so the exception alone can't tell us why.
            # Consult oMLX's own server log for the real cause instead of
            # guessing OOM (which is frequently wrong — e.g. the VLM Metal
            # stream bug fails with plenty of memory free).
            try:
                from backend.omlx_provisioner import diagnose_omlx_stream_drop

                return diagnose_omlx_stream_drop()
            except Exception:  # noqa: BLE001
                logger.debug("oMLX log diagnosis failed", exc_info=True)
                return (
                    "The local LLM server (oMLX) dropped the streaming "
                    "connection mid-response. Restart oMLX from Settings → "
                    "LLM → oMLX and retry; if it persists, check the oMLX "
                    "server log (~/.omlx/logs/server.log).",
                    "llm_connection",
                )
        return (f"Lost connection to the LLM provider: {msg}", "llm_connection")

    if "RateLimitError" in exc_type or "rate_limit" in msg.lower() or "429" in msg:
        return ("LLM rate limit exceeded. The request will be retried automatically, or try again shortly.", "llm_rate_limit")

    if "AuthenticationError" in exc_type or "Could not resolve authentication" in msg:
        return (
            "No valid API key or credentials configured. "
            "Go to Settings and configure your Anthropic API key or Bedrock credentials.",
            "llm_auth",
        )

    if "overloaded" in msg.lower() or "529" in msg:
        return ("The LLM provider is temporarily overloaded. Please try again in a moment.", "llm_overloaded")

    if "ConnectError" in msg or "connection failed" in msg.lower():
        return (f"Failed to connect to a service: {msg}", "connection")

    return (msg, "internal")


async def _persist_subagent_to_transcript(session_id: str, item: dict) -> None:
    """Mirror a subagent queue event into the session transcript ``.jsonl``.

    Subagent steps are otherwise written only to ``.messages.json``; the run
    timeline is built from the transcript, so without this they never appear.
    Only items carrying a ``subagent`` marker are persisted — orchestrator
    events are already written to the transcript by ``stream_message`` and
    must not be duplicated here.
    """
    metadata = item.get("metadata") or {}
    subagent = metadata.get("subagent")
    if not subagent:
        return

    from backend.session_transcript import append_event_async as _transcript

    rtype = item.get("type")
    try:
        if rtype == "agent":
            content = item.get("content")
            if not content:
                return
            await _transcript(
                session_id, "assistant", content,
                role="assistant", metadata={"subagent": subagent},
            )
        elif rtype == "tool_call":
            await _transcript(
                session_id, "tool_call", metadata.get("args") or {},
                tool_name=item.get("content") or "",
                tool_call_id=metadata.get("tool_call_id"),
                metadata={"subagent": subagent, "args": metadata.get("args") or {}},
            )
        elif rtype == "tool_result":
            tr_meta: dict[str, Any] = {"subagent": subagent}
            if metadata.get("images"):
                tr_meta["images"] = metadata["images"]
            await _transcript(
                session_id, "tool_result", item.get("content") or "",
                tool_name=metadata.get("name") or "tool",
                tool_call_id=metadata.get("tool_call_id"),
                metadata=tr_meta,
            )
    except Exception:  # noqa: BLE001 — transcript mirroring must never break streaming
        logger.debug("Failed to mirror subagent event to transcript", exc_info=True)


class _PersistingSubagentQueue:
    """Wraps an asyncio.Queue to also persist subagent messages to the
    session's messages file so they survive page navigations."""

    __slots__ = ("_inner", "_session_id")

    def __init__(self, inner: asyncio.Queue, session_id: str) -> None:
        self._inner = inner
        self._session_id = session_id

    async def put(self, item: dict) -> None:
        await _append_message_async(self._session_id, item)
        await _persist_subagent_to_transcript(self._session_id, item)
        await self._inner.put(item)


class _LazyPersistingSubagentQueue:
    """Like _PersistingSubagentQueue but resolves the WS queue lazily.

    For scheduled runs the WebSocket queue may not exist when streaming
    begins.  This wrapper always persists to disk and forwards to the WS
    queue if a client has connected by the time a message is produced.
    """

    __slots__ = ("_session_id",)

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id

    async def put(self, item: dict) -> None:
        await _append_message_async(self._session_id, item)
        await _persist_subagent_to_transcript(self._session_id, item)
        queue = message_queues.get(self._session_id)
        if queue is not None:
            await queue.put(item)


async def _run_agent_stream_loop(
    session_id: str,
    stream_iter: Any,
    queue: asyncio.Queue,
    label: str,
) -> None:
    """Generic runner: drain *stream_iter*, relay to *queue*, handle errors."""
    # A fresh run clears any stale stop request from a previous turn so the
    # cooperative-cancellation checks don't abort this run immediately.
    stop_requested.discard(session_id)
    # Likewise clear any loop-guard escalation abort flag from a prior turn.
    from backend.state import loop_abort_requested
    loop_abort_requested.pop(session_id, None)
    persisting_queue = _PersistingSubagentQueue(queue, session_id)
    token = set_subagent_queue(persisting_queue, session_id=session_id)
    sent_terminal = False
    was_cancelled = False
    has_pending_context = False
    try:
        async for response in stream_iter:
            rtype = response.get("type")
            if rtype in ("done", "hitl_request", "ask_user", "stopped"):
                sent_terminal = True
            await queue.put(response)
    except asyncio.CancelledError:
        was_cancelled = True
        logger.info("%s cancelled (stopped) for session %s", label, session_id)
    except Exception as exc:
        user_msg, error_code = _friendly_error(exc)
        logger.warning(
            "%s error [%s] %s: %s",
            label, session_id, type(exc).__name__, exc,
            exc_info=True,
        )
        # Stamp error status on the session
        session = session_mgr.get_session(session_id)
        if session:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            session.status = "error"
            session.error = str(exc)
            session.error_code = error_code
            session.finished_at = now
            session.duration_ms = int((now - session.created_at).total_seconds() * 1000)
            await session.save_meta_async()
            # Best-effort: analyze the failure for a prompt fix (gated by
            # evaluation.analyze_errors). Reads from the just-saved meta.
            await session_mgr.maybe_analyze_error(session_id)
        await queue.put({"type": "error", "content": user_msg, "metadata": {"error_code": error_code}})
    finally:
        reset_subagent_queue(token, session_id=session_id)
        # Check for pending context before deciding whether to send done.
        # If context is queued and no terminal event was sent (e.g. no HITL pause),
        # suppress done so the frontend stays in streaming mode while the context
        # stream fires immediately after.
        if not was_cancelled and not sent_terminal:
            ctx_q = context_queues.get(session_id)
            has_pending_context = bool(ctx_q and not ctx_q.empty())
        if not sent_terminal and not was_cancelled and not has_pending_context:
            await queue.put({"type": "done", "content": ""})
        running_tasks.pop(session_id, None)

    # Drain any context the user injected while the agent was running.
    # Merge all queued messages into a single turn to avoid rapid-fire sessions.
    if not was_cancelled and not sent_terminal and has_pending_context:
        ctx_q = context_queues.get(session_id)
        if ctx_q:
            contexts: list[str] = []
            while not ctx_q.empty():
                try:
                    contexts.append(ctx_q.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if contexts:
                combined = "\n\n".join(contexts)
                new_task = asyncio.create_task(
                    _run_agent_stream(session_id, combined, queue)
                )
                running_tasks[session_id] = new_task


async def _run_agent_stream(session_id: str, query: str, queue: asyncio.Queue) -> None:
    """Run the agent in a background task, pushing responses to the queue."""
    # Create the context queue eagerly (setdefault) so it exists for the whole
    # run and is the SAME object the WS handler appends to when the user injects
    # context mid-run.  Using .get() here would return None (the queue is only
    # lazily created when context first arrives, which is *after* this task
    # starts), silently disabling Level 2 injection.
    ctx_q = context_queues.setdefault(session_id, asyncio.Queue())
    await _run_agent_stream_loop(
        session_id,
        session_mgr.stream_message(session_id, query, context_queue=ctx_q),
        queue,
        "Agent stream",
    )


async def _run_agent_edit(
    session_id: str,
    message_index: int,
    new_content: str,
    queue: asyncio.Queue,
) -> None:
    """Edit a user message and replay the agent from that checkpoint."""
    await _run_agent_stream_loop(
        session_id, session_mgr.stream_edit(session_id, message_index, new_content), queue, "Agent edit",
    )


async def _run_agent_resume(
    session_id: str,
    decisions: list,
    queue: asyncio.Queue,
) -> None:
    """Resume the agent after a HITL decision."""
    await _run_agent_stream_loop(
        session_id, session_mgr.stream_resume(session_id, decisions), queue, "HITL resume",
    )


@ws_router.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    await websocket.accept()

    if session_id not in message_queues:
        message_queues[session_id] = asyncio.Queue()
    queue = message_queues[session_id]

    # Re-emit any pending interrupt on (re)connect.  In-memory queues are
    # lost on backend restart and individual messages can be lost if WS
    # send fails mid-flight, so the LangGraph checkpoint is the single
    # source of truth for "is the agent waiting on the user?".  If the
    # last persisted message is the same interrupt, skip — the client
    # already has it from history.
    try:
        pending_msg = await session_mgr.get_pending_interrupt(session_id)
    except Exception:
        logger.debug("get_pending_interrupt failed for %s", session_id, exc_info=True)
        pending_msg = None
    if pending_msg is not None:
        already_persisted = False
        try:
            history = await _load_messages_async(session_id)
            if history:
                last = history[-1]
                if (
                    last.get("type") == pending_msg.get("type")
                    and last.get("content") == pending_msg.get("content")
                    and last.get("metadata") == pending_msg.get("metadata")
                ):
                    already_persisted = True
        except Exception:
            logger.debug("history check failed for %s", session_id, exc_info=True)
        if not already_persisted:
            await _append_message_async(session_id, pending_msg)
        await queue.put(pending_msg)

    try:
        while True:
            ws_task = asyncio.create_task(websocket.receive_text())
            queue_task = asyncio.create_task(queue.get())

            done, pending = await asyncio.wait(
                {ws_task, queue_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()

            for task in done:
                if task is ws_task:
                    data = task.result()
                    msg = json.loads(data)
                    msg_type = msg.get("type")

                    # add_context: enqueue for injection without interrupting the
                    # running agent.  An acknowledgement event is pushed back to the
                    # client immediately so it can render the context bubble.
                    if msg_type == "add_context":
                        content = msg.get("content", "").strip()
                        if content:
                            ctx_q = context_queues.setdefault(
                                session_id, asyncio.Queue()
                            )
                            await ctx_q.put(content)
                            ack = {"type": "context_received", "content": content}
                            await _append_message_async(session_id, ack)
                            await queue.put(ack)
                        # Do not cancel the running task or create a new one.
                        continue

                    prev_task = running_tasks.get(session_id)
                    if prev_task and not prev_task.done():
                        prev_task.cancel()
                        try:
                            await prev_task
                        except (asyncio.CancelledError, Exception):
                            pass

                    if msg_type == "edit":
                        agent_task = asyncio.create_task(
                            _run_agent_edit(
                                session_id,
                                msg["message_index"],
                                msg["content"],
                                queue,
                            )
                        )
                    elif msg_type == "hitl_response":
                        agent_task = asyncio.create_task(
                            _run_agent_resume(
                                session_id,
                                msg.get("decisions", []),
                                queue,
                            )
                        )
                    else:
                        content = msg.get("content", "")
                        agent_task = asyncio.create_task(
                            _run_agent_stream(session_id, content, queue)
                        )
                    running_tasks[session_id] = agent_task
                elif task is queue_task:
                    response = task.result()
                    await websocket.send_json(response)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: %s (agent continues in background)", session_id)
        task = running_tasks.get(session_id)
        if not task or task.done():
            message_queues.pop(session_id, None)
    except Exception as exc:
        logger.exception("WebSocket error: %s", exc)
        try:
            await websocket.send_json({"type": "error", "content": str(exc)})
        except Exception:
            pass
