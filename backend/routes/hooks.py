"""HTTP hook receiver endpoints for Claude Code.

Claude Code can be configured to POST events to these endpoints via its
``type: "http"`` hook system.  Each endpoint pushes the event into the
:mod:`~tools.transcripts.hook_event_buffer` singleton so the eval MCP
server can react instantly instead of polling the JSONL file.

All endpoints return 200 immediately.  Claude Code treats non-2xx and
connection failures as non-blocking errors, so the backend being down never
disrupts a Claude Code session.

See ``backend/config.py`` → ``ClaudeHookConfig.http_hooks_enabled`` for
the toggle that gates these endpoints.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, Request, Response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hooks/claude", tags=["hooks"])

_auto_monitor_sessions: dict[str, str] = {}
_auto_monitor_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_enabled() -> bool:
    """Check whether HTTP hook receiver is turned on in settings."""
    try:
        from backend.config import AppConfig
        cfg = AppConfig.load()
        return cfg.claude_hook.enabled and cfg.claude_hook.http_hooks_enabled
    except Exception:
        logger.debug("_is_enabled check failed", exc_info=True)
        return False


def _disabled_response() -> Response:
    return Response(status_code=200, content="{}", media_type="application/json")


async def _read_payload(request: Request) -> dict[str, Any]:
    """Read and parse the JSON body from Claude Code."""
    body = await request.body()
    try:
        return json.loads(body) if body else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning("Malformed hook payload (Content-Length: %d)", len(body))
        return {}


def _push(payload: dict[str, Any]) -> None:
    """Push an event to the hook buffer."""
    session_id = payload.get("session_id", "")
    if not session_id:
        logger.debug("Hook event missing session_id — ignored")
        return

    from tools.transcripts.hook_event_buffer import hook_buffer
    hook_buffer.push(session_id, payload)

    event_name = payload.get("hook_event_name", "?")
    tool_name = payload.get("tool_name", "")
    suffix = f" ({tool_name})" if tool_name else ""
    logger.info("Hook: %s%s [session=%s]", event_name, suffix, session_id[:12])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/post-tool-use")
async def post_tool_use(request: Request) -> Response:
    """Receive PostToolUse events — a tool call completed successfully."""
    if not _is_enabled():
        return _disabled_response()
    payload = await _read_payload(request)
    _push(payload)
    return Response(status_code=200, content="{}", media_type="application/json")


@router.post("/post-tool-use-failure")
async def post_tool_use_failure(request: Request) -> Response:
    """Receive PostToolUseFailure events — a tool call failed."""
    if not _is_enabled():
        return _disabled_response()
    payload = await _read_payload(request)
    _push(payload)
    return Response(status_code=200, content="{}", media_type="application/json")


@router.post("/stop")
async def stop(request: Request) -> Response:
    """Receive Stop events — Claude finished responding.

    When quality gating is enabled, this endpoint may return a
    ``decision: "block"`` response that tells Claude Code to continue.
    """
    if not _is_enabled():
        return _disabled_response()

    payload = await _read_payload(request)
    _push(payload)

    decision = _maybe_quality_gate(payload)
    if decision is not None:
        return Response(
            status_code=200,
            content=json.dumps(decision),
            media_type="application/json",
        )

    return Response(status_code=200, content="{}", media_type="application/json")


@router.post("/subagent-stop")
async def subagent_stop(request: Request) -> Response:
    """Receive SubagentStop events — a subagent finished."""
    if not _is_enabled():
        return _disabled_response()
    payload = await _read_payload(request)
    _push(payload)
    return Response(status_code=200, content="{}", media_type="application/json")


@router.post("/session-start")
async def session_start(request: Request) -> Response:
    """Receive SessionStart events — a session began or resumed."""
    if not _is_enabled():
        return _disabled_response()
    payload = await _read_payload(request)
    _push(payload)

    if _is_auto_monitor_enabled():
        claude_session_id = payload.get("session_id", "")
        transcript_path = payload.get("transcript_path", "")
        if claude_session_id and transcript_path:
            asyncio.create_task(
                _auto_start_eval_session(claude_session_id, transcript_path),
                name=f"auto-monitor-{claude_session_id[:12]}",
            )

    return Response(status_code=200, content="{}", media_type="application/json")


@router.post("/session-end")
async def session_end(request: Request) -> Response:
    """Receive SessionEnd events — a session terminated."""
    if not _is_enabled():
        return _disabled_response()
    payload = await _read_payload(request)
    _push(payload)

    claude_session_id = payload.get("session_id", "")
    if claude_session_id:
        _auto_monitor_sessions.pop(claude_session_id, None)

    return Response(status_code=200, content="{}", media_type="application/json")


@router.get("/status")
async def hook_status() -> dict[str, Any]:
    """Health check: return whether the hook receiver is enabled and
    how many sessions are currently being tracked."""
    enabled = _is_enabled()
    count = 0
    if enabled:
        from tools.transcripts.hook_event_buffer import hook_buffer
        count = hook_buffer.active_session_count()
    return {
        "enabled": enabled,
        "active_sessions": count,
        "auto_monitor": {
            "enabled": _is_auto_monitor_enabled(),
            "active": dict(_auto_monitor_sessions),
        },
    }


# ---------------------------------------------------------------------------
# Auto-monitor
# ---------------------------------------------------------------------------

def _is_auto_monitor_enabled() -> bool:
    """Check whether auto-monitor is turned on in settings."""
    try:
        from backend.config import AppConfig
        cfg = AppConfig.load()
        return (
            cfg.claude_hook.enabled
            and cfg.claude_hook.http_hooks_enabled
            and cfg.claude_hook.auto_monitor_enabled
        )
    except Exception:
        logger.debug("_is_auto_monitor_enabled check failed", exc_info=True)
        return False


def _count_active_auto_sessions() -> int:
    """Count auto-monitor sessions that are still running."""
    from backend.state import running_tasks

    stale = [
        cid for cid, sid in _auto_monitor_sessions.items()
        if sid not in running_tasks or running_tasks[sid].done()
    ]
    for cid in stale:
        _auto_monitor_sessions.pop(cid, None)

    return len(_auto_monitor_sessions)


async def _auto_start_eval_session(
    claude_session_id: str,
    transcript_path: str,
) -> None:
    """Create an eval-agent session to monitor a Claude Code session.

    Called as a fire-and-forget ``asyncio.create_task`` from the
    ``session_start`` handler.  Guards against duplicates and respects
    the ``max_auto_sessions`` cap.
    """
    from backend.config import AppConfig
    from backend.routes.sessions import _LazyPersistingSubagentQueue
    from backend.state import running_tasks, session_mgr
    from backend.streaming_subagent import reset_subagent_queue, set_subagent_queue

    async with _auto_monitor_lock:
        if claude_session_id in _auto_monitor_sessions:
            logger.debug(
                "Auto-monitor: session %s already tracked — skipping",
                claude_session_id[:12],
            )
            return

        try:
            cfg = AppConfig.load()
        except Exception:
            logger.warning("Auto-monitor: failed to load config", exc_info=True)
            return

        max_sessions = cfg.claude_hook.max_auto_sessions
        active = _count_active_auto_sessions()
        if active >= max_sessions:
            logger.info(
                "Auto-monitor: cap reached (%d/%d) — skipping %s",
                active, max_sessions, claude_session_id[:12],
            )
            return

        agent_name = cfg.claude_hook.auto_monitor_agent or "claude-session-eval-agent"
        try:
            session = await session_mgr.create_session(
                config=cfg,
                agent_name=agent_name,
                trigger_source="claude-hook",
            )
        except Exception:
            logger.error(
                "Auto-monitor: failed to create session for %s",
                claude_session_id[:12],
                exc_info=True,
            )
            return

        _auto_monitor_sessions[claude_session_id] = session.id
        logger.info(
            "Auto-monitor: created eval session %s for Claude session %s (transcript: %s)",
            session.id[:12],
            claude_session_id[:12],
            transcript_path,
        )

    prompt = (
        f"A new Claude Code session has started and is being monitored via HTTP hooks. "
        f"Evaluate this session in real-time.\n\n"
        f"Session path: {transcript_path}\n\n"
        f"Monitor the session, wait for activity, then evaluate the quality and "
        f"efficiency of the agent's work. Report your findings when the session ends "
        f"or after significant milestones."
    )

    lazy_queue = _LazyPersistingSubagentQueue(session.id)
    token = set_subagent_queue(lazy_queue, session_id=session.id)

    async def _run() -> None:
        try:
            async for resp in session_mgr.stream_message(session.id, prompt):
                await lazy_queue.put(resp)
        except asyncio.CancelledError:
            logger.info("Auto-monitor session %s cancelled", session.id[:12])
        except Exception as exc:
            logger.error(
                "Auto-monitor session %s failed: %s",
                session.id[:12], exc,
                exc_info=True,
            )
        finally:
            reset_subagent_queue(token, session_id=session.id)
            running_tasks.pop(session.id, None)

    task = asyncio.create_task(_run(), name=f"auto-eval-{session.id[:12]}")
    running_tasks[session.id] = task


# ---------------------------------------------------------------------------
# Quality gating
# ---------------------------------------------------------------------------

def _maybe_quality_gate(payload: dict[str, Any]) -> dict[str, Any] | None:
    """If quality gating is enabled and the session has tool errors,
    return a ``decision: "block"`` dict.  Otherwise return ``None``."""
    try:
        from backend.config import AppConfig
        cfg = AppConfig.load()
    except Exception:
        logger.debug("Quality gate config load failed", exc_info=True)
        return None

    if not cfg.claude_hook.quality_gate_enabled:
        return None

    if payload.get("stop_hook_active"):
        return None

    session_id = payload.get("session_id", "")
    if not session_id:
        return None

    from tools.transcripts.hook_event_buffer import hook_buffer
    trajectory = hook_buffer.get_tool_trajectory(session_id)

    failures = [
        e for e in trajectory
        if e.get("hook_event_name") == "PostToolUseFailure"
    ]

    if not failures:
        return None

    threshold = cfg.claude_hook.quality_gate_threshold
    total = len(trajectory)
    fail_count = len(failures)
    error_rate = fail_count / total if total > 0 else 0.0

    if error_rate <= (1.0 - threshold):
        return None

    tool_names = ", ".join(
        sorted({e.get("tool_name", "?") for e in failures})
    )
    reason = (
        f"Quality gate: {fail_count}/{total} tool calls failed "
        f"(error rate {error_rate:.0%}, threshold {1 - threshold:.0%}). "
        f"Failed tools: {tool_names}. "
        f"Please address the errors before finishing."
    )

    logger.info("Quality gate BLOCKED stop for session %s: %s", session_id[:12], reason)

    return {
        "decision": "block",
        "reason": reason,
    }
