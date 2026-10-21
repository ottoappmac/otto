"""Per-run state the middleware reads. Process-local on purpose.

LangGraph ``interrupt()`` inside a sub-agent does not reach the parent
graph (the nested ``ainvoke`` returns as if the child finished). Approvals
are therefore an asyncio future the websocket resolves while the stream
task is still blocked inside the tool. That future is not checkpointed:
a backend restart drops it, and the card answers with an error instead
of running the tool.

Session grants and taint are dicts, not context vars, so parallel
sub-agents on the same session see them. Invocation grants are a
context var set at the start of each agent run, so "allow desktop
control for this run" does not leak into a sibling agent.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import uuid
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_scope: contextvars.ContextVar["RunScope | None"] = contextvars.ContextVar(
    "otto_action_scope", default=None,
)
_invocation_grants: contextvars.ContextVar[set[str] | None] = contextvars.ContextVar(
    "otto_invocation_grants", default=None,
)

_session_grants: dict[str, set[str]] = {}
_tainted: set[str] = set()
_counts: dict[tuple[str, str], int] = {}
_pending: dict[str, dict[str, asyncio.Future]] = {}
# Cards the stream loop should yield while a tool call is waiting.
# Putting them only on the websocket queue would hide them from
# stream_message consumers, and yielding them twice would show two cards.
_approval_events: dict[str, list[dict[str, Any]]] = {}


@dataclass(frozen=True)
class RunScope:
    session_id: str
    attended: bool
    allowlist: frozenset[str]
    agent_name: str | None = None
    # agent | plan | debug. Plan denies every category that changes something.
    mode: str = "agent"


def current_scope() -> RunScope | None:
    return _scope.get()


def set_scope(scope: RunScope) -> contextvars.Token:
    return _scope.set(scope)


def reset_scope(token: contextvars.Token) -> None:
    _scope.reset(token)


def begin_invocation() -> contextvars.Token:
    """A fresh grant set for one orchestrator turn or one sub-agent run."""
    return _invocation_grants.set(set())


def end_invocation(token: contextvars.Token) -> None:
    _invocation_grants.reset(token)


def invocation_grants() -> set[str]:
    current = _invocation_grants.get()
    if current is None:
        # Tests and tool calls outside a stream. A private set so a grant
        # here cannot leak into another call.
        fresh: set[str] = set()
        _invocation_grants.set(fresh)
        return fresh
    return current


def session_grants(session_id: str) -> set[str]:
    return _session_grants.setdefault(session_id, set())


def grant_session(session_id: str, category: str) -> None:
    session_grants(session_id).add(category)


def grant_invocation(category: str) -> None:
    invocation_grants().add(category)


def mark_tainted(session_id: str) -> None:
    if session_id:
        _tainted.add(session_id)


def is_tainted(session_id: str) -> bool:
    return session_id in _tainted


def bump_count(session_id: str, category: str) -> int:
    key = (session_id, category)
    _counts[key] = _counts.get(key, 0) + 1
    return _counts[key]


def current_count(session_id: str, category: str) -> int:
    return _counts.get((session_id, category), 0)


def clear_session(session_id: str) -> None:
    """Drop grants, taint, and counters when a session is forgotten."""
    _session_grants.pop(session_id, None)
    _tainted.discard(session_id)
    for key in [key for key in _counts if key[0] == session_id]:
        _counts.pop(key, None)
    _pending.pop(session_id, None)
    _approval_events.pop(session_id, None)


def publish_approval(session_id: str, message: dict[str, Any]) -> None:
    _approval_events.setdefault(session_id, []).append(message)


def pop_approvals(session_id: str) -> list[dict[str, Any]]:
    return _approval_events.pop(session_id, [])


def has_pending(session_id: str, request_id: str) -> bool:
    fut = _pending.get(session_id, {}).get(request_id)
    return fut is not None and not fut.done()


def resolve(session_id: str, request_id: str, decisions: list[dict[str, Any]]) -> bool:
    """Complete a waiting approval. False when nothing is waiting."""
    fut = _pending.get(session_id, {}).get(request_id)
    if fut is None or fut.done():
        return False
    fut.set_result(decisions)
    return True


async def request_approval(
    session_id: str,
    message: dict[str, Any],
) -> list[dict[str, Any]]:
    """Publish ``message`` and wait until the websocket resolves it.

    Polls the stop flag so a stopped session does not wait forever.
    """
    request_id = str((message.get("metadata") or {}).get("request_id") or uuid.uuid4().hex)
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    _pending.setdefault(session_id, {})[request_id] = fut

    # The stream loop yields this. The websocket runner forwards that
    # yield, so putting it on the queue here as well would duplicate the card.
    publish_approval(session_id, message)

    try:
        from backend.session_manager import _append_message_async
        await _append_message_async(session_id, message)
    except Exception:
        logger.debug("action policy: could not persist approval card", exc_info=True)

    try:
        while True:
            from backend.state import stop_requested
            if session_id in stop_requested:
                return [{"type": "reject", "message": "Stopped before approval"}]
            try:
                return await asyncio.wait_for(asyncio.shield(fut), timeout=0.4)
            except asyncio.TimeoutError:
                continue
    finally:
        pending = _pending.get(session_id)
        if pending is not None:
            pending.pop(request_id, None)
            if not pending:
                _pending.pop(session_id, None)
