"""Shared singleton state used by route modules and the main app."""

from __future__ import annotations

import asyncio

from backend.session_manager import SessionManager
from backend.mcp_manager import MCPManager

session_mgr = SessionManager()
mcp_mgr = MCPManager()

running_tasks: dict[str, asyncio.Task] = {}
message_queues: dict[str, asyncio.Queue] = {}
context_queues: dict[str, asyncio.Queue] = {}

# Session ids whose HITL resume has been accepted and is still running.
# Duplicate ``hitl_response`` messages (Always-allow + the 2s message poll
# wiping client-side ``resolved``) must not cancel that task — doing so
# SIGTERMs the in-flight ``execute`` and restarts it in a loop.
hitl_resume_inflight: set[str] = set()

# Session ids the user has asked to stop.  Long-running, non-cancellable
# work (e.g. the macOS desktop agent, which blocks in ``asyncio.to_thread``
# wrapping pyautogui / Accessibility / osascript calls that cannot be
# force-killed) checks this set cooperatively at step boundaries and
# unwinds promptly instead of running to completion after a /stop.
stop_requested: set[str] = set()

# Loop-guard escalations, keyed by session id then subagent invocation id.
# A value is the short human-readable reason.  Set by ``ToolLoopGuard``'s
# escalation callback (via ``streaming_subagent.request_loop_abort_current``)
# when a model keeps looping despite repeated corrective messages.  Each
# subagent stream checks only its own invocation id, so one runaway subagent
# unwinds with a partial answer and its siblings keep running.  User Stop
# still cancels the whole session via ``stop_requested``.
loop_abort_requested: dict[str, dict[str, str]] = {}

# Subagent invocation tasks keyed by session id.  Parallel subagents are
# scheduled as their own asyncio tasks; cancelling the top-level run task
# does not propagate to these orphans, so /stop cancels them explicitly.
subagent_tasks: dict[str, set[asyncio.Task]] = {}
