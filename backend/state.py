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

# Session ids the user has asked to stop.  Long-running, non-cancellable
# work (e.g. the macOS desktop agent, which blocks in ``asyncio.to_thread``
# wrapping pyautogui / Accessibility / osascript calls that cannot be
# force-killed) checks this set cooperatively at step boundaries and
# unwinds promptly instead of running to completion after a /stop.
stop_requested: set[str] = set()

# Session ids whose loop guard has escalated past its limit, mapped to a short
# human-readable reason.  Set by ``ToolLoopGuard``'s escalation callback (via
# ``streaming_subagent.request_loop_abort``) when a model keeps looping despite
# repeated corrective messages.  Checked cooperatively at step boundaries — the
# same mechanism as ``stop_requested`` — so a runaway run unwinds gracefully
# with a partial answer instead of burning the whole recursion budget.
loop_abort_requested: dict[str, str] = {}

# Subagent invocation tasks keyed by session id.  Parallel subagents are
# scheduled as their own asyncio tasks; cancelling the top-level run task
# does not propagate to these orphans, so /stop cancels them explicitly.
subagent_tasks: dict[str, set[asyncio.Task]] = {}
