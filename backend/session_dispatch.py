"""Helpers that kick off agent runs against an existing session.

This module exists to keep agent-facing tools (e.g.
``spawn_followup_session``) decoupled from the route layer where the
WebSocket queue and ``running_tasks`` registry live.  Tools call into
this module; this module reaches into the route helpers via lazy
imports so we don't reintroduce the cyclic dependency that would arise
from importing route handlers at module-load time.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


async def kick_off_message(session_id: str, prompt: str) -> None:
    """Schedule the agent task for *session_id* with *prompt* as the first
    user message.

    Mirrors what the WebSocket handler does on a normal incoming user
    message: ensures a queue exists, cancels any in-flight task on the
    same session (there shouldn't be one for a freshly-spawned child,
    but the guard makes this safe to call repeatedly), and registers
    the new task in ``running_tasks`` so the existing ``/stop`` and
    cleanup paths work for spawned sessions too.

    The frontend picks up the streaming output by opening the chat
    WebSocket for *session_id* — the queue will replay everything
    that's accumulated by the time the user navigates over.
    """
    from backend.routes.sessions import _run_agent_stream
    from backend.state import message_queues, running_tasks

    if session_id not in message_queues:
        message_queues[session_id] = asyncio.Queue()
    queue = message_queues[session_id]

    prev = running_tasks.get(session_id)
    if prev and not prev.done():
        prev.cancel()
        try:
            await prev
        except (asyncio.CancelledError, Exception):
            pass

    task = asyncio.create_task(_run_agent_stream(session_id, prompt, queue))
    running_tasks[session_id] = task
    logger.info("Kicked off message on session %s (prompt %d chars)", session_id, len(prompt))
