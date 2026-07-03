"""LangChain tool: ``spawn_followup_session``.

When the orchestrator creates new MCP tools or sub-agents mid-turn, the
graph for the current session is already compiled with the *old* tool
list — the new resources won't be callable until the graph is rebuilt.
Rebuilding mid-stream would mean rewinding the LangGraph checkpoint
and re-firing the user message inside the same session (see Option C
in earlier design notes), which works but adds complexity around
checkpoint correctness, infinite-loop prevention, and partial-state
discard.

The cleaner alternative wired up here is to **hand off to a fresh
session**.  The orchestrator finishes its current turn ("I built the
tool — continuing in a new session…"), the backend creates a child
session whose graph already includes the new resources, and the child
auto-fires the original (or rephrased) user prompt.  The parent session
stays in history with its build narrative; the child session contains
the actual answer.

Trade-offs:
* Pro: no checkpoint surgery; uses the same paths every normal session
  uses.  Easier to reason about and debug.
* Pro: failure is visible — if the child's graph is also missing a
  tool, the agent can spawn a grandchild (capped) instead of looping
  invisibly inside one session.
* Con: the child doesn't see the parent's chat history.  The
  orchestrator should pass enough context in *prompt* to make the
  child's first turn self-contained.

The chain depth is capped by
:attr:`backend.session_manager.SessionManager.MAX_SESSION_CHAIN_DEPTH`
to prevent runaway recursion.
"""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def build_spawn_tools(parent_session_id: str) -> list:
    """Return the agent-facing session-spawn tools.

    The parent session ID is closed over so the tool body doesn't need
    LangGraph runtime state to know which session it's spawning from —
    it's resolved at graph-build time in
    :meth:`backend.session_manager.SessionManager._build_graph`.
    """

    @tool
    async def spawn_followup_session(prompt: str) -> str:
        """Hand off the current request to a fresh session.

        Use this when you've created new MCP tools or sub-agents during
        this turn that aren't bound to the current graph yet, OR when
        you want a clean context for a follow-up task that the user
        approved.  The new session:

        * Inherits your selected agent (if any).
        * Shares this session's files directory, so anything you wrote
          during the build phase (plans, scratch notes, downloaded
          assets) is immediately available.
        * Starts with ``prompt`` as its first user message.
        * Is linked back to this session via ``parent_session_id`` so
          the UI can show a "← from parent" badge.

        The chain has a hard depth cap (default 2 — parent → child →
        grandchild).  Beyond that this tool returns an error so a
        looping agent can't spawn forever.

        Args:
            prompt: First user message for the child session.  Must be
                    self-contained — the child does NOT inherit this
                    session's chat history.  Typical pattern: paraphrase
                    the original user request, optionally adding a note
                    about which new tool to use ("Charge $10 to the test
                    Stripe customer using the stripe-mcp tools just
                    installed.").

        Returns:
            JSON describing the spawned session (id, title, agent_name).
            Tell the user about the new session in your reply so they
            know to follow it.
        """
        # Async on purpose: ``kick_off_message`` registers an
        # ``asyncio.create_task`` for the child agent loop in
        # ``running_tasks``.  If this tool were sync, LangChain would
        # invoke it via ``run_coro_sync`` → ``asyncio.run(...)`` on a
        # *fresh* event loop, the child task would be created on that
        # temporary loop, and ``asyncio.run`` would cancel the task
        # the instant the tool returned.  Running the tool natively
        # async puts both the spawn coroutine *and* the child task on
        # the same FastAPI main loop, so the child survives this
        # tool's return and the orchestrator finishing its turn.
        if not prompt or not prompt.strip():
            return "Error: prompt must be a non-empty string."

        from backend.session_dispatch import kick_off_message
        from backend.state import session_mgr

        try:
            child = await session_mgr.spawn_child_session(
                parent_session_id=parent_session_id,
                prompt=prompt,
            )
        except ValueError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            logger.exception("spawn_followup_session failed")
            return f"Error: {exc}"

        try:
            await kick_off_message(child.id, prompt)
        except Exception as exc:
            logger.exception("kick_off_message failed for %s", child.id)
            return f"Error: child session created but failed to start: {exc}"

        return json.dumps({
            "child_session_id": child.id,
            "title": child.title,
            "agent_name": child.agent_name,
            "chain_depth": child.chain_depth,
            "parent_session_id": parent_session_id,
        }, indent=2)

    return [spawn_followup_session]
