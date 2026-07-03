"""Streaming wrapper for subagents.

Relays intermediate tool calls and results from a subagent execution to the
chat session's WebSocket queue in real time, so users see progress instead of
waiting in silence while a subagent works.

The primary channel between the deep-nested ``ainvoke`` call and the WebSocket
queue is a ``contextvars.ContextVar`` set at the WebSocket handler layer.  It
propagates through all nested ``await`` calls under normal asyncio scheduling.

A secondary, explicit registry (``_queue_registry``) keyed by session ID is
maintained alongside the contextvar.  ``StreamingSubagentRunnable.ainvoke``
consults it when ``_subagent_queue.get()`` returns ``None`` — which can happen
when tasks are scheduled through alternative event loop implementations (e.g.
uvloop) that do not reliably copy contextvar state into nested tasks.
"""

from __future__ import annotations

import asyncio  # noqa: E402 (needed for Queue type hint)
import contextvars
import logging
import time
from typing import Any, Optional

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig

from backend.utils import extract_text_content

logger = logging.getLogger(__name__)

_subagent_queue: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "_subagent_queue",
    default=None,
)

# Session-keyed registry used as a fallback when contextvar propagation fails.
# Keys are session IDs; values are the same queue objects stored in the contextvar.
_queue_registry: dict[str, Any] = {}

# Session id of the subagent invocation running in the current asyncio context.
# Per-connection MCP loop guards (in ``backend.mcp_manager``) are created once at
# connect time and shared across sessions, so they cannot bind to a single
# session id.  Instead they resolve the *current* session at escalation time via
# this contextvar (set at the subagent ``ainvoke`` boundary) so a runaway loop
# aborts the right run.  See :func:`request_loop_abort_current`.
_current_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_session_id",
    default=None,
)


def set_current_session(session_id: str | None) -> contextvars.Token:
    """Bind *session_id* as the session for the current asyncio context.

    Returns a token to pass to :func:`reset_current_session`.
    """
    return _current_session_id.set(session_id)


def reset_current_session(token: contextvars.Token) -> None:
    """Restore the previously-bound current session id."""
    _current_session_id.reset(token)


def set_subagent_queue(queue: Any, *, session_id: str | None = None) -> contextvars.Token:
    """Bind *queue* as the destination for intermediate subagent events.

    When *session_id* is provided the queue is also stored in a module-level
    registry so it can be located as a fallback when contextvar propagation
    fails (e.g. with certain uvloop task-scheduling configurations).
    """
    if session_id is not None:
        _queue_registry[session_id] = queue
    return _subagent_queue.set(queue)


def reset_subagent_queue(token: contextvars.Token, *, session_id: str | None = None) -> None:
    """Restore the previous queue (or ``None``) and evict the registry entry."""
    _subagent_queue.reset(token)
    if session_id is not None:
        _queue_registry.pop(session_id, None)


def _stop_requested(session_id: str | None) -> bool:
    """Return ``True`` when the user has asked to stop *session_id*."""
    if not session_id:
        return False
    from backend.state import stop_requested
    return session_id in stop_requested


def request_loop_abort(session_id: str | None, reason: str) -> None:
    """Mark *session_id* for a cooperative abort after a loop-guard escalation.

    Called from ``ToolLoopGuard``'s ``on_escalate`` callback (wired in
    ``session_manager._apply_universal_loop_guard``) when a model keeps
    looping past the escalation limit.  The run loop checks this at the next
    step boundary and unwinds gracefully with whatever it has so far."""
    if not session_id:
        return
    from backend.state import loop_abort_requested
    loop_abort_requested[session_id] = reason
    logger.warning(
        "loop-abort requested for session %s: %s", session_id, reason
    )


def request_loop_abort_current(reason: str) -> None:
    """Mark the *current* asyncio context's session for a cooperative abort.

    Resolves the session id from the :data:`_current_session_id` contextvar
    (bound at the subagent ``ainvoke`` boundary) so per-connection MCP loop
    guards — which are shared across sessions and never see a session id at
    construction time — can still trigger the cooperative abort for the run
    that is actually looping.  No-op when no session is bound to the context.
    """
    request_loop_abort(_current_session_id.get(), reason)


def _loop_abort_requested(session_id: str | None) -> str | None:
    """Return the abort reason if a loop-guard escalation flagged the run."""
    if not session_id:
        return None
    from backend.state import loop_abort_requested
    return loop_abort_requested.get(session_id)


def _register_subagent_task(session_id: str | None) -> Any:
    """Track the current asyncio task so /stop can cancel orphaned subagents.

    Returns a token (the task) to pass back to
    :func:`_deregister_subagent_task`, or ``None`` when there is no session
    id or running task to track.
    """
    if not session_id:
        return None
    task = asyncio.current_task()
    if task is None:
        return None
    from backend.state import subagent_tasks
    subagent_tasks.setdefault(session_id, set()).add(task)
    return task


def _deregister_subagent_task(session_id: str | None, token: Any) -> None:
    """Untrack a task previously registered via :func:`_register_subagent_task`."""
    if not session_id or token is None:
        return
    from backend.state import subagent_tasks
    tasks = subagent_tasks.get(session_id)
    if tasks is None:
        return
    tasks.discard(token)
    if not tasks:
        subagent_tasks.pop(session_id, None)


def _resolve_queue_from_registry(agent_name: str, session_id: str | None = None) -> Any:
    """Return a queue from the registry when the contextvar failed to propagate.

    When *session_id* is provided (extracted from the ``RunnableConfig``'s
    ``thread_id``), the lookup is exact — no ambiguity even when dozens of
    sessions stream concurrently.  Falls back to the single-entry heuristic
    only when the caller cannot supply a session ID.
    """
    # When the session id is known, the lookup MUST be exact.  A known
    # session id that is absent from the registry means that session has
    # already been stopped/ended (its ``_run_agent_stream_loop`` finally
    # evicted it) — routing its straggler subagent events to whatever other
    # session happens to be live would "bleed" them into an unrelated chat.
    # Return ``None`` so the subagent runs without live streaming instead.
    if session_id is not None:
        if session_id in _queue_registry:
            logger.warning(
                "[%s] contextvar missed; resolved queue via session_id=%s",
                agent_name,
                session_id,
            )
            return _queue_registry[session_id]
        logger.warning(
            "[%s] contextvar missed and session_id=%s is not registered "
            "(session stopped/ended); skipping live streaming to avoid "
            "cross-session bleed",
            agent_name,
            session_id,
        )
        return None

    # Session id genuinely unavailable (no thread_id in the RunnableConfig).
    # Fall back to the single-entry heuristic only — it is unambiguous when
    # exactly one session is streaming, and unsafe otherwise.
    if not _queue_registry:
        return None

    if len(_queue_registry) == 1:
        queue = next(iter(_queue_registry.values()))
        logger.warning(
            "[%s] _subagent_queue contextvar did not propagate and no "
            "session_id was available; falling back to session registry "
            "(single-entry heuristic)",
            agent_name,
        )
        return queue

    logger.error(
        "[%s] _subagent_queue contextvar did not propagate, no session_id "
        "was available, and %d sessions are active — cannot safely determine "
        "which queue to use; subagent will run without live streaming",
        agent_name,
        len(_queue_registry),
    )
    return None


class StreamingSubagentRunnable(Runnable):
    """Wraps a compiled agent graph so ``ainvoke`` streams intermediate events.

    When the ``_subagent_queue`` context var holds a queue, ``ainvoke`` calls
    ``astream(stream_mode="values")`` on the inner graph and pushes every
    tool-call / tool-result to the queue.  The final state dict is still
    returned normally so the ``task`` tool's ``Command`` / ``ToolMessage``
    flow works unchanged.

    When no queue is available (e.g. non-WebSocket execution), ``ainvoke``
    falls through to the inner graph's standard ``ainvoke``.

    When a :class:`~backend.playwright_pool.PlaywrightPool` is attached, each
    invocation acquires an isolated browser instance for the duration of the
    run, preventing concurrent subagents from sharing a single browser.
    """

    def __init__(
        self,
        graph: Runnable,
        agent_name: str,
        *,
        pw_pool: Any = None,
    ) -> None:
        self._graph = graph
        self._agent_name = agent_name
        self._pw_pool = pw_pool  # Optional[PlaywrightPool]
        self._active: int = 0
        self._seq: int = 0

    def _enter(self) -> str:
        """Claim a display name for a new invocation.

        Resets the sequence counter when no other invocations are active
        (i.e. start of a new parallel batch).  Returns a display name
        like ``"general-purpose #2"`` so concurrent invocations of the
        same subagent are visually distinguishable in the UI.
        """
        if self._active == 0:
            self._seq = 0
        self._active += 1
        self._seq += 1
        return f"{self._agent_name} #{self._seq}"

    def _exit(self) -> None:
        self._active -= 1

    @property
    def InputType(self) -> Any:
        return self._graph.InputType

    @property
    def OutputType(self) -> Any:
        return self._graph.OutputType

    def invoke(self, input: dict, config: Optional[RunnableConfig] = None, **kw: Any) -> dict:
        return self._graph.invoke(input, config, **kw)

    async def ainvoke(self, input: dict, config: Optional[RunnableConfig] = None, **kw: Any) -> dict:
        queue = _subagent_queue.get()
        thread_id = (config or {}).get("configurable", {}).get("thread_id")
        if queue is None:
            queue = _resolve_queue_from_registry(self._agent_name, session_id=thread_id)
        display_name = self._enter()

        # Register this invocation's task so /stop can cancel it even when it
        # runs as an orphaned parallel task that the top-level run's
        # cancellation never reaches.
        sub_token = _register_subagent_task(thread_id)

        # Per-invocation desktop-lease owner. The macos-native MCP connection is
        # shared across sessions, so without this every concurrent desktop agent
        # would share one (re-entrant ⇒ useless) lease owner and fight over the
        # foreground. A unique-but-stable id per run makes the lease serialize
        # them. Harmless for non-desktop subagents (only desktop tools read it).
        from backend.builtin_mcps.macos_osascript._desktop_lock import (
            active_desktop_owner,
        )
        owner_token = active_desktop_owner.set(
            f"desktop:{thread_id or 'nosession'}:{display_name}"
        )

        # Bind the running session so per-connection MCP loop guards can
        # resolve which run to abort when they escalate (see
        # ``request_loop_abort_current``).
        session_token = set_current_session(thread_id)

        pw_instance = None
        pw_token = None
        try:
            if self._pw_pool is not None:
                from backend.playwright_pool import active_pw_instance
                pw_instance = await self._pw_pool.acquire()
                pw_token = active_pw_instance.set(pw_instance)
                logger.info(
                    "[%s] acquired Playwright instance on port %d",
                    display_name, pw_instance.port,
                )

            return await self._ainvoke_inner(input, config, queue, display_name, thread_id, **kw)
        finally:
            reset_current_session(session_token)
            active_desktop_owner.reset(owner_token)
            if pw_token is not None:
                from backend.playwright_pool import active_pw_instance
                active_pw_instance.reset(pw_token)
            if pw_instance is not None:
                await self._pw_pool.release(pw_instance)
                logger.info(
                    "[%s] released Playwright instance on port %d",
                    display_name, pw_instance.port,
                )
            _deregister_subagent_task(thread_id, sub_token)
            self._exit()

    @staticmethod
    def _merge_recursion_limit(config: Optional[RunnableConfig], limit: int) -> dict:
        """Return a plain config dict with ``recursion_limit`` injected."""
        base: dict = dict(config) if config else {}
        base["recursion_limit"] = limit
        return base

    async def _ainvoke_inner(
        self,
        input: dict,
        config: Optional[RunnableConfig],
        queue: Optional[asyncio.Queue],
        display_name: str,
        session_id: Optional[str] = None,
        **kw: Any,
    ) -> dict:
        from utilities.environment import Environment
        run_cfg = self._merge_recursion_limit(config, Environment.get_recursion_limit())

        if queue is None:
            logger.info("[%s] started (no queue)", display_name)
            t0 = time.monotonic()
            try:
                result = await self._graph.ainvoke(input, run_cfg, **kw)
                logger.info("[%s] finished in %.1fs", display_name, time.monotonic() - t0)
                return result
            except Exception:
                logger.exception("[%s] failed after %.1fs", display_name, time.monotonic() - t0)
                raise

        logger.info("[%s] started (streaming)", display_name)
        t0 = time.monotonic()
        result: dict = {}
        printed = 0
        steps = 0

        try:
            async for chunk in self._graph.astream(input, run_cfg, stream_mode="values", **kw):
                # Cooperative stop: the desktop agent blocks in
                # ``asyncio.to_thread`` calls that cannot be force-cancelled,
                # so honour a /stop request at the next step boundary by
                # unwinding instead of running the whole task to completion.
                if _stop_requested(session_id):
                    logger.info(
                        "[%s] stop requested for session %s; aborting subagent",
                        display_name, session_id,
                    )
                    raise asyncio.CancelledError()

                # Loop-guard escalation: a model that keeps looping past the
                # escalation limit is unwound gracefully here (unlike /stop,
                # we keep the partial result rather than cancelling) so the
                # parent gets a best-effort answer instead of a runaway run.
                _abort_reason = _loop_abort_requested(session_id)
                if _abort_reason is not None:
                    logger.warning(
                        "[%s] loop-abort for session %s (%s); ending subagent "
                        "with partial result",
                        display_name, session_id, _abort_reason,
                    )
                    break

                result = chunk
                if "messages" not in chunk:
                    continue

                new_msgs = chunk["messages"][printed:]
                printed = len(chunk["messages"])
                steps += len(new_msgs)

                for msg in new_msgs:
                    if isinstance(msg, AIMessage):
                        text = extract_text_content(msg.content)
                        if text:
                            await queue.put({
                                "type": "agent",
                                "content": text,
                                "metadata": {"subagent": display_name},
                            })

                        if msg.tool_calls:
                            tool_names = [tc.get("name", "") for tc in msg.tool_calls]
                            logger.debug("[%s] tool calls: %s", display_name, tool_names)
                            for tc in msg.tool_calls:
                                tc_meta: dict[str, Any] = {
                                    "args": tc.get("args", {}),
                                    "subagent": display_name,
                                }
                                tc_id = tc.get("id")
                                if tc_id:
                                    tc_meta["tool_call_id"] = tc_id
                                await queue.put({
                                    "type": "tool_call",
                                    "content": tc.get("name", ""),
                                    "metadata": tc_meta,
                                })

                    elif isinstance(msg, ToolMessage):
                        if isinstance(msg.content, list):
                            parts = [
                                b.get("text", "")
                                for b in msg.content
                                if isinstance(b, dict) and b.get("type") == "text"
                            ]
                            preview = " ".join(parts)[:500]
                        else:
                            preview = str(msg.content)[:500]

                        images: list[dict[str, str]] = []
                        if isinstance(msg.content, list):
                            for block in msg.content:
                                if not isinstance(block, dict):
                                    continue
                                # LangChain v1 standard image block.
                                if block.get("type") == "image" and block.get("base64"):
                                    images.append({
                                        "base64": block["base64"],
                                        "mime_type": block.get("mime_type", "image/png"),
                                    })
                                # OpenAI-style image_url block with a base64 data
                                # URL (what capture_app_screenshot / read_screen
                                # vision combo emit).
                                elif block.get("type") == "image_url":
                                    url = (block.get("image_url") or {}).get("url", "")
                                    if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
                                        header, b64 = url.split(";base64,", 1)
                                        mime = header[len("data:"):] or "image/png"
                                        images.append({"base64": b64, "mime_type": mime})

                        metadata: dict[str, Any] = {
                            "name": getattr(msg, "name", None) or "tool",
                            "subagent": display_name,
                        }
                        if getattr(msg, "tool_call_id", None):
                            metadata["tool_call_id"] = msg.tool_call_id
                        if images:
                            metadata["images"] = images

                        await queue.put({
                            "type": "tool_result",
                            "content": preview,
                            "metadata": metadata,
                        })

            logger.info("[%s] finished in %.1fs (%d steps)", display_name, time.monotonic() - t0, steps)
        except Exception:
            logger.exception("[%s] failed after %.1fs (%d steps)", display_name, time.monotonic() - t0, steps)
            raise

        return result
