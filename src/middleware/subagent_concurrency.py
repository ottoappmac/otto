"""Cap how many ``task`` (subagent) calls run at the same time.

**Why this exists**

When the orchestrator emits several ``task(...)`` calls in one turn, the
agent runtime executes them concurrently.  On a hosted API that is fine, but
on a local inference server (oMLX, exo, MLX) every subagent is a separate
long-context request: a research fan-out of six subagents holds six
~50–70k-token KV caches at once, which blows through the server's memory
budget and surfaces as ``Prefill context too large for available memory``.

**What it does**

Wraps the ``task`` tool with a semaphore so at most ``max_parallel`` subagents
run at once.  Excess calls simply wait for a free slot — nothing is dropped,
the model is not asked to retry, and no extra tokens are spent.  All other
tools pass through untouched.

The limit is scoped to one middleware instance (one orchestrator graph /
session), so unrelated sessions never block each other and a session that
spawns a child session cannot deadlock on a shared slot.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

logger = logging.getLogger(__name__)

TASK_TOOL_NAME = "task"


class SubagentConcurrencyMiddleware(AgentMiddleware):
    """Limit concurrent ``task`` tool executions to *max_parallel*.

    ``max_parallel <= 0`` disables the limit (pass-through).
    """

    def __init__(self, max_parallel: int) -> None:
        super().__init__()
        self._max_parallel = max(0, int(max_parallel))
        self._thread_sem = (
            threading.BoundedSemaphore(self._max_parallel)
            if self._max_parallel
            else None
        )
        # asyncio primitives bind to an event loop on first use, so create
        # one lazily per running loop rather than at construction time.
        self._async_sems: dict[int, asyncio.Semaphore] = {}

    @property
    def max_parallel(self) -> int:
        return self._max_parallel

    def _applies(self, request: ToolCallRequest) -> bool:
        if not self._max_parallel:
            return False
        name = (request.tool_call or {}).get("name")
        if name is None and request.tool is not None:
            name = getattr(request.tool, "name", None)
        return name == TASK_TOOL_NAME

    def _async_sem(self) -> asyncio.Semaphore:
        loop_id = id(asyncio.get_running_loop())
        sem = self._async_sems.get(loop_id)
        if sem is None:
            sem = asyncio.Semaphore(self._max_parallel)
            self._async_sems[loop_id] = sem
        return sem

    # ── AgentMiddleware hooks ──────────────────────────────────────────────

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        if not self._applies(request):
            return handler(request)
        assert self._thread_sem is not None
        with self._thread_sem:
            return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        if not self._applies(request):
            return await handler(request)
        async with self._async_sem():
            return await handler(request)


def maybe_for_environment() -> SubagentConcurrencyMiddleware | None:
    """Build the middleware from ``Environment``; ``None`` when unlimited."""
    from utilities.environment import Environment

    limit = Environment.get_max_parallel_subagents()
    if limit <= 0:
        return None
    logger.info("SubagentConcurrencyMiddleware: capping parallel subagents at %d", limit)
    return SubagentConcurrencyMiddleware(limit)
