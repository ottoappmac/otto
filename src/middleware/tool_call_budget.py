"""Per-run tool-call budget guard.

``recursion_limit`` (default 10 000 graph steps) is far too coarse to catch a
run that thrashes — e.g. the stock-report run that fired 185 tool calls
(dozens of redundant ``browser_navigate`` / ``browser_snapshot`` /
``web_research``) before producing a broken answer.  The per-tool
:class:`tools.loop_guard.ToolLoopGuard` only trips on *repeated identical*
calls, and observation tools are largely exempt, so non-identical churn sails
straight past it.

This middleware adds a simple, provider-agnostic ceiling on the number of tool
calls in a single run:

* at the **soft budget** — append a one-shot transient nudge steering the model
  to stop exploring and produce its final answer / write its files now;
* at the **hard budget** — short-circuit with a terminal ``AIMessage`` (no tool
  calls) so the agent loop ends gracefully with whatever it has, instead of
  burning the full recursion budget.

The count is derived from the request's message history (tool calls in
``AIMessage``s since the last genuine user turn), so it is correctly scoped per
run and per graph invocation — parallel subagents that share a session id are
never conflated, and a multi-turn chat resets each user message.  Transient
nudges injected by this guard or the repeated-thought guard are recognised and
do **not** reset the counter.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Optional

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

logger = logging.getLogger(__name__)

_SOFT_NUDGE_TEXT = (
    "You have made many tool calls in this run. Stop exploring and converge "
    "now: produce your final answer and write any required output files with "
    "the information you already have. Do not start new research or browser "
    "navigation unless it is strictly required to finish."
)

_TERMINAL_TEXT = (
    "Stopping: this run reached its tool-call budget without converging. "
    "Returning the results gathered so far rather than continuing to call "
    "tools."
)


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    try:
        from middleware._react_core import content_to_text

        return content_to_text(content)
    except Exception:  # pragma: no cover — defensive
        return content if isinstance(content, str) else str(content)


def _injected_nudge_texts() -> set[str]:
    """Texts known to be transient guard nudges (not real user turns)."""
    texts = {_SOFT_NUDGE_TEXT}
    try:
        from middleware.repeated_thought_guard import _NUDGE_TEXT

        texts.add(_NUDGE_TEXT)
    except Exception:  # pragma: no cover — defensive
        pass
    return texts


def count_run_tool_calls(messages: list[BaseMessage]) -> int:
    """Count tool calls in the current run (since the last genuine user turn).

    Walks the whole history but resets the tally whenever a *genuine* user
    ``HumanMessage`` is seen, so a multi-turn chat is scoped per user message.
    Injected guard nudges (which are also ``HumanMessage``s) do not reset it.
    """
    injected = _injected_nudge_texts()
    count = 0
    for msg in messages:
        if isinstance(msg, HumanMessage):
            if _content_to_text(msg.content) not in injected:
                count = 0
        elif isinstance(msg, AIMessage):
            count += len(getattr(msg, "tool_calls", None) or [])
    return count


class ToolCallBudgetMiddleware(AgentMiddleware):
    """Nudge then gracefully stop a run that exceeds its tool-call budget."""

    def __init__(
        self,
        *,
        soft_budget: int | None = None,
        hard_budget: int | None = None,
    ) -> None:
        from utilities.environment import Environment

        self._soft = (
            soft_budget if soft_budget is not None
            else Environment.get_tool_call_soft_budget()
        )
        self._hard = (
            hard_budget if hard_budget is not None
            else Environment.get_tool_call_hard_budget()
        )

    # ── AgentMiddleware hooks ───────────────────────────────────────────────

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        decision = self._decide(request)
        if decision == "abort":
            return self._terminal_response()
        if decision == "nudge":
            request = self._with_nudge(request)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        decision = self._decide(request)
        if decision == "abort":
            return self._terminal_response()
        if decision == "nudge":
            request = self._with_nudge(request)
        return await handler(request)

    # ── Private helpers ─────────────────────────────────────────────────────

    def _decide(self, request: ModelRequest) -> Optional[str]:
        messages = list(getattr(request, "messages", None) or [])
        count = count_run_tool_calls(messages)

        if self._hard and count >= self._hard:
            logger.warning(
                "ToolCallBudget: %d tool calls >= hard budget %d — ending run "
                "with partial result.",
                count, self._hard,
            )
            return "abort"
        if self._soft and count >= self._soft:
            logger.warning(
                "ToolCallBudget: %d tool calls >= soft budget %d — nudging to "
                "converge.",
                count, self._soft,
            )
            return "nudge"
        return None

    def _with_nudge(self, request: ModelRequest) -> ModelRequest:
        messages = list(getattr(request, "messages", None) or [])
        messages.append(HumanMessage(content=_SOFT_NUDGE_TEXT))
        try:
            return request.override(messages=messages)
        except Exception:  # pragma: no cover — defensive
            return request

    def _terminal_response(self) -> ModelResponse:
        return ModelResponse(
            result=[AIMessage(content=_TERMINAL_TEXT)],
            structured_response=None,
        )
