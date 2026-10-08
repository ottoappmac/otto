"""Apply Quick-run limits once a turn's depth has been chosen.

Deep is today's behaviour. Quick adds a short instruction, and rejects
``write_todos`` plus any task call after the first, so a trivial turn
cannot fan out. The smaller tool-call budget lives on
:class:`middleware.tool_call_budget.ToolCallBudgetMiddleware`, which reads
the same ``run_depth`` marker.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)

QUICK_ADDENDUM = (
    "<run_depth>\n"
    "This turn is a QUICK run. Answer directly.\n"
    "- Prefer one or two tool calls. Do not explore.\n"
    "- Do not call write_todos.\n"
    "- Do not fan out. At most one task call, and only if you cannot do the "
    "work with your own tools.\n"
    "- If you cannot finish in a few steps, give the best answer you have "
    "and say what is left.\n"
    "</run_depth>"
)

QUICK_SOFT_NUDGE = (
    "This is a Quick run and you have used several tool calls. Stop and "
    "answer now with what you already have. If the task needs more work, "
    "say so — the user can switch to Deep."
)

_ADDENDUM_START = "<run_depth>"
_ADDENDUM_END = "</run_depth>"
_TASK_TOOL = "task"
_TODO_TOOL = "write_todos"


def latest_run_depth(messages: list[Any]) -> str | None:
    """The depth stamped on the latest genuine user turn, if any."""
    for message in reversed(list(messages or [])):
        if not isinstance(message, HumanMessage):
            continue
        depth = (getattr(message, "additional_kwargs", None) or {}).get("run_depth")
        if depth in ("quick", "deep"):
            return depth
    return None


class RunDepthMiddleware(AgentMiddleware):
    """Quick-run prompt addendum and tool restrictions. Orchestrator only."""

    def _with_addendum(self, request: ModelRequest) -> ModelRequest:
        messages = list(getattr(request, "messages", None) or [])
        if latest_run_depth(messages) != "quick":
            return request
        existing = _system_text(request.system_message)
        if _ADDENDUM_START in existing and _ADDENDUM_END in existing:
            return request
        text = f"{existing}\n\n{QUICK_ADDENDUM}".strip() if existing else QUICK_ADDENDUM
        try:
            return request.override(system_message=SystemMessage(content=text))
        except Exception:  # pragma: no cover — defensive
            logger.debug("Run depth: could not override system message", exc_info=True)
            return request

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._with_addendum(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._with_addendum(request))

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        rejection = self._rejection(request)
        if rejection is not None:
            return rejection
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        rejection = self._rejection(request)
        if rejection is not None:
            return rejection
        return await handler(request)

    def _rejection(self, request: ToolCallRequest) -> ToolMessage | None:
        if latest_run_depth(_state_messages(request)) != "quick":
            return None
        tool_call = getattr(request, "tool_call", None) or {}
        name = tool_call.get("name") or ""
        if name == _TODO_TOOL:
            return _tool_error(
                tool_call,
                "This is a Quick run. Do not call write_todos. Answer directly.",
            )
        if name == _TASK_TOOL and _task_index(request) >= 1:
            return _tool_error(
                tool_call,
                "This is a Quick run. Do not fan out to another subagent. "
                "Answer directly with what you have.",
            )
        return None


def _task_index(request: ToolCallRequest) -> int:
    """How many earlier ``task`` calls this run already contains.

    Parallel calls in one assistant message are ordered by id, so the
    first is allowed and the rest are rejected.
    """
    current_id = (getattr(request, "tool_call", None) or {}).get("id")
    ids: list[str] = []
    counting = False
    for message in _state_messages(request):
        if isinstance(message, HumanMessage):
            depth = (getattr(message, "additional_kwargs", None) or {}).get("run_depth")
            text = message.content if isinstance(message.content, str) else ""
            if depth in ("quick", "deep") or (
                text and not text.startswith("Date check failed.")
            ):
                # A genuine user turn starts a new count. Nudges have neither
                # a depth stamp nor are the user's text; reset only when this
                # looks like a real turn (has a depth stamp or isn't a nudge).
                if depth in ("quick", "deep"):
                    ids = []
                    counting = True
            continue
        if not counting:
            continue
        if isinstance(message, AIMessage):
            for call in getattr(message, "tool_calls", None) or []:
                if (call.get("name") or "") == _TASK_TOOL:
                    ids.append(call.get("id") or "")
    if current_id and current_id in ids:
        return ids.index(current_id)
    return len(ids)


def _state_messages(request: ToolCallRequest) -> list[Any]:
    state = getattr(request, "state", None)
    messages = None
    if isinstance(state, dict):
        messages = state.get("messages")
    else:
        messages = getattr(state, "messages", None)
    return list(messages or [])


def _tool_error(tool_call: dict, text: str) -> ToolMessage:
    return ToolMessage(
        content=text,
        name=tool_call.get("name") or "",
        tool_call_id=tool_call.get("id") or "",
        status="error",
    )


def _system_text(system_message: Any) -> str:
    if system_message is None:
        return ""
    try:
        from middleware._react_core import content_to_text

        return content_to_text(system_message.content)
    except Exception:  # pragma: no cover — defensive
        content = getattr(system_message, "content", "")
        return content if isinstance(content, str) else str(content)
