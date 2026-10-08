"""Inject a fresh calendar and reject answers whose weekdays are wrong.

:class:`DateContextMiddleware` appends today's date and a 14-month calendar
to the system message on every model call, so a long-lived session never
keeps a stale "today".

:class:`DateValidationMiddleware` checks the final answer once and asks the
model to rewrite it when a weekday does not match the date. File writes
(``write_file`` / ``edit_file``) are rejected with the same correction so
the file is not saved until the weekdays are right.
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

# Prefix of the transient correction. Other guards treat a human message
# that starts with this as an injected nudge, not a new user turn.
DATE_CORRECTION_PREFIX = "Date check failed."

_FILE_TOOLS = frozenset({"write_file", "edit_file", "edit"})
# Only the text being written is checked. ``old_string`` is the search
# text the model is matching, not the new content.
_CONTENT_KEYS = ("content", "file_text", "new_string", "new_content", "text")


def _system_text(system_message: Any) -> str:
    if system_message is None:
        return ""
    try:
        from middleware._react_core import content_to_text

        return content_to_text(system_message.content)
    except Exception:  # pragma: no cover — defensive
        content = getattr(system_message, "content", "")
        return content if isinstance(content, str) else str(content)


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    try:
        from middleware._react_core import content_to_text

        return content_to_text(content)
    except Exception:  # pragma: no cover — defensive
        return str(content)


class DateContextMiddleware(AgentMiddleware):
    """Append a fresh calendar to the system message on every model call."""

    def _with_date(self, request: ModelRequest) -> ModelRequest:
        from backend.date_context import build_date_block, strip_date_block

        block = build_date_block()
        existing = strip_date_block(_system_text(request.system_message))
        text = f"{existing}\n\n{block}".strip() if existing else block
        try:
            return request.override(system_message=SystemMessage(content=text))
        except Exception:  # pragma: no cover — defensive
            logger.debug("Date context: could not override system message", exc_info=True)
            return request

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._with_date(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._with_date(request))


class DateValidationMiddleware(AgentMiddleware):
    """One retry on a bad final answer, and a tool error on a bad file write."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        response = handler(request)
        retried = self._retry_request(request, response)
        if retried is None:
            return response
        return handler(retried)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        response = await handler(request)
        retried = self._retry_request(request, response)
        if retried is None:
            return response
        return await handler(retried)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        rejection = self._file_rejection(request)
        if rejection is not None:
            return rejection
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        rejection = self._file_rejection(request)
        if rejection is not None:
            return rejection
        return await handler(request)

    def _retry_request(self, request: ModelRequest, response: ModelResponse) -> ModelRequest | None:
        messages = list(getattr(request, "messages", None) or [])
        if _already_corrected(messages):
            return None
        result = list(getattr(response, "result", None) or [])
        if len(result) != 1 or not isinstance(result[0], AIMessage):
            return None
        if getattr(result[0], "tool_calls", None):
            return None
        text = _message_text(result[0])
        correction = _correction_for(text)
        if correction is None:
            return None
        logger.info("Date validation: retrying final answer (%d chars)", len(text))
        messages.append(result[0])
        messages.append(HumanMessage(content=correction))
        try:
            return request.override(messages=messages)
        except Exception:  # pragma: no cover — defensive
            logger.debug("Date validation: could not build retry request", exc_info=True)
            return None

    def _file_rejection(self, request: ToolCallRequest) -> ToolMessage | None:
        tool_call = getattr(request, "tool_call", None) or {}
        name = tool_call.get("name") or ""
        if name not in _FILE_TOOLS:
            return None
        args = tool_call.get("args") or {}
        if not isinstance(args, dict):
            return None
        chunks = [args[key] for key in _CONTENT_KEYS if isinstance(args.get(key), str)]
        correction = _correction_for("\n".join(chunks))
        if correction is None:
            return None
        logger.info("Date validation: rejecting %s with wrong weekdays", name)
        return ToolMessage(
            content=(
                f"{correction}\n"
                "The file was not written. Rewrite it with the corrected weekdays "
                "and call the tool again."
            ),
            name=name,
            tool_call_id=tool_call.get("id") or "",
            status="error",
        )


def _already_corrected(messages: list[Any]) -> bool:
    for message in messages[-4:]:
        if isinstance(message, HumanMessage) and _message_text(message).startswith(DATE_CORRECTION_PREFIX):
            return True
    return False


def _correction_for(text: str) -> str | None:
    if not text or not text.strip():
        return None
    from backend.date_context import find_date_mismatches, format_date_correction

    mismatches = find_date_mismatches(text)
    if not mismatches:
        return None
    return format_date_correction(mismatches)
