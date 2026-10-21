"""Skip a second copy of a file the model already has in front of it.

``read_file`` is cheap on disk and expensive in the next prompt: the body is
prefilled again on every later turn. When the same path, offset, and limit
are already in the conversation, nothing has written that path since, and a
fresh read returns the same text, the tool result is a one-line note instead
of a second copy.

This is not the loop guard. A repeated read is not a stuck tool, and tripping
the guard would evict caches and count toward aborting the run. A read after
``edit_file`` / ``write_file``, a different offset, or a read whose earlier
body was replaced (so it is no longer the text on disk) goes through normally.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

UNCHANGED_NOTE = "Unchanged since your earlier read; the content is above."

_READ_TOOLS = frozenset({"read_file"})
_WRITE_TOOLS = frozenset({"write_file", "edit_file", "edit"})


def maybe_for_provider(provider: str | None) -> "DuplicateReadMiddleware | None":
    """oMLX only. Rewriting a tool result is pointless for hosts that cache elsewhere."""
    if (provider or "").strip().lower() != "omlx":
        return None
    return DuplicateReadMiddleware()


class DuplicateReadMiddleware(AgentMiddleware):
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        key = _read_key(request)
        if key is None:
            return handler(request)
        result = handler(request)
        return _maybe_note(request, key, result)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        key = _read_key(request)
        if key is None:
            return await handler(request)
        result = await handler(request)
        return _maybe_note(request, key, result)


def _maybe_note(
    request: ToolCallRequest,
    key: tuple[str, int, int],
    result: ToolMessage | Command[Any],
) -> ToolMessage | Command[Any]:
    if not isinstance(result, ToolMessage) or getattr(result, "status", None) == "error":
        return result
    previous = _visible_body(_state_messages(request)).get(key)
    fresh = _text(result.content).strip()
    if not previous or not fresh or previous != fresh:
        return result
    tool_call = getattr(request, "tool_call", None) or {}
    return ToolMessage(
        content=UNCHANGED_NOTE,
        name=tool_call.get("name") or "read_file",
        tool_call_id=tool_call.get("id") or "",
    )


def _visible_body(messages: list[Any]) -> dict[tuple[str, int, int], str]:
    """Last full ``read_file`` body for each (path, offset, limit).

    A later write or edit of that path drops the body. The unchanged-note
    itself is not a body, so a read whose earlier text was replaced still
    misses and reads the file.
    """
    calls: dict[str, dict[str, Any]] = {}
    bodies: dict[tuple[str, int, int], str] = {}
    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls or []:
                call_id = str(tool_call.get("id") or "")
                if call_id:
                    calls[call_id] = tool_call
            continue
        if not isinstance(message, ToolMessage):
            continue
        tool_call = calls.get(str(message.tool_call_id or ""))
        name = ""
        args: dict[str, Any] = {}
        if tool_call is not None:
            name = str(tool_call.get("name") or "")
            args = _args(tool_call)
        if not name:
            name = str(message.name or "")
        if name in _WRITE_TOOLS:
            path = _path(args)
            if path:
                bodies = {key: body for key, body in bodies.items() if key[0] != path}
            continue
        if name not in _READ_TOOLS or getattr(message, "status", None) == "error":
            continue
        key = _key(args)
        if key is None:
            continue
        body = _text(message.content).strip()
        if not body or body == UNCHANGED_NOTE:
            continue
        bodies[key] = body
    return bodies


def _read_key(request: ToolCallRequest) -> tuple[str, int, int] | None:
    tool_call = getattr(request, "tool_call", None) or {}
    if (tool_call.get("name") or "") not in _READ_TOOLS:
        return None
    return _key(_args(tool_call))


def _args(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call.get("args")
    if raw is None:
        raw = tool_call.get("arguments")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return raw if isinstance(raw, dict) else {}


def _path(args: dict[str, Any]) -> str:
    return str(args.get("file_path") or args.get("path") or "")


def _key(args: dict[str, Any]) -> tuple[str, int, int] | None:
    path = _path(args)
    if not path:
        return None
    return (path, _as_int(args.get("offset")), _as_int(args.get("limit")))


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in ("text", None):
                parts.append(str(block.get("text") or ""))
        return "\n".join(parts)
    return str(content or "")


def _state_messages(request: ToolCallRequest) -> list[Any]:
    state = getattr(request, "state", None)
    messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
    return list(messages or [])
