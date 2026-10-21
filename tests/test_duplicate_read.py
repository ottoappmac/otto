"""Identical read_file calls return a short note instead of a second body."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from middleware.duplicate_read import UNCHANGED_NOTE, DuplicateReadMiddleware, maybe_for_provider
from tools.loop_guard import OBSERVATION_TOOLS, ToolLoopGuard

_BODY = "line one\nline two\n"


def _read(path: str, offset: int, limit: int, call_id: str) -> dict:
    return {
        "name": "read_file",
        "args": {"file_path": path, "offset": offset, "limit": limit},
        "id": call_id,
        "type": "tool_call",
    }


def _request(tool_call: dict, messages: list) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call=tool_call,
        tool=None,
        state={"messages": messages},
        runtime=MagicMock(),
    )


def _prior(path: str, offset: int, limit: int, content: str, call_id: str = "call_1") -> list:
    return [
        AIMessage(content="", tool_calls=[_read(path, offset, limit, call_id)]),
        ToolMessage(content=content, name="read_file", tool_call_id=call_id),
    ]


def _run(messages: list, tool_call: dict, content: str) -> ToolMessage:
    def handler(_request):
        return ToolMessage(
            content=content,
            name="read_file",
            tool_call_id=tool_call["id"],
        )

    result = DuplicateReadMiddleware().wrap_tool_call(
        _request(tool_call, messages), handler,
    )
    assert isinstance(result, ToolMessage)
    return result


def test_identical_read_with_body_still_visible_returns_the_note():
    guard = ToolLoopGuard()
    before = guard._trip_count
    result = _run(_prior("notes.md", 0, 100, _BODY), _read("notes.md", 0, 100, "call_2"), _BODY)
    assert result.content == UNCHANGED_NOTE
    assert result.status != "error"
    assert guard._trip_count == before
    assert "read_file" in OBSERVATION_TOOLS


def test_read_after_edit_returns_the_file():
    messages = _prior("notes.md", 0, 100, _BODY) + [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "edit_file",
                "args": {"file_path": "notes.md", "old_string": "line", "new_string": "line"},
                "id": "call_edit",
                "type": "tool_call",
            }],
        ),
        ToolMessage(content="updated", name="edit_file", tool_call_id="call_edit"),
    ]
    result = _run(messages, _read("notes.md", 0, 100, "call_2"), _BODY)
    assert result.content == _BODY


def test_different_offset_reads_normally():
    result = _run(_prior("notes.md", 0, 100, _BODY), _read("notes.md", 40, 100, "call_2"), _BODY)
    assert result.content == _BODY


def test_read_after_stubbed_body_reads_normally():
    messages = _prior("notes.md", 0, 100, "(earlier read omitted)")
    result = _run(messages, _read("notes.md", 0, 100, "call_2"), _BODY)
    assert result.content == _BODY


def test_note_does_not_hide_an_earlier_full_body():
    messages = _prior("notes.md", 0, 100, _BODY) + [
        AIMessage(content="", tool_calls=[_read("notes.md", 0, 100, "call_note")]),
        ToolMessage(content=UNCHANGED_NOTE, name="read_file", tool_call_id="call_note"),
    ]
    result = _run(messages, _read("notes.md", 0, 100, "call_3"), _BODY)
    assert result.content == UNCHANGED_NOTE


def test_only_omlx_gets_the_middleware():
    assert maybe_for_provider("omlx") is not None
    assert maybe_for_provider("openai") is None
    assert maybe_for_provider("mlx") is None
    assert maybe_for_provider(None) is None
