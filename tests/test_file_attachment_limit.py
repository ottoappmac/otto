"""Tests for ``FileAttachmentLimitMiddleware``.

Some OpenAI-compatible servers (e.g. the local oMLX server) count ``read_file``
"file" content blocks cumulatively across the whole conversation and reject
requests once that running total crosses a threshold — even on turns that
read no new files. The middleware evicts the oldest file blocks once the
total exceeds the cap so replayed history never re-triggers the limit.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from middleware.file_attachment_limit import FileAttachmentLimitMiddleware


def _file_tool_message(call_id: str) -> ToolMessage:
    return ToolMessage(
        tool_call_id=call_id,
        name="read_file",
        content=[{"type": "file", "base64": "AAAA", "mime_type": "application/pdf", "filename": f"{call_id}.pdf"}],
    )


def test_noop_when_under_limit():
    mw = FileAttachmentLimitMiddleware(max_attachments=3)
    messages = [
        SystemMessage(content="sys"),
        HumanMessage(content="read my files"),
        _file_tool_message("c1"),
        _file_tool_message("c2"),
    ]
    assert mw._limit_attachments(messages) is None


def test_evicts_oldest_blocks_once_over_limit():
    mw = FileAttachmentLimitMiddleware(max_attachments=2)
    messages = [
        SystemMessage(content="sys"),
        _file_tool_message("c1"),
        _file_tool_message("c2"),
        _file_tool_message("c3"),
    ]

    out = mw._limit_attachments(messages)
    assert out is not None

    tool_msgs = [m for m in out if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 3

    # Oldest is evicted, most recent two are kept intact.
    assert tool_msgs[0].content[0]["type"] == "text"
    assert tool_msgs[1].content[0]["type"] == "file"
    assert tool_msgs[2].content[0]["type"] == "file"


def test_noop_when_no_file_blocks():
    mw = FileAttachmentLimitMiddleware(max_attachments=1)
    messages = [
        SystemMessage(content="sys"),
        HumanMessage(content="hello"),
        AIMessage(content="hi"),
        ToolMessage(tool_call_id="c1", content="plain text result"),
    ]
    assert mw._limit_attachments(messages) is None


def test_does_not_mutate_input_messages():
    mw = FileAttachmentLimitMiddleware(max_attachments=1)
    tm1 = _file_tool_message("c1")
    tm2 = _file_tool_message("c2")
    messages = [HumanMessage(content="q"), tm1, tm2]

    mw._limit_attachments(messages)

    assert tm1.content[0]["type"] == "file"
    assert tm2.content[0]["type"] == "file"
