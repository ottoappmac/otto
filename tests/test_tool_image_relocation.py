"""Tests for ``ToolImageRelocationMiddleware``.

OpenAI-compatible servers (openai / omlx / exo providers) only render image
content that appears in a *user* message; images returned inside a
``ToolMessage`` (e.g. from ``view_image``) are silently dropped.  The
middleware relocates those images into a following ``HumanMessage`` while
preserving the assistant→tool ordering the API requires.
"""

from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.content import create_image_block, create_text_block

from middleware.tool_image_relocation import ToolImageRelocationMiddleware


def _image_tool_message(call_id: str, path: str, b64: str) -> ToolMessage:
    return ToolMessage(
        tool_call_id=call_id,
        content=[
            create_text_block(text=f"Image: {path} (image/jpeg, 100 KB)"),
            create_image_block(base64=b64, mime_type="image/jpeg"),
        ],
    )


def _has_image(blocks) -> bool:
    return any(
        isinstance(b, dict) and b.get("type") in ("image", "image_url")
        for b in blocks
    )


def test_images_moved_from_tool_results_into_user_message():
    mw = ToolImageRelocationMiddleware()
    ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "view_image", "args": {}, "id": "c1", "type": "tool_call"},
            {"name": "view_image", "args": {}, "id": "c2", "type": "tool_call"},
        ],
    )
    messages = [
        SystemMessage(content="sys"),
        HumanMessage(content="what is in the pdf"),
        ai,
        _image_tool_message("c1", "/doc_images/p1_img1.jpg", "AAA"),
        _image_tool_message("c2", "/doc_images/p1_img2.jpg", "BBB"),
    ]

    out = mw._relocate(messages)
    assert out is not None

    tool_msgs = [m for m in out if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 2
    # Tool results keep their text but no longer carry image blocks.
    for tm in tool_msgs:
        assert not _has_image(tm.content)
        assert any(b.get("type") == "text" for b in tm.content)

    # A single user message now carries both images, placed after the tools.
    relocated = [
        m for m in out if isinstance(m, HumanMessage) and isinstance(m.content, list)
    ]
    assert len(relocated) == 1
    images = [b for b in relocated[0].content if _has_image([b])]
    assert len(images) == 2
    # Ordering: AIMessage → ToolMessages → relocated HumanMessage.
    assert isinstance(out[-1], HumanMessage)
    assert out.index(ai) < out.index(tool_msgs[-1]) < out.index(relocated[0])


def test_noop_when_no_tool_images():
    mw = ToolImageRelocationMiddleware()
    messages = [
        SystemMessage(content="sys"),
        HumanMessage(content="hello"),
        AIMessage(content="hi"),
        ToolMessage(tool_call_id="c1", content="plain text result"),
    ]
    assert mw._relocate(messages) is None


def test_does_not_mutate_input_messages():
    mw = ToolImageRelocationMiddleware()
    tm = _image_tool_message("c1", "/doc_images/p1_img1.jpg", "AAA")
    messages = [HumanMessage(content="q"), AIMessage(content=""), tm]

    mw._relocate(messages)

    # The original tool message still carries its image (transform is on a copy).
    assert _has_image(tm.content)
