"""``OldImagePruningMiddleware`` keeps only the newest tool-result images in the replayed history.

A benchmark run died when a local server ran out of memory after 39 ``view_image`` results, long before
the summarization (context-length) trigger. Older images become a text note; the tool-call pairing and
user-sent images stay.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from middleware.old_image_pruning import OldImagePruningMiddleware, maybe_for_agent


def _img(call_id: str, n: int = 1) -> ToolMessage:
    blocks = [{"type": "text", "text": f"{call_id} image"}]
    blocks += [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{call_id}{i}"}} for i in range(n)]
    return ToolMessage(tool_call_id=call_id, name="view_image", content=blocks)


def _count(messages) -> int:
    return sum(1 for m in messages if isinstance(m, ToolMessage) and isinstance(m.content, list)
               for b in m.content if isinstance(b, dict) and b.get("type") in ("image", "image_url"))


def test_noop_at_or_under_limit():
    mw = OldImagePruningMiddleware(max_images=3)
    assert mw._prune([HumanMessage(content="hi"), _img("a"), _img("b"), _img("c")]) is None


def test_keeps_newest_and_notes_the_rest():
    mw = OldImagePruningMiddleware(max_images=2)
    msgs = [HumanMessage(content="hi")] + [_img(c) for c in "abcde"]
    out = mw._prune(msgs)
    assert _count(out) == 2 and len(out) == len(msgs)
    kept = [m.tool_call_id for m in out if isinstance(m, ToolMessage)
            and any(isinstance(b, dict) and b.get("type") == "image_url" for b in m.content)]
    assert kept == ["d", "e"]
    first = out[1]
    assert first.tool_call_id == "a" and any("older image removed" in b.get("text", "") for b in first.content)
    assert first.content[0]["text"] == "a image"                       # its own text survives
    assert _count(msgs) == 5                                           # input list not mutated


def test_several_images_in_one_message():
    mw = OldImagePruningMiddleware(max_images=2)
    out = mw._prune([_img("a", 3), _img("b", 2)])
    assert _count(out) == 2
    assert all(any(isinstance(b, dict) and b.get("type") == "image_url" for b in m.content) for m in out[1:])


def test_user_images_untouched_and_idempotent():
    mw = OldImagePruningMiddleware(max_images=1)
    user = HumanMessage(content=[{"type": "image_url", "image_url": {"url": "data:x"}}])
    out = mw._prune([user, _img("a"), _img("b"), AIMessage(content="ok")])
    assert out[0] is user and _count(out) == 1
    assert mw._prune(out) is None


def test_only_for_image_heavy_agents():
    assert maybe_for_agent("benchcad-agent") is not None
    assert maybe_for_agent("cad-agent") is None and maybe_for_agent(None) is None
