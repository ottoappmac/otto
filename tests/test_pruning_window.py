"""The sliding message window must be switchable off, and off for oMLX."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from backend.session_manager import _playwright_pruning
from middleware.playwright_pruning import PlaywrightSnapshotPruningMiddleware, _prune


def _history(steps: int) -> list:
    messages: list = [SystemMessage(content="sys"), HumanMessage(content="go")]
    for i in range(steps):
        call = {"name": "read_file", "args": {"file_path": f"/f{i}"}, "id": f"c{i}"}
        messages.append(AIMessage(content="", tool_calls=[call]))
        messages.append(ToolMessage(content=f"body {i}", tool_call_id=f"c{i}"))
    return messages


def test_default_window_drops_oldest():
    pruned = _prune(_history(30), 40)
    assert len(pruned) <= 40
    assert not any(getattr(m, "content", "") == "body 0" for m in pruned)


def test_no_window_keeps_prefix_stable_across_steps():
    a = _prune(_history(30), None)
    b = _prune(_history(31), None)
    assert len(b) == len(a) + 2
    assert [m.content for m in b[: len(a)]] == [m.content for m in a]


def test_omlx_model_gets_no_window():
    class _OMLXChat: ...

    assert _playwright_pruning(_OMLXChat()).max_messages is None


def test_other_models_keep_default_window():
    class Other: ...

    mw = _playwright_pruning(Other())
    assert isinstance(mw, PlaywrightSnapshotPruningMiddleware)
    assert mw.max_messages == 40
