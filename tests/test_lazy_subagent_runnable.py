"""Tests for lazy subagent compilation wrapper."""

from __future__ import annotations

from langchain_core.runnables import Runnable

from backend.session_manager import _LazySubagentRunnable


def test_lazy_subagent_runnable_is_a_runnable():
    lazy = _LazySubagentRunnable(lambda: Runnable(), agent_name="test-agent")
    assert isinstance(lazy, Runnable)


def test_lazy_subagent_runnable_supports_with_config():
    built = {"value": 0}

    class _Counter(Runnable):
        def invoke(self, input, config=None, **kwargs):
            built["value"] += 1
            return {"messages": []}

    lazy = _LazySubagentRunnable(lambda: _Counter(), agent_name="counter")
    bound = lazy.with_config({"tags": ["subagent"]})
    assert hasattr(bound, "invoke")
    bound.invoke({"messages": []})
    assert built["value"] == 1
