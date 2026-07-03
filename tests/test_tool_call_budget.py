"""Unit tests for the per-run tool-call budget guard.

Covers:

* :func:`count_run_tool_calls` — counts tool calls in the current run, resets
  on a genuine user turn, and ignores injected guard nudges.
* :class:`ToolCallBudgetMiddleware` — pass-through below budget, a transient
  nudge at the soft budget, and a graceful terminal stop at the hard budget
  (sync + async).
* The :class:`utilities.environment.Environment` budget getters.
"""

from __future__ import annotations

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from middleware.tool_call_budget import (
    _SOFT_NUDGE_TEXT,
    _TERMINAL_TEXT,
    ToolCallBudgetMiddleware,
    count_run_tool_calls,
)


def _ai(n_calls: int = 1, text: str = "thinking") -> AIMessage:
    calls = [
        {"name": "web_research", "args": {"q": f"{i}"}, "id": f"c{i}", "type": "tool_call"}
        for i in range(n_calls)
    ]
    return AIMessage(content=text, tool_calls=calls)


class _Req:
    def __init__(self, messages) -> None:
        self.messages = messages

    def override(self, *, messages=None, **_kw):
        return _Req(messages if messages is not None else self.messages)


def _handler_factory():
    calls: list[_Req] = []

    def handler(req: _Req) -> ModelResponse:
        calls.append(req)
        return ModelResponse(result=[AIMessage(content="real")], structured_response=None)

    return handler, calls


# ── counting ─────────────────────────────────────────────────────────────


def test_count_sums_tool_calls():
    msgs = [HumanMessage(content="do it"), _ai(3), ToolMessage(content="r", tool_call_id="c0"), _ai(2)]
    assert count_run_tool_calls(msgs) == 5


def test_count_resets_on_new_user_turn():
    msgs = [HumanMessage(content="turn 1"), _ai(5), HumanMessage(content="turn 2"), _ai(2)]
    assert count_run_tool_calls(msgs) == 2


def test_count_ignores_injected_nudges():
    # An injected soft-nudge HumanMessage must NOT reset the run counter.
    msgs = [
        HumanMessage(content="do it"),
        _ai(40),
        HumanMessage(content=_SOFT_NUDGE_TEXT),
        _ai(45),
    ]
    assert count_run_tool_calls(msgs) == 85


# ── middleware ───────────────────────────────────────────────────────────


def _guard(soft=80, hard=150):
    return ToolCallBudgetMiddleware(soft_budget=soft, hard_budget=hard)


def test_passthrough_below_soft_budget():
    handler, calls = _handler_factory()
    req = _Req([HumanMessage(content="go"), _ai(10)])
    out = _guard().wrap_model_call(req, handler)
    assert len(calls) == 1
    assert not any(isinstance(m, HumanMessage) and m.content == _SOFT_NUDGE_TEXT
                   for m in calls[0].messages)
    assert out.result[0].content == "real"


def test_nudges_at_soft_budget():
    handler, calls = _handler_factory()
    req = _Req([HumanMessage(content="go"), _ai(80)])
    out = _guard(soft=80, hard=150).wrap_model_call(req, handler)
    assert len(calls) == 1
    assert calls[0].messages[-1].content == _SOFT_NUDGE_TEXT
    assert out.result[0].content == "real"


def test_aborts_at_hard_budget():
    handler, calls = _handler_factory()
    req = _Req([HumanMessage(content="go"), _ai(150)])
    out = _guard(soft=80, hard=150).wrap_model_call(req, handler)
    assert calls == []  # model never called
    assert out.result[0].content == _TERMINAL_TEXT
    assert not out.result[0].tool_calls


def test_zero_budgets_disable_guard():
    handler, calls = _handler_factory()
    req = _Req([HumanMessage(content="go"), _ai(500)])
    out = ToolCallBudgetMiddleware(soft_budget=0, hard_budget=0).wrap_model_call(req, handler)
    assert len(calls) == 1
    assert out.result[0].content == "real"


async def test_async_aborts_at_hard_budget():
    calls: list = []

    async def ahandler(req):
        calls.append(req)
        return ModelResponse(result=[AIMessage(content="real")], structured_response=None)

    req = _Req([HumanMessage(content="go"), _ai(200)])
    out = await _guard(soft=80, hard=150).awrap_model_call(req, ahandler)
    assert calls == []
    assert out.result[0].content == _TERMINAL_TEXT


# ── environment knobs ──────────────────────────────────────────────────────


def test_environment_budget_getters(monkeypatch):
    from utilities.environment import Environment

    monkeypatch.setenv("TOOL_CALL_SOFT_BUDGET", "30")
    monkeypatch.setenv("TOOL_CALL_HARD_BUDGET", "60")
    assert Environment.get_tool_call_soft_budget() == 30
    assert Environment.get_tool_call_hard_budget() == 60


def test_environment_budget_getters_bad_values(monkeypatch):
    from utilities.environment import Environment

    monkeypatch.setenv("TOOL_CALL_SOFT_BUDGET", "not-a-number")
    monkeypatch.setenv("TOOL_CALL_HARD_BUDGET", "-5")
    # Falls back to default on garbage; clamps negatives to 0.
    assert Environment.get_tool_call_soft_budget() == int(Environment.TOOL_CALL_SOFT_BUDGET)
    assert Environment.get_tool_call_hard_budget() == 0
