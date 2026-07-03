"""Unit tests for the repeated-thought / repeated-action loop guard.

Covers:

* :class:`RepeatedThoughtGuard` signature + consecutive-repeat counting,
  including that interleaved tool/nudge messages don't reset the count and
  that differing tool args break the streak.
* :class:`RepeatedThoughtGuardMiddleware` behaviour: below threshold it is a
  pass-through, at the nudge threshold it appends a transient corrective
  message, and at the abort threshold it short-circuits with a terminal
  response that ends the run (without calling the model).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain.agents.middleware.types import ModelResponse

from middleware.repeated_thought_guard import (
    _NUDGE_TEXT,
    _TERMINAL_TEXT,
    RepeatedThoughtGuard,
    RepeatedThoughtGuardMiddleware,
)

_THOUGHT = (
    "I can see the Company filter dropdown is open. Let me close it and try "
    "the All filters button."
)


def _ai(text: str = _THOUGHT, ref: str | None = "e1121") -> AIMessage:
    """Build an AIMessage mimicking the LinkedIn browser_click loop turn."""
    tool_calls = []
    if ref is not None:
        tool_calls = [{
            "name": "browser_click",
            "args": {"ref": ref},
            "id": f"call_{ref}",
            "type": "tool_call",
        }]
    return AIMessage(content=text, tool_calls=tool_calls)


def _tool(ref: str = "e1121") -> ToolMessage:
    return ToolMessage(content="### Snapshot ...", tool_call_id=f"call_{ref}")


# ── signature ───────────────────────────────────────────────────────────────

def test_signature_identical_thought_and_action_match():
    assert RepeatedThoughtGuard.signature(_ai()) == RepeatedThoughtGuard.signature(_ai())


def test_signature_differs_on_different_args():
    a = RepeatedThoughtGuard.signature(_ai(ref="e1121"))
    b = RepeatedThoughtGuard.signature(_ai(ref="e2222"))
    assert a != b


def test_signature_differs_on_different_thought():
    a = RepeatedThoughtGuard.signature(_ai(text="thought one"))
    b = RepeatedThoughtGuard.signature(_ai(text="thought two"))
    assert a != b


def test_signature_whitespace_and_case_insensitive():
    a = RepeatedThoughtGuard.signature(_ai(text="Hello   World", ref=None))
    b = RepeatedThoughtGuard.signature(_ai(text="hello world", ref=None))
    assert a == b


def test_signature_empty_message_is_none():
    assert RepeatedThoughtGuard.signature(_ai(text="", ref=None)) is None


def test_signature_non_ai_message_is_none():
    assert RepeatedThoughtGuard.signature(HumanMessage(content="hi")) is None


# ── consecutive_repeats ─────────────────────────────────────────────────────

def test_consecutive_repeats_counts_identical_ai_messages():
    msgs = [_ai(), _tool(), _ai(), _tool(), _ai()]
    count, sig = RepeatedThoughtGuard.consecutive_repeats(msgs)
    assert count == 3
    assert sig == RepeatedThoughtGuard.signature(_ai())


def test_consecutive_repeats_skips_interleaved_nudges():
    # A nudge HumanMessage between repeats must not reset the count.
    msgs = [_ai(), _tool(), HumanMessage(content=_NUDGE_TEXT), _ai(), _tool(), _ai()]
    count, _ = RepeatedThoughtGuard.consecutive_repeats(msgs)
    assert count == 3


def test_consecutive_repeats_breaks_on_different_action():
    msgs = [_ai(ref="e1121"), _tool(), _ai(ref="e2222"), _tool(), _ai(ref="e2222")]
    count, sig = RepeatedThoughtGuard.consecutive_repeats(msgs)
    # Latest two share e2222; the e1121 before them breaks the streak.
    assert count == 2
    assert sig == RepeatedThoughtGuard.signature(_ai(ref="e2222"))


def test_consecutive_repeats_empty_history():
    count, sig = RepeatedThoughtGuard.consecutive_repeats([])
    assert count == 0
    assert sig is None


# ── cyclic_repeats ───────────────────────────────────────────────────────────

def _alternating(refs: list[str]) -> list:
    """Build [ai(refs[0]), tool, ai(refs[1]), tool, ...] in order."""
    msgs: list = []
    for ref in refs:
        msgs.append(_ai(ref=ref))
        msgs.append(_tool(ref=ref))
    return msgs


def test_cyclic_repeats_period1_matches_consecutive():
    # Period-1 loop: cyclic with max_period>1 must report the same count as
    # the consecutive-streak detector (smallest-period tie-break).
    msgs = [_ai(), _tool(), _ai(), _tool(), _ai()]
    c_consec, _ = RepeatedThoughtGuard.consecutive_repeats(msgs)
    c_cyclic, sig = RepeatedThoughtGuard.cyclic_repeats(msgs, max_period=4)
    assert c_consec == 3
    assert c_cyclic == 3
    assert sig is not None


def test_cyclic_repeats_detects_period2_cycle():
    # A,B,A,B,A,B — consecutive_repeats sees only 1; cyclic sees the (A,B)
    # cycle repeating 3 times.
    msgs = _alternating(["e1", "e2", "e1", "e2", "e1", "e2"])
    c_consec, _ = RepeatedThoughtGuard.consecutive_repeats(msgs)
    c_cyclic, sig = RepeatedThoughtGuard.cyclic_repeats(msgs, max_period=4)
    assert c_consec == 1
    assert c_cyclic == 3
    assert sig is not None


def test_cyclic_repeats_detects_period3_cycle():
    msgs = _alternating(["e1", "e2", "e3", "e1", "e2", "e3"])
    c_cyclic, _ = RepeatedThoughtGuard.cyclic_repeats(msgs, max_period=4)
    assert c_cyclic == 2


def test_cyclic_repeats_max_period1_ignores_alternation():
    # With max_period=1 a period-2 alternation must NOT be flagged (count 1).
    msgs = _alternating(["e1", "e2", "e1", "e2"])
    count, _ = RepeatedThoughtGuard.cyclic_repeats(msgs, max_period=1)
    assert count == 1


def test_cyclic_repeats_empty_history():
    count, sig = RepeatedThoughtGuard.cyclic_repeats([], max_period=4)
    assert count == 0
    assert sig is None


# ── middleware ──────────────────────────────────────────────────────────────

class _Req:
    """Minimal ModelRequest stand-in: only ``messages`` + ``override`` are used."""

    def __init__(self, messages: list) -> None:
        self.messages = messages

    def override(self, *, messages=None, **_kw):
        return _Req(messages if messages is not None else self.messages)


def _handler_factory():
    calls: list[_Req] = []

    def handler(req: _Req) -> ModelResponse:
        calls.append(req)
        return ModelResponse(
            result=[AIMessage(content="real model output")],
            structured_response=None,
        )

    return handler, calls


def _guard(nudge_at=2, abort_at=4):
    # recovery_temperature=0.0 keeps the MLX temp-bump path a no-op in tests.
    return RepeatedThoughtGuardMiddleware(
        nudge_at=nudge_at,
        abort_at=abort_at,
        recovery_temperature=0.0,
        recovery_temperature_turns=1,
    )


def test_middleware_passthrough_below_threshold():
    handler, calls = _handler_factory()
    req = _Req([_ai(), _tool()])  # count == 1, below nudge_at=2
    out = _guard().wrap_model_call(req, handler)
    assert len(calls) == 1
    # No nudge appended.
    assert not any(isinstance(m, HumanMessage) for m in calls[0].messages)
    assert out.result[0].content == "real model output"


def test_middleware_nudges_at_threshold():
    handler, calls = _handler_factory()
    req = _Req([_ai(), _tool(), _ai()])  # count == 2 == nudge_at
    out = _guard().wrap_model_call(req, handler)
    assert len(calls) == 1
    # The handler saw a transient nudge appended as the last message.
    assert isinstance(calls[0].messages[-1], HumanMessage)
    assert calls[0].messages[-1].content == _NUDGE_TEXT
    assert out.result[0].content == "real model output"


def test_middleware_aborts_at_threshold():
    handler, calls = _handler_factory()
    # count == 4 == abort_at — should short-circuit without calling the model.
    req = _Req([_ai(), _tool(), _ai(), _tool(), _ai(), _tool(), _ai()])
    out = _guard().wrap_model_call(req, handler)
    assert calls == []  # model never called
    assert len(out.result) == 1
    assert out.result[0].content == _TERMINAL_TEXT
    # Terminal message carries no tool calls, so the ReAct router ends the run.
    assert not out.result[0].tool_calls


async def test_middleware_async_aborts_at_threshold():
    handler_sync, _ = _handler_factory()
    calls: list = []

    async def ahandler(req):
        calls.append(req)
        return ModelResponse(
            result=[AIMessage(content="real")], structured_response=None,
        )

    req = _Req([_ai(), _tool(), _ai(), _tool(), _ai(), _tool(), _ai()])
    out = await _guard().awrap_model_call(req, ahandler)
    assert calls == []
    assert out.result[0].content == _TERMINAL_TEXT


def test_middleware_disabled_thresholds_passthrough():
    handler, calls = _handler_factory()
    guard = RepeatedThoughtGuardMiddleware(
        nudge_at=None, abort_at=None, recovery_temperature=0.0,
    )
    req = _Req([_ai(), _tool(), _ai(), _tool(), _ai(), _tool(), _ai()])
    out = guard.wrap_model_call(req, handler)
    assert len(calls) == 1
    assert out.result[0].content == "real model output"


# ── middleware: period-N cycles ──────────────────────────────────────────────

def _guard_p(nudge_at=2, abort_at=4, max_period=4):
    return RepeatedThoughtGuardMiddleware(
        nudge_at=nudge_at,
        abort_at=abort_at,
        max_period=max_period,
        recovery_temperature=0.0,
        recovery_temperature_turns=1,
    )


def test_middleware_nudges_on_period2_cycle():
    handler, calls = _handler_factory()
    # (A,B) repeated twice == count 2 == nudge_at — a period-2 loop that the
    # old consecutive-only guard would have scored as 1 (pass-through).
    req = _Req(_alternating(["e1", "e2", "e1", "e2"]))
    out = _guard_p().wrap_model_call(req, handler)
    assert len(calls) == 1
    assert isinstance(calls[0].messages[-1], HumanMessage)
    assert calls[0].messages[-1].content == _NUDGE_TEXT
    assert out.result[0].content == "real model output"


def test_middleware_aborts_on_period2_cycle():
    handler, calls = _handler_factory()
    # (A,B) repeated four times == count 4 == abort_at — short-circuits.
    req = _Req(_alternating(["e1", "e2", "e1", "e2", "e1", "e2", "e1", "e2"]))
    out = _guard_p().wrap_model_call(req, handler)
    assert calls == []
    assert out.result[0].content == _TERMINAL_TEXT
    assert not out.result[0].tool_calls


def test_middleware_period2_passthrough_when_max_period1():
    handler, calls = _handler_factory()
    # With max_period=1 the same period-2 loop is invisible — pass-through.
    req = _Req(_alternating(["e1", "e2", "e1", "e2", "e1", "e2", "e1", "e2"]))
    out = _guard_p(max_period=1).wrap_model_call(req, handler)
    assert len(calls) == 1
    assert out.result[0].content == "real model output"
