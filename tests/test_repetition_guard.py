"""Unit tests for the intra-message repetition guard.

Covers:

* :func:`is_degenerate_repetition` — fires on single-sentence loops and short
  multi-sentence cycles, stays quiet on healthy / short / legitimately
  repetitive-but-valid prose.
* :class:`RepetitionGuardMiddleware` — replaces a degenerate, tool-call-free
  ``AIMessage`` with a recovery message, while leaving healthy output and any
  message carrying tool calls untouched (sync + async).
"""

from __future__ import annotations

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage

from middleware.repetition_guard import (
    _RECOVERY_MSG,
    RepetitionGuardMiddleware,
    is_degenerate_repetition,
)

# The exact loop observed in session 2fe2cb15 (two sentences repeated forever).
_LOOP_UNIT = (
    "The charts were generated in the virtual filesystem but the sandbox "
    "can't see them. Let me check the real session files directory and copy "
    "the charts there.\n\n"
)


# ── detector ────────────────────────────────────────────────────────────────

def test_detects_real_world_two_sentence_loop():
    text = _LOOP_UNIT * 80
    assert is_degenerate_repetition(text)


def test_detects_single_sentence_loop():
    text = ("Let me check the real session files directory now. " * 60)
    assert is_degenerate_repetition(text)


def test_ignores_short_text():
    # Even if repetitive, short outputs must never trip (avoid false positives).
    assert not is_degenerate_repetition(_LOOP_UNIT * 1)


def test_ignores_healthy_varied_prose():
    text = (
        "First, I researched undervalued ASX stocks and found three candidates. "
        "Then I cross-checked their P/B ratios against the criteria. "
        "BHP failed the dividend screen, while Fortescue passed every filter. "
        "Next I moved on to the SGX market and evaluated five more names. "
        "Finally I wrote the report and generated the four charts as requested. "
        "Each section includes commentary explaining the selection rationale."
    )
    assert not is_degenerate_repetition(text)


def test_ignores_long_unique_list():
    # A long bullet list of *distinct* lines is valid, not a loop.
    text = "\n".join(f"- candidate number {i} with a unique rationale here" for i in range(60))
    assert not is_degenerate_repetition(text)


# ── middleware ──────────────────────────────────────────────────────────────

class _Req:
    """Minimal ModelRequest stand-in (the guard never reads the request)."""

    def __init__(self, messages=None) -> None:
        self.messages = messages or []


def _resp(content: str, *, tool_calls=None) -> ModelResponse:
    msg = AIMessage(content=content, tool_calls=tool_calls or [])
    return ModelResponse(result=[msg], structured_response=None)


def test_middleware_replaces_degenerate_output():
    mw = RepetitionGuardMiddleware()
    out = mw.wrap_model_call(_Req(), lambda r: _resp(_LOOP_UNIT * 80))
    assert out.result[-1].content == _RECOVERY_MSG
    assert not out.result[-1].tool_calls


def test_middleware_passthrough_healthy_output():
    mw = RepetitionGuardMiddleware()
    out = mw.wrap_model_call(_Req(), lambda r: _resp("All done — wrote /output/report.md."))
    assert out.result[-1].content == "All done — wrote /output/report.md."


def test_middleware_ignores_message_with_tool_calls():
    # A degenerate-looking blob that still carried a usable tool call is left
    # alone — progress is being made and the MLX path handles oversized args.
    mw = RepetitionGuardMiddleware()
    tc = [{"name": "execute", "args": {"command": "ls"}, "id": "c1", "type": "tool_call"}]
    out = mw.wrap_model_call(_Req(), lambda r: _resp(_LOOP_UNIT * 80, tool_calls=tc))
    assert out.result[-1].content == _LOOP_UNIT * 80
    assert out.result[-1].tool_calls == tc


def test_middleware_handles_empty_result():
    mw = RepetitionGuardMiddleware()
    empty = ModelResponse(result=[], structured_response=None)
    out = mw.wrap_model_call(_Req(), lambda r: empty)
    assert out is empty


async def test_middleware_async_replaces_degenerate_output():
    mw = RepetitionGuardMiddleware()

    async def handler(r):
        return _resp(_LOOP_UNIT * 80)

    out = await mw.awrap_model_call(_Req(), handler)
    assert out.result[-1].content == _RECOVERY_MSG
