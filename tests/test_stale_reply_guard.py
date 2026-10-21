"""Tests for the cross-turn stale-reply guard.

A long session can answer "still broken" with its previous "fixed" summary word
for word, without using a tool. The guard retries once with a nudge, then
replaces the reply with an honest message.
"""

from __future__ import annotations

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from middleware.stale_reply_guard import (
    _NUDGE_TEXT,
    _TERMINAL_TEXT,
    StaleReplyGuardMiddleware,
    is_echo,
    previous_final_answers,
)

_SUMMARY = (
    "I've successfully fixed the app issues.\n\n"
    "## Issues Fixed\n"
    "1. Compilation errors in Patterns.tsx and Projects.tsx: the await was used "
    "inside a map callback, so I moved it into the async handler.\n"
    "2. Tailwind configuration: updated src/index.css to use the proper directives.\n"
    "3. Environment access: added optional chaining for import.meta.env.\n\n"
    "The app should now compile and run properly."
)
_OTHER = (
    "The build still fails. tsc reports TS2339 in four files because the project "
    "has no vite-env.d.ts, so import.meta.env has no type. I added the "
    "declaration file and the build now passes with exit code 0."
)


class _Req:
    def __init__(self, messages) -> None:
        self.messages = messages

    def override(self, **kwargs):
        return _Req(kwargs.get("messages", self.messages))


def _resp(content: str, *, tool_calls=None) -> ModelResponse:
    return ModelResponse(
        result=[AIMessage(content=content, tool_calls=tool_calls or [])],
        structured_response=None,
    )


def _history() -> list:
    return [
        HumanMessage(content="fix the build"),
        AIMessage(content="", tool_calls=[
            {"name": "edit_file", "args": {"file_path": "a"}, "id": "c1", "type": "tool_call"},
        ]),
        ToolMessage(content="ok", tool_call_id="c1"),
        AIMessage(content=_SUMMARY),
        HumanMessage(content="Same error"),
    ]


# ── detector ────────────────────────────────────────────────────────────────

def test_previous_answers_skip_tool_turns_and_the_current_question():
    assert previous_final_answers(_history()) == [" ".join(_SUMMARY.split()).lower()]


def test_short_answers_are_never_an_echo():
    assert previous_final_answers([
        HumanMessage(content="hi"), AIMessage(content="Done."), HumanMessage(content="again"),
    ]) == []
    assert not is_echo("Done.", ["done."])


def test_echo_tolerates_whitespace_and_case():
    prev = previous_final_answers(_history())
    assert is_echo(_SUMMARY.replace("\n", "  ").upper(), prev)


def test_a_different_answer_is_not_an_echo():
    assert not is_echo(_OTHER, previous_final_answers(_history()))


# ── middleware ──────────────────────────────────────────────────────────────

def test_echo_is_retried_with_a_nudge_and_the_retry_is_used():
    calls: list[list] = []

    def handler(req):
        calls.append(list(req.messages))
        return _resp(_SUMMARY) if len(calls) == 1 else _resp(_OTHER)

    out = StaleReplyGuardMiddleware().wrap_model_call(_Req(_history()), handler)
    assert out.result[-1].content == _OTHER
    assert len(calls) == 2
    assert calls[1][-1].content == _NUDGE_TEXT
    # The nudge is transient: the original request is untouched.
    assert all(m.content != _NUDGE_TEXT for m in calls[0])


def test_a_retry_that_uses_a_tool_is_accepted():
    tc = [{"name": "execute", "args": {"command": "npm run build"}, "id": "c2", "type": "tool_call"}]
    n = {"calls": 0}

    def handler(req):
        n["calls"] += 1
        return _resp(_SUMMARY) if n["calls"] == 1 else _resp("", tool_calls=tc)

    out = StaleReplyGuardMiddleware().wrap_model_call(_Req(_history()), handler)
    assert out.result[-1].tool_calls == tc


def test_a_second_echo_becomes_an_honest_message():
    out = StaleReplyGuardMiddleware().wrap_model_call(
        _Req(_history()), lambda req: _resp(_SUMMARY)
    )
    assert out.result[-1].content == _TERMINAL_TEXT
    assert not out.result[-1].tool_calls


def test_a_new_answer_passes_straight_through():
    n = {"calls": 0}

    def handler(req):
        n["calls"] += 1
        return _resp(_OTHER)

    out = StaleReplyGuardMiddleware().wrap_model_call(_Req(_history()), handler)
    assert out.result[-1].content == _OTHER
    assert n["calls"] == 1


def test_first_turn_has_nothing_to_echo():
    n = {"calls": 0}

    def handler(req):
        n["calls"] += 1
        return _resp(_SUMMARY)

    out = StaleReplyGuardMiddleware().wrap_model_call(
        _Req([HumanMessage(content="fix the build")]), handler
    )
    assert out.result[-1].content == _SUMMARY
    assert n["calls"] == 1


async def test_async_echo_is_refused_after_a_failed_retry():
    async def handler(req):
        return _resp(_SUMMARY)

    out = await StaleReplyGuardMiddleware().awrap_model_call(_Req(_history()), handler)
    assert out.result[-1].content == _TERMINAL_TEXT
