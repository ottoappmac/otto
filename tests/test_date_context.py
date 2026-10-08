"""Weekday lookup, calendar injection, and date-validation retries."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from backend.date_context import (
    build_date_block,
    describe_dates,
    find_date_mismatches,
)
from middleware.date_context import DateContextMiddleware, DateValidationMiddleware


NOW = datetime(2026, 10, 8, 9, 0)


def test_paris_friday_is_actually_saturday():
    text = (
        "Paris Itinerary — 6–9 November 2026\n"
        "Day 1 — Friday 7 November: Eiffel Tower\n"
    )
    found = find_date_mismatches(text, NOW)
    assert len(found) == 1
    assert found[0].iso == "2026-11-07"
    assert found[0].claimed == "Friday"
    assert found[0].actual == "Saturday"


def test_correct_weekday_is_not_flagged():
    text = "Saturday 7 November 2026 is the first full day."
    assert find_date_mismatches(text, NOW) == []


def test_friday_night_is_not_a_date():
    assert find_date_mismatches("Let's do Friday night in the Marais.", NOW) == []


def test_missing_year_uses_the_nearest_explicit_year():
    text = "Trip in November 2026. Day 2 — Saturday 8 November."
    found = find_date_mismatches(text, NOW)
    assert len(found) == 1
    assert found[0].iso == "2026-11-08"
    assert found[0].actual == "Sunday"


def test_missing_year_uses_the_next_occurrence():
    # 1 January 2027 is a Friday. No year is written in the text.
    found = find_date_mismatches("We leave Thursday 1 January.", NOW)
    assert len(found) == 1
    assert found[0].iso == "2027-01-01"
    assert found[0].actual == "Friday"


def test_month_boundary():
    # 1 December 2026 is a Tuesday.
    found = find_date_mismatches("Monday 1 December 2026", NOW)
    assert len(found) == 1
    assert found[0].iso == "2026-12-01"
    assert found[0].actual == "Tuesday"


def test_parenthetical_and_abbreviated_forms():
    text = "Anchors: Friday (7 Nov) and Sat, 8 Nov 2026."
    # The 2026 nearest the first pair is the one on the second pair.
    found = find_date_mismatches(text, NOW)
    isos = {item.iso for item in found}
    assert "2026-11-07" in isos
    assert "2026-11-08" in isos


def test_calendar_contains_today_and_a_later_month():
    block = build_date_block(NOW)
    assert "Thursday, October 08, 2026" in block
    assert "November 2027" in block
    assert "Never compute weekdays mentally" in block


def test_describe_dates_relative_and_absolute():
    text = describe_dates("2026-11-07; next Friday; tomorrow", NOW)
    assert "2026-11-07 Saturday" in text
    assert "2026-10-09 Friday" in text  # tomorrow
    # 8 Oct 2026 is Thursday, so next Friday is 9 Oct — already covered —
    # "next Friday" from Thursday is the next day only if strictly after today
    # and the delta is not zero. Thursday -> Friday is 1 day, which is "next".
    assert "next Friday: 2026-10-09 Friday" in text


class _Req:
    def __init__(self, messages, system_message=None):
        self.messages = messages
        self.system_message = system_message

    def override(self, *, messages=None, system_message=None, **_kw):
        return _Req(
            messages if messages is not None else self.messages,
            system_message if system_message is not None else self.system_message,
        )


def test_date_context_middleware_appends_calendar():
    seen = {}

    def handler(req):
        seen["text"] = req.system_message.content
        return ModelResponse(result=[AIMessage(content="ok")], structured_response=None)

    req = _Req([HumanMessage(content="hi")], SystemMessage(content="You are Otto."))
    DateContextMiddleware().wrap_model_call(req, handler)
    assert "You are Otto." in seen["text"]
    assert "<current_date>" in seen["text"]
    assert "Today is" in seen["text"]


def test_validation_retries_a_wrong_final_answer_once():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return ModelResponse(
                result=[AIMessage(content="Day 1 — Friday 7 November 2026")],
                structured_response=None,
            )
        assert any(
            isinstance(m, HumanMessage) and str(m.content).startswith("Date check failed.")
            for m in req.messages
        )
        return ModelResponse(
            result=[AIMessage(content="Day 1 — Saturday 7 November 2026")],
            structured_response=None,
        )

    req = _Req([HumanMessage(content="plan it")])
    out = DateValidationMiddleware().wrap_model_call(req, handler)
    assert calls["n"] == 2
    assert out.result[0].content.startswith("Day 1 — Saturday")


def test_validation_does_not_retry_a_correct_answer():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return ModelResponse(
            result=[AIMessage(content="Saturday 7 November 2026")],
            structured_response=None,
        )

    out = DateValidationMiddleware().wrap_model_call(
        _Req([HumanMessage(content="plan it")]), handler,
    )
    assert calls["n"] == 1
    assert "Saturday" in out.result[0].content


def test_validation_rejects_a_file_write():
    def handler(_req):
        raise AssertionError("file tool should not run")

    request = ToolCallRequest(
        tool_call={
            "name": "write_file",
            "args": {"file_path": "/output/itinerary.md", "content": "Friday 7 November 2026"},
            "id": "call_1",
            "type": "tool_call",
        },
        tool=None,
        state={},
        runtime=MagicMock(),
    )
    result = DateValidationMiddleware().wrap_tool_call(request, handler)
    assert isinstance(result, ToolMessage)
    assert "2026-11-07 is a Saturday" in result.content
    assert result.tool_call_id == "call_1"


def test_validation_allows_a_correct_file_write():
    def handler(req):
        return ToolMessage(content="wrote", tool_call_id=req.tool_call["id"])

    request = ToolCallRequest(
        tool_call={
            "name": "edit_file",
            "args": {
                "file_path": "/output/itinerary.md",
                "old_string": "x",
                "new_string": "Saturday 7 November 2026",
            },
            "id": "call_2",
            "type": "tool_call",
        },
        tool=None,
        state={},
        runtime=MagicMock(),
    )
    result = DateValidationMiddleware().wrap_tool_call(request, handler)
    assert result.content == "wrote"
