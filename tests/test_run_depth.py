"""Auto asks the model; an explicit Quick or Deep never does."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from backend.run_depth import (
    FALLBACK_REASON,
    UNATTENDED_REASON,
    depth_notice,
    judge_run_depth,
    parse_depth_reply,
    resolve_run_depth,
)
from middleware.run_depth import QUICK_ADDENDUM, RunDepthMiddleware, latest_run_depth
from middleware.tool_call_budget import ToolCallBudgetMiddleware, _QUICK_TERMINAL_TEXT


class _Judge:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0

    async def ainvoke(self, _messages):
        self.calls += 1
        text = self.replies.pop(0)
        return AIMessage(content=text)


def test_depth_notice_names_the_mode():
    assert depth_notice("quick", "user", "") == "Running Quick."
    assert depth_notice("deep", "user", "") == "Running Deep."
    assert depth_notice("deep", "agent", "Needs several steps.") == (
        "Auto chose Deep. Needs several steps."
    )
    assert depth_notice("deep", "user", UNATTENDED_REASON) == (
        f"Running Deep. {UNATTENDED_REASON}"
    )


def test_parse_depth_reply():
    assert parse_depth_reply("QUICK\nShort lookup.") == ("quick", "Short lookup.")
    assert parse_depth_reply("DEEP\nNeeds research.") == ("deep", "Needs research.")
    assert parse_depth_reply("I think we should explore.") is None


def test_explicit_mode_does_not_call_the_model():
    judge = _Judge(["QUICK\nunused"])

    async def run():
        return await resolve_run_depth(
            requested="deep",
            query="what time is it",
            llm=judge,
            unattended=False,
            default_mode="auto",
            has_attachments=False,
            prior_user="",
            prior_depth="",
        )

    mode, source, reason, response = asyncio.run(run())
    assert (mode, source, response) == ("deep", "user", None)
    assert reason == ""
    assert judge.calls == 0


def test_auto_uses_the_model_reply():
    judge = _Judge(["DEEP\nThis is a multi-day itinerary."])

    async def run():
        return await judge_run_depth(
            judge,
            query="Plan a Paris trip for 6-9 November",
            has_attachments=False,
            prior_user="",
            prior_depth="",
        )

    mode, reason, _response = asyncio.run(run())
    assert mode == "deep"
    assert "itinerary" in reason
    assert judge.calls == 1


def test_auto_retries_once_then_falls_back_to_quick():
    judge = _Judge(["maybe", "still no"])

    async def run():
        return await judge_run_depth(
            judge,
            query="hello",
            has_attachments=False,
            prior_user="",
            prior_depth="",
        )

    mode, reason, _response = asyncio.run(run())
    assert mode == "quick"
    assert reason == FALLBACK_REASON
    assert judge.calls == 2


def test_unattended_run_is_deep_without_a_judge():
    judge = _Judge(["QUICK\nunused"])

    async def run():
        return await resolve_run_depth(
            requested="auto",
            query="morning digest",
            llm=judge,
            unattended=True,
            default_mode="auto",
            has_attachments=False,
            prior_user="",
            prior_depth="",
        )

    mode, source, _reason, response = asyncio.run(run())
    assert mode == "deep"
    assert source == "user"
    assert response is None
    assert judge.calls == 0


def test_omitted_depth_uses_the_configured_default():
    judge = _Judge(["DEEP\nunused"])

    async def run():
        return await resolve_run_depth(
            requested=None,
            query="hi",
            llm=judge,
            unattended=False,
            default_mode="quick",
            has_attachments=False,
            prior_user="",
            prior_depth="",
        )

    mode, source, _reason, response = asyncio.run(run())
    assert (mode, source, response) == ("quick", "user", None)
    assert judge.calls == 0


class _Req:
    def __init__(self, messages, system_message=None):
        self.messages = messages
        self.system_message = system_message

    def override(self, *, messages=None, system_message=None, **_kw):
        return _Req(
            messages if messages is not None else self.messages,
            system_message if system_message is not None else self.system_message,
        )


def _user(text: str, depth: str) -> HumanMessage:
    return HumanMessage(content=text, additional_kwargs={"run_depth": depth})


def test_latest_run_depth_ignores_a_later_nudge():
    messages = [
        _user("plan the trip", "deep"),
        AIMessage(content="working", tool_calls=[{
            "name": "web_research", "args": {}, "id": "c0", "type": "tool_call",
        }]),
        HumanMessage(content="Date check failed. 7 Nov is a Saturday."),
        _user("what is 2+2", "quick"),
    ]
    assert latest_run_depth(messages) == "quick"


def test_quick_addendum_is_added_and_deep_is_unchanged():
    seen = {}

    def handler(req):
        seen["text"] = req.system_message.content if req.system_message else ""
        return ModelResponse(result=[AIMessage(content="ok")], structured_response=None)

    RunDepthMiddleware().wrap_model_call(
        _Req([_user("hi", "quick")], SystemMessage(content="Base.")),
        handler,
    )
    assert "Base." in seen["text"]
    assert QUICK_ADDENDUM in seen["text"]

    seen.clear()
    RunDepthMiddleware().wrap_model_call(
        _Req([_user("hi", "deep")], SystemMessage(content="Base.")),
        handler,
    )
    assert seen["text"] == "Base."


def _tool_request(name: str, call_id: str, messages: list) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": {}, "id": call_id, "type": "tool_call"},
        tool=None,
        state={"messages": messages},
        runtime=MagicMock(),
    )


def test_quick_rejects_todos_and_a_second_task():
    mw = RunDepthMiddleware()
    messages = [_user("hi", "quick")]

    def boom(_req):
        raise AssertionError("tool should not run")

    todo = mw.wrap_tool_call(_tool_request("write_todos", "t1", messages), boom)
    assert isinstance(todo, ToolMessage)
    assert "write_todos" in todo.content

    first = AIMessage(content="", tool_calls=[{
        "name": "task", "args": {"description": "one"}, "id": "a", "type": "tool_call",
    }, {
        "name": "task", "args": {"description": "two"}, "id": "b", "type": "tool_call",
    }])
    ran = {}

    def handler(req):
        ran["id"] = req.tool_call["id"]
        return ToolMessage(content="ok", tool_call_id=req.tool_call["id"])

    state = [_user("hi", "quick"), first]
    allowed = mw.wrap_tool_call(_tool_request("task", "a", state), handler)
    rejected = mw.wrap_tool_call(_tool_request("task", "b", state), handler)
    assert allowed.content == "ok"
    assert isinstance(rejected, ToolMessage)
    assert "fan out" in rejected.content


def test_deep_does_not_restrict_tools():
    def handler(req):
        return ToolMessage(content="ok", tool_call_id=req.tool_call["id"])

    messages = [_user("research this", "deep")]
    out = RunDepthMiddleware().wrap_tool_call(
        _tool_request("write_todos", "t1", messages), handler,
    )
    assert out.content == "ok"


def test_quick_budget_stops_early(monkeypatch):
    monkeypatch.setenv("QUICK_TOOL_CALL_SOFT_BUDGET", "2")
    monkeypatch.setenv("QUICK_TOOL_CALL_HARD_BUDGET", "4")

    def _ai(n: int) -> AIMessage:
        return AIMessage(
            content="thinking",
            tool_calls=[
                {"name": "web_research", "args": {"q": str(i)}, "id": f"c{i}", "type": "tool_call"}
                for i in range(n)
            ],
        )

    calls = []

    def handler(req):
        calls.append(req)
        return ModelResponse(result=[AIMessage(content="real")], structured_response=None)

    out = ToolCallBudgetMiddleware(soft_budget=80, hard_budget=150).wrap_model_call(
        _Req([_user("hi", "quick"), _ai(4)]),
        handler,
    )
    assert calls == []
    assert out.result[0].content == _QUICK_TERMINAL_TEXT
