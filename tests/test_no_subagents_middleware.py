"""NoSubagentsMiddleware hides deepagents' ``task`` tool from an agent."""

import asyncio
from types import SimpleNamespace

from langchain_core.messages import SystemMessage, ToolMessage

from backend.session_manager import _AGENTS_WITHOUT_MEMORY, _AGENTS_WITHOUT_SUBAGENTS
from middleware.no_subagents import NoSubagentsMiddleware


class _Request:
    def __init__(self, tools, system_message):
        self.tools = tools
        self.system_message = system_message

    def override(self, **kw):
        return _Request(kw.get("tools", self.tools), kw.get("system_message", self.system_message))


def _system():
    return SystemMessage(content_blocks=[
        {"type": "text", "text": "You fix bugs."},
        {"type": "text", "text": "\n\n## `task` (subagent spawner)\n\nUse task.\n\nAvailable subagent types:\n- general-purpose: x"},
        {"type": "text", "text": "\n\n## Date\nToday."},
    ])


def test_model_request_loses_task_tool_and_prompt():
    seen = {}

    def handler(req):
        seen["req"] = req
        return "ok"

    tools = [SimpleNamespace(name="task"), SimpleNamespace(name="execute"), {"name": "grep"}]
    NoSubagentsMiddleware().wrap_model_call(_Request(tools, _system()), handler)
    req = seen["req"]
    assert [getattr(t, "name", None) or t["name"] for t in req.tools] == ["execute", "grep"]
    text = "".join(b["text"] for b in req.system_message.content_blocks)
    assert "subagent spawner" not in text
    assert "You fix bugs." in text and "## Date" in text


def test_task_call_is_refused_without_running():
    def handler(req):
        raise AssertionError("task must not run")

    req = SimpleNamespace(tool_call={"name": "task", "id": "c1", "args": {}})
    out = asyncio.run(NoSubagentsMiddleware().awrap_tool_call(req, handler))
    assert isinstance(out, ToolMessage) and out.status == "error" and out.tool_call_id == "c1"


def test_other_tools_pass_through():
    req = SimpleNamespace(tool_call={"name": "execute", "id": "c2", "args": {}})
    assert NoSubagentsMiddleware().wrap_tool_call(req, lambda r: "ran") == "ran"


def test_bench_agent_works_alone_without_memory():
    assert "swebench-coding-agent" in _AGENTS_WITHOUT_SUBAGENTS
    assert "swebench-coding-agent" in _AGENTS_WITHOUT_MEMORY
    assert "coding-agent" not in _AGENTS_WITHOUT_SUBAGENTS | _AGENTS_WITHOUT_MEMORY
