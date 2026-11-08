"""The benchmark agent runs with a narrow toolbox; other agents are untouched."""

from types import SimpleNamespace

from backend.session_manager import _AGENT_TOOL_ALLOWLIST, _restrict_agent_tools


def _tools(*names):
    return [SimpleNamespace(name=n) for n in names]


def test_unlisted_agents_keep_every_tool():
    tools = _tools("run_tests", "send_email", "create_schedule")
    assert _restrict_agent_tools("coding-agent", tools) is tools
    assert _restrict_agent_tools(None, tools) is tools


def test_swebench_agent_keeps_only_allowlisted_tools():
    tools = _tools("run_tests", "search_code", "send_email", "create_schedule", "browser_click")
    kept = _restrict_agent_tools("swebench-coding-agent", tools)
    assert [t.name for t in kept] == ["run_tests", "search_code"]


def test_allowlist_has_no_unrelated_tools():
    allowed = _AGENT_TOOL_ALLOWLIST["swebench-coding-agent"]
    assert not any(n.startswith(("browser", "send_", "create_", "schedule")) for n in allowed)
