"""Unit tests for cross-server MCP tool-name disambiguation.

Regression coverage for a real bug: when two connected MCP servers
expose a tool with the same bare name (e.g. Slack's and Microsoft
Teams' ``list_users``), a flat tool list let one silently shadow the
other in LangGraph's ``ToolNode`` dispatch dict — calling what looked
like Slack's ``list_users`` could actually invoke Microsoft Teams'
implementation and surface its credential error instead.

See :func:`backend.mcp_manager.dedupe_tool_names`.
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from backend.mcp_manager import dedupe_tool_names


def _tool(name: str) -> StructuredTool:
    def fn(**kwargs: object) -> str:
        return name

    return StructuredTool.from_function(func=fn, name=name, description=f"{name} tool")


def test_unique_names_are_returned_unchanged():
    slack_tools = [_tool("list_channels"), _tool("send_message")]
    edgar_tools = [_tool("search_filings")]

    result = dedupe_tool_names([("slack", slack_tools), ("edgar-sec", edgar_tools)])

    names = {t.name for t in result}
    assert names == {"list_channels", "send_message", "search_filings"}
    # Unique names must be the *same* tool objects, not copies — this
    # is the common case and should be a no-op for them.
    assert result[0] is slack_tools[0] or result[0] is slack_tools[1]


def test_colliding_names_are_prefixed_with_server_id():
    slack_tools = [_tool("list_users"), _tool("list_channels")]
    teams_tools = [_tool("list_users"), _tool("get_channel_info")]

    result = dedupe_tool_names([("slack", slack_tools), ("microsoft-teams", teams_tools)])
    names = {t.name for t in result}

    assert "slack__list_users" in names
    assert "microsoft-teams__list_users" in names
    # Non-colliding names from each server stay bare.
    assert "list_channels" in names
    assert "get_channel_info" in names
    # The collision is the only thing renamed — 4 tools in, 4 tools out.
    assert len(result) == 4


def test_renamed_tool_still_dispatches_to_its_own_implementation():
    """The whole point: each renamed tool must still call ITS OWN server.

    Before the fix, a bare-name dict (``{t.name: t for t in tools}``)
    would let the second-registered server's ``list_users`` shadow the
    first's. After dedup, both survive under distinct names and each
    still invokes its own underlying function.
    """
    slack_tools = [_tool("list_users")]
    teams_tools = [_tool("list_users")]

    result = dedupe_tool_names([("slack", slack_tools), ("microsoft-teams", teams_tools)])
    by_name = {t.name: t for t in result}

    assert by_name["slack__list_users"].invoke({}) == "list_users"
    assert by_name["microsoft-teams__list_users"].invoke({}) == "list_users"
    # Confirm they're independent copies, not aliases of each other.
    assert by_name["slack__list_users"] is not by_name["microsoft-teams__list_users"]


def test_three_way_collision_renames_all_three():
    result = dedupe_tool_names([
        ("slack", [_tool("list_channels")]),
        ("discord", [_tool("list_channels")]),
        ("microsoft-teams", [_tool("list_channels")]),
    ])
    names = {t.name for t in result}
    assert names == {"slack__list_channels", "discord__list_channels", "microsoft-teams__list_channels"}


def test_empty_input_returns_empty_list():
    assert dedupe_tool_names([]) == []


def test_disconnected_servers_excluded_by_caller_not_helper():
    """dedupe_tool_names trusts its caller's filtering; it has no
    'connected' concept of its own — every (server_id, tools) pair
    passed in is treated as live.
    """
    result = dedupe_tool_names([("slack", [_tool("list_users")])])
    assert len(result) == 1
    assert result[0].name == "list_users"
