"""The agent's tool list must not depend on which MCP server connects first.

``MCPManager.connect_all`` connects every server concurrently, and
``connect`` registers each connection when it finishes — so the order of
``get_all_tools()`` (and therefore of the tool schemas rendered into the
system prompt) used to follow connection *completion* order.  Two sessions
with the very same servers then got byte-different prompts, which defeats
every prompt-prefix cache (a local MLX model re-prefilled ~33k tool tokens
for every new chat).
"""

from __future__ import annotations

import asyncio

from langchain_core.tools import StructuredTool

from backend.config import MCPServerConfig
from backend.mcp_manager import MCPConnection, MCPManager


def _tool(name: str) -> StructuredTool:
    def fn(**kwargs: object) -> str:
        return name

    return StructuredTool.from_function(func=fn, name=name, description=f"{name} tool")


def _configs(*ids: str) -> list[MCPServerConfig]:
    return [MCPServerConfig(id=i, name=i, transport="stdio", command="true", enabled=True) for i in ids]


def _manager_with_delays(delays: dict[str, float]) -> MCPManager:
    """An MCPManager whose ``connect`` finishes after a per-server delay."""
    manager = MCPManager()

    async def fake_connect(cfg: MCPServerConfig, **_: object) -> None:
        await asyncio.sleep(delays[cfg.id])
        manager._connections[cfg.id] = MCPConnection(
            config=cfg, tools=[_tool(f"{cfg.id}_tool")], connected=True,
        )

    manager.connect = fake_connect  # type: ignore[method-assign]
    return manager


async def test_tool_order_follows_config_order_not_connect_completion():
    configs = _configs("mail", "notes", "calendar")
    # The last configured server connects first, the first one last.
    manager = _manager_with_delays({"mail": 0.06, "notes": 0.03, "calendar": 0.0})

    await manager.connect_all(configs)

    assert [t.name for t in manager.get_all_tools()] == ["mail_tool", "notes_tool", "calendar_tool"]


async def test_two_sessions_with_different_timings_get_identical_tool_lists():
    configs = _configs("a", "b", "c", "d")
    first = _manager_with_delays({"a": 0.0, "b": 0.02, "c": 0.04, "d": 0.06})
    second = _manager_with_delays({"a": 0.06, "b": 0.04, "c": 0.02, "d": 0.0})

    await first.connect_all(configs)
    await second.connect_all(configs)

    assert [t.name for t in first.get_all_tools()] == [t.name for t in second.get_all_tools()]
