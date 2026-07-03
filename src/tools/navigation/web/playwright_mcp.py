"""Playwright MCP client — browser automation via accessibility snapshots.

Connects to a running Playwright MCP service (``@playwright/mcp``) over
HTTP through the existing ``MCPHelper`` / ``MultiServerMCPClient``
infrastructure.

Start the service first::

    ./scripts/start_playwright_mcp.sh          # standalone
    ./scripts/start_all.sh                      # or with all services

Usage with Anthropic computer-use agent::

    from tools.navigation.web.playwright_mcp import create_playwright_mcp_client
    from tools.anthropic.tool_collection import ToolCollection

    mcps = create_playwright_mcp_client()
    collection = await ToolCollection.from_params(tools=my_tools, mcps=mcps)

Standalone usage with MCPHelper::

    from tools.navigation.web.playwright_mcp import create_playwright_mcp_client
    from tools.anthropic.mcps import MCPHelper

    mcps = create_playwright_mcp_client()
    helper = MCPHelper(mcps)
    await helper.connect_all()
    tools = helper.get_tools()   # LangChain-compatible BaseTool list
    # ... use tools in any LangGraph agent ...
    await helper.close()

As a context manager::

    async with PlaywrightMCP() as pw:
        tools = pw.get_tools()          # LangChain BaseTool list
        mcp_map, params = pw.get_anthropic_params()  # Anthropic format
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from anthropic.types.beta import BetaToolParam
from mcp.client.session import ClientSession
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from tools.anthropic.mcps import MCPHelper

logger = logging.getLogger(__name__)

_SERVER_NAME = "playwright"

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8931


def create_playwright_mcp_client(
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> MultiServerMCPClient:
    """Create a ``MultiServerMCPClient`` that connects to Playwright MCP.

    Connects via ``streamable_http`` to a running Playwright MCP service.
    Start the service with ``scripts/start_playwright_mcp.sh``.

    Args:
        host: Hostname of the Playwright MCP service.
              Env: ``PLAYWRIGHT_MCP_HOST``. Default: ``localhost``.
        port: Port of the Playwright MCP service.
              Env: ``PLAYWRIGHT_MCP_PORT``. Default: ``8931``.

    Returns:
        A ``MultiServerMCPClient`` ready to pass to ``MCPHelper`` or
        ``ToolCollection.from_params(mcps=...)``.
    """
    if host is None:
        host = os.getenv("PLAYWRIGHT_MCP_HOST", DEFAULT_HOST)
    if port is None:
        port_env = os.getenv("PLAYWRIGHT_MCP_PORT", str(DEFAULT_PORT))
        port = int(port_env)

    url = f"http://{host}:{port}/mcp"
    logger.info("Playwright MCP: connecting to %s", url)

    return MultiServerMCPClient(
        connections={
            _SERVER_NAME: {
                "transport": "streamable_http",
                "url": url,
            }
        }
    )


class PlaywrightMCP:
    """Async context manager for Playwright MCP browser automation.

    Connects to the running Playwright MCP service, loads tools,
    and cleans up on exit.

    Usage::

        async with PlaywrightMCP() as pw:
            tools = pw.get_tools()
            # Use tools in a LangGraph agent...

        # Or for Anthropic computer-use agent:
        async with PlaywrightMCP() as pw:
            mcp_map, mcp_params = pw.get_anthropic_params()
    """

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        self._host = host
        self._port = port
        self._helper: Optional[MCPHelper] = None

    async def __aenter__(self) -> PlaywrightMCP:
        mcps = create_playwright_mcp_client(host=self._host, port=self._port)
        self._helper = MCPHelper(mcps)
        await self._helper.connect_all()
        logger.info("Playwright MCP connected — %d tools loaded", len(self.get_tools()))
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._helper:
            await self._helper.close()
            self._helper = None

    def _check_connected(self) -> MCPHelper:
        if self._helper is None:
            raise RuntimeError("PlaywrightMCP is not connected. Use 'async with PlaywrightMCP() as pw:'")
        return self._helper

    def get_tools(self) -> List[BaseTool]:
        """Return LangChain-compatible tools for use in any LangGraph agent."""
        return self._check_connected().get_tools()

    def get_sessions(self) -> Dict[str, ClientSession]:
        """Return the raw MCP sessions (server_name -> ClientSession)."""
        return self._check_connected().get_sessions()

    def get_anthropic_params(self) -> Tuple[Dict[str, ClientSession], List[BetaToolParam]]:
        """Return (mcp_map, mcp_params) for Anthropic computer-use agents.

        These are already built during ``connect_all`` via
        ``MCPHelper.build_mcp_map_and_params``.
        """
        helper = self._check_connected()
        return helper.mcp_map, helper.mcp_params

    def get_helper(self) -> MCPHelper:
        """Return the underlying ``MCPHelper`` for advanced usage."""
        return self._check_connected()

    def get_mcp_client(self) -> MultiServerMCPClient:
        """Return the ``MultiServerMCPClient`` for passing to ``ToolCollection``."""
        return self._check_connected().mcps
