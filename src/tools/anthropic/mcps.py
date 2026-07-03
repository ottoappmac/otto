import os
from contextlib import AsyncExitStack
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Tuple, List, Literal, Optional

from anthropic.types.beta import BetaToolParam

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from langchain_core.tools import BaseTool

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


DEFAULT_ENCODING = "utf-8"
DEFAULT_HTTP_TIMEOUT = 5
DEFAULT_SSE_READ_TIMEOUT = 60 * 5
DEFAULT_STREAMABLE_HTTP_TIMEOUT = timedelta(seconds=30)
DEFAULT_STREAMABLE_HTTP_SSE_READ_TIMEOUT = timedelta(seconds=60 * 5)

EncodingErrorHandler = Literal["strict", "ignore", "replace"]


class MCPHelper:
    def __init__(self, mcps: MultiServerMCPClient):
        self.mcps = mcps
        self.exit_stack = AsyncExitStack()
        self.sessions: Dict[str, ClientSession] = {}
        self.tools_by_server: Dict[str, list[BaseTool]] = {}
        self.mcp_map = {}
        self.mcp_params = []

    async def connect_all(self):
        for server_name, connection in self.mcps.connections.items():
            transport = connection.get("transport")
            if not transport:
                raise ValueError(f"Missing 'transport' in connection for server '{server_name}'")
            conn_kwargs = {k: v for k, v in connection.items() if k != "transport"}
            await self._connect_to_server(server_name, transport, **conn_kwargs)

    async def _connect_to_server(self, server_name: str, transport: str, **kwargs: Any):
        if transport == "stdio":
            await self._connect_stdio(server_name, **kwargs)
        elif transport == "sse":
            await self._connect_sse(server_name, **kwargs)
        elif transport == "streamable_http":
            await self._connect_streamable_http(server_name, **kwargs)
        elif transport == "websocket":
            await self._connect_websocket(server_name, **kwargs)
        else:
            raise ValueError(f"Unsupported transport: {transport}")

    async def _initialize(self, server_name: str, session: ClientSession):
        await session.initialize()
        self.sessions[server_name] = session
        tools = await load_mcp_tools(session)
        self.tools_by_server[server_name] = tools

    async def _connect_stdio(
        self,
        server_name: str,
        command: str,
        args: list[str],
        env: Optional[dict[str, str]] = None,
        cwd: Optional[str | Path] = None,
        encoding: str = DEFAULT_ENCODING,
        encoding_error_handler: EncodingErrorHandler = "strict",
        session_kwargs: Optional[dict[str, Any]] = None,
        **_,
    ):
        env = env or {}
        env.setdefault("PATH", os.environ.get("PATH", ""))

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
            cwd=cwd,
            encoding=encoding,
            encoding_error_handler=encoding_error_handler,
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        read, write = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(read, write, **(session_kwargs or {})))
        await self._initialize(server_name, session)

    async def _connect_sse(
        self,
        server_name: str,
        url: str,
        headers: Optional[dict[str, Any]] = None,
        timeout: float = DEFAULT_HTTP_TIMEOUT,
        sse_read_timeout: float = DEFAULT_SSE_READ_TIMEOUT,
        session_kwargs: Optional[dict[str, Any]] = None,
        **_,
    ):
        sse_transport = await self.exit_stack.enter_async_context(
            sse_client(url, headers, timeout, sse_read_timeout)
        )
        read, write = sse_transport
        session = await self.exit_stack.enter_async_context(ClientSession(read, write, **(session_kwargs or {})))
        await self._initialize(server_name, session)

    async def _connect_streamable_http(
        self,
        server_name: str,
        url: str,
        headers: Optional[dict[str, Any]] = None,
        timeout: timedelta = DEFAULT_STREAMABLE_HTTP_TIMEOUT,
        sse_read_timeout: timedelta = DEFAULT_STREAMABLE_HTTP_SSE_READ_TIMEOUT,
        session_kwargs: Optional[dict[str, Any]] = None,
        **_,
    ):
        transport = await self.exit_stack.enter_async_context(
            streamablehttp_client(url, headers, timeout, sse_read_timeout)
        )
        read, write, _ = transport
        session = await self.exit_stack.enter_async_context(ClientSession(read, write, **(session_kwargs or {})))
        await self._initialize(server_name, session)

    async def _connect_websocket(
        self,
        server_name: str,
        url: str,
        session_kwargs: Optional[dict[str, Any]] = None,
        **_,
    ):
        try:
            from mcp.client.websocket import websocket_client
        except ImportError:
            raise ImportError("Install with `pip install mcp[ws]` to use websocket transport")

        ws_transport = await self.exit_stack.enter_async_context(websocket_client(url))
        read, write = ws_transport
        session = await self.exit_stack.enter_async_context(ClientSession(read, write, **(session_kwargs or {})))
        await self._initialize(server_name, session)

    async def build_mcp_map_and_params(self) -> Tuple[Dict[str, ClientSession], List[BetaToolParam]]:
        await self.connect_all()
        for server_name, session in self.sessions.items():
            response = await session.list_tools()

            for tool in response.tools:
                self.mcp_map[tool.name] = session
                self.mcp_params.append(
                    BetaToolParam({
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": tool.inputSchema
                    })
                )
        return self.mcp_map, self.mcp_params

    def get_tools(self) -> list[BaseTool]:
        return [tool for tools in self.tools_by_server.values() for tool in tools]

    def get_sessions(self) -> Dict[str, ClientSession]:
        return self.sessions

    async def close(self):
        await self.exit_stack.aclose()
