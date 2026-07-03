"""Playwright browser isolation pool for concurrent subagent use.

When multiple subagents need browser automation simultaneously, sharing a
single Playwright MCP server causes navigation/state conflicts.  This module
provides a pool of isolated Playwright MCP instances — each with its own
Chromium browser — that subagents acquire for the duration of their invocation.

Architecture::

    PlaywrightPool       — manages lifecycle of N ephemeral PW MCP processes
    PlaywrightInstance   — one process + MCP connection + tool set
    proxy_playwright_tools() — wraps PW tools in-place to route through pool
    active_pw_instance   — contextvars.ContextVar scoped per subagent

Integration happens in :class:`StreamingSubagentRunnable`: before running the
subagent graph it acquires an instance, sets the context var, and releases on
completion.  Proxy tools check the context var and delegate to the acquired
instance's tools, falling back to the primary (shared) instance when no pool
instance is active (e.g. orchestrator direct use).
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import shutil
import socket
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Context variable — set per subagent invocation by StreamingSubagentRunnable
# ---------------------------------------------------------------------------

active_pw_instance: contextvars.ContextVar[PlaywrightInstance | None] = (
    contextvars.ContextVar("active_pw_instance", default=None)
)


# ---------------------------------------------------------------------------
# Instance dataclass
# ---------------------------------------------------------------------------

@dataclass
class PlaywrightInstance:
    """An isolated Playwright MCP server with its own browser."""

    port: int
    process: Any = None          # ManagedProcess
    helper: Any = None           # MCPHelper
    tools: dict[str, BaseTool] = field(default_factory=dict)
    data_dirs: list[Path] = field(default_factory=list)

    async def shutdown(self) -> None:
        if self.helper is not None:
            try:
                await self.helper.close()
            except Exception:
                logger.debug("Error closing PW helper on port %d", self.port, exc_info=True)
            self.helper = None
        if self.process is not None:
            try:
                await self.process.stop()
            except Exception:
                logger.debug("Error stopping PW process on port %d", self.port, exc_info=True)
            self.process = None
        if self.data_dirs:
            await asyncio.to_thread(self._remove_data_dirs)

    def _remove_data_dirs(self) -> None:
        for d in self.data_dirs:
            try:
                shutil.rmtree(d, ignore_errors=True)
                logger.debug("Removed ephemeral dir: %s", d.name)
            except Exception:
                logger.debug("Failed to remove %s", d.name, exc_info=True)


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------

def _default_max_instances() -> int:
    """Max concurrent isolated browsers.

    Overridable via ``PLAYWRIGHT_POOL_MAX_INSTANCES`` so deployments with more
    RAM can isolate larger parallel subagent batches. When more subagents run
    in parallel than the limit, the extras block on :meth:`PlaywrightPool.acquire`
    until a browser frees up (serialized) rather than silently sharing one.
    """
    raw = os.environ.get("PLAYWRIGHT_POOL_MAX_INSTANCES", "")
    try:
        val = int(raw)
        if val >= 1:
            return val
    except (TypeError, ValueError):
        pass
    return 5


_DEFAULT_MAX_INSTANCES = _default_max_instances()


def _find_free_port() -> int:
    """Bind to port 0 and let the OS assign an ephemeral port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class PlaywrightPool:
    """Pool of isolated Playwright MCP instances for concurrent subagent use.

    Instances are created lazily on first :meth:`acquire` and reused across
    subagent invocations within the same session.  All instances are torn down
    when :meth:`shutdown` is called (triggered by session close).

    Args:
        base_config: The primary Playwright ``MCPServerConfig`` (used as
            template for excluded_tools and env-var settings).
        max_instances: Maximum number of concurrent browser instances.
    """

    def __init__(self, base_config: Any, *, max_instances: int = _DEFAULT_MAX_INSTANCES) -> None:
        self._base_config = base_config
        self._max_instances = max_instances
        self._available: asyncio.Queue[PlaywrightInstance] = asyncio.Queue()
        self._all: list[PlaywrightInstance] = []
        self._lock = asyncio.Lock()

    async def startup(self) -> None:
        """Clean up orphaned ephemeral dirs left behind by crashed processes."""
        await asyncio.to_thread(self._cleanup_orphaned_dirs)

    @staticmethod
    def _is_pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    @staticmethod
    def _cleanup_orphaned_dirs() -> None:
        """Remove ephemeral dirs whose owning process is no longer running.

        Each ephemeral dir contains a ``.pid`` file written at creation time.
        If the PID is no longer alive the dir is orphaned and safe to remove.
        Dirs without a ``.pid`` marker are also removed (legacy leftovers).
        """
        from backend.config import get_app_data_dir

        app_data = get_app_data_dir()
        for pattern in ("playwright-profile-ephemeral-*", "playwright-output-ephemeral-*"):
            for d in app_data.glob(pattern):
                pid_file = d / ".pid"
                if pid_file.exists():
                    try:
                        pid = int(pid_file.read_text().strip())
                    except (ValueError, OSError):
                        pid = None
                    if pid is not None and PlaywrightPool._is_pid_alive(pid):
                        continue
                shutil.rmtree(d, ignore_errors=True)
                logger.debug("Cleaned up orphaned dir: %s", d.name)

    async def acquire(self) -> PlaywrightInstance:
        """Get an available instance, creating a new one if under the limit."""
        try:
            return self._available.get_nowait()
        except asyncio.QueueEmpty:
            pass

        async with self._lock:
            if len(self._all) < self._max_instances:
                instance = await self._create_instance()
                self._all.append(instance)
                return instance

        return await self._available.get()

    async def release(self, instance: PlaywrightInstance) -> None:
        """Return an instance to the pool for reuse."""
        await self._available.put(instance)

    async def shutdown(self) -> None:
        """Stop all managed browser instances and clean up."""
        for inst in list(self._all):
            await inst.shutdown()
        self._all.clear()
        while not self._available.empty():
            try:
                self._available.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _create_instance(self) -> PlaywrightInstance:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        from backend.config import MCPServerConfig, get_app_data_dir
        from backend.mcp_manager import ManagedProcess, _build_playwright_command
        from tools.anthropic.mcps import MCPHelper

        port = await asyncio.to_thread(_find_free_port)
        instance_id = f"ephemeral-{port}"
        url = f"http://localhost:{port}/mcp"

        app_data = get_app_data_dir()
        data_dirs = [
            app_data / f"playwright-profile-{instance_id}",
            app_data / f"playwright-output-{instance_id}",
        ]

        config = MCPServerConfig(
            id=f"playwright-{instance_id}",
            name=f"Playwright ({instance_id})",
            transport="streamable_http",
            url=url,
            enabled=True,
            auto_start=True,
            builtin=True,
            excluded_tools=list(self._base_config.excluded_tools),
        )

        cmd = _build_playwright_command(config, instance_id=instance_id)
        proc = ManagedProcess(server_id=config.id, command=cmd, health_url=url)
        await proc.start()

        # Write a PID marker so the startup sweep can distinguish orphaned
        # dirs from dirs still owned by a live process.
        pid = proc.pid
        if pid is not None:
            def _write_pid_markers() -> None:
                for d in data_dirs:
                    try:
                        (d / ".pid").write_text(str(pid))
                    except Exception:
                        pass
            await asyncio.to_thread(_write_pid_markers)

        client = MultiServerMCPClient(
            connections={"playwright": {"transport": "streamable_http", "url": url}}
        )
        helper = MCPHelper(client)
        await helper.connect_all()

        raw_tools = helper.get_tools()
        excluded = set(config.excluded_tools)
        raw_tools = [t for t in raw_tools if t.name not in excluded]
        for t in raw_tools:
            t.handle_tool_error = True
            _strip_none_args(t)

        tools_dict = {t.name: t for t in raw_tools}

        instance = PlaywrightInstance(
            port=port, process=proc, helper=helper, tools=tools_dict,
            data_dirs=data_dirs,
        )
        logger.info(
            "Created ephemeral Playwright instance on port %d (%d tools)",
            port, len(tools_dict),
        )
        return instance


# ---------------------------------------------------------------------------
# Tool proxying
# ---------------------------------------------------------------------------

def proxy_playwright_tools(tools: list[BaseTool]) -> None:
    """Wrap Playwright tools **in-place** to route through the pool.

    After calling this, the tool objects in *tools* will:

    - Check :data:`active_pw_instance` on every invocation.
    - If an instance is set (subagent context), delegate to that instance's
      matching tool — giving each subagent its own isolated browser.
    - Otherwise fall back to the original (primary) tool, preserving
      single-user / orchestrator behaviour unchanged.

    Call this once on the primary PW tool list before passing tools to
    subagent builders and the orchestrator.
    """
    for tool in tools:
        _proxy_tool_in_place(tool)


def _proxy_tool_in_place(tool: BaseTool) -> None:
    if not isinstance(tool, StructuredTool) or tool.coroutine is None:
        return

    tool_name = tool.name
    original_coro = tool.coroutine

    @wraps(original_coro)
    async def _proxy(_name: str = tool_name, _orig: Any = original_coro, **kwargs: Any) -> Any:
        instance = active_pw_instance.get()
        if instance is not None:
            target = instance.tools.get(_name)
            if target is not None:
                return await target.coroutine(**kwargs)
        return await _orig(**kwargs)

    tool.coroutine = _proxy


# ---------------------------------------------------------------------------
# Helpers (duplicated from mcp_manager to avoid circular import)
# ---------------------------------------------------------------------------

def _strip_none_args(tool: BaseTool) -> None:
    """Drop ``None``-valued kwargs before forwarding to the MCP server."""
    if not isinstance(tool, StructuredTool) or tool.coroutine is None:
        return
    original = tool.coroutine

    @wraps(original)
    async def _cleaned(*args: Any, **kwargs: Any) -> Any:
        cleaned = {k: v for k, v in kwargs.items() if v is not None}
        return await original(*args, **cleaned)

    tool.coroutine = _cleaned
