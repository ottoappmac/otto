"""MacArena "Otto-native" track: Otto's macOS desktop agent, with its tool calls run in the benchmark VM.

Like for like with the app: the benchmark agent gets Otto's ``macos-desktop-agent`` prompt and the very tool
objects ``MCPManager._load_builtin_macos_native`` builds (names, schemas, descriptions, loop guard, desktop
lock, the vision ``read_screen`` swap). The one change is where each call runs. ``install()`` replaces
``MacOSToolkit`` with a subclass whose tools send the call to the MacArena bridge (``/agent/native``), which
counts the step and runs it with Otto's unmodified toolkit inside the VM (``scripts/macarena/guest``).

The agent's AppleScript tools (the ``macos-osascript`` MCP) get the same treatment: ``install()`` also makes the
backend load that server through ``RemoteOsascriptHelper``, which starts no process on this computer. It builds the
tools from the server module's own FastMCP schemas (so names, arguments and descriptions are Otto's) and sends
each call to the bridge, which runs the server's function inside the VM. Only the VM set (``server.VM_TOOLS``) is
offered: the others read files on this host.

Nothing in the app imports this module's toolkit code. Only ``scripts/macarena/native_backend.py`` calls
``install()``, in the isolated benchmark backend; ``backend/macarena_guard_middleware.py`` reads ``ACTIVE`` so
the benchmark agent's desktop and AppleScript tools are refused unless the swap is in place (they must never
act on the host).
"""

from __future__ import annotations

import contextvars
import json
import os
from pathlib import Path
from typing import Any

AGENT = "macarena-desktop-agent"

# Set by install(). False in the app, so the guard refuses this agent's desktop tools there.
ACTIVE = False

# Every tool the macos-native connection can give an agent (text set + vision variants).
# tests/test_macarena_native.py checks this against the real toolkit.
NATIVE_TOOL_NAMES = frozenset({
    "activate_app", "batch_actions", "click", "click_app_dom", "click_at", "click_text", "double_click",
    "double_click_at", "find_text_on_screen", "get_control_value", "get_screen_controls", "hotkey", "launch_app",
    "list_apps", "open_app", "press_control", "read_app_dom", "read_screen", "right_click", "right_click_at",
    "scroll", "scroll_at", "spotlight_search", "type_into_control", "type_text", "wait_for_controls",
    "capture_app_screenshot",
    "observe", "menu_select", "menu_search", "read_dialog", "fill_dialog", "read_table", "get_control_values",
    "wait_until", "app_guide", "drag", "set_text_at", "write_cells", "select_text",
})
# The macos-osascript tools that run in the VM (the server's ``VM_TOOLS``; a test checks they match).
OSASCRIPT_SERVER_ID = "macos-osascript"
OSASCRIPT_TOOL_NAMES = frozenset({"run_osascript", "inspect_app_dictionary", "dump_ax_tree", "query_mail_store"})
REMOTE_TOOL_NAMES = NATIVE_TOOL_NAMES | OSASCRIPT_TOOL_NAMES
FINISH_TOOLS = frozenset({"vm_done", "vm_fail"})
# Procedural memory (``backend/procedural_memory``): runs on this host but only reads a frozen snapshot, so it cannot
# change the VM. Offered only when the backend serves one (OTTO_PROCEDURE_MODE=serve); the bridge counts each call as
# a step, like any other call, so memory-on and memory-off runs have the same budget.
MEMORY_TOOL_NAMES = frozenset({"recall_procedure"})
# No shell tool of its own: a VM shell (``vm_shell``) was tried and removed after one check, where it led to 36 calls
# poking at system databases, a deleted login item and an overclaimed result. Commands go through ``run_osascript``.
AGENT_TOOLS = REMOTE_TOOL_NAMES | FINISH_TOOLS | MEMORY_TOOL_NAMES

# MacOSToolkit constructor arguments, sent with every call so the guest builds its toolkit exactly as the
# host backend resolved it (Environment values, defaults).
CONFIG_FIELDS = ("ax_ipc_timeout", "scan_max_depth", "scan_max_elements", "scan_max_workers", "scan_time_budget",
                 "ax_bridge_init_delay", "ax_bridge_max_wait", "launch_delay", "activate_delay", "action_delay",
                 "scan_cache_ttl", "skip_apps")

REMOTE_TIMEOUT = 300.0
# Steps left at which the budget note tells the agent to wrap up (finish, or report the task infeasible).
LOW_BUDGET = 3

# Id of the agent turn (the model response) whose tool call is running. MacarenaGuardMiddleware sets it before each
# call; the bridge counts steps per turn when started with --step-unit turns.
TURN: contextvars.ContextVar[str | None] = contextvars.ContextVar("macarena_turn", default=None)


def bridge_settings() -> tuple[str, str]:
    """(bridge url, agent token): env first, then the benchmark profile's ~/.macarena-bridge.json
    (written by scripts/macarena/start_backend.sh, mode 600). Same source as the macarena-vm MCP."""
    url = os.environ.get("MACARENA_BRIDGE_URL", "")
    token = os.environ.get("MACARENA_AGENT_TOKEN", "")
    if not (url and token):
        path = Path(os.environ.get("MACARENA_BRIDGE_FILE") or Path.home() / ".macarena-bridge.json")
        try:
            data = json.loads(path.read_text())
            url, token = url or str(data.get("url", "")), token or str(data.get("token", ""))
        except (OSError, ValueError):
            pass
    return (url or "http://127.0.0.1:18095").rstrip("/"), token


def budget_note(res: dict) -> str:
    if "steps_left" not in res:
        return ""
    left = res.get("steps_left")
    note = (f"[benchmark] steps used {res.get('steps_used')}, "
            + ("no step limit" if left is None else f"steps left {left}"))
    if res.get("finished"):
        note += f". RUN OVER ({res['finished']}): no more actions will run; stop and give your final answer."
    elif isinstance(left, int) and 0 < left <= LOW_BUDGET:
        note += (f". ONLY {left} LEFT: finish now. If the task is done, call vm_done; if it cannot be done "
                 "here, call vm_fail now, before the steps run out.")
    return note


async def forward(toolkit: Any, tool_id: str, name: str, args: dict) -> Any:
    """Run one tool call in the VM through the bridge. Errors raise ToolException, which the tool's
    ``handle_tool_error=True`` turns into an error result, as for a local failure."""
    result, note = await _post(tool_id, name, args, toolkit.remote_config,
                               bool(getattr(toolkit, "vision_mode", False)))
    if not note:
        return result
    if isinstance(result, list):
        return [*result, {"type": "text", "text": note}]
    return f"{'' if result is None else result}\n\n{note}"


async def _post(tool_id: str, name: str, args: dict, config: dict, vision_mode: bool) -> tuple[Any, str]:
    """(the tool's result, the step-budget note or ""), or ToolException when the call failed or was refused."""
    import httpx
    from langchain_core.tools import ToolException

    url, token = bridge_settings()
    if not token:
        raise ToolException("the benchmark VM bridge is not configured (MACARENA_AGENT_TOKEN or "
                            "~/.macarena-bridge.json); nothing was run")
    body = {"tool_id": tool_id, "tool": name, "args": args, "config": config, "vision_mode": vision_mode}
    if TURN.get():
        body["turn"] = TURN.get()
    try:
        async with httpx.AsyncClient(timeout=REMOTE_TIMEOUT) as client:
            r = await client.post(url + "/agent/native", json=body, headers={"Authorization": f"Bearer {token}"})
        res = r.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ToolException(f"cannot reach the benchmark VM: {type(exc).__name__}: {exc}") from exc
    note = budget_note(res)
    if res.get("notice"):                      # e.g. several actions failed in a row (scripts/macarena/bridge.py)
        note = f"{res['notice']}\n{note}" if note else str(res["notice"])
    if not res.get("ok"):
        raise ToolException(str(res.get("error") or "unknown error") + (f"\n{note}" if note else ""))
    return res.get("result"), note


TICK_TIMEOUT = 10.0


def _tick_body(kind: str, tool: str | None, turn: str | None) -> dict:
    body: dict = {"kind": kind}
    if tool:
        body["tool"] = tool
    if turn:
        body["turn"] = turn
    return body


def tick(kind: str, tool: str | None = None, turn: str | None = None) -> dict | None:
    """Report one model call (``kind="model"``) or tool call (``"tool"``) to the bridge (``/agent/tick``). Returns
    the bridge's answer, or None when it cannot be reached: bookkeeping never breaks the agent."""
    import httpx
    url, token = bridge_settings()
    if not token:
        return None
    try:
        r = httpx.post(url + "/agent/tick", json=_tick_body(kind, tool, turn),
                       headers={"Authorization": f"Bearer {token}"}, timeout=TICK_TIMEOUT)
        return r.json()
    except (httpx.HTTPError, ValueError):
        return None


async def atick(kind: str, tool: str | None = None, turn: str | None = None) -> dict | None:
    """``tick`` for the async agent loop."""
    import httpx
    url, token = bridge_settings()
    if not token:
        return None
    try:
        async with httpx.AsyncClient(timeout=TICK_TIMEOUT) as client:
            r = await client.post(url + "/agent/tick", json=_tick_body(kind, tool, turn),
                                  headers={"Authorization": f"Bearer {token}"})
        return r.json()
    except (httpx.HTTPError, ValueError):
        return None


async def report_memory(record: dict) -> None:
    """Send one ``recall_procedure`` call's record (query, candidates, returned item ids) to the bridge, which keeps
    it for the run's result row and its ``memory_calls.jsonl``. Best effort."""
    import httpx
    url, token = bridge_settings()
    if not token:
        return
    body = {"kind": "memory", "record": record}
    if TURN.get():
        body["turn"] = TURN.get()
    try:
        async with httpx.AsyncClient(timeout=TICK_TIMEOUT) as client:
            await client.post(url + "/agent/tick", json=body, headers={"Authorization": f"Bearer {token}"})
    except httpx.HTTPError:
        return


def _remote(tool: Any, tool_id: str, toolkit: Any) -> None:
    name = tool.name

    async def run_in_vm(**kwargs: Any) -> Any:
        return await forward(toolkit, tool_id, name, kwargs)

    tool.coroutine = run_in_vm
    tool.func = None
    tool.metadata = {**(tool.metadata or {}), "macarena_remote": True}


def tool_ids(toolkit: Any) -> list[tuple[str, Any]]:
    out = [(f"tools:{t.name}", t) for t in toolkit.tools]
    out += [(f"vision:{t.name}", t) for t in toolkit.vision_tools]
    if getattr(toolkit, "read_screen_vision", None) is not None:
        out.append(("read_screen_vision", toolkit.read_screen_vision))
    return out


def remote_toolkit_class(base: type) -> type:
    class RemoteMacOSToolkit(base):  # type: ignore[misc, valid-type]
        """Otto's MacOSToolkit with every tool run in the MacArena VM. Builds the real tools (so names,
        schemas and descriptions are Otto's), then points each one at the bridge."""

        def __init__(self, **kwargs: Any):
            super().__init__(**kwargs)
            self.remote_config = {f: sorted(getattr(self, f)) if f == "skip_apps" else getattr(self, f)
                                  for f in CONFIG_FIELDS}
            for tool_id, t in tool_ids(self):
                _remote(t, tool_id, self)

    RemoteMacOSToolkit.__qualname__ = RemoteMacOSToolkit.__name__ = "RemoteMacOSToolkit"
    return RemoteMacOSToolkit


async def forward_osascript(name: str, args: dict) -> tuple[str, None]:
    """One AppleScript tool call, run in the VM. Returns what the MCP tool returns: the server's result dict as
    JSON text, and no artifact."""
    result, note = await _post(f"osascript:{name}", name, args, {}, False)
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, indent=2)
    return (f"{text}\n\n{note}" if note else text), None


def _remote_osascript_tool(spec: Any) -> Any:
    from langchain_core.tools import StructuredTool
    name = spec.name

    async def run_in_vm(**kwargs: Any) -> tuple[str, None]:
        return await forward_osascript(name, kwargs)

    return StructuredTool(name=name, description=spec.description or "", args_schema=spec.inputSchema,
                          coroutine=run_in_vm, response_format="content_and_artifact",
                          metadata={"macarena_remote": True})


class RemoteOsascriptHelper:
    """Stands in for the ``macos-osascript`` MCP client in the benchmark backend: no subprocess, tools from the
    server module's own schemas, every call run in the VM. Has the three methods the MCP manager uses."""

    def __init__(self) -> None:
        self._tools: list[Any] = []

    async def connect_all(self) -> None:
        from backend.builtin_mcps.macos_osascript import server
        specs = await server.mcp.list_tools()
        self._tools = [_remote_osascript_tool(s) for s in specs if s.name in server.VM_TOOLS]

    def get_tools(self) -> list[Any]:
        return list(self._tools)

    async def close(self) -> None:
        return None


def install() -> type:
    """Make the backend build remote toolkits. Call before the backend loads any MCP server.
    ``MCPManager._load_builtin_macos_native`` imports ``MacOSToolkit`` from ``tools.navigation.computer`` at
    call time, so patching that name is enough; ``tools.navigation`` re-exports it and is patched too."""
    global ACTIVE
    import tools.navigation as nav
    import tools.navigation.computer as comp

    base = comp.MacOSToolkit
    cls = base if getattr(base, "__name__", "") == "RemoteMacOSToolkit" else remote_toolkit_class(base)
    comp.MacOSToolkit = cls
    if hasattr(nav, "MacOSToolkit"):
        nav.MacOSToolkit = cls
    _install_remote_osascript()
    ACTIVE = True
    return cls


def _install_remote_osascript() -> None:
    """Load the ``macos-osascript`` server through ``RemoteOsascriptHelper``. The manager's own loader still wraps
    the tools (loop guard, redactor, desktop lease), exactly as for the real server."""
    from backend import mcp_manager

    real = mcp_manager.MCPManager._create_helper
    if getattr(real, "__macarena_remote__", False):
        return

    async def create_helper(self: Any, config: Any) -> Any:
        if config.id == OSASCRIPT_SERVER_ID:
            return RemoteOsascriptHelper()
        return await real(self, config)

    create_helper.__macarena_remote__ = True  # type: ignore[attr-defined]
    mcp_manager.MCPManager._create_helper = create_helper  # type: ignore[method-assign]
