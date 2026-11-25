#!/usr/bin/env python3
"""Built-in MCP server: a remote macOS VM, for the MacArena benchmark.

Every tool sends one structured GUI action (or a screenshot request) to the MacArena *bridge*
(``scripts/macarena/bridge.py``), which validates it and runs it in the benchmark's macOS VM. Nothing here
touches the machine Otto runs on. The tools are raw mouse and keyboard input, the same action space MacArena
gives its baseline agents; there is deliberately no shell, file or AppleScript tool.

Coordinates are pixels in the screenshot the tools return (origin top-left).

Config. Otto resets a built-in MCP's ``env`` from the registry every time it loads its config, and tool servers
only inherit a small safe subset of the backend's environment, so the per-run bridge address and token are
read from, in this order:
  1. env MACARENA_BRIDGE_URL / MACARENA_AGENT_TOKEN (tests, manual runs)
  2. a JSON file {"url": ..., "token": ...}: $MACARENA_BRIDGE_FILE, else ~/.macarena-bridge.json
     (scripts/macarena/start_backend.sh writes it, mode 600, into the benchmark backend's own HOME)
The token is the bridge's *agent* token, not its control token.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP, Image

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("otto.mcp.macarena_vm")

mcp = FastMCP("MacArena VM")


def _settings() -> tuple[str, str]:
    """(bridge url, agent token). Either may be empty when not configured."""
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


def _post(path: str, body: dict) -> dict:
    base, token = _settings()
    if not token:
        return {"ok": False, "error": "the VM bridge token is not configured for this tool server "
                                      "(MACARENA_AGENT_TOKEN or ~/.macarena-bridge.json)"}
    try:
        r = httpx.post(base + path, json=body, headers={"Authorization": f"Bearer {token}"}, timeout=180)
    except httpx.HTTPError as exc:
        return {"ok": False, "error": f"cannot reach the VM bridge: {type(exc).__name__}: {exc}"}
    try:
        return r.json()
    except ValueError:
        return {"ok": False, "error": f"bridge returned HTTP {r.status_code}"}


def _budget_line(res: dict) -> str:
    if "steps_left" not in res:
        return ""
    left = res.get("steps_left")
    line = f"steps used {res.get('steps_used')}, " + ("no step limit" if left is None else f"steps left {left}")
    if res.get("finished"):
        line += f". RUN OVER ({res['finished']}): no more actions will run; stop and give your final answer."
    return line


def _result(res: dict, what: str) -> list:
    """Text for the agent, plus the new screenshot when the bridge sent one.

    Always a bare ``list`` (also for errors): FastMCP serialises a ``list[Any]`` / union return annotation
    as structured output and fails on the Image object, which only showed up over real stdio.
    """
    if not res.get("ok"):
        return [f"Error: {res.get('error') or 'unknown error'}" + (f"\n{_budget_line(res)}" if _budget_line(res) else "")]
    out: list[Any] = [f"{what}. {_budget_line(res)}".strip()]
    if res.get("screenshot"):
        out.append(Image(data=base64.b64decode(res["screenshot"]), format="png"))
    return out


def _act(action: dict, what: str) -> list:
    return _result(_post("/agent/act", {"action": action}), what)


@mcp.tool()
def vm_screenshot() -> list:
    """Take a screenshot of the VM's screen. Free: it does not use a step. Coordinates in every other
    tool are pixels in this image, origin top-left."""
    res = _post("/agent/observe", {"kind": "screenshot"})
    if not res.get("ok"):
        return f"Error: {res.get('error')}"
    size = res.get("size") or ["?", "?"]
    return [f"Screenshot {size[0]}x{size[1]}. {_budget_line(res)}",
            Image(data=base64.b64decode(res["screenshot"]), format="png")]


@mcp.tool()
def vm_accessibility_tree() -> str:
    """The VM's accessibility tree as text, when the run provides it. Free: it does not use a step."""
    res = _post("/agent/observe", {"kind": "tree"})
    if not res.get("ok"):
        return f"Error: {res.get('error')}"
    return (res.get("tree") or "(empty tree)")[:60000]


@mcp.tool()
def vm_click(x: int, y: int, button: str = "left") -> list:
    """Click at pixel (x, y). button is left, right or middle. Uses one step; returns the new screenshot."""
    return _act({"type": "click", "x": x, "y": y, "button": button}, f"Clicked ({x}, {y})")


@mcp.tool()
def vm_double_click(x: int, y: int) -> list:
    """Double-click at pixel (x, y). Uses one step; returns the new screenshot."""
    return _act({"type": "double_click", "x": x, "y": y}, f"Double-clicked ({x}, {y})")


@mcp.tool()
def vm_right_click(x: int, y: int) -> list:
    """Right-click at pixel (x, y). Uses one step; returns the new screenshot."""
    return _act({"type": "right_click", "x": x, "y": y}, f"Right-clicked ({x}, {y})")


@mcp.tool()
def vm_move_to(x: int, y: int) -> list:
    """Move the pointer to (x, y) without clicking (for hover menus). Uses one step."""
    return _act({"type": "move_to", "x": x, "y": y}, f"Moved to ({x}, {y})")


@mcp.tool()
def vm_drag(from_x: int, from_y: int, to_x: int, to_y: int) -> list:
    """Press at (from_x, from_y), drag to (to_x, to_y) and release. Uses one step."""
    return _act({"type": "drag_to", "from_x": from_x, "from_y": from_y, "x": to_x, "y": to_y},
                f"Dragged ({from_x}, {from_y}) -> ({to_x}, {to_y})")


@mcp.tool()
def vm_scroll(amount: int, x: int | None = None, y: int | None = None) -> list:
    """Scroll by `amount` clicks (positive up, negative down), at (x, y) if given. Uses one step."""
    action: dict[str, Any] = {"type": "scroll", "amount": amount}
    if x is not None and y is not None:
        action.update(x=x, y=y)
    return _act(action, f"Scrolled {amount}")


@mcp.tool()
def vm_type(text: str) -> list:
    """Type text into the focused field (click it first). Uses one step. Use vm_press for Enter, Tab, etc."""
    return _act({"type": "type", "text": text}, f"Typed {len(text)} characters")


@mcp.tool()
def vm_press(key: str) -> list:
    """Press one key: enter, tab, escape, space, backspace, delete, up/down/left/right, home, end,
    pageup, pagedown, f1-f12, or a single character. Uses one step."""
    return _act({"type": "press", "key": key}, f"Pressed {key}")


@mcp.tool()
def vm_hotkey(keys: list[str]) -> list:
    """Press a key combination, e.g. ["command", "c"] or ["command", "shift", "g"]. Uses one step."""
    return _act({"type": "hotkey", "keys": keys}, "Pressed " + "+".join(keys))


@mcp.tool()
def vm_wait() -> list:
    """Wait a few seconds for the screen to change (a loading app, a dialog). Uses one step."""
    return _act({"type": "wait"}, "Waited")


@mcp.tool()
def vm_done() -> str:
    """Say the task is finished. The result is checked on the VM's final state whether or not you call this."""
    res = _post("/agent/act", {"action": {"type": "done"}})
    return "Marked done. Give your final answer now." if res.get("ok") else f"Error: {res.get('error')}"


@mcp.tool()
def vm_fail(reason: str = "") -> str:
    """Say the task cannot be done (it is impossible or not possible in this environment). Some tasks are
    infeasible on purpose and only pass if you say so; do not use this just because a step was hard."""
    res = _post("/agent/act", {"action": {"type": "fail"}})
    return "Marked as infeasible. Give your final answer now." if res.get("ok") else f"Error: {res.get('error')}"


if __name__ == "__main__":
    mcp.run()
