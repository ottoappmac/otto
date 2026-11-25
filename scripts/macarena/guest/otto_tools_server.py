#!/usr/bin/env python3
"""Runs Otto's own macOS desktop tools inside the MacArena VM, for the Otto-native track.

The bridge uploads this file with an unmodified copy of Otto's toolkit (``src/tools/navigation/computer``,
``src/utilities``) and of Otto's AppleScript tools (``src/macos_osascript``, the ``macos-osascript`` MCP server's
module, imported without the MCP SDK) and launches it through MacArena's in-guest server, so it runs in the same login session
and with the same Accessibility and Screen Recording permissions as MacArena's own actions. The agent, its
prompt and its model stay on the host; only each tool call runs here, because the apps being tested are here.

    POST /call    {"tool_id", "tool", "args", "config", "vision_mode", "budget"}
                  -> {"ok", "cost", "result" | "error"}; refused (nothing run) when the call costs > budget.
                  ``tool_id`` is ``tools:<name>`` / ``vision:<name>`` for the desktop toolkit and
                  ``osascript:<name>`` for the AppleScript tools.
    GET  /health  -> {"ok", "trusted", "screen_capture", "screen", "automation"}

Every request needs ``Authorization: Bearer <token>`` (the token file is written by the bridge per run).
The tools run on this process's main thread's event loop, as they do in Otto's backend; HTTP is served from
a second thread and calls are handled one at a time.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import logging
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # native_tools.py sits next to guest/
import native_tools  # noqa: E402

logger = logging.getLogger("otto.guest_tools")
CALL_TIMEOUT = 240.0
OSASCRIPT_PREFIX = "osascript:"


def controls_delta(prev: list | None, cur: list) -> str | None:
    """The toolkit's own delta (``macos_ax_struct.controls_delta``), or None (show everything) without it."""
    try:
        from tools.navigation.computer.macos_ax_struct import controls_delta as delta
    except ImportError:
        return None
    return delta(prev, cur, lambda it: f"[{it[0]}] {it[1]} '{it[2]}'")


def append_text(result: Any, text: str) -> Any:
    """``result`` with ``text`` after it: a string gets it appended, a list of content blocks gets a text block."""
    if isinstance(result, list):
        return [*result, {"type": "text", "text": text}]
    return f"{'' if result is None else result}\n\n{text}"


def jsonable(value: Any) -> Any:
    """Tool output as JSON: str and LangChain content blocks pass through; anything else becomes text."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


class GuestTools:
    """The toolkit and the call logic, separate from HTTP so tests can use a stand-in toolkit."""

    def __init__(self, make_toolkit: Callable[[dict], Any], parse_batch: Callable | None,
                 loop: asyncio.AbstractEventLoop, osascript_tools: dict[str, Callable] | None = None):
        self.make_toolkit, self.parse_batch, self.loop = make_toolkit, parse_batch, loop
        # name -> the macos-osascript server's own coroutine function (``run_osascript``, ...).
        self.osascript_tools = dict(osascript_tools or {})
        self.toolkit: Any = None
        self.config: dict | None = None
        self.lock = threading.Lock()
        # app -> [(index, role, label)] of the last controls listing the agent got for it, for observe_after deltas.
        self.last_seen: dict[str, list[tuple[int, str, str]]] = {}

    def _tools(self) -> dict[str, Any]:
        tk = self.toolkit
        out = {f"tools:{t.name}": t for t in tk.tools}
        out.update({f"vision:{t.name}": t for t in tk.vision_tools})
        if getattr(tk, "read_screen_vision", None) is not None:
            out["read_screen_vision"] = tk.read_screen_vision
        return out

    def call(self, body: dict) -> dict:
        tool_id, tool = str(body.get("tool_id") or ""), str(body.get("tool") or "")
        args = body.get("args") or {}
        if not isinstance(args, dict):
            return {"ok": False, "error": "args must be an object", "cost": 0, "refused": True}
        if tool_id.startswith(OSASCRIPT_PREFIX):
            return self._call_osascript(tool_id, tool, args, int(body.get("budget", 0)))
        with self.lock:
            config = body.get("config") or {}
            if self.toolkit is None or config != self.config:
                # Same constructor arguments as the host backend resolved for its own toolkit.
                self.toolkit, self.config = self.make_toolkit(config), config
            tools = self._tools()
            t = tools.get(tool_id)
            if t is None or t.name != tool:
                return {"ok": False, "error": f"unknown tool {tool_id!r}", "cost": 0, "refused": True}
            n = native_tools.cost(tool, args, self.parse_batch)
            budget = int(body.get("budget", 0))
            if n > budget:
                return {"ok": False, "refused": True, "cost": n,
                        "error": f"this call needs {n} step(s) but {budget} are left; nothing was run"}
            self.toolkit.vision_mode = bool(body.get("vision_mode"))
            fut = asyncio.run_coroutine_threadsafe(t.ainvoke(args), self.loop)
            try:
                result = fut.result(timeout=CALL_TIMEOUT)
            except Exception as exc:  # noqa: BLE001 — the tool's own error, shown to the agent as in Otto
                fut.cancel()
                return {"ok": False, "cost": n, "error": f"{type(exc).__name__}: {exc}"}
            out = {"ok": True, "cost": n, "result": jsonable(result)}
            if tool in ("get_screen_controls", "wait_for_controls") and args.get("app_name"):
                self._remember(str(args["app_name"]))
            if body.get("observe_after") and tool in native_tools.OBSERVE_AFTER:
                seen = self._observe(tools)
                if seen is not None:
                    out["result"], out["observed"] = append_text(out["result"], seen), True
            return out

    def _observe(self, tools: dict[str, Any]) -> str | None:
        """The frontmost app's controls, through Otto's own ``get_screen_controls`` (so the indices the agent sees
        are the ones ``press_control`` uses next). None when there is nothing to show."""
        scan = tools.get("tools:get_screen_controls")
        name_of = getattr(type(self.toolkit), "_frontmost_app_name", None)
        app = name_of() if callable(name_of) else ""
        if scan is None or not app:
            return None
        extra = self._situation(tools)
        dialog = tools.get("tools:read_dialog")
        if dialog is not None and "\nDialog:" in extra:
            # A dialog is open: its fields are what the agent acts on next, and the window behind it (a sheet of
            # cells, toolbars) is noise. read_dialog's indices work with press_control as well.
            fut = asyncio.run_coroutine_threadsafe(dialog.ainvoke({"app_name": app}), self.loop)
            try:
                text = str(fut.result(timeout=CALL_TIMEOUT))
            except Exception:  # noqa: BLE001 — fall back to the full read below
                fut.cancel()
            else:
                on_top = [ln for ln in extra.splitlines() if ln.startswith("On top of it:")]
                return f"Dialog now ({app}), read after the action:\n{text}" + "".join("\n" + ln for ln in on_top)
        fut = asyncio.run_coroutine_threadsafe(scan.ainvoke({"app_name": app}), self.loop)
        try:
            text = str(fut.result(timeout=CALL_TIMEOUT))
        except Exception as exc:  # noqa: BLE001 — the action ran; only the extra read failed
            fut.cancel()
            return f"(could not read the controls of {app}: {type(exc).__name__})"
        prev = self.last_seen.get(app.lower())
        cur = self._remember(app)
        delta = controls_delta(prev, cur) if cur is not None else None
        if delta is not None:
            return f"Controls now ({app}), read after the action:\n{delta}{extra}"
        limit = native_tools.OBSERVE_AFTER_MAX_CHARS
        if len(text) > limit:
            text = text[:limit] + "\n... (cut; call get_screen_controls with role= or label= to see the rest)"
        return f"Controls now ({app}), read after the action:\n{text}{extra}"

    def _remember(self, app: str) -> list[tuple[int, str, str]] | None:
        """Record the controls of ``app``'s last scan (from the toolkit's scan cache) as what the agent saw."""
        get = getattr(self.toolkit, "_cache_get", None)
        cached = get(app) if callable(get) else None
        if not cached:
            return None
        items = [(int(c["index"]), str(c.get("role", "")), str(c.get("label", "")))
                 for c in cached[0] if not c.get("_structural") and "index" in c]
        self.last_seen[app.lower()] = items
        return items

    def _situation(self, tools: dict[str, Any]) -> str:
        """The dialog and on-top lines of ``observe`` when there is something to report, else ""."""
        obs = tools.get("tools:observe")
        if obs is None:
            return ""
        fut = asyncio.run_coroutine_threadsafe(obs.ainvoke({}), self.loop)
        try:
            text = str(fut.result(timeout=CALL_TIMEOUT))
        except Exception:  # noqa: BLE001 — optional extra
            fut.cancel()
            return ""
        keep = [ln for ln in text.splitlines()
                if (ln.startswith("Dialog:") and ln != "Dialog: none") or ln.startswith("On top of it:")]
        return ("\n" + "\n".join(keep)) if keep else ""

    def _call_osascript(self, tool_id: str, tool: str, args: dict, budget: int) -> dict:
        """One AppleScript tool call: the server function itself, called with the agent's arguments. Its result
        is the same dict the MCP server would send; the host turns it back into the tool's text."""
        fn = self.osascript_tools.get(tool_id[len(OSASCRIPT_PREFIX):])
        if fn is None or tool_id[len(OSASCRIPT_PREFIX):] != tool:
            return {"ok": False, "error": f"unknown tool {tool_id!r}", "cost": 0, "refused": True}
        n = native_tools.cost(tool, args)
        if n > budget:
            return {"ok": False, "refused": True, "cost": n,
                    "error": f"this call needs {n} step(s) but {budget} are left; nothing was run"}
        with self.lock:
            try:
                coro = fn(**args)
            except TypeError as exc:                 # wrong argument names: nothing ran
                return {"ok": False, "cost": n, "error": f"TypeError: {exc}"}
            fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
            try:
                result = fut.result(timeout=CALL_TIMEOUT)
            except Exception as exc:  # noqa: BLE001
                fut.cancel()
                return {"ok": False, "cost": n, "error": f"{type(exc).__name__}: {exc}"}
            return {"ok": True, "cost": n, "result": jsonable(result)}


def cocoa_pump() -> Callable[[float], None] | None:
    """A function that turns this thread's Cocoa run loop for a moment, or None where it cannot.

    macOS refreshes NSWorkspace's list of running applications only while a run loop turns on the main thread.
    A process that never turns one (this server's main thread runs asyncio) keeps the list it saw at start-up, so
    Otto's app lookup says an app launched later "is not running". Found on a real VM: a fresh Python process saw
    VS Code, this server did not."""
    try:
        from Foundation import NSDate, NSDefaultRunLoopMode, NSRunLoop
    except Exception:  # noqa: BLE001
        return None
    rl = NSRunLoop.currentRunLoop()

    def pump(seconds: float) -> None:
        rl.runMode_beforeDate_(NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(seconds))
    return pump


def run_loop(loop: asyncio.AbstractEventLoop, pump: Callable[[float], None] | None,
             stop: threading.Event | None = None, slice_s: float = 0.02) -> None:
    """Run the asyncio loop forever, in short slices with a Cocoa run-loop turn between them."""
    if pump is None:
        loop.run_forever()
        return
    while stop is None or not stop.is_set():
        loop.run_until_complete(asyncio.sleep(slice_s))
        pump(slice_s)


_AUTOMATION: dict | None = None
_AUTOMATION_PROBE = 'tell application "System Events" to get name of first process'


def automation(run: Callable[..., Any] = subprocess.run) -> dict:
    """Whether this process may send Apple Events (Automation permission), for the AppleScript tools. Probed
    once with System Events, which every UI script goes through; a definite answer is kept. Other apps have
    their own grant, which ``preflight.py`` checks per task. ``automation`` is None when the probe timed out,
    which usually means a permission dialog is waiting in the VM for a click nobody will make."""
    global _AUTOMATION
    if _AUTOMATION is not None:
        return _AUTOMATION
    try:
        r = run(["osascript", "-e", _AUTOMATION_PROBE], capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        return {"automation": None, "automation_error": "the System Events probe timed out (a dialog may be open)"}
    except OSError as exc:
        _AUTOMATION = {"automation": False, "automation_error": repr(exc)}
        return _AUTOMATION
    if r.returncode == 0:
        _AUTOMATION = {"automation": True}
    else:
        _AUTOMATION = {"automation": False, "automation_error": (r.stderr or r.stdout or "").strip()[-300:]}
    return _AUTOMATION


def health() -> dict:
    out: dict[str, Any] = {"ok": True, "python": sys.version.split()[0]}
    try:
        from ApplicationServices import AXIsProcessTrusted
        out["trusted"] = bool(AXIsProcessTrusted())
    except Exception as exc:  # noqa: BLE001
        out["trusted"], out["trusted_error"] = False, repr(exc)
    try:
        from Quartz import CGPreflightScreenCaptureAccess
        out["screen_capture"] = bool(CGPreflightScreenCaptureAccess())
    except Exception as exc:  # noqa: BLE001
        out["screen_capture"], out["screen_capture_error"] = False, repr(exc)
    try:
        import pyautogui
        out["screen"] = list(pyautogui.size())
    except Exception as exc:  # noqa: BLE001
        out["screen_error"] = repr(exc)
    out.update(automation())
    return out


def make_handler(guest: GuestTools, token: str, health_fn: Callable[[], dict] = health):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            logger.debug(fmt, *a)

        def _send(self, code: int, body: dict):
            data = json.dumps(body, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _route(self, method: str):
            if not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {token}"):
                return self._send(401, {"ok": False, "error": "unauthorized"})
            try:
                if method == "GET" and self.path == "/health":
                    return self._send(200, health_fn())
                if method == "POST" and self.path == "/call":
                    n = int(self.headers.get("Content-Length") or 0)
                    if n > 2_000_000:
                        return self._send(413, {"ok": False, "error": "body too large"})
                    return self._send(200, guest.call(json.loads(self.rfile.read(n) or b"{}")))
                self._send(404, {"ok": False, "error": "not found"})
            except Exception as exc:  # noqa: BLE001
                logger.exception("request failed")
                self._send(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

        def do_GET(self): self._route("GET")
        def do_POST(self): self._route("POST")

    return H


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=18100)
    p.add_argument("--token-file", required=True)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    token = Path(a.token_file).read_text().strip()
    if len(token) < 16:
        p.error("token too short")

    # Otto's toolkit, unmodified (PYTHONPATH has the bundle's src/ and deps/).
    from tools.navigation.computer.macos_tools import MacOSToolkit, _parse_batch_actions

    # Otto's AppleScript tools: the macos-osascript server module, without the MCP SDK. Calls here are handled
    # one at a time and the host already holds its desktop lease for the agent, so the module must not lock.
    os.environ["OTTO_DESKTOP_LEASE_BY_HOST"] = "1"
    from macos_osascript import server as osa
    osascript_tools = {name: getattr(osa, name) for name in osa.VM_TOOLS}

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    def make_toolkit(cfg: dict) -> Any:
        cfg = dict(cfg)
        if "skip_apps" in cfg:
            cfg["skip_apps"] = set(cfg["skip_apps"] or ())
        return MacOSToolkit(**cfg)

    guest = GuestTools(make_toolkit, _parse_batch_actions, loop, osascript_tools)
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(guest, token))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    logger.info("Otto desktop tools on :%d %s", a.port, health())
    run_loop(loop, cocoa_pump())


if __name__ == "__main__":
    main()
