#!/usr/bin/env python
"""Bridge between Otto and MacArena's VM environment.

Runs in MacArena's own Python environment (it imports ``desktop_env``) and owns the one ``DesktopEnv``.
Otto's ``macarena-vm`` MCP calls the ``/agent/*`` endpoints; ``run.py`` calls the ``/ctl/*`` endpoints.
Nothing the agent can reach does more than send one validated GUI action or ask for a screenshot.

What it enforces, and why:
  * Every action goes through ``env.step`` and counts as one step, whatever the model did in its turn.
    MacArena counts one ``predict()`` call as a step, and a baseline may return several actions in it, so this
    cap is the stricter of the two. The cap is on actions, so a score here is not inflated by the cap.
  * Screenshot / accessibility-tree reads do not count as steps (MacArena gives the agent a free observation
    every step) but are capped at ``--max-obs`` so a model cannot loop on them.
  * After the step budget is spent, or after done / fail, further actions are refused and not executed.
  * Two tokens: the agent's cannot reach ``/ctl/*`` (reset, evaluate, recording).
  * Per-run output in MacArena's own layout (``traj.jsonl``, ``step_N_*.png``, ``result.txt``, ``recording.mp4``).

    python scripts/macarena/bridge.py --macarena-dir ../MacArena --vm osworld --port 18095 \
        --agent-token $A --control-token $C
    python scripts/macarena/bridge.py --fake-env ...      # no VM: blank screen, score 0. Wiring smoke test only.

Otto-native track (``--native``): instead of mouse and keyboard actions, the agent uses Otto's own macOS desktop
tools. After every reset the bridge uploads Otto's toolkit and ``guest/otto_tools_server.py`` into the VM
(``--native-bundle``) and starts it through MacArena's in-guest server; ``/agent/native`` runs one tool call there.
Reads (``native_tools.OBSERVATION``) are free and capped like screenshots; every acting call is one step and
``batch_actions`` one per action inside it. ``/agent/act`` then accepts only done / fail, and ``/agent/observe``
nothing. Steps are recorded with a screenshot in ``traj.jsonl``; every call, reads included, in ``native_calls.jsonl``.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import hmac
import json
import logging
import os
import re
import shlex
import secrets
import struct
import sys
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import actions  # noqa: E402
import native_tools  # noqa: E402
import preflight  # noqa: E402

logger = logging.getLogger("macarena.bridge")

# Same as MacArena's runners/lib_run_single.py, used between run 1 and run 2 when not resetting.
CLOSE_ALL_APPS_COMMAND = """
osascript -e '
tell application "System Events"
    set quitapps to name of every application process whose visible is true and name is not "Finder"
    repeat with closeall in quitapps
        try
            do shell script "killall -9 " & quoted form of closeall
        end try
    end repeat
end tell
'
"""


END_TOOLS = frozenset({"vm_done", "vm_fail"})   # how the agent ends a run (``/agent/act`` done / fail)
# Host-side tools that still cost a step (they never reach /agent/native): procedural memory lookups. Counting them keeps
# a memory-on run on the same budget as a memory-off one.
MEMORY_TOOLS = frozenset({"recall_procedure"})
TURN_BATCH_MAX = 100         # turns mode: actions one batch_actions call may hold (they all belong to one turn)
UNLIMITED = 10 ** 9           # stands in for "no step limit" in comparisons (max_steps 0)
UNLIMITED_READS = 600         # reads stay capped when steps are not, so a loop of free reads still ends
# After this many acting calls fail in a row, the next failure's result says so (runs showed up to 6 failed
# AppleScript UI-scripting variations in a row, each with different arguments, which an identical-call loop guard
# cannot see). 0 turns it off.
FAIL_NOTICE_AFTER = 4
# After this many identical acting calls in a row (same tool, same arguments), the result says so. A failure streak
# misses these: one run clicked the same point about 25 times and each click "succeeded". Moving keys and scrolls
# repeat on purpose and are not counted. 0 turns it off.
REPEAT_NOTICE_AFTER = 3
REPEATABLE_KEYS = frozenset({"up", "down", "left", "right", "tab", "shift+tab", "return", "enter", "pageup",
                             "pagedown", "space", "delete", "backspace", "esc", "escape"})

# ── Starting state (Otto-native track) ──────────────────────────────────────
# Screenshot agents get the screen as their first observation; Otto's desktop agent starts from its tools alone,
# and runs showed it guessing an app (Safari, Numbers) while the task's page or file was already open in another
# (Chrome, LibreOffice), which the checker reads. So the bridge reads what is on screen after setup and the runner
# puts it in the task message: the same facts a first screenshot shows, as text. None of these reads needs a
# permission the VM's harness server lacks: ``lsappinfo`` lists the visible apps front to back, the window list
# gives on-screen window titles, and Chrome's own debugging port (the harness starts Chrome with it) lists its tabs.
START_STATE_SCRIPT = r"""
echo '##APPS'; lsappinfo visibleProcessList 2>/dev/null
echo '##WINDOWS'; osascript -l JavaScript -e 'ObjC.import("CoreGraphics"); var a = ObjC.deepUnwrap(ObjC.castRefToObject($.CGWindowListCopyWindowInfo($.kCGWindowListOptionOnScreenOnly | $.kCGWindowListExcludeDesktopElements, 0))); JSON.stringify(a.filter(function (w) { return w.kCGWindowLayer == 0; }).map(function (w) { return [w.kCGWindowOwnerName || "", w.kCGWindowName || ""]; }))' 2>/dev/null
echo '##TABS'; curl -s -m 3 http://localhost:1337/json/list 2>/dev/null
echo '##END'
"""
START_STATE_MAX_TABS = 8
# Update nags that open with the apps a task starts and are not part of it. Chrome's "Can't update Chrome" window
# sat in front at the start of 12 tasks; agents spent a step on it or typed into it. Closed through the VM tool
# server (it has the Accessibility and Automation grants), before the starting state is read; a window that is
# not there is no error. Native track only: the standard track's agent sees the screen as the paper's agents do.
DISMISS_NAGS_SCRIPT = """
tell application "System Events"
    if exists process "Google Chrome" then
        tell process "Google Chrome"
            repeat with w in (every window whose name is "Can't update Chrome")
                try
                    click button "Close" of w
                end try
            end repeat
        end tell
    end if
end tell
"""
# Task setup (``pre_command``). MacArena starts it with /setup/launch and does not wait or look at the result, and
# the bridge then waited a fixed ``settle``. Two ways that broke runs (2026-11-24 macOSWorld runs):
#   * 29 file_management tasks start `osascript ... tell application "Finder" to close windows && rm -rf ... && cp
#     -r ... && open <folder>`. In 29 of 29 runs the folder never opened: the first step failed, so the copy never
#     ran and the task's files did not exist. Nothing said so.
#   * 4e6b97d1's setup sleeps 15 s while settle is 10 s, so the agent started while the spreadsheet was being
#     filled.
# "sync" (default) runs pre_command through /run_bash_script and waits for it, after Finder answers AppleScript,
# keeps its output and exit code in setup.json, checks the folders it opens exist, and runs it once more when that
# is safe (``setup_rerunnable``). "launch" keeps MacArena's way. The mode is recorded with every result.
SETUP_MODES = ("sync", "launch")
SETUP_TIMEOUT = 180                 # the longest pre_command sleeps 15 s; app launches add some
FINDER_WAIT = 45                    # seconds to wait for Finder to answer before running the setup anyway
FINDER_READY_SCRIPT = "osascript -e 'tell application \"Finder\" to count windows' 2>&1"
_OPEN_TARGET = re.compile(r"""(?:^|[;&|]\s*)open\s+(?!-)("[^"]+"|'[^']+'|/[^\s;&|]+)""")
_OSASCRIPT_E = re.compile(r"""osascript\s+-e\s+'([^']*)'""")


def setup_targets(command: str) -> list[str]:
    """Absolute paths the setup opens with plain ``open <path>`` (not ``open -a App``): what must exist after it."""
    out = []
    for m in _OPEN_TARGET.finditer(command or ""):
        p = m.group(1).strip("\"'")
        if p.startswith("/") and p not in out:
            out.append(p)
    return out


def setup_rerunnable(command: str) -> bool:
    """True when running the setup twice gives the same state as once: its AppleScript only closes windows or quits
    apps. A setup that makes documents or sets values (Keynote, Numbers) would make a second copy, so it is not."""
    scripts = _OSASCRIPT_E.findall(command or "")
    return all(re.search(r"\b(close|quit)\b", s) and not re.search(r"\b(make|set|keystroke|click)\b", s)
               for s in scripts)


def _rc(r: dict) -> Any:
    """A script result's exit code; MacArena's FakeEnv-style answers give only ``status``."""
    rc = r.get("returncode")
    return 0 if rc is None and r.get("status") == "success" else rc


def _tail(text: Any, n: int = 2000) -> str:
    s = str(text or "")
    return s if len(s) <= n else "…" + s[-n:]


# System notices that are not part of a task (the same set as macos_ax_struct.NOTICE_OWNERS; the bridge does not
# import the toolkit). The "Accessibility Access" warning only offers buttons that change privacy settings.
NOTICE_OWNERS = frozenset({"universalAccessAuthWarn"})


def _sections(output: str) -> dict[str, str]:
    out: dict[str, str] = {}
    cur = None
    for line in (output or "").splitlines():
        if line.startswith("##") and line[2:].strip().isupper():
            cur = line[2:].strip()
            out[cur] = ""
        elif cur:
            out[cur] += line + "\n"
    return out


def _clip(text: str, n: int = 80) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def parse_start_state(output: str) -> str:
    """The text block the task message gets, from ``START_STATE_SCRIPT``'s output; "" when nothing could be read.
    Facts only: which apps are open front to back, their window titles, and Chrome's tabs."""
    import re
    sec = _sections(output)
    apps = re.findall(r'"([^"]+)"', sec.get("APPS", ""))
    windows: list[tuple[str, str]] = []
    try:
        windows = [(str(o), str(t)) for o, t in json.loads(sec.get("WINDOWS", "").strip() or "[]")]
    except (ValueError, TypeError):
        windows = []
    tabs: list[dict] = []
    try:
        tabs = [t for t in json.loads(sec.get("TABS", "").strip() or "[]")
                if isinstance(t, dict) and t.get("type") == "page" and not str(t.get("url", "")).startswith("devtools:")]
    except (ValueError, TypeError):
        tabs = []
    if not apps and not windows and not tabs:
        return ""
    # lsappinfo writes spaces in app names as underscores ("Google_Chrome"); use the window list's spelling.
    owners = {o.replace(" ", "_"): o for o, _ in windows}
    apps = [owners.get(a, a.replace("_", " ") if a.replace("_", " ") in owners.values() else a) for a in apps]
    titles ={o: [t for oo, t in windows if oo == o and t] for o in {o for o, _ in windows}}
    lines = []
    for i, app in enumerate(apps):
        wins = titles.get(app) or []
        shown = "; windows: " + ", ".join(f'"{_clip(t)}"' for t in wins[:3]) if wins else ""
        lines.append(f"- {'Frontmost app' if i == 0 else 'Also open'}: {app}{shown}")
    others = [(o, t) for o, t in windows if o not in apps and t and o not in NOTICE_OWNERS]
    notices = [(o, t) for o, t in windows if o not in apps and t and o in NOTICE_OWNERS]
    if others:
        lines.append("- Other windows on screen: " + ", ".join(f'"{_clip(t)}" ({o})' for o, t in others[:3]))
    if notices:
        lines.append("- Also on screen: " + ", ".join(f'"{_clip(t)}" ({o})' for o, t in notices[:2])
                     + ", a macOS permission notice that is not part of the task: leave it, do not activate or "
                       "close it")
    if tabs:
        chrome_titles = set(titles.get("Google Chrome") or [])
        lines.append("- Google Chrome tabs:")
        for t in tabs[:START_STATE_MAX_TABS]:
            mark = " (showing)" if t.get("title") in chrome_titles else ""
            lines.append(f'  - "{_clip(t.get("title") or "")}" {t.get("url", "")}{mark}')
    return "What is on the test Mac's screen as the task starts:\n" + "\n".join(lines)


# ── Script calls the native track refuses ───────────────────────────────────
# The VM's own ports serve the benchmark harness (5000) and Chrome's debugger (1337, and 9222 forwarded to it):
# reading them is reading the harness, not doing the task. One run fetched the harness's page in a loop.
HARNESS_PORT_RE = r"(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])\s*:\s*(?:5000|1337|9222)\b"
# Chrome tasks are judged from Chrome's own state (its open tab, settings, history), so a page fetched with curl
# or wget cannot be the result; runs spent their budget that way. Only those tasks: elsewhere a download by curl
# can be the right way to get a file.
WEB_FETCH_RE = r"(?<![\w./-])(?:curl|wget)(?![\w-])"
BROWSER_JUDGED = frozenset({("osworld", "chrome")})


# Folders in the VM that belong to the harness, not to any task: MacArena's server and its files, the pristine
# copy every file task's setup restores from, and Otto's tool server. No instruction names them (checked over all
# 163 task files); only setup and grading commands do. Runs on 2026-11-24 read the server's code for ~30 steps and
# deleted a file from the backup copy when a setup had failed. A search that does not name them (``find ~``) can still
# list them; this stops the agent working in them, not every way of seeing them.
HARNESS_PATH_RE = r"(?:~|/Users/[^/\s'\"]+)/(?:vm_files|Benchmark_Backup)\b|/tmp/otto-bench\b"


def harness_path_refusal(args: dict) -> str | None:
    """The refusal text when any argument of a native call names a harness folder (``HARNESS_PATH_RE``)."""
    import re
    text = json.dumps(args or {}, ensure_ascii=False)
    if re.search(HARNESS_PATH_RE, text):
        return ("Refused: that folder belongs to the benchmark's harness on this Mac (its server, or the backup "
                "copy the task's setup restores from), not to the task. Work with the files and windows the task "
                "gave you. Nothing was run.")
    return None


def script_refusal(tool: str, args: dict, kind: tuple[str, str] | None) -> str | None:
    """The refusal text for a native call the benchmark does not run, else None."""
    import re
    harness = harness_path_refusal(args)
    if harness:
        return harness
    if tool != "run_osascript":
        return None
    # A shell command given as ``shell_command`` runs like one inside the script: check both.
    script = "\n".join(str((args or {}).get(k) or "") for k in ("script", "shell_command"))
    if re.search(HARNESS_PORT_RE, script, re.I):
        return ("Refused: that address is the benchmark's own harness on this Mac (its control server or Chrome's "
                "debugger), not part of the task. Nothing was run.")
    if kind in BROWSER_JUDGED and re.search(WEB_FETCH_RE, script):
        return ("Refused: this task is judged from the browser's own state (the tab it shows, its settings or "
                "history), so a page fetched with curl or wget cannot count. Do it in the browser window. Nothing "
                "was run.")
    return None


LOCAL_NETWORK_BLOCKED = """\
This process cannot reach the benchmark VM at {url}: {error}

That is macOS Local Network privacy blocking this Python ({python}), not a VM problem. It happens when the
bench is started from a terminal whose app grants Local Network access to its own Apple-signed tools but not to
ad-hoc-signed interpreters such as uv's or pyenv's Python (seen in the Claude desktop app's terminal: curl
reaches the VM, these Pythons get Errno 65). Start the bench from Terminal.app or iTerm instead, and allow the
Local Network prompt the first time; or start it from a session whose app already has that access."""


def local_network_error(url: str, attempts: int = 3, wait: float = 5.0, get=None) -> str | None:
    """None if ``url`` answers, else the error text when it is macOS's Local Network block (Errno 65,
    "No route to host"). Other failures return None: the VM may still be starting, and MacArena retries."""
    import requests
    get = get or (lambda u: requests.get(u, timeout=5))
    last = None
    for i in range(attempts):
        try:
            get(url)
            return None
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
            if "Errno 65" not in last and "No route to host" not in last:
                return None
        if i + 1 < attempts:
            time.sleep(wait)
    return last


def png_size(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    return struct.unpack(">II", data[16:24])


def blank_png(w: int = 1280, h: int = 720) -> bytes:
    def chunk(tag: bytes, body: bytes) -> bytes:
        c = struct.pack(">I", len(body)) + tag + body
        return c + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\xff\xff\xff" * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


class FakeEnv:
    """A stand-in for DesktopEnv: blank screen, records actions, scores 0. For wiring tests only."""

    class _Ctl:
        def __init__(self, env): self.env = env
        def get_screenshot(self): return self.env.png
        def get_accessibility_tree(self): return "<fake accessibility tree/>"
        def get_vm_screen_size(self): return {"width": self.env.size[0], "height": self.env.size[1]}
        def start_recording(self): pass
        def end_recording(self, dest): Path(dest).write_bytes(b"")
        def run_bash_script(self, script, timeout=30, working_dir=None): return {"status": "success", "output": ""}

    def __init__(self, size=(1280, 720)):
        self.size, self.png = size, blank_png(*size)
        self.controller = self._Ctl(self)
        self.actions: list[str] = []
        self.resets = 0

    def reset(self, task_config=None, **_):
        self.resets += 1
        self.actions.clear()
        return self._get_obs()

    def _get_obs(self):
        return {"screenshot": self.png, "accessibility_tree": "<fake/>", "terminal": None, "instruction": None}

    def step(self, action, pause=2):
        self.actions.append(action)
        return self._get_obs(), 0, action in ("DONE", "FAIL"), ({"done": True} if action == "DONE" else {})

    def evaluate(self):
        return 0.0


class GuestTools:
    """Deploys and calls Otto's desktop tool server inside the VM (Otto-native track).

    Uses only MacArena's own in-guest endpoints (/setup/execute, /setup/upload, /setup/launch), so the server
    starts in the same session, with the same permissions, as MacArena's own actions."""

    GUEST_DIR = "/tmp/otto-bench"

    def __init__(self, env: Any, bundle: Path, python: str, port: int = 18100, timeout: float = 300.0):
        self.env, self.bundle, self.python, self.port, self.timeout = env, Path(bundle), python, port, timeout
        self.url: str | None = None
        self.token: str | None = None

    def _ctl(self) -> str:
        return self.env.controller.http_server.rstrip("/")

    def _execute(self, cmd: list[str]) -> dict:
        import requests
        r = requests.post(self._ctl() + "/setup/execute", json={"command": cmd, "shell": False}, timeout=180)
        r.raise_for_status()
        out = r.json()
        if out.get("returncode") not in (0, None):
            raise RuntimeError(f"{cmd[0]} failed in the VM: {out.get('error') or out.get('output')}")
        return out

    def _upload(self, data: bytes, dest: str) -> None:
        import requests
        r = requests.post(self._ctl() + "/setup/upload", data={"file_path": dest},
                          files={"file_data": ("f", data)}, timeout=300)
        r.raise_for_status()

    def deploy(self) -> dict:
        from urllib.parse import urlparse
        d = self.GUEST_DIR
        self.token = secrets.token_hex(16)
        # [s]erver: the pattern must not match the shell running pkill.
        self._execute(["/bin/sh", "-c", f"pkill -f 'otto_tools_serve[r].py' ; rm -rf {d} ; mkdir -p {d}"])
        self._upload(self.bundle.read_bytes(), f"{d}/bundle.tgz")
        self._upload(self.token.encode(), f"{d}/token")
        self._execute(["/bin/sh", "-c", f"chmod 600 {d}/token && tar -xzf {d}/bundle.tgz -C {d}"])
        self._replace_clicker()
        self.url = f"http://{urlparse(self._ctl()).hostname}:{self.port}"
        # Seen on a real VM: right after a boot the same server, image and launch path reported no Accessibility
        # permission once and had it on another boot. So wait for it, and restart the server once, before calling
        # it an infrastructure failure.
        health: dict = {}
        for attempt in (1, 2):
            if attempt == 2:
                self._execute(["/bin/sh", "-c", "pkill -f 'otto_tools_serve[r].py' ; sleep 1"])
            self._launch()
            health = self._wait(trusted_within=45.0)
            if health.get("trusted"):
                health["attempts"] = attempt
                if health.get("automation") is not True:
                    # Not fatal: the desktop tools still work, and the agent is told to use them on -1743.
                    logger.warning("VM tool server cannot send Apple Events (Automation), so the AppleScript "
                                   "tools will fail: %s. %s", health.get("automation_error"), preflight.FIX_HINT)
                return health
            logger.warning("VM tool server without Accessibility permission (attempt %d): %s", attempt, health)
        raise RuntimeError(f"the VM tool server has no Accessibility permission: {health}")

    def _replace_clicker(self) -> None:
        """Swap MacArena's permission clicker for one that leaves app dialogs alone.

        MacArena starts ``allow_permission.py`` (five copies) with every fresh VM; it clicks any button named OK,
        Allow or Always Allow in any process, so app dialogs with an OK button (Format Cells, Rename Sheet...)
        were confirmed with their defaults within about 0.25 s of opening. ``guest/permission_clicker.py`` clicks
        the same buttons only in background agents, where the system's permission prompts live."""
        import requests
        d = self.GUEST_DIR
        # [n]/[r]/[d]: the patterns must not match the shell running pkill. The loop's own osascript child outlives
        # it (seen live: one still clicking OK half a minute later), so it is matched by its script's text.
        self._execute(["/bin/sh", "-c", "pkill -f 'allow_permissio[n].py' ; pkill -f 'permission_clicke[r].py' ; "
                       "pkill -f 'No permission dialog foun[d]' ; true"])
        requests.post(self._ctl() + "/setup/launch", json={"shell": True, "command": (
            f"cd {d} && nohup {self.python} guest/permission_clicker.py >> {d}/clicker.log 2>&1 &")},
            timeout=60).raise_for_status()

    def _launch(self) -> None:
        import requests
        d = self.GUEST_DIR
        requests.post(self._ctl() + "/setup/launch", json={"shell": True, "command": (
            f"cd {d} && OTTO_DESKTOP_LEASE_BY_HOST=1 PYTHONPATH={d}/deps:{d}/src nohup {self.python} guest/otto_tools_server.py "
            f"--port {self.port} --token-file {d}/token >> {d}/server.log 2>&1 &")}, timeout=60).raise_for_status()

    def _wait(self, trusted_within: float) -> dict:
        """Health once the server answers; keeps asking while it reports no Accessibility permission."""
        import requests
        last, health, t_up = "no answer", {}, None
        deadline = time.time() + 60
        while time.time() < deadline or (t_up is not None and time.time() < t_up + trusted_within):
            try:
                r = requests.get(self.url + "/health", headers=self._auth(), timeout=10)
                if r.status_code == 200:
                    health = r.json()
                    t_up = t_up or time.time()
                    if health.get("trusted"):
                        return health
                    if time.time() > t_up + trusted_within:
                        return health
                else:
                    last = f"HTTP {r.status_code}"
            except requests.RequestException as exc:
                last = repr(exc)
            time.sleep(1)
        if health:
            return health
        raise RuntimeError(f"the VM tool server did not start ({last}); see {self.GUEST_DIR}/server.log in the VM")

    def _auth(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}

    def call(self, body: dict) -> dict:
        import requests
        if not self.url:
            raise RuntimeError("the VM tool server is not deployed")
        r = requests.post(self.url + "/call", json=body, headers=self._auth(), timeout=self.timeout)
        r.raise_for_status()
        return r.json()


class FakeGuest:
    """Stand-in for the VM tool server (``--fake-env --native``): canned text, Otto's step costs."""

    def __init__(self):
        self.calls: list[dict] = []
        self.setup_calls: list[dict] = []     # the harness's own calls at reset (closing nags), not the agent's
        self.deploys = 0

    def deploy(self) -> dict:
        self.deploys += 1
        return {"ok": True, "trusted": True, "fake": True}

    def call(self, body: dict) -> dict:
        if (body.get("args") or {}).get("script") == DISMISS_NAGS_SCRIPT:
            self.setup_calls.append(body)
            return {"ok": True, "cost": 0, "result": ""}
        self.calls.append(body)
        tool = body.get("tool", "")
        n = native_tools.cost(tool, body.get("args") or {},
                              lambda raw: [(ln.split("(")[0].strip(), {}) for ln in raw.splitlines() if ln.strip()])
        if n > int(body.get("budget", 0)):
            return {"ok": False, "refused": True, "cost": n, "error": f"needs {n} steps; nothing was run"}
        out = {"ok": True, "cost": n, "result": f"(fake VM) {tool} ok"}
        if body.get("observe_after") and tool in native_tools.OBSERVE_AFTER:     # as the real guest does
            out["result"] += "\n\nControls now (Fake), read after the action:\n[1] B'OK'"
            out["observed"] = True
        return out


class Bridge:
    """The logic, separate from HTTP so tests can drive it with FakeEnv."""

    def __init__(self, env: Any, *, max_steps: int = 15, max_obs: int | None = None, use_a11y: bool = False,
                 pause: float = 3.0, settle: float = 10.0, coord_scale: tuple[float, float] | None = None,
                 vm_name: str = "", guest: Any = None, record: bool = False, observe_after: bool = True,
                 fail_notice_after: int = FAIL_NOTICE_AFTER, step_unit: str = "actions",
                 repeat_notice_after: int = REPEAT_NOTICE_AFTER, dismiss_nags: bool = True,
                 setup_mode: str = "sync"):
        self.env, self.max_steps, self.vm_name = env, max(max_steps, 0), vm_name
        if step_unit not in ("actions", "turns"):
            raise ValueError("step_unit must be 'actions' or 'turns'")
        # "actions": every acting call is one step (the stricter rule). "turns": one step per agent turn, i.e. per
        # model response that calls any tool, however many calls it makes (the paper's cycle: one observation, one
        # model call). Only the Otto-native track can tell turns apart; the standard track has one action per turn.
        self.step_unit = step_unit if guest is not None else "actions"
        # max_steps 0 = no step limit: the run ends on vm_done / vm_fail or the runner's time limit. ``cap`` is
        # the number every comparison uses, so one place decides what "no limit" means.
        self.cap = self.max_steps if self.max_steps > 0 else UNLIMITED
        self.record = record                        # save per-step PNGs and recording.mp4 (off by default)
        self.guest = guest                          # set: Otto-native track
        self.track = "native" if guest is not None else "standard"
        # Otto-native: after an acting call, return the frontmost app's controls with the result (one free read),
        # so the agent does not spend a turn on get_screen_controls. Recorded with every run (``state``).
        self.observe_after = bool(observe_after) and self.track == "native"
        self.fail_notice_after = max(0, int(fail_notice_after))     # 0 = no notice
        self.repeat_notice_after = max(0, int(repeat_notice_after))  # 0 = no notice
        self.dismiss_nags = bool(dismiss_nags) and self.track == "native"
        if setup_mode not in SETUP_MODES:
            raise ValueError(f"setup_mode must be one of {SETUP_MODES}")
        self.setup_mode = setup_mode
        self.setup: dict = {}                   # what the task's pre_command did this run (``_run_setup``)
        self.max_obs = max_obs if max_obs is not None else (3 * max_steps if max_steps > 0 else UNLIMITED_READS)
        self.use_a11y, self.pause, self.settle = use_a11y, pause, settle
        self.coord_scale_override = coord_scale
        self.lock = threading.RLock()
        self.task: dict | None = None
        self.kind: tuple[str, str] | None = None     # (source, domain) of the task, from its path
        self.start_state = ""
        self.run_dir: Path | None = None
        self.armed = False
        self._clear()

    def _clear(self):
        self.steps = self.obs_calls = self.invalid = 0
        self.finished: str | None = None        # None | done | fail | budget
        self.size = (0, 0)
        self.scale = (1.0, 1.0)
        self.last_error: str | None = None
        self.eval_error: str | None = None
        self.native_calls = 0
        self.guest_health: dict | None = None
        self.cur_turn: str | None = None         # turns mode: id of the agent turn that last used a step
        self.model_calls = 0                     # model responses, reported by the backend (/agent/tick)
        self.tool_messages = 0                   # tool calls the model made (each is one tool-result message)
        self.fail_streak: list[str] = []         # tools of the acting calls that failed in a row
        self.fail_notices = 0
        self.repeat_key: str | None = None       # the last acting call (tool and arguments), for repeat notices
        self.repeat_count = 0
        self.repeat_notices = 0
        self.memory_calls = 0                    # recall_procedure calls (each a step)
        self.memory_turns: set[str] = set()      # turns that included one
        self.memory_used: list[dict] = []        # every item returned, in order
        self.memory_abstained = 0                # calls that returned nothing (below the floor)
        self.memory_tokens = 0

    # ── control side ────────────────────────────────────────────────────────

    def reset(self, task: dict, run_dir: Path, *, fresh: bool = True,
              kind: tuple[str, str] | None = None) -> dict:
        """Prepare one run. ``fresh`` resets the VM (copy-on-use); otherwise only quit visible apps, which is
        what MacArena's runner does before run 2 (state from run 1, such as files, then carries over).
        ``kind`` is the task's (source, domain). On the native track the answer has ``start_state``, the screen
        after setup as text (``parse_start_state``)."""
        with self.lock:
            self.armed = False
            self._clear()
            self.task, self.run_dir, self.kind, self.start_state = task, Path(run_dir), kind, ""
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.setup = {"mode": self.setup_mode, "status": "none"}
            if fresh:
                pre = task.get("pre_command") if self.setup_mode == "sync" else None
                if isinstance(pre, str) and pre.strip():
                    # MacArena would only start it; run it here, after its other setup, and wait for it.
                    self.env.reset(task_config={k: v for k, v in task.items() if k != "pre_command"})
                    self.setup = self._run_setup(pre)
                    self.env.is_environment_used = True     # so the next reset reverts to a clean VM copy
                else:
                    self.env.reset(task_config=task)
                    if task.get("pre_command"):
                        self.setup = {"mode": self.setup_mode, "status": "launched"}
            else:
                self.env.controller.run_bash_script(CLOSE_ALL_APPS_COMMAND)
            time.sleep(self.settle)                 # MacArena waits 10 s for the environment to be ready
            shot = self.env.controller.get_screenshot()
            if not shot:
                raise RuntimeError("the VM returned no screenshot after reset")
            self.size = png_size(shot)
            self.scale = self.coord_scale_override or self._scale_from_vm()
            if self.guest is not None:
                self.guest_health = self.guest.deploy()     # a fresh toolkit for every run, also without a reset
                if self.dismiss_nags:
                    self._dismiss_nags()
                self.start_state = self._read_start_state()
            if self.record:
                self.env.controller.start_recording()
            self.armed = True
            logger.info("armed %s (screenshot %s, scale %s)", task.get("id"), self.size, self.scale)
            return {**self.state(), "start_state": self.start_state}

    def _run_setup(self, command: str) -> dict:
        """Run the task's pre_command and wait for it; record what happened (also in ``setup.json``).

        status "ok": exit code 0 and every folder it opens exists. "failed": otherwise, after one more try when
        ``setup_rerunnable``. A failed setup is recorded, not hidden: the run goes on as MacArena's would, and the
        report shows the score with and without these runs."""
        rec: dict = {"mode": "sync", "status": "failed", "attempts": [], "targets": setup_targets(command),
                     "rerunnable": setup_rerunnable(command)}
        tries = 2 if rec["rerunnable"] else 1
        for n in range(1, tries + 1):
            rec["finder_wait_s"] = self._wait_for_finder()
            t0 = time.monotonic()
            r = self._run_script_once(command, SETUP_TIMEOUT)
            missing = self._missing_paths(rec["targets"])
            attempt = {"n": n, "returncode": _rc(r), "seconds": round(time.monotonic() - t0, 1),
                       "output": _tail(r.get("output")), "error": _tail(r.get("error")), "missing": missing}
            rec["attempts"].append(attempt)
            if attempt["returncode"] == 0 and not missing:
                rec["status"] = "ok"
                break
            logger.warning("setup attempt %d failed (exit %s, missing %s): %s", n, attempt["returncode"], missing,
                           _tail(f"{r.get('output') or ''}{r.get('error') or ''}", 300))
            if n < tries:
                time.sleep(5)
        last = rec["attempts"][-1]
        if rec["status"] != "ok":
            rec["reason"] = (f"missing {', '.join(last['missing'])}" if last["missing"] else
                             f"exit code {last['returncode']}")
        if self.run_dir is not None:
            (self.run_dir / "setup.json").write_text(json.dumps({"command": command, **rec}, indent=1),
                                                     encoding="utf-8")
        return {k: rec[k] for k in ("mode", "status", "rerunnable", "finder_wait_s")} | {
            "attempts": len(rec["attempts"]), "reason": rec.get("reason"), "returncode": last["returncode"],
            "missing": last["missing"]}

    def _wait_for_finder(self) -> float:
        """Wait until Finder answers AppleScript (a fresh VM copy may not be ready), up to ``FINDER_WAIT`` s.
        Returns the seconds waited; on giving up the setup runs anyway and its own result says what happened."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < FINDER_WAIT:
            if _rc(self._run_script_once(FINDER_READY_SCRIPT, 10)) == 0:
                break
            time.sleep(2)
        return round(time.monotonic() - t0, 1)

    def _missing_paths(self, paths: list[str]) -> list[str]:
        if not paths:
            return []
        script = "; ".join(f"[ -e {shlex.quote(p)} ] || echo {shlex.quote(p)}" for p in paths)
        r = self._run_script_once(script, 15)
        if _rc(r) != 0:
            return list(paths)                  # could not check: count them as missing rather than guess
        return [ln.strip() for ln in str(r.get("output") or "").splitlines() if ln.strip()]

    def _dismiss_nags(self) -> None:
        """Close update nags (``DISMISS_NAGS_SCRIPT``) in the VM. Best effort: a failure only leaves them open."""
        try:
            self.guest.call({"tool_id": "osascript:run_osascript", "tool": "run_osascript",
                             "args": {"script": DISMISS_NAGS_SCRIPT, "timeout_seconds": 10}, "budget": 1})
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not close update nags: %s", exc)

    def _read_start_state(self) -> str:
        """The screen after setup as text, read once; "" when the VM could not say (the run goes on without it)."""
        try:
            r = self._run_script_once(START_STATE_SCRIPT, 20)
            text = parse_start_state(str(r.get("output") or ""))
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read the starting state: %s", exc)
            return ""
        if self.run_dir is not None and text:
            (self.run_dir / "start_state.txt").write_text(text + "\n", encoding="utf-8")
        return text

    def _scale_from_vm(self) -> tuple[float, float]:
        """Screenshot pixels -> pyautogui coordinates. 1.0 when they match; UNVERIFIED on a real macOS VM
        (a Retina screenshot can be larger than the logical screen). Override with --coord-scale."""
        try:
            vs = self.env.controller.get_vm_screen_size()
            if isinstance(vs, dict):
                vw, vh = vs.get("width"), vs.get("height")
            else:
                vw, vh = vs
            if vw and vh and self.size[0] and self.size[1]:
                return (vw / self.size[0], vh / self.size[1])
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read the VM screen size (%s); assuming screenshot == screen", exc)
        return (1.0, 1.0)

    def _run_script_once(self, script: str, timeout: int) -> dict:
        """One shell script in the VM, once. MacArena's own ``run_bash_script`` retries a timed-out script three
        times, which turns one hung permission dialog into minutes; a probe wants a single short answer."""
        ctl = self.env.controller
        base = getattr(ctl, "http_server", None)
        if not isinstance(base, str):
            return ctl.run_bash_script(script, timeout=timeout) or {}
        import requests
        for attempt in range(6):                       # the VM's server may still be starting: wait for it
            try:
                started = time.monotonic()
                r = requests.post(base.rstrip("/") + "/run_bash_script", json={"script": script, "timeout": timeout},
                                  timeout=timeout + 15)
                if r.status_code == 200:
                    return r.json()
                # The in-VM server answers HTTP 500 when the script outlives its timeout (a dialog nobody answers).
                hung = time.monotonic() - started >= timeout - 1
                return {"returncode": -1, "output": "Script execution timed out" if hung else f"HTTP {r.status_code}"}
            except requests.exceptions.Timeout:
                return {"returncode": -1, "output": "Script execution timed out"}
            except requests.exceptions.RequestException as exc:
                if attempt == 5:
                    return {"returncode": -1, "output": f"VM unreachable: {type(exc).__name__}"}
                time.sleep(5)
        return {"returncode": -1, "output": "VM unreachable"}

    def preflight(self, apps: list[str], timeout: int = 20) -> dict:
        """Ask each app for its name the way a task's setup and grading scripts do, from the same in-VM server, so
        a missing Automation permission (-1743) or a missing app shows up now and not as a silent 0.0 later.
        Starts the apps, so the next reset must begin from a clean VM copy."""
        with self.lock:
            if self.armed:
                raise RuntimeError("a run is armed; check permissions before the first task")
            results: dict[str, dict] = {}
            for app in sorted(set(apps)):
                try:
                    r = self._run_script_once(preflight.probe_command(app), timeout)
                except Exception as exc:  # noqa: BLE001
                    r = {"returncode": -1, "output": f"{type(exc).__name__}: {exc}"}
                results[app] = preflight.classify(r.get("returncode"), f"{r.get('output') or ''}{r.get('error') or ''}")
            self.env.is_environment_used = True     # MacArena then reverts to a clean copy at the next reset
            return results

    def evaluate(self) -> dict:
        with self.lock:
            if self.task is None:
                raise RuntimeError("no task is armed")
            try:
                r = self.env.evaluate()
                if isinstance(r, tuple):            # macOSWorld evaluator returns (0, error_json) on failure
                    r, self.eval_error = r[0], str(r[1])
                score = float(r or 0.0)
            except Exception as exc:  # noqa: BLE001
                score, self.eval_error = 0.0, f"{type(exc).__name__}: {exc}"
            return {"score": score, "eval_error": self.eval_error}

    def finish(self, score: float) -> dict:
        with self.lock:
            self.armed = False
            # MacArena marks the VM "used" only when it runs a setup step, env.step() or a postconfig. A task with no
            # setup whose agent works through the native tools (and an infeasible task) sets none of those, so the
            # next reset skipped the revert and started on this task's leftovers (a copied photo.jpg showed up).
            self.env.is_environment_used = True
            if self.run_dir is None:
                return {}
            if self.record:
                try:
                    self.env.controller.end_recording(str(self.run_dir / "recording.mp4"))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("recording not saved: %s", exc)
            (self.run_dir / "result.txt").write_text(f"{score}\n")
            return self.state()

    def state(self) -> dict:
        return {"vm": self.vm_name, "track": self.track, "task_id": (self.task or {}).get("id"), "armed": self.armed,
                "steps": self.steps, "native_calls": self.native_calls, "guest": self.guest_health,
                "max_steps": self.max_steps, "obs_calls": self.obs_calls, "max_obs": self.max_obs,
                "invalid_actions": self.invalid, "finished": self.finished, "screenshot_size": list(self.size),
                "scale": list(self.scale), "eval_error": self.eval_error, "observe_after": self.observe_after,
                "fail_notice_after": self.fail_notice_after, "fail_notices": self.fail_notices,
                "repeat_notice_after": self.repeat_notice_after, "repeat_notices": self.repeat_notices,
                "dismiss_nags": self.dismiss_nags, "setup_mode": self.setup_mode, "setup": dict(self.setup),
                "step_unit": self.step_unit, "model_calls": self.model_calls, "tool_messages": self.tool_messages,
                "dialog_messages": self.model_calls + self.tool_messages,
                "memory_calls": self.memory_calls, "memory_turns": len(self.memory_turns),
                "memory_used": list(self.memory_used), "memory_abstained": self.memory_abstained,
                "memory_tokens": self.memory_tokens}

    def _repeat_notice(self, tool: str, args: dict) -> str:
        """Track identical acting calls in a row; the text to add when the same call keeps being made."""
        if not self.repeat_notice_after:
            return ""
        if tool in ("scroll", "scroll_at") or (tool == "hotkey" and str((args or {}).get("keys", "")).strip().lower()
                                               in REPEATABLE_KEYS):
            self.repeat_key, self.repeat_count = None, 0
            return ""
        key = f"{tool}:{json.dumps(args or {}, sort_keys=True, default=str)}"
        self.repeat_count = self.repeat_count + 1 if key == self.repeat_key else 1
        self.repeat_key = key
        n = self.repeat_count
        if n < self.repeat_notice_after or n % self.repeat_notice_after:
            return ""
        self.repeat_notices += 1
        return (f"[benchmark] This is the same {tool} call {n} times in a row. If the screen did not change after "
                "the last one, repeating it will not help: read the screen and try a different way (another "
                "control, a menu or keyboard shortcut, typing the value, or another tool).")

    def _streak_notice(self, tool: str, failed: bool) -> str:
        """Track acting calls that fail in a row; the text to add to this result when the streak is long."""
        if not failed:
            self.fail_streak = []
            return ""
        self.fail_streak.append(tool)
        n = len(self.fail_streak)
        if not self.fail_notice_after or n < self.fail_notice_after or n % self.fail_notice_after:
            return ""
        self.fail_notices += 1
        kinds = ", ".join(sorted(set(self.fail_streak[-n:])))
        return (f"[benchmark] The last {n} actions failed ({kinds}). Trying more variations of the same kind rarely "
                "works: read the screen again and switch to a different kind of approach (on-screen text with "
                "click_text / type_text, a keyboard shortcut or menu, or another app).")

    # ── agent side ──────────────────────────────────────────────────────────

    def observe(self, kind: str) -> dict:
        with self.lock:
            if not self.armed:
                return {"ok": False, "error": "no task is running"}
            if self.guest is not None:
                return {"ok": False, "error": "not available in the Otto-native track; use the desktop tools"}
            if self.obs_calls >= self.max_obs:
                return {"ok": False, "error": f"observation limit ({self.max_obs}) reached; act or finish"}
            self.obs_calls += 1
            if kind == "tree":
                if not self.use_a11y:
                    return {"ok": False, "error": "the accessibility tree is not available in this run; use the screenshot"}
                tree = self.env.controller.get_accessibility_tree()
                return {"ok": True, "tree": tree or "", **self._budget()}
            if kind != "screenshot":
                return {"ok": False, "error": "kind must be 'screenshot' or 'tree'"}
            shot = self.env.controller.get_screenshot()
            if not shot:
                return {"ok": False, "error": "no screenshot available"}
            return {"ok": True, "screenshot": base64.b64encode(shot).decode(), "size": list(self.size), **self._budget()}

    def _budget(self) -> dict:
        left = None if self.max_steps == 0 else max(self.max_steps - self.steps, 0)      # None: no step limit
        return {"steps_used": self.steps, "steps_left": left, "finished": self.finished}

    def act(self, action: dict) -> dict:
        with self.lock:
            if not self.armed:
                return {"ok": False, "error": "no task is running"}
            if self.finished:
                return {"ok": False, "error": f"the run is over ({self.finished}); nothing was executed", **self._budget()}
            if self.guest is not None and (action or {}).get("type") not in ("done", "fail"):
                return {"ok": False, "error": "only done / fail here in the Otto-native track; nothing was executed",
                        **self._budget()}
            try:
                cmd = actions.to_command(action, self.size, self.scale)
            except actions.ActionError as exc:
                self.invalid += 1
                if self.invalid >= self.cap:
                    self.finished = "budget"
                self.last_error = str(exc)
                return {"ok": False, "error": f"{exc}. Nothing was executed and no step was used.", **self._budget()}
            if self.step_unit == "actions":         # turns mode: done / fail ends the run and uses no step
                self.steps += 1
            try:
                obs, _reward, done, info = self.env.step(cmd, self.pause)
            except Exception as exc:  # noqa: BLE001 — the step may or may not have run; it still counts
                self.last_error = f"{type(exc).__name__}: {exc}"
                self._log(self.steps, action, cmd, None, False, {"error": self.last_error})
                if self.steps >= self.cap:
                    self.finished = "budget"
                return {"ok": False, "error": f"the VM failed to run the action ({self.last_error})", **self._budget()}
            if done:
                self.finished = "fail" if (info or {}).get("fail") or cmd == "FAIL" else "done"
            elif self.steps >= self.cap:
                self.finished = "budget"
            shot = (obs or {}).get("screenshot")
            self._log(self.steps, action, cmd, shot, done, info)
            out = {"ok": True, "action": cmd, **self._budget()}
            if shot and not self.finished:
                out["screenshot"] = base64.b64encode(shot).decode()
            return out

    def tick(self, body: dict) -> dict:
        """Bookkeeping the backend reports for the Otto-native agent: one ``model`` tick per model response and one
        ``tool`` tick per tool call it made, so a run records its model calls and dialog messages (macOSWorld's
        "15 screenshots or 30 dialog turns" is its 15-model-call loop seen two ways).

        ``vm_done`` / ``vm_fail`` reach the bridge through the macarena-vm MCP, which cannot carry the turn id, so
        their tool tick does the turn accounting: in turns mode, ending the run in a new turn uses a step, and a new
        turn after the last one the budget allows is refused (macOSWorld's agent can only say DONE / FAIL inside
        its 15 calls)."""
        with self.lock:
            if not self.armed:
                return {"ok": False, "error": "no task is running"}
            kind = body.get("kind")
            if kind == "model":
                self.model_calls += 1
                return {"ok": True}
            if kind == "memory":
                return self._memory_record(body)
            if kind != "tool":
                return {"ok": False, "error": "kind must be 'model', 'tool' or 'memory'"}
            self.tool_messages += 1
            tool, turn = str(body.get("tool") or ""), body.get("turn")
            if tool in MEMORY_TOOLS:
                return self._memory_step(tool, turn)
            if (tool in END_TOOLS and self.step_unit == "turns" and turn is not None and str(turn) != self.cur_turn
                    and not self.finished):
                if self.steps >= self.cap:
                    self.finished = "budget"
                    return {"ok": False, "error": f"no turns left ({self.max_steps}); {tool} was not accepted",
                            **self._budget()}
                self.steps += 1
                self.cur_turn = str(turn)
            return {"ok": True, **self._budget()}

    def _memory_step(self, tool: str, turn: "str | None") -> dict:
        """A procedural-memory lookup costs a step: one per call by actions, or the turn's step by turns (free when
        the turn already used one). Past the budget it is refused; the run is not ended by it."""
        if self.finished:
            return {"ok": False, "error": f"the run is over ({self.finished}); {tool} was not run", **self._budget()}
        by_turn = self.step_unit == "turns" and turn is not None
        if by_turn and str(turn) == self.cur_turn:
            self.memory_calls += 1
            self.memory_turns.add(str(turn))
            return {"ok": True, **self._budget()}
        if self.steps >= self.cap:
            return {"ok": False, "error": f"no steps left ({self.max_steps}); {tool} was not run", **self._budget()}
        self.steps += 1
        if by_turn:
            self.cur_turn = str(turn)
        self.memory_calls += 1
        if turn is not None:
            self.memory_turns.add(str(turn))
        if self.steps >= self.cap and self.step_unit == "actions":
            self.finished = "budget"
        return {"ok": True, **self._budget()}

    def _memory_record(self, body: dict) -> dict:
        rec = body.get("record") if isinstance(body.get("record"), dict) else {}
        used = [u for u in (rec.get("used") or []) if isinstance(u, dict)]
        for u in used:
            self.memory_used.append({**u, "step": self.steps, "query": str(rec.get("query") or "")[:200],
                                     "detail": rec.get("detail") or "recipe"})
        if rec.get("abstained"):
            self.memory_abstained += 1
        self.memory_tokens += int(rec.get("tokens") or 0)
        if self.run_dir is not None:
            with (self.run_dir / "memory_calls.jsonl").open("a") as f:
                f.write(json.dumps({"ts": time.time(), "steps_used": self.steps, "turn": body.get("turn"), **rec},
                                   default=str) + "\n")
        return {"ok": True}

    def native(self, body: dict) -> dict:
        """One call of Otto's desktop tools, run in the VM (Otto-native track)."""
        with self.lock:
            if self.guest is None:
                return {"ok": False, "error": "this bridge runs the standard track (no --native)"}
            if not self.armed:
                return {"ok": False, "error": "no task is running"}
            if self.finished:
                return {"ok": False, "error": f"the run is over ({self.finished}); nothing was executed", **self._budget()}
            tool, args = str(body.get("tool") or ""), body.get("args") or {}
            if tool not in native_tools.ALL_TOOLS or not isinstance(args, dict):
                self.invalid += 1
                return {"ok": False, "error": f"unknown tool {tool!r}; nothing was executed", **self._budget()}
            observe = tool in native_tools.OBSERVATION
            refused = script_refusal(tool, args, self.kind)
            turn = body.get("turn")
            by_turn = self.step_unit == "turns" and turn is not None     # no turn id: count actions as before
            if by_turn and str(turn) != self.cur_turn:
                if self.steps >= self.cap:          # a new turn after the last one the budget allows
                    self.finished = "budget"
                    return {"ok": False, "error": f"no turns left ({self.max_steps}); nothing was executed",
                            **self._budget()}
                self.steps += 1
                self.cur_turn = str(turn)
            if refused:                             # like a refusal by the VM: counted invalid, nothing run
                self.invalid += 1
                if self.invalid >= self.cap:
                    self.finished = "budget"
                self._log_native(tool, args, 0, {"ok": False, "refused": True, "error": refused})
                return {"ok": False, "error": refused, **self._budget()}
            if observe:
                if self.obs_calls >= self.max_obs:
                    return {"ok": False, "error": f"read limit ({self.max_obs}) reached; act or finish", **self._budget()}
                self.obs_calls += 1
            budget = 0 if observe else (TURN_BATCH_MAX if by_turn else self.cap - self.steps)
            self.native_calls += 1
            call = {"tool_id": str(body.get("tool_id") or ""), "tool": tool, "args": args,
                    "config": body.get("config") or {}, "vision_mode": bool(body.get("vision_mode")), "budget": budget,
                    "observe_after": (self.observe_after and tool in native_tools.OBSERVE_AFTER
                                      and self.obs_calls < self.max_obs)}
            try:
                res = self.guest.call(call)
            except Exception as exc:  # noqa: BLE001 — an action may or may not have run; it still counts
                res = {"ok": False, "cost": 0 if observe else 1,
                       "error": f"the VM tool server failed ({type(exc).__name__}: {exc})"}
            if res.get("refused"):
                self.invalid += 1
                if self.invalid >= self.cap:
                    self.finished = "budget"
                self._log_native(tool, args, 0, res)
                return {"ok": False, "error": res.get("error") or "refused; nothing was executed", **self._budget()}
            n = 0 if observe else max(1, int(res.get("cost") or 1))
            if not by_turn:
                self.steps += n
            if res.get("observed"):                 # the controls returned with this action: one free read
                self.obs_calls += 1
            notice = "" if observe else self._streak_notice(tool, not native_tools.succeeded(res))
            if not observe:
                notice = "\n".join(x for x in (notice, self._repeat_notice(tool, args)) if x)
            if self.steps >= self.cap and not by_turn:      # by turn: the next new turn is the one refused
                self.finished = "budget"
            self._log_native(tool, args, n, res)
            if n:
                shot = None
                try:
                    if not self.record:
                        raise LookupError("screenshots are not being recorded")
                    shot = self.env.controller.get_screenshot()
                except LookupError:
                    pass
                except Exception as exc:  # noqa: BLE001 — the record only
                    logger.warning("no screenshot for the record: %s", exc)
                self._log(self.steps, {"tool": tool, "args": args, "cost": n},
                          f"otto:{tool}", shot, False, {"ok": bool(res.get("ok")), "error": res.get("error")})
            extra = {"notice": notice} if notice else {}
            if res.get("ok"):
                return {"ok": True, "result": res.get("result"), **extra, **self._budget()}
            return {"ok": False, "error": res.get("error") or "the tool failed", **extra, **self._budget()}

    def _log_native(self, tool: str, args: dict, cost: int, res: dict) -> None:
        if self.run_dir is None:
            return
        result = res.get("result")
        preview = result if isinstance(result, str) else json.dumps(result, default=str)
        with (self.run_dir / "native_calls.jsonl").open("a") as f:
            f.write(json.dumps({"n": self.native_calls, "ts": time.time(), "tool": tool, "args": args, "cost": cost,
                                "steps_used": self.steps, "ok": bool(res.get("ok")), "refused": bool(res.get("refused")),
                                "error": res.get("error"), "result": (preview or "")[:4000]}, default=str) + "\n")

    def _log(self, n: int, action: dict, cmd: str, shot: bytes | None, done: bool, info: dict) -> None:
        if self.run_dir is None:
            return
        ts = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
        name = None
        if shot and self.record:
            name = f"step_{n}_{ts}.png"
            (self.run_dir / name).write_bytes(shot)
        with (self.run_dir / "traj.jsonl").open("a") as f:
            f.write(json.dumps({"step_num": n, "action_timestamp": ts, "action": cmd, "agent_action": action,
                                "reward": 0, "done": done, "info": info, "screenshot_file": name}, default=str) + "\n")


# ── HTTP ────────────────────────────────────────────────────────────────────

def make_handler(bridge: Bridge, agent_token: str, control_token: str, load_task):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):  # quiet
            logger.debug(fmt, *a)

        def _auth(self, token: str) -> bool:
            got = self.headers.get("Authorization", "")
            return hmac.compare_digest(got, f"Bearer {token}")

        def _send(self, code: int, body: dict):
            data = json.dumps(body, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 1_000_000:
                raise ValueError("body too large")
            return json.loads(self.rfile.read(n) or b"{}")

        def _route(self, method: str):
            path = self.path.split("?")[0]
            try:
                if path.startswith("/agent/"):
                    if not self._auth(agent_token):
                        return self._send(401, {"ok": False, "error": "unauthorized"})
                    body = self._body() if method == "POST" else {}
                    if path == "/agent/observe":
                        return self._send(200, bridge.observe(str(body.get("kind", "screenshot"))))
                    if path == "/agent/act":
                        return self._send(200, bridge.act(body.get("action")))
                    if path == "/agent/native":
                        return self._send(200, bridge.native(body))
                    if path == "/agent/tick":
                        return self._send(200, bridge.tick(body))
                elif path.startswith("/ctl/"):
                    if not self._auth(control_token):
                        return self._send(401, {"ok": False, "error": "unauthorized"})
                    body = self._body() if method == "POST" else {}
                    if path == "/ctl/state":
                        return self._send(200, bridge.state())
                    if path == "/ctl/reset":
                        task = load_task(body["task_path"])
                        tp = Path(body["task_path"])             # .../<source>/<domain>/<id>.json
                        return self._send(200, bridge.reset(task, Path(body["run_dir"]), fresh=bool(body.get("fresh", True)),
                                                            kind=(tp.parent.parent.name, tp.parent.name)))
                    if path == "/ctl/preflight":
                        return self._send(200, bridge.preflight(list(body.get("apps") or [])))
                    if path == "/ctl/evaluate":
                        return self._send(200, bridge.evaluate())
                    if path == "/ctl/finish":
                        return self._send(200, bridge.finish(float(body.get("score", 0.0))))
                    if path == "/ctl/health":
                        return self._send(200, {"ok": True})
                self._send(404, {"ok": False, "error": "not found"})
            except Exception as exc:  # noqa: BLE001
                logger.exception("request failed")
                self._send(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

        def do_GET(self): self._route("GET")
        def do_POST(self): self._route("POST")

    return H


SKIP_KNOWN_HOSTS = ["-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR"]


def skip_known_hosts(opts: list | None = None) -> list:
    """Make MacArena's UTM provider ignore ``~/.ssh/known_hosts``. Every task's VM copy gets a new SSH host key but
    reuses an address from UTM's small pool, so a stored key for that address no longer matches. MacArena already
    passes ``StrictHostKeyChecking=no``, but with a mismatching stored key ssh still turns password login off and
    the copy cannot be reached ("REMOTE HOST IDENTIFICATION HAS CHANGED"), which stopped the bridge at startup.
    These are throwaway local VMs, so their keys are not recorded at all (``bench.sh`` does the same for its own
    ssh). Changes the provider's option list in place; MacArena's files are not edited."""
    if opts is None:
        from desktop_env.providers.utm import provider as utm  # type: ignore
        opts = utm._SSH_OPTS
    if "UserKnownHostsFile=/dev/null" not in opts:
        opts.extend(SKIP_KNOWN_HOSTS)
    return opts


# OSWorld's setup starts Linux programs by name. They do not exist on macOS, so the task starts with no app open and the
# agent spends its 15 turns finding and opening the file (or opens it in the wrong app). The same apps are launched with
# ``open -a``, which gives the start state the task assumes. Only the three Linux names below change; every other
# command, and every MacArena and macOSWorld task, runs exactly as written.
MAC_LAUNCH = {"gimp": "GIMP", "soffice": "LibreOffice", "libreoffice": "LibreOffice"}
ACTIVATE_WAIT = 5.0   # seconds for the launched app to open its window before it is brought forward
_OFFICE_FLAGS = {"--calc", "--writer", "--impress", "--draw", "--base", "--math"}


def macos_launch(command, shell: bool = False):
    """The command to run in the VM for an OSWorld ``launch`` step: ``open -a <App> [files]`` for gimp / soffice, else
    the command unchanged."""
    if shell or not isinstance(command, list) or not command or command[0] not in MAC_LAUNCH:
        return command
    return ["open", "-a", MAC_LAUNCH[command[0]]] + [c for c in command[1:] if c not in _OFFICE_FLAGS]


def translate_launch_setup() -> None:
    """Make MacArena's setup controller start gimp / soffice the macOS way (see ``macos_launch``). Patches the class in
    this process only; MacArena's files are not edited."""
    from desktop_env.controllers.setup import SetupController  # type: ignore
    orig = SetupController._launch_setup
    if getattr(orig, "_otto_macos", False):
        return

    def _launch_setup(self, command, shell=False):
        translated = macos_launch(command, shell)
        out = orig(self, translated, shell)
        if translated is not command:
            # ``open -a`` returns before the window exists, and a program started over SSH may not come to the front.
            # Wait for the window, then bring the app forward, as the task's Linux start state has it.
            time.sleep(ACTIVATE_WAIT)
            try:
                self._execute_setup(["osascript", "-e", f'tell application "{translated[2]}" to activate'], shell=False)
            except Exception as e:  # never fail a task's setup over this
                print(f"could not bring {translated[2]} to the front: {e}", file=sys.stderr)
        return out
    _launch_setup._otto_macos = True  # type: ignore[attr-defined]
    SetupController._launch_setup = _launch_setup


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--macarena-dir", default=os.environ.get("MACARENA_DIR"), help="MacArena checkout")
    p.add_argument("--vm", default="osworld", help="UTM VM name: 'osworld' for OSWorld/macOSWorld tasks, 'macarena' for MacArena's own")
    p.add_argument("--provider", default="utm")
    p.add_argument("--port", type=int, default=18095)
    p.add_argument("--agent-token", default=os.environ.get("MACARENA_AGENT_TOKEN"))
    p.add_argument("--control-token", default=os.environ.get("MACARENA_CONTROL_TOKEN"))
    p.add_argument("--max-steps", type=int, default=0,
                   help="steps per run; 0 = no limit (the MacArena paper uses 15)")
    p.add_argument("--max-obs", type=int)
    p.add_argument("--a11y", action="store_true", help="give the agent the accessibility tree (MacArena: optional)")
    p.add_argument("--pause", type=float, default=3.0, help="seconds after each action (MacArena used 3.0)")
    p.add_argument("--settle", type=float, default=10.0)
    p.add_argument("--coord-scale", type=float, nargs=2, metavar=("SX", "SY"))
    p.add_argument("--fake-env", action="store_true", help="no VM: wiring smoke test only")
    p.add_argument("--record", action="store_true",
                   help="save a screenshot after every step and recording.mp4 (off by default: review only, ~1.7 GB per 7 runs)")
    p.add_argument("--step-unit", choices=("actions", "turns"), default="actions",
                   help="what one step is: every action (default, stricter) or one agent turn (Otto-native only)")
    p.add_argument("--native", action="store_true", help="Otto-native track: Otto's desktop tools run in the VM")
    p.add_argument("--no-observe-after", action="store_true",
                   help="Otto-native: do not return the app's controls with each action (the tools as the app has them)")
    p.add_argument("--fail-notice-after", type=int, default=FAIL_NOTICE_AFTER,
                   help="add a notice after this many failed actions in a row (0 = never)")
    p.add_argument("--repeat-notice-after", type=int, default=REPEAT_NOTICE_AFTER,
                   help="add a notice after this many identical actions in a row (0 = never)")
    p.add_argument("--setup-mode", choices=SETUP_MODES, default="sync",
                   help="sync (default): run each task's pre_command and wait for it, check it, keep its output in "
                        "setup.json; launch: start it without waiting, as MacArena does")
    p.add_argument("--keep-nags", action="store_true",
                   help="Otto-native: leave update nags (Chrome's \"Can't update Chrome\") open at the start")
    p.add_argument("--native-bundle", help="tarball of Otto's toolkit + the guest server (bench.sh builds it)")
    p.add_argument("--guest-python", default="/opt/homebrew/Caskroom/miniconda/base/envs/server/bin/python3",
                   help="the VM's Python that runs MacArena's own server (has PyObjC and the permissions)")
    p.add_argument("--guest-port", type=int, default=18100)
    a = p.parse_args()
    if a.native and not a.fake_env and not (a.native_bundle and Path(a.native_bundle).is_file()):
        p.error("--native needs --native-bundle pointing at a built bundle (bench.sh up --native builds it)")
    if not a.agent_token or not a.control_token or a.agent_token == a.control_token:
        p.error("--agent-token and --control-token are required and must differ")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    if a.fake_env:
        env = FakeEnv()
        a.settle = min(a.settle, 0.0)
    else:
        if not a.macarena_dir:
            p.error("--macarena-dir (or MACARENA_DIR) is required")
        sys.path.insert(0, str(Path(a.macarena_dir).resolve()))
        os.chdir(a.macarena_dir)                    # MacArena resolves cache/ and logs/ relative to itself
        from desktop_env.desktop_env import DesktopEnv  # type: ignore
        if a.provider == "utm":
            skip_known_hosts()
        translate_launch_setup()
        env = DesktopEnv(action_space="pyautogui", provider_name=a.provider, path_to_vm=a.vm,
                         snapshot_name=a.vm, require_a11y_tree=a.a11y)
        # Fail fast with the cause, instead of every task stalling in MacArena's silent connect retries.
        url = env.controller.http_server.rstrip("/") + "/screenshot"
        blocked = local_network_error(url)
        if blocked:
            logger.error(LOCAL_NETWORK_BLOCKED.format(url=url, error=blocked[-160:], python=sys.executable))
            env.close()                             # do not leave the VM copy running
            raise SystemExit(3)

    guest = None
    if a.native:
        guest = FakeGuest() if a.fake_env else GuestTools(env, Path(a.native_bundle).resolve(), a.guest_python,
                                                          a.guest_port)
    bridge = Bridge(env, step_unit=a.step_unit, max_steps=a.max_steps, max_obs=a.max_obs, use_a11y=a.a11y, pause=a.pause,
                    settle=a.settle, coord_scale=tuple(a.coord_scale) if a.coord_scale else None,
                    vm_name=a.vm, guest=guest, record=a.record, observe_after=not a.no_observe_after,
                    fail_notice_after=a.fail_notice_after, repeat_notice_after=a.repeat_notice_after,
                    dismiss_nags=not a.keep_nags, setup_mode=a.setup_mode)

    def load_task(path: str) -> dict:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    import signal

    def _term(*_):                                  # SIGTERM (bench.sh stop) runs the `finally` below, so the
        signal.signal(signal.SIGTERM, signal.SIG_IGN)   # VM copy is stopped and deleted instead of left running;
        signal.signal(signal.SIGINT, signal.SIG_IGN)    # a second signal must not interrupt that clean-up
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, _term)

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(bridge, a.agent_token, a.control_token, load_task))
    logger.info("bridge on 127.0.0.1:%d (vm=%s, track=%s, max_steps=%s, fake=%s)", a.port, a.vm, bridge.track,
                a.max_steps or "none", a.fake_env)
    try:
        srv.serve_forever()
    finally:
        if hasattr(env, "close"):
            logger.info("closing the environment (stops and deletes the VM copy)")
            env.close()


if __name__ == "__main__":
    main()
