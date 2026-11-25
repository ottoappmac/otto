"""The combined macOS agent drives the screen from two processes: the
``macos-native`` tools run in the backend, the ``macos-osascript`` tools in an
MCP subprocess.  The desktop lease is a cross-process ``flock`` with per-owner
re-entrancy only *inside* one process, so a subprocess script that needs the
screen would wait on the agent's own lease (up to the idle timeout, or fail
with "desktop busy").  The fix: the backend takes the lease for the osascript
tools with the agent's own owner id, and tells the subprocess not to lock.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

import backend.builtin_mcps.macos_osascript._desktop_lock as dl
from tests.test_desktop_lease import _hard_reset

ROOT = Path(__file__).resolve().parents[1]
LOCK_MODULE_DIR = ROOT / "backend" / "builtin_mcps" / "macos_osascript"

# What the osascript subprocess does for a screen-needing script: import the bundled sibling module and
# take ``desktop_lock()`` with its default per-process owner.
_CHILD = """
import asyncio, sys
sys.path.insert(0, sys.argv[1])
import _desktop_lock as dl
async def go():
    try:
        async with dl.desktop_lock(wait_timeout=1):
            print("ran")
    except dl.DesktopBusy:
        print("busy")
asyncio.run(go())
"""


@pytest.fixture(autouse=True)
def _lease(tmp_path, monkeypatch):
    monkeypatch.setenv("OTTO_DESKTOP_LOCK_PATH", str(tmp_path / "desktop.lock"))
    monkeypatch.delenv(dl.HOST_HOLDS_LEASE_ENV, raising=False)
    _hard_reset()
    yield
    _hard_reset()


def _child(env_extra: dict[str, str]) -> str:
    env = {**os.environ, **env_extra}
    out = subprocess.run([sys.executable, "-c", _CHILD, str(LOCK_MODULE_DIR)], env=env,
                         capture_output=True, text=True, timeout=30)
    return out.stdout.strip()


def test_subprocess_lock_blocks_on_the_agents_own_lease():
    """The bug: the agent holds the lease in the backend; its own osascript subprocess can't get it."""
    async def go():
        await dl.acquire_desktop("desktop:s1:macos-desktop-agent", wait_timeout=2)
        await dl.end_desktop_call("desktop:s1:macos-desktop-agent")       # idle, but within LEASE_IDLE_SECS
        return await asyncio.to_thread(_child, {})
    assert asyncio.run(go()) == "busy"


def test_subprocess_skips_lock_when_host_holds_it():
    async def go():
        await dl.acquire_desktop("desktop:s1:macos-desktop-agent", wait_timeout=2)
        try:
            return await asyncio.to_thread(_child, {dl.HOST_HOLDS_LEASE_ENV: "1"})
        finally:
            await dl.end_desktop_call("desktop:s1:macos-desktop-agent")
    assert asyncio.run(go()) == "ran"


# ── Backend side: the osascript tools take the lease under the agent's owner ──

def _fake_osascript_tool(name: str, calls: list):
    from langchain_core.tools import StructuredTool

    async def run(**kwargs):
        calls.append(kwargs)
        return ("ok", None)
    return StructuredTool.from_function(coroutine=run, name=name, description=name,
                                        response_format="content_and_artifact")


def _wrapped(name: str, calls: list):
    from backend import mcp_manager as mm
    t = _fake_osascript_tool(name, calls)
    mm._wrap_with_desktop_lock(t, "macos-osascript-conn", mm._OSASCRIPT_LOCK_RULES[name])
    return t


def test_agent_scripts_run_under_its_own_lease():
    """An agent that holds the lease for its UI tools runs its screen-needing script at once."""
    calls: list = []
    t = _wrapped("run_osascript", calls)
    owner = "desktop:s1:macos-desktop-agent"

    async def go():
        token = dl.active_desktop_owner.set(owner)
        try:
            await dl.acquire_desktop(owner, wait_timeout=2)                 # a macos-native call in flight
            try:
                out = await asyncio.wait_for(
                    t.coroutine(script='tell application "System Events" to keystroke "a"'), 2)
            finally:
                await dl.end_desktop_call(owner)
            return out, dl._held_owner
        finally:
            dl.active_desktop_owner.reset(token)
    out, held = asyncio.run(go())
    assert out == ("ok", None) and len(calls) == 1
    assert held == "desktop:s1:macos-desktop-agent"


def test_background_script_skips_the_lease():
    """A dictionary-verb script needs no screen: it runs even while another agent holds the lease."""
    calls: list = []
    t = _wrapped("run_osascript", calls)

    async def go():
        await dl.acquire_desktop("desktop:other-agent", wait_timeout=2)     # another agent mid-call
        return await asyncio.wait_for(t.coroutine(script='tell application "Music" to get name of current track'), 2)
    assert asyncio.run(go()) == ("ok", None) and len(calls) == 1


def test_screen_script_waits_for_another_agent_and_reports_busy(monkeypatch):
    """Two different agents still serialize; a timeout returns the busy text in the tool's (content, artifact) shape."""
    import functools
    monkeypatch.setattr(dl, "acquire_desktop", functools.partial(dl.acquire_desktop, wait_timeout=0.3))
    calls: list = []
    t = _wrapped("dump_ax_tree", calls)

    async def go():
        await dl.acquire_desktop("desktop:other-agent", wait_timeout=2)
        token = dl.active_desktop_owner.set("desktop:s1:macos-desktop-agent")
        try:
            return await t.coroutine(app_name="Finder")
        finally:
            dl.active_desktop_owner.reset(token)
    out = asyncio.run(go())
    assert isinstance(out, tuple) and out[1] is None and "Desktop is busy" in out[0]
    assert calls == []


def test_osascript_subprocess_is_spawned_with_the_host_lease_flag(monkeypatch):
    from backend.config import MCPServerConfig
    from backend import mcp_manager as mm
    import langchain_mcp_adapters.client as lmc

    seen: dict = {}

    class FakeClient:
        def __init__(self, connections):
            seen.update(connections)
    monkeypatch.setattr(lmc, "MultiServerMCPClient", FakeClient)
    monkeypatch.setattr(mm, "_hydrate_secrets", lambda cfg: {})
    mgr = mm.MCPManager()
    for sid in ("macos-osascript", "slack"):
        cfg = MCPServerConfig(id=sid, name=sid, transport="stdio", command=sys.executable,
                              args=["-c", "pass"], builtin=False, enabled=True)
        asyncio.run(mgr._create_helper(cfg))
    assert seen["macos-osascript"]["env"][dl.HOST_HOLDS_LEASE_ENV] == "1"
    assert dl.HOST_HOLDS_LEASE_ENV not in (seen["slack"]["env"] or {})


def test_loader_locks_only_the_osascript_screen_tools(monkeypatch):
    from backend.config import MCPServerConfig
    from backend import mcp_manager as mm

    names = ["run_osascript", "run_osascript_file", "dump_ax_tree", "inspect_app_dictionary", "query_mail_store"]

    class FakeHelper:
        def get_tools(self):
            return [_fake_osascript_tool(n, []) for n in names]

    async def fake_create(self, config):
        return FakeHelper()

    async def fake_connect(helper, label):
        return helper

    monkeypatch.setattr(mm.MCPManager, "_create_helper", fake_create)
    monkeypatch.setattr(mm._OwnedMCPHelper, "connect", staticmethod(fake_connect))
    locked = {}
    real = mm._wrap_with_desktop_lock

    def spy(tool, owner, needs_lock=None):
        locked[tool.name] = needs_lock
        real(tool, owner, needs_lock)
    monkeypatch.setattr(mm, "_wrap_with_desktop_lock", spy)

    for sid in ("macos-osascript", "other-server"):
        cfg = MCPServerConfig(id=sid, name=sid, transport="stdio", command="x", builtin=True, enabled=True)
        asyncio.run(mm.MCPManager()._load_generic(cfg, mm.MCPConnection(config=cfg)))
    assert set(locked) == {"run_osascript", "run_osascript_file", "dump_ax_tree"}
    assert locked["run_osascript"]({"script": 'tell application "Notes" to get name of notes'}) is False
    assert locked["run_osascript"]({"script": 'tell application "System Events" to keystroke "x"'}) is True
