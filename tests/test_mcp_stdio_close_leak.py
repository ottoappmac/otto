"""Stdio MCP servers must die with their connection — whichever task closes it.

Session graphs connect their MCP servers inside short-lived tasks
(``SessionManager._build_graph`` → ``asyncio.shield(MCPManager.connect_all)``
→ one ``asyncio.gather`` task per server) and close them later from whatever
task rebuilds, evicts or deletes the session.  The anyio cancel scopes that
``stdio_client`` / ``ClientSession`` open must be exited by the task that
entered them, so every one of those closes used to run cross-task.

These tests drive a real stdio MCP server (a tiny FastMCP script in a temp
dir) through that same code path and check the OS process table rather than
our own bookkeeping.  The probe server — and the helper it can spawn,
standing in for ``npx`` → ``node`` or a worker pool — carry the probe
script's unique temp path on their command lines, so ``ps`` finds exactly
them (no psutil needed).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from backend import mcp_manager
from backend.config import MCPServerConfig
from backend.mcp_manager import ManagedProcess, MCPManager

if sys.platform == "win32":
    pytest.skip("relies on ps and POSIX process groups", allow_module_level=True)

_PROBE_SERVER = '''\
import subprocess
import sys
import time

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("leak-probe")


@mcp.tool()
def ping() -> str:
    """Answer pong."""
    return "pong"


if "--spawn-helper" in sys.argv:
    subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(600)", sys.argv[0]],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
if "--slow-start" in sys.argv:
    time.sleep(2)
mcp.run()
'''

_HTTP_SERVER = '''\
import sys

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("leak-probe-http", port=int(sys.argv[1]), log_level="WARNING")


@mcp.tool()
def ping() -> str:
    """Answer pong."""
    return "pong"


mcp.run(transport="streamable-http")
'''


def _live(marker: str) -> list[int]:
    """PIDs of running (non-zombie) processes with *marker* in their command."""
    out = subprocess.run(
        ["ps", "-A", "-o", "pid=,stat=,command="],
        capture_output=True,
        text=True,
    ).stdout
    pids = []
    for line in out.splitlines():
        fields = line.split(None, 2)
        if len(fields) == 3 and marker in fields[2] and not fields[1].startswith("Z"):
            pids.append(int(fields[0]))
    return pids


async def _live_after(marker: str, *, within: float) -> list[int]:
    """The *marker* processes still running after waiting up to *within* s."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + within
    live = _live(marker)
    while live and loop.time() < deadline:
        await asyncio.sleep(0.1)
        live = _live(marker)
    return live


@pytest.fixture
def probe_script(tmp_path: Path):
    """The probe server script; its path marks every process it starts."""
    script = tmp_path / "leak_probe_server.py"
    script.write_text(_PROBE_SERVER)
    # Failed connects (the timeout test) must not trip the circuit breaker
    # for the tests that follow.
    mcp_manager.reset_circuit_breaker("leak-probe")
    yield str(script)
    mcp_manager.reset_circuit_breaker("leak-probe")
    # Never leave a process behind, even when an assertion failed.
    for pid in _live(str(script)):
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)


def _stdio_config(script: str, *flags: str) -> MCPServerConfig:
    return MCPServerConfig(
        id="leak-probe",
        name="leak-probe",
        transport="stdio",
        command=sys.executable,
        args=[script, *flags],
    )


async def _connect_like_build_graph(
    mgr: MCPManager, config: MCPServerConfig, **kwargs: float,
) -> None:
    """Connect the way ``_build_graph`` does: via a shielded ``connect_all``."""
    await asyncio.shield(mgr.connect_all([config], skip_process_start=True, **kwargs))


async def _close_from_another_task(mgr: MCPManager) -> None:
    await asyncio.create_task(mgr.close())


@pytest.mark.parametrize(
    "flags",
    [(), ("--spawn-helper",)],
    ids=["server", "server-and-its-helper"],
)
async def test_close_from_another_task_terminates_the_server_process_tree(
    probe_script: str, flags: tuple[str, ...],
) -> None:
    mgr = MCPManager()
    await _connect_like_build_graph(mgr, _stdio_config(probe_script, *flags))
    conn = mgr.connections["leak-probe"]
    assert conn.connected, conn.error
    assert len(_live(probe_script)) == 1 + len(flags)

    await _close_from_another_task(mgr)

    assert await _live_after(probe_script, within=3.0) == []


async def test_repeated_rebuilds_do_not_accumulate_server_processes(
    probe_script: str,
) -> None:
    """Three session rebuilds' worth of connect → close cycles leave nothing behind."""
    live_after_each_close = []
    for _ in range(3):
        mgr = MCPManager()
        await _connect_like_build_graph(mgr, _stdio_config(probe_script, "--spawn-helper"))
        assert mgr.connections["leak-probe"].connected
        await _close_from_another_task(mgr)
        live_after_each_close.append(len(await _live_after(probe_script, within=3.0)))

    assert live_after_each_close == [0, 0, 0]


async def test_close_from_another_task_leaves_the_connecting_task_alone(
    probe_script: str,
) -> None:
    """Someone else's close must not cancel a still-running task that connected.

    Mirrors ``POST /api/mcp/{id}/connect``: the request task connects, then
    keeps working (``refresh_tools``) while another request may disconnect.
    """
    mgr = MCPManager()
    connected = asyncio.Event()
    release = asyncio.Event()

    async def connect_then_keep_working() -> str:
        await mgr.connect(_stdio_config(probe_script))
        connected.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            return "cancelled"
        return "finished"

    owner = asyncio.create_task(connect_then_keep_working())
    await connected.wait()
    await _close_from_another_task(mgr)
    await asyncio.sleep(0.2)  # room for a stray cancellation to land
    release.set()

    assert await owner == "finished"


async def test_timed_out_connect_does_not_leave_the_server_running(
    probe_script: str,
) -> None:
    """A server too slow to initialize is abandoned by the build — and must die."""
    # Cold imports on a first connect can outlast the 1s timeout before
    # anything is spawned; warm them so the timeout lands mid-initialize.
    import tools.anthropic.mcps  # noqa: F401

    mgr = MCPManager()
    await _connect_like_build_graph(
        mgr, _stdio_config(probe_script, "--slow-start"), timeout=1.0,
    )
    assert not mgr.connections["leak-probe"].connected
    assert _live(probe_script), "the slow server should still be starting"

    assert await _live_after(probe_script, within=8.0) == []


async def test_close_kills_the_server_when_the_client_shutdown_hangs(
    probe_script: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timeout + hard-kill fallback: a wedged teardown can't keep the server alive."""
    from tools.anthropic import mcps

    real_close = mcps.MCPHelper.close

    async def wedged_close(self) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            await real_close(self)  # let the SDK unwind once we give up waiting

    monkeypatch.setattr(mcps.MCPHelper, "close", wedged_close)
    monkeypatch.setattr(mcp_manager, "_MCP_CLOSE_TIMEOUT_SECS", 1.0, raising=False)

    mgr = MCPManager()
    await _connect_like_build_graph(mgr, _stdio_config(probe_script))
    assert mgr.connections["leak-probe"].connected
    assert _live(probe_script)

    # asyncio.wait (unlike wait_for) doesn't cancel on timeout: close() has
    # to come back on its own.
    closing = asyncio.create_task(mgr.close())
    done, _ = await asyncio.wait({closing}, timeout=5)
    try:
        assert closing in done, "close() hung on a wedged client shutdown"
        assert await _live_after(probe_script, within=3.0) == []
    finally:
        closing.cancel()
        with contextlib.suppress(BaseException):
            await closing


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def test_streamable_http_connection_still_works_and_closes_cleanly(
    tmp_path: Path,
) -> None:
    """No regression for HTTP MCPs or for ``ManagedProcess`` start/stop."""
    script = tmp_path / "leak_probe_http.py"
    script.write_text(_HTTP_SERVER)
    port = _free_port()
    url = f"http://127.0.0.1:{port}/mcp"
    server = ManagedProcess(
        server_id="leak-probe-http",
        command=[sys.executable, str(script), str(port)],
        health_url=url,
    )
    await server.start(ready_timeout=30)
    try:
        mgr = MCPManager()
        config = MCPServerConfig(
            id="leak-probe-http", name="leak-probe-http",
            transport="streamable_http", url=url,
        )
        await _connect_like_build_graph(mgr, config)
        conn = mgr.connections["leak-probe-http"]
        assert conn.connected, conn.error
        ping = next(t for t in conn.tools if t.name == "ping")
        assert "pong" in str(await asyncio.create_task(ping.ainvoke({})))

        await _close_from_another_task(mgr)
        assert mgr.connections == {}
    finally:
        await server.stop()
    assert not server.running
