"""Hermetic tests for the MacArena Otto-native track (Otto's macOS desktop agent, tools run in the VM).

No VM, no model, and nothing acts on this computer: the bridge uses ``FakeEnv`` and a stand-in for the VM's tool
server, and Otto's real toolkit is only built (its tools' names, schemas and wrappers), never run here. These
tests do NOT show that the toolkit works inside a real MacArena VM; see scripts/macarena/README.md "Status".
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import types
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MA = ROOT / "scripts" / "macarena"
for p in (str(MA), str(MA / "guest"), str(ROOT / "src"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import bridge as br  # noqa: E402
import native_tools  # noqa: E402
import otto_tools_server as gs  # noqa: E402
import permission_clicker as pc  # noqa: E402
import report  # noqa: E402

from backend import macarena_native  # noqa: E402


def _load_macarena_run():
    """scripts/macarena/run.py under its own name: a plain ``import run`` would shadow scripts/swebench/run.py
    for every test that runs later (test_swebench_docker_env imports ``run``)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("macarena_run", MA / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runpy_ = _load_macarena_run()
# Everything is imported; take the script folders off sys.path so they cannot shadow other suites' modules.
sys.path[:] = [p for p in sys.path if p not in (str(MA), str(MA / "guest"))]

TASK = {"id": "t1", "instruction": "do it", "evaluator": []}


def real_toolkit():
    from tools.navigation.computer.macos_tools import MacOSToolkit
    return MacOSToolkit()


def parse_batch():
    from tools.navigation.computer.macos_tools import _parse_batch_actions
    return _parse_batch_actions


# ── the step model is Otto's own read / act split ───────────────────────────

def test_tool_sets_match_the_real_toolkit_and_the_action_policy():
    tk = real_toolkit()
    names = {t.name for t in tk.tools} | {t.name for t in tk.vision_tools}
    from backend.builtin_mcps.macos_osascript import server as osa
    osa_names = set(osa.VM_TOOLS)
    assert names == set(macarena_native.NATIVE_TOOL_NAMES) == set(native_tools.ALL_TOOLS) - osa_names
    assert osa_names == set(macarena_native.OSASCRIPT_TOOL_NAMES) == set(native_tools.OSASCRIPT_TOOLS)
    assert not (native_tools.OBSERVATION & native_tools.ACTION)
    from backend.action_policy import catalog
    def is_read(category):              # ungated, or a gated read (mail_read: reading mail changes nothing)
        return category is None or category.endswith("_read")
    for server, mine in (("macos-native", names), ("macos-osascript", osa_names)):
        cats = {n: c for (s, n), c in catalog._SERVER_TOOLS.items() if s == server and n in mine}
        assert {n for n, c in cats.items() if not is_read(c)} == set(native_tools.ACTION) & mine
        assert {n for n, c in cats.items() if is_read(c)} == (set(native_tools.OBSERVATION) & mine) - {
            "capture_app_screenshot"}                                              # not in the catalog; a read


def test_cost_reads_free_actions_one_batches_per_action():
    assert native_tools.cost("get_screen_controls", {}) == 0
    assert native_tools.cost("press_control", {"index": 1}) == 1
    # get_control_value is the one read Otto's batch parser accepts; it is free inside a batch too.
    steps = "press_control(index=5)\nget_control_value(index=2)\ntype_into_control(index=3, text='a;b(c)')"
    assert [n for n, _ in parse_batch()(steps)] == ["press_control", "get_control_value", "type_into_control"]
    assert native_tools.cost("batch_actions", {"steps": steps}, parse_batch()) == 2
    assert native_tools.cost("batch_actions", {"steps": "get_control_value(index=1)"}, parse_batch()) == 1
    assert native_tools.cost("batch_actions", {"steps": steps}) == 1                     # no parser: the minimum


# ── bridge, native track ────────────────────────────────────────────────────

def make_bridge(tmp_path, max_steps=3, **kw):
    g = br.FakeGuest()
    b = br.Bridge(br.FakeEnv(), max_steps=max_steps, settle=0.0, pause=0.0, guest=g, **kw)
    b.reset(TASK, tmp_path / "run_1")
    return b, g


def call(b, tool, **args):
    return b.native({"tool": tool, "tool_id": f"tools:{tool}", "args": args, "config": {"scan_max_depth": 9}})


def test_reset_deploys_the_guest_server_every_run(tmp_path):
    b, g = make_bridge(tmp_path)
    b.reset(TASK, tmp_path / "run_2", fresh=False)
    assert g.deploys == 2 and b.state()["track"] == "native" and b.state()["guest"]["trusted"]


def test_reads_are_free_and_capped_actions_count(tmp_path):
    b, g = make_bridge(tmp_path, max_steps=3, max_obs=2)
    assert call(b, "list_apps")["steps_used"] == 0
    assert call(b, "get_screen_controls", app_name="Notes")["ok"]
    r = call(b, "get_screen_controls", app_name="Notes")
    assert not r["ok"] and "read limit" in r["error"] and len(g.calls) == 2            # third read never sent
    r = call(b, "press_control", index=4)
    assert r["ok"] and r["steps_used"] == 1 and "(fake VM) press_control" in r["result"]
    assert g.calls[-1]["config"] == {"scan_max_depth": 9} and g.calls[-1]["budget"] == 3


def test_budget_is_enforced_and_extra_calls_are_not_sent(tmp_path):
    b, g = make_bridge(tmp_path, max_steps=2)
    call(b, "click", index=1)
    call(b, "hotkey", keys="cmd+s")
    assert b.finished == "budget"
    sent = len(g.calls)
    for tool in ("click", "get_screen_controls"):
        r = call(b, tool, index=1)
        assert not r["ok"] and "run is over" in r["error"]
    assert len(g.calls) == sent


def test_batch_costs_one_per_action_and_an_oversized_batch_runs_nothing(tmp_path):
    b, g = make_bridge(tmp_path, max_steps=3)
    r = call(b, "batch_actions", steps="click(index=1)\nclick(index=2)\nclick(index=3)\nclick(index=4)")
    assert not r["ok"] and "nothing was run" in r["error"] and b.steps == 0 and b.invalid == 1
    r = call(b, "batch_actions", steps="click(index=1)\nclick(index=2)")
    assert r["ok"] and r["steps_used"] == 2


def test_unknown_tool_and_guest_failure(tmp_path):
    b, g = make_bridge(tmp_path, max_steps=3)
    r = call(b, "execute", command="ls")
    assert not r["ok"] and "unknown tool" in r["error"] and b.steps == 0 and not g.calls

    def boom(body):
        raise ConnectionError("down")
    g.call = boom
    r = call(b, "click", index=1)
    assert not r["ok"] and "tool server failed" in r["error"] and r["steps_used"] == 1     # may have run: counts
    r = call(b, "list_apps")
    assert not r["ok"] and r["steps_used"] == 1                                           # a failed read: free


def test_only_done_and_fail_on_the_action_endpoint_and_no_screenshots(tmp_path):
    b, g = make_bridge(tmp_path)
    r = b.act({"type": "click", "x": 1, "y": 1})
    assert not r["ok"] and "only done / fail" in r["error"] and not b.env.actions
    assert not b.observe("screenshot")["ok"]
    assert b.act({"type": "done"})["ok"] and b.finished == "done" and b.env.actions == ["DONE"]
    assert not call(b, "click", index=1)["ok"]


def test_standard_bridge_refuses_native_calls(tmp_path):
    b = br.Bridge(br.FakeEnv(), max_steps=3, settle=0, pause=0)
    b.reset(TASK, tmp_path / "r")
    assert b.state()["track"] == "standard"
    assert not call(b, "click", index=1)["ok"]
    assert b.act({"type": "click", "x": 1, "y": 1})["ok"]                                  # unchanged


def test_records(tmp_path):
    b, g = make_bridge(tmp_path, max_steps=3, record=True)
    call(b, "get_screen_controls", app_name="Notes")
    call(b, "press_control", index=2)
    run = tmp_path / "run_1"
    calls = [json.loads(x) for x in (run / "native_calls.jsonl").read_text().splitlines()]
    traj = [json.loads(x) for x in (run / "traj.jsonl").read_text().splitlines()]
    assert [c["tool"] for c in calls] == ["get_screen_controls", "press_control"] and calls[1]["cost"] == 1
    assert len(traj) == 1 and traj[0]["action"] == "otto:press_control" and traj[0]["screenshot_file"]
    assert (run / traj[0]["screenshot_file"]).is_file()


# ── the guest tool server ───────────────────────────────────────────────────

class FakeTool:
    def __init__(self, name, fn=None):
        self.name, self.fn, self.calls = name, fn, []

    async def ainvoke(self, args):
        self.calls.append(args)
        if self.fn:
            return self.fn(args)
        return f"{self.name} ran"


class FakeKit:
    built: list = []

    def __init__(self, cfg):
        FakeKit.built.append(cfg)
        self.vision_mode = False
        self.tools = [FakeTool("press_control"), FakeTool("get_screen_controls"), FakeTool("batch_actions"),
                      FakeTool("click", lambda a: (_ for _ in ()).throw(RuntimeError("no such index")))]
        self.vision_tools = [FakeTool("capture_app_screenshot",
                                      lambda a: [{"type": "image", "base64": "AA==", "mime_type": "image/png"}])]
        self.read_screen_vision = FakeTool("read_screen")


@pytest.fixture()
def guest():
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    FakeKit.built = []
    yield gs.GuestTools(FakeKit, parse_batch(), loop)
    loop.call_soon_threadsafe(loop.stop)


def test_guest_runs_tools_with_the_hosts_config_and_vision_mode(guest):
    r = guest.call({"tool_id": "tools:press_control", "tool": "press_control", "args": {"index": 2},
                    "config": {"a": 1}, "vision_mode": True, "budget": 5})
    assert r == {"ok": True, "cost": 1, "result": "press_control ran"} and guest.toolkit.vision_mode is True
    guest.call({"tool_id": "tools:get_screen_controls", "tool": "get_screen_controls", "args": {}, "config": {"a": 1},
                "budget": 0})
    assert FakeKit.built == [{"a": 1}]                                       # same config: same toolkit (indices kept)
    guest.call({"tool_id": "tools:get_screen_controls", "tool": "get_screen_controls", "args": {}, "config": {"a": 2},
                "budget": 0})
    assert FakeKit.built == [{"a": 1}, {"a": 2}]


def test_guest_refuses_over_budget_unknown_and_mismatched_tools(guest):
    base = {"config": {}, "args": {}}
    r = guest.call({**base, "tool_id": "tools:press_control", "tool": "press_control", "budget": 0})
    assert r["refused"] and not guest.toolkit.tools[0].calls
    steps = "press_control(index=1)\npress_control(index=2)"
    r = guest.call({**base, "tool_id": "tools:batch_actions", "tool": "batch_actions", "args": {"steps": steps},
                    "budget": 1})
    assert r["refused"] and r["cost"] == 2
    assert guest.call({**base, "tool_id": "tools:nope", "tool": "nope", "budget": 9})["refused"]
    assert guest.call({**base, "tool_id": "tools:click", "tool": "press_control", "budget": 9})["refused"]


def test_guest_tool_errors_and_content_blocks(guest):
    r = guest.call({"tool_id": "tools:click", "tool": "click", "args": {"index": 9}, "config": {}, "budget": 3})
    assert not r["ok"] and r["cost"] == 1 and "no such index" in r["error"]
    r = guest.call({"tool_id": "vision:capture_app_screenshot", "tool": "capture_app_screenshot", "args": {},
                    "config": {}, "budget": 0})
    assert r["ok"] and r["result"][0]["type"] == "image"
    r = guest.call({"tool_id": "read_screen_vision", "tool": "read_screen", "args": {}, "config": {}, "budget": 0})
    assert r["ok"] and r["cost"] == 0


def test_run_loop_serves_calls_and_turns_the_cocoa_run_loop():
    """Without the pump the guest's app list goes stale (seen on a real VM); calls must still run meanwhile."""
    loop = asyncio.new_event_loop()
    stop, turns = threading.Event(), []
    t = threading.Thread(target=gs.run_loop, args=(loop, lambda s: turns.append(s), stop), daemon=True)
    t.start()
    try:
        fut = asyncio.run_coroutine_threadsafe(asyncio.sleep(0.05, result="ran"), loop)
        assert fut.result(timeout=5) == "ran"
        import time
        deadline = time.time() + 5
        while len(turns) < 3 and time.time() < deadline:
            time.sleep(0.01)
        assert len(turns) >= 3
    finally:
        stop.set()
        t.join(timeout=5)
    assert not t.is_alive()


def test_run_loop_without_a_pump_is_plain_asyncio():
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=gs.run_loop, args=(loop, None), daemon=True)
    t.start()
    try:
        assert asyncio.run_coroutine_threadsafe(asyncio.sleep(0, result=1), loop).result(timeout=5) == 1
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=5)


def test_guest_http_needs_the_token(guest):
    import httpx
    srv = ThreadingHTTPServer(("127.0.0.1", 0), gs.make_handler(guest, "T" * 32, lambda: {"ok": True, "trusted": True}))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert httpx.get(url + "/health").status_code == 401
        assert httpx.post(url + "/call", json={}, headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert httpx.get(url + "/health", headers={"Authorization": "Bearer " + "T" * 32}).json()["trusted"]
    finally:
        srv.shutdown()


# ── host side: the remote toolkit inside Otto's own loader ──────────────────

@pytest.fixture()
def native_served(tmp_path, monkeypatch):
    taskfile = tmp_path / "task.json"
    taskfile.write_text(json.dumps(TASK))
    g = br.FakeGuest()
    b = br.Bridge(br.FakeEnv(), max_steps=3, settle=0, pause=0, guest=g)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), br.make_handler(b, "AGENT", "CONTROL",
                                                               lambda p: json.loads(Path(p).read_text())))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    monkeypatch.setenv("MACARENA_BRIDGE_URL", url)
    monkeypatch.setenv("MACARENA_AGENT_TOKEN", "AGENT")
    monkeypatch.setenv("OTTO_DESKTOP_LOCK_PATH", str(tmp_path / "desktop.lock"))
    b.reset(TASK, tmp_path / "run_1")
    yield b, g
    srv.shutdown()


@pytest.fixture()
def installed(monkeypatch):
    """install(), undone after the test: the patched names and ACTIVE are restored."""
    import tools.navigation as nav
    import tools.navigation.computer as comp
    from backend.mcp_manager import MCPManager
    monkeypatch.setattr(comp, "MacOSToolkit", comp.MacOSToolkit)
    monkeypatch.setattr(nav, "MacOSToolkit", nav.MacOSToolkit)
    monkeypatch.setattr(MCPManager, "_create_helper", MCPManager._create_helper)
    monkeypatch.setattr(macarena_native, "ACTIVE", False)
    return macarena_native.install()


def load_native_tools():
    """Otto's own loader for the macos-native connection, unmodified."""
    from backend.config import MCPServerConfig
    from backend.mcp_manager import MCPManager
    conn = types.SimpleNamespace(_loop_guard=None)
    cfg = MCPServerConfig(id="macos-native", name="macOS Desktop", transport="stdio", builtin=True, enabled=True)
    tools = asyncio.run(MCPManager._load_builtin_macos_native(None, cfg, conn))
    return tools, conn


def test_install_makes_ottos_loader_build_remote_tools(installed):
    import tools.navigation.computer as comp
    assert comp.MacOSToolkit is installed and macarena_native.ACTIVE
    tools, conn = load_native_tools()
    local = {t.name: t for t in real_toolkit().tools}
    assert {t.name for t in tools} == set(local)
    for t in tools:
        assert t.metadata.get("macarena_remote") is True
        assert t.description == local[t.name].description                     # Otto's tool, as the app has it
        assert t.args_schema is local[t.name].args_schema
    assert conn.macos_native_vision["read_screen"].metadata["macarena_remote"]
    assert all(t.metadata["macarena_remote"] for t in conn.macos_native_vision["extra"])


def test_remote_tool_call_goes_through_the_bridge_with_ottos_wrappers(installed, native_served):
    b, g = native_served
    tools, conn = load_native_tools()
    by = {t.name: t for t in tools}
    out = asyncio.run(by["press_control"].ainvoke({"index": 7}))
    assert "(fake VM) press_control ok" in out and "[benchmark] steps used 1, steps left 2" in out
    body = g.calls[-1]
    assert body["tool_id"] == "tools:press_control" and body["args"] == {"index": 7}
    host_kit = conn.macos_native_vision["toolkit"]
    assert body["config"] == host_kit.remote_config                        # the host's resolved toolkit settings
    assert set(body["config"]) == set(macarena_native.CONFIG_FIELDS) and body["vision_mode"] is False
    conn.macos_native_vision["toolkit"].vision_mode = True
    asyncio.run(by["get_screen_controls"].ainvoke({"app_name": "Notes"}))
    assert g.calls[-1]["vision_mode"] is True


def test_remote_error_becomes_a_tool_error_not_a_host_action(installed, native_served, monkeypatch):
    b, g = native_served
    tools, _ = load_native_tools()
    by = {t.name: t for t in tools}
    b.finished = "done"
    out = asyncio.run(by["click"].ainvoke({"index": 1}))
    assert "run is over" in str(out) and len(g.calls) == 0
    monkeypatch.setenv("MACARENA_AGENT_TOKEN", "")
    monkeypatch.setenv("MACARENA_BRIDGE_FILE", "/nonexistent")
    out = asyncio.run(by["click"].ainvoke({"index": 1}))
    assert "not configured" in str(out)


# ── guard, registration, run.py, report ─────────────────────────────────────

def test_guard_for_the_desktop_agent(monkeypatch):
    from backend.macarena_guard_middleware import refusal
    remote = types.SimpleNamespace(metadata={"macarena_remote": True})
    local = types.SimpleNamespace(metadata={})
    A = "macarena-desktop-agent"
    monkeypatch.setattr(macarena_native, "ACTIVE", True)
    assert refusal(A, "press_control", remote) is None
    assert refusal(A, "press_control", local)                       # a host-side tool of the same name: refused
    assert refusal(A, "press_control", None)
    assert refusal(A, "vm_done") is None and refusal(A, "vm_fail") is None
    assert refusal(A, "run_osascript", remote) is None               # the VM-backed AppleScript tool
    assert refusal(A, "run_osascript", local)                       # the host's own osascript server: refused
    for t in ("execute", "write_file", "read_file", "task", "vm_click", "vm_screenshot", "run_osascript_file",
              "query_outlook_store"):
        assert refusal(A, t, remote)
    monkeypatch.setattr(macarena_native, "ACTIVE", False)
    assert refusal(A, "press_control", remote)                      # swap not installed: never run
    assert refusal("macos-desktop-agent", "press_control", local) is None       # the app's agent: untouched
    assert refusal("macarena-agent", "vm_click") is None and refusal("macarena-agent", "press_control", remote)


def test_desktop_agent_is_ottos_desktop_agent_plus_benchmark_facts():
    from backend import agent_library, session_manager
    P = agent_library._MACARENA_DESKTOP_AGENT_PROMPT
    # Built from the app agent's own sections (not a copy), so the shared text stays identical.
    assert P.startswith(agent_library._macos_agent_prompt(scripts=True, files=False))
    for sec in (agent_library._MACOS_SEC_ROUTE, agent_library._MACOS_SEC_UI_PATH, agent_library._MACOS_SEC_HONESTY,
                agent_library._MACOS_SEC_COMPLETION):
        assert sec in P and sec in agent_library._MACOS_DESKTOP_AGENT_PROMPT
    # The script path, minus the tool that reads host files (not in the VM set).
    assert "## Script path" in P and "`run_osascript`" in P and "run_osascript_file" not in P
    assert "-1743" in P and "do that part with the desktop tools" in P       # nobody can grant permission
    assert "vm_done" in P
    assert "macarena-desktop-agent" in agent_library._BUILTIN_AGENT_VERSIONS
    assert session_manager._AGENT_TOOL_ALLOWLIST["macarena-desktop-agent"] == macarena_native.AGENT_TOOLS
    assert "macarena-desktop-agent" in session_manager._AGENTS_WITHOUT_MEMORY
    assert "macarena-desktop-agent" in session_manager._AGENTS_WITHOUT_SUBAGENTS
    assert "macos-desktop-agent" not in session_manager._AGENT_TOOL_ALLOWLIST      # the app's agent: untouched


def test_seed_lists_the_desktop_tools_and_finish_tools():
    src = (ROOT / "backend" / "agent_library.py").read_text()
    i = src.index('name="macarena-desktop-agent"')
    assert 'tools=["macos-native", "macos-osascript", "macarena-vm"]' in src[i:i + 600]


def test_run_prompt_and_agent_per_track():
    task = {"instruction": "Rename the file"}
    assert runpy_.AGENTS == {"standard": "macarena-agent", "native": "macarena-desktop-agent"}
    p = runpy_.prompt_for(task, 15, "native")
    assert "Rename the file" in p and "15" in p and "vm_done" in p and "vm_screenshot" not in p
    assert "vm_screenshot" in runpy_.prompt_for(task, 15)                        # standard: unchanged


def test_report_labels_the_track_and_refuses_mixing():
    rows = [{"id": "a", "run": 1, "score": 1.0, "status": "ok", "source": "macarena", "domain": "d", "steps": 3,
             "track": "native"}]
    s = report.summarize(rows)
    assert s["track"] == "native" and "NOT comparable" in report.render(s)
    assert report.summarize([{k: v for k, v in rows[0].items() if k != "track"}])["track"] == "standard"
    with pytest.raises(ValueError):
        report.summarize(rows + [{**rows[0], "id": "b", "track": "standard"}])


def test_no_screenshot_is_taken_by_default(tmp_path):
    b, g = make_bridge(tmp_path, max_steps=3)
    taken = []
    b.env.controller.get_screenshot = lambda: taken.append(1) or b""
    call(b, "press_control", index=2)
    run = tmp_path / "run_1"
    assert not taken and not list(run.glob("*.png"))
    assert json.loads((run / "traj.jsonl").read_text().splitlines()[0])["screenshot_file"] is None
    assert (run / "native_calls.jsonl").exists()


# ── the model is not offered tools it may not use ───────────────────────────

def _offered(agent_name):
    """Tool names and prompt headings the model gets from a real deepagents agent with the guard attached."""
    import tempfile
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from backend.macarena_guard_middleware import MacarenaGuardMiddleware

    class Fake(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self

    seen = {}

    class Rec(AgentMiddleware):                     # innermost: sees what the model would be given
        def wrap_model_call(self, request, handler):
            seen["tools"] = [getattr(t, "name", None) or t.get("name") for t in request.tools]
            seen["prompt"] = "\n".join(str(b.get("text", "")) for b in request.system_message.content_blocks)
            return handler(request)

    @tool
    def press_control(index: int) -> str:
        """press"""
        return "ok"

    @tool
    def vm_done(summary: str = "") -> str:
        """done"""
        return "ok"

    agent = create_deep_agent(
        model=Fake(messages=iter([AIMessage(content="hi")])), tools=[press_control, vm_done], system_prompt="SYS",
        middleware=[MacarenaGuardMiddleware(agent_name), Rec()],
        backend=FilesystemBackend(root_dir=tempfile.mkdtemp(), virtual_mode=True))
    agent.invoke({"messages": [("user", "x")]})
    return seen


def test_native_agent_is_offered_only_desktop_and_finish_tools():
    seen = _offered("macarena-desktop-agent")
    assert sorted(seen["tools"]) == ["press_control", "vm_done", "write_todos"]   # planning list kept
    assert "## `write_todos`" in seen["prompt"]
    for gone in ("Following Conventions", "`task`", "read_file", "execute"):
        assert gone not in seen["prompt"], gone
    assert "SYS" in seen["prompt"]                                      # its own prompt stays


def test_other_agents_keep_every_tool():
    seen = _offered("some-other-agent")
    assert {"write_todos", "ls", "read_file", "write_file", "glob", "press_control"} <= set(seen["tools"])
    assert "Following Conventions" in seen["prompt"]


def test_standard_track_agent_is_offered_only_vm_tools():
    seen = _offered("macarena-agent")                      # paper-comparable track: no planning tool either
    assert not set(seen["tools"]) & {"ls", "write_file", "write_todos", "glob", "task"}
    assert "## `write_todos`" not in seen["prompt"]


def test_write_todos_is_allowed_only_for_the_native_agent():
    from backend.macarena_guard_middleware import refusal
    assert refusal("macarena-desktop-agent", "write_todos") is None
    assert refusal("macarena-agent", "write_todos") is not None
    assert refusal("macarena-desktop-agent", "execute") is not None


def test_benchmark_prompt_does_not_point_at_file_or_shell_tools():
    from backend.agent_library import _MACARENA_DESKTOP_AGENT_PROMPT as P, _MACOS_DESKTOP_AGENT_PROMPT as app
    assert "SESSION_FILES" not in P and "## Files" not in P and "Use `execute`" not in P
    # It never names the host's `execute` (runs showed the name alone prompted calls) and offers no shell tool.
    assert "`execute`" not in P and "vm_shell" not in P and "`do shell script`" in P
    assert "`execute`" in app and "## Files" in app                      # the app's own agent is unchanged
    assert "## Completion" in P and "## Benchmark run (MacArena)" in P


def test_benchmark_prompts_do_not_claim_impossible_tasks():
    """MacArena has no infeasible tasks and its evaluator never rewards vm_fail, so the prompts must not say so."""
    from backend.agent_library import _MACARENA_AGENT_PROMPT as standard, _MACARENA_DESKTOP_AGENT_PROMPT as native
    for p in (standard, native):
        assert "impossible on purpose" not in p and "only pass if you say so" not in p
        assert "`vm_fail` earns no credit" in p


def test_native_no_step_limit_costs_steps_but_never_runs_out(tmp_path):
    b, g = make_bridge(tmp_path, max_steps=0)
    for i in range(25):
        r = call(b, "press_control", index=i)
        assert r["ok"] and r["steps_left"] is None and r["finished"] is None
    assert b.state()["steps"] == 25
    assert macarena_native.budget_note({"steps_used": 25, "steps_left": None, "finished": None}).endswith("no step limit")


def test_native_prompt_guides_the_agent_without_leaking_task_details():
    from backend.agent_library import _MACARENA_DESKTOP_AGENT_PROMPT as P
    for needle in ("`do shell script`", "write_todos", "Verify, then stop", "Do not repeat what failed",
                   "Do not conclude that something is missing", "make no more tool calls", "stand-ins",
                   "A failed check is not a pass", "Use the thing the task names",
                   "change the kind of approach", "Operation not permitted",
                   "Use a shell only", "call `vm_done` at once",
                   "Never read a table one cell at a time", "`find_text_on_screen`",
                   "Never estimate a position from a picture"):
        assert needle in P, needle
    for task_specific in ("vault", "Obsidian", "Reminders", "Keynote", ".tar", "Important Documents", "Anaconda",
                          "Calculator", "Automator", "PNG"):
        assert task_specific not in P, task_specific          # general advice only; nothing taken from a checker
    assert "no shell" not in P and "impossible on purpose" not in P


# ── AppleScript tools: the macos-osascript server's functions, run in the VM ─

def load_osascript_tools():
    """Otto's own generic MCP loader for the macos-osascript connection (helper swapped by install())."""
    from backend.config import MCPServerConfig
    from backend.mcp_manager import MCPConnection, MCPManager
    cfg = MCPServerConfig(id="macos-osascript", name="macOS osascript", transport="stdio", command="never-run",
                          builtin=True, enabled=True)
    return asyncio.run(MCPManager()._load_generic(cfg, MCPConnection(config=cfg)))


def test_install_loads_osascript_tools_from_the_servers_own_schemas(installed):
    from backend.builtin_mcps.macos_osascript import server as osa
    specs = {s.name: s for s in asyncio.run(osa.mcp.list_tools())}
    tools = load_osascript_tools()
    assert {t.name for t in tools} == set(osa.VM_TOOLS)                    # no host-file tools
    for t in tools:
        assert t.metadata.get("macarena_remote") is True
        assert t.description == specs[t.name].description                  # Otto's tool, as the app has it
        assert t.args_schema == specs[t.name].inputSchema
        assert t.response_format == "content_and_artifact"


def test_remote_osascript_call_goes_through_the_bridge(installed, native_served):
    b, g = native_served
    by = {t.name: t for t in load_osascript_tools()}
    script = 'tell application "Finder" to get name of front window'
    out = asyncio.run(by["run_osascript"].ainvoke({"script": script}))
    assert "(fake VM) run_osascript ok" in out and "[benchmark] steps used 1, steps left 2" in out
    body = g.calls[-1]
    assert body["tool_id"] == "osascript:run_osascript" and body["args"]["script"] == script
    out = asyncio.run(by["inspect_app_dictionary"].ainvoke({"app_name": "Finder"}))
    assert "steps used 1," in out                                           # a read: free
    assert b.state()["steps"] == 1


def test_remote_osascript_result_dict_becomes_the_servers_json_text(monkeypatch):
    async def fake_post(tool_id, name, args, config, vision_mode):
        assert tool_id == "osascript:inspect_app_dictionary" and config == {} and vision_mode is False
        return {"ok": True, "has_app_specific_suite": False}, "[benchmark] steps used 0, steps left 3"
    monkeypatch.setattr(macarena_native, "_post", fake_post)
    text, artifact = asyncio.run(macarena_native.forward_osascript("inspect_app_dictionary", {"app_name": "X"}))
    assert artifact is None
    head, note = text.split("\n\n[benchmark]")
    assert json.loads(head) == {"ok": True, "has_app_specific_suite": False}


@pytest.fixture()
def osa_guest():
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    calls: list = []

    async def run_osascript(script: str, language: str = "AppleScript", timeout_seconds: int = 30):
        calls.append(("run_osascript", script))
        return {"ok": True, "stdout": "done\n"}

    async def inspect_app_dictionary(app_name: str):
        calls.append(("inspect_app_dictionary", app_name))
        return {"ok": True, "has_app_specific_suite": True}

    FakeKit.built = []
    g = gs.GuestTools(FakeKit, parse_batch(), loop,
                      {"run_osascript": run_osascript, "inspect_app_dictionary": inspect_app_dictionary})
    yield g, calls
    loop.call_soon_threadsafe(loop.stop)


def test_guest_runs_osascript_functions_and_counts_steps(osa_guest):
    g, calls = osa_guest
    r = g.call({"tool_id": "osascript:run_osascript", "tool": "run_osascript", "args": {"script": "x"}, "budget": 1})
    assert r == {"ok": True, "cost": 1, "result": {"ok": True, "stdout": "done\n"}}
    r = g.call({"tool_id": "osascript:inspect_app_dictionary", "tool": "inspect_app_dictionary",
                "args": {"app_name": "Mail"}, "budget": 0})
    assert r["ok"] and r["cost"] == 0
    assert calls == [("run_osascript", "x"), ("inspect_app_dictionary", "Mail")]
    assert FakeKit.built == []                                    # never builds (or rebuilds) the desktop toolkit


def test_guest_refuses_osascript_over_budget_unknown_and_bad_args(osa_guest):
    g, calls = osa_guest
    r = g.call({"tool_id": "osascript:run_osascript", "tool": "run_osascript", "args": {"script": "x"}, "budget": 0})
    assert r["refused"] and r["cost"] == 1
    r = g.call({"tool_id": "osascript:run_osascript_file", "tool": "run_osascript_file", "args": {}, "budget": 5})
    assert r["refused"]                                           # not in the VM set
    r = g.call({"tool_id": "osascript:run_osascript", "tool": "inspect_app_dictionary", "args": {}, "budget": 5})
    assert r["refused"]                                           # id and name disagree
    r = g.call({"tool_id": "osascript:run_osascript", "tool": "run_osascript", "args": {"nope": 1}, "budget": 5})
    assert not r["ok"] and "TypeError" in r["error"]
    assert calls == []


def test_guest_automation_probe(monkeypatch):
    import subprocess
    monkeypatch.setattr(gs, "_AUTOMATION", None)
    ok = lambda *a, **k: subprocess.CompletedProcess(a[0], 0, "Finder\n", "")
    assert gs.automation(ok) == {"automation": True}
    assert gs.automation(lambda *a, **k: pytest.fail("a definite answer is kept")) == {"automation": True}
    monkeypatch.setattr(gs, "_AUTOMATION", None)
    denied = lambda *a, **k: subprocess.CompletedProcess(a[0], 1, "", "execution error: Not authorized (-1743)")
    r = gs.automation(denied)
    assert r["automation"] is False and "-1743" in r["automation_error"]
    monkeypatch.setattr(gs, "_AUTOMATION", None)

    def hangs(*a, **k):
        raise subprocess.TimeoutExpired(a[0], 15)
    assert gs.automation(hangs)["automation"] is None and gs._AUTOMATION is None          # asked again next time


def test_osascript_server_imports_without_the_mcp_sdk():
    """The VM has no MCP SDK: the guest imports the server module the way the bundle lays it out."""
    import subprocess
    code = (
        "import builtins, sys, asyncio\n"
        "real = builtins.__import__\n"
        "def imp(name, *a, **k):\n"
        "    if name == 'mcp' or name.startswith('mcp.'):\n"
        "        raise ImportError(name)\n"
        "    return real(name, *a, **k)\n"
        "builtins.__import__ = imp\n"
        f"sys.path.insert(0, {str(ROOT / 'backend' / 'builtin_mcps')!r})\n"
        "from macos_osascript import server as s\n"
        "print(type(s.mcp).__name__, all(asyncio.iscoroutinefunction(getattr(s, n)) for n in s.VM_TOOLS))\n"
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60)
    assert out.stdout.strip() == "_NoServer True", out.stderr


def test_bundle_ships_the_osascript_module():
    sh = (MA / "bench.sh").read_text()
    assert 'backend/builtin_mcps/macos_osascript/{server,_helpers,_desktop_lock}.py "$st/src/macos_osascript/"' in sh
    src = (MA / "bridge.py").read_text()
    assert "OTTO_DESKTOP_LEASE_BY_HOST=1 PYTHONPATH=" in src


# ── harness changes from the run analysis ───────────────────────────────────

def test_host_sections_are_dropped_for_benchmark_agents_only():
    from langchain_core.messages import SystemMessage
    from backend.macarena_guard_middleware import drop_sections, strip_hidden_prompt
    host = "/Users/someone/git/otto-macarena-runs/home/Library/Application Support/Otto/sessions/x/files"
    prompt = ("# macOS Agent\nDo the task.\n\n## Ambient Scheduling\nOffer a schedule.\n\n"
              f"## Session Context\n- Save under `/output/`.\n- The real session files directory is `{host}`.\n\n"
              "## Data Integrity (NON-NEGOTIABLE)\n- Report only what tools returned.\n\n"
              "## Parallel Subagents & Output Files\n- Distinct paths.\n\n"
              "## When the user says a fix did not work\n- Re-read.\n\n## Benchmark run (MacArena)\n- Rules.\n")
    out = drop_sections(prompt)
    assert host not in out and "/output/" not in out and "schedule" not in out and "Re-read" not in out
    assert "Do the task." in out and "## Data Integrity" in out and "## Benchmark run (MacArena)" in out
    msg = strip_hidden_prompt(SystemMessage(content_blocks=[{"type": "text", "text": prompt}]))
    assert host not in msg.content_blocks[0]["text"]
    same = SystemMessage(content_blocks=[{"type": "text", "text": "no host sections here"}])
    assert strip_hidden_prompt(same) is same                         # nothing to drop: the message is untouched


def test_session_context_reaches_other_agents_but_not_the_benchmark_agent():
    def offered(agent):
        import tempfile
        from deepagents import create_deep_agent
        from deepagents.backends import FilesystemBackend
        from langchain.agents.middleware import AgentMiddleware
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langchain_core.messages import AIMessage
        from backend.macarena_guard_middleware import MacarenaGuardMiddleware

        class Fake(GenericFakeChatModel):
            def bind_tools(self, tools, **kw):
                return self
        seen = {}

        class Rec(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                seen["prompt"] = "\n".join(str(b.get("text", "")) for b in request.system_message.content_blocks)
                return handler(request)
        agent_ = create_deep_agent(model=Fake(messages=iter([AIMessage(content="hi")])), tools=[],
                                   system_prompt="SYS\n\n## Session Context\n- files at /Users/host/x\n\n## Keep\nk",
                                   middleware=[MacarenaGuardMiddleware(agent), Rec()],
                                   backend=FilesystemBackend(root_dir=tempfile.mkdtemp(), virtual_mode=True))
        agent_.invoke({"messages": [("user", "x")]})
        return seen["prompt"]
    assert "/Users/host/x" not in offered("macarena-desktop-agent") and "## Keep" in offered("macarena-desktop-agent")
    assert "/Users/host/x" not in offered("macarena-agent")
    assert "/Users/host/x" in offered("some-other-agent")            # the app's agents are untouched


def test_no_tools_are_offered_once_the_run_is_finished():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from backend.macarena_guard_middleware import run_finished
    call = AIMessage(content="", tool_calls=[{"name": "vm_done", "args": {}, "id": "c1"}])
    ok = ToolMessage(content="Marked done. Give your final answer now.", name="vm_done", tool_call_id="c1")
    err = ToolMessage(content="Error: no task is running", name="vm_done", tool_call_id="c1")
    other = ToolMessage(content="Marked done.", name="press_control", tool_call_id="c2")
    assert run_finished([HumanMessage("x"), call, ok])
    assert not run_finished([HumanMessage("x"), call, err])          # refused finish: keep the tools
    assert not run_finished([other]) and not run_finished([])
    fail = ToolMessage(content=[{"type": "text", "text": "Marked as infeasible. Give your final answer now."}],
                       name="vm_fail", tool_call_id="c3")
    assert run_finished([fail])


def test_guard_offers_no_tools_after_finish_for_benchmark_agents_only():
    from types import SimpleNamespace
    from langchain_core.messages import SystemMessage, ToolMessage
    from backend.macarena_guard_middleware import MacarenaGuardMiddleware
    done = ToolMessage(content="Marked done. Give your final answer now.", name="vm_done", tool_call_id="c1")

    class Req(SimpleNamespace):
        def override(self, **kw):
            return Req(**{**self.__dict__, **kw})
    tools = [SimpleNamespace(name="press_control"), SimpleNamespace(name="vm_done"), SimpleNamespace(name="ls")]
    req = Req(tools=tools, messages=[done], system_message=SystemMessage(content="SYS"))
    assert MacarenaGuardMiddleware("macarena-desktop-agent")._offer(req).tools == []
    before = Req(tools=tools, messages=[], system_message=SystemMessage(content="SYS"))
    assert [t.name for t in MacarenaGuardMiddleware("macarena-desktop-agent")._offer(before).tools] == [
        "press_control", "vm_done"]
    assert MacarenaGuardMiddleware("some-other-agent")._offer(req).tools == tools


def test_execute_refusal_points_at_applescript_and_not_at_vm_fail():
    from backend.macarena_guard_middleware import refusal
    msg = refusal("macarena-desktop-agent", "execute")
    assert "do shell script" in msg and "vm_fail" not in msg and "impossible" not in msg
    assert refusal("macarena-desktop-agent", "vm_shell")            # removed: refused like any unknown tool
    assert refusal("macarena-agent", "vm_shell")


def fail_osascript(b):
    return b.native({"tool": "run_osascript", "tool_id": "osascript:run_osascript", "args": {"script": "x"}})


def test_failure_streak_notice_after_four_failed_actions(tmp_path):
    b, g = make_bridge(tmp_path, max_steps=0, repeat_notice_after=0)      # the same call each time: no repeat notice
    real = g.call
    g.call = lambda body: ({"ok": True, "cost": 1, "result": {"ok": False, "stderr": "Can't get group 5 (-1728)"}}
                           if body["tool"] == "run_osascript" else real(body))
    for _ in range(3):
        assert "notice" not in fail_osascript(b)
    r = fail_osascript(b)
    assert "The last 4 actions failed (run_osascript)" in r["notice"] and "different kind" in r["notice"]
    assert "notice" not in fail_osascript(b)                          # every 4th, not every call after
    assert call(b, "list_apps")["ok"] and "notice" not in call(b, "list_apps")   # reads neither count nor reset
    for _ in range(3):                                                # failures 6, 7, 8
        r = fail_osascript(b)
    assert "The last 8 actions failed" in r["notice"] and b.state()["fail_notices"] == 2
    assert call(b, "press_control", index=1)["ok"]                    # a success resets the streak
    for _ in range(3):
        assert "notice" not in fail_osascript(b)
    assert "The last 4 actions failed" in fail_osascript(b)["notice"]  # counted from the reset, not from 8


def test_failure_streak_counts_tool_errors_and_can_be_turned_off(tmp_path):
    b, g = make_bridge(tmp_path, max_steps=0, fail_notice_after=0, repeat_notice_after=0)
    g.call = lambda body: {"ok": False, "cost": 1, "error": "no such index"}
    assert all("notice" not in call(b, "click", index=9) for _ in range(9))
    b2, g2 = make_bridge(tmp_path / "b2", max_steps=0, fail_notice_after=2, repeat_notice_after=0)
    g2.call = lambda body: {"ok": False, "cost": 1, "error": "no such index"}
    call(b2, "click", index=9)
    assert "(click)" in call(b2, "click", index=9)["notice"]


def test_observe_after_is_requested_only_for_acting_tools_and_counts_a_read(tmp_path):
    b, g = make_bridge(tmp_path, max_steps=0)
    r = call(b, "press_control", index=2)
    assert g.calls[-1]["observe_after"] is True and "Controls now (Fake)" in r["result"]
    assert b.state()["obs_calls"] == 1 and r["steps_used"] == 1     # the read is free but counted against the cap
    call(b, "get_screen_controls", app_name="Notes")
    assert g.calls[-1]["observe_after"] is False                      # reads never trigger another read
    call(b, "launch_app", app_name="Notes")
    assert g.calls[-1]["observe_after"] is False                      # already returns the screen
    assert b.state()["observe_after"] is True


def test_observe_after_off_and_at_the_read_cap(tmp_path):
    b, g = make_bridge(tmp_path, max_steps=0, observe_after=False)
    r = call(b, "press_control", index=2)
    assert g.calls[-1]["observe_after"] is False and "Controls now" not in r["result"]
    assert b.state()["observe_after"] is False
    b2, g2 = make_bridge(tmp_path / "b2", max_steps=5, max_obs=1)
    call(b2, "press_control", index=1)                                # uses the one read
    call(b2, "press_control", index=1)
    assert g2.calls[-1]["observe_after"] is False                     # cap reached: no extra read
    std = br.Bridge(br.FakeEnv(), max_steps=3, settle=0.0, pause=0.0)
    assert std.observe_after is False                                 # standard track: never


class ObsKit(FakeKit):
    front = "Notes"

    @staticmethod
    def _frontmost_app_name():
        return ObsKit.front

    def __init__(self, cfg):
        super().__init__(cfg)
        self.tools = [FakeTool("press_control"), FakeTool("list_apps"),
                      FakeTool("get_screen_controls", lambda a: f"[1] B'Save' in {a['app_name']}" + "x" * ObsKit.pad)]
    pad = 0


@pytest.fixture()
def obs_guest():
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    ObsKit.front, ObsKit.pad = "Notes", 0
    yield gs.GuestTools(ObsKit, parse_batch(), loop)
    loop.call_soon_threadsafe(loop.stop)


def gcall(g, tool, observe=True, **args):
    return g.call({"tool_id": f"tools:{tool}", "tool": tool, "args": args, "config": {}, "budget": 5,
                   "observe_after": observe})


def test_guest_appends_the_frontmost_apps_controls_after_an_action(obs_guest):
    r = gcall(obs_guest, "press_control", index=1)
    assert r["observed"] is True and r["result"].startswith("press_control ran")
    assert "Controls now (Notes), read after the action:\n[1] B'Save' in Notes" in r["result"]
    assert "observed" not in gcall(obs_guest, "press_control", observe=False, index=1)
    assert "observed" not in gcall(obs_guest, "list_apps")            # not an acting tool
    ObsKit.front = ""
    assert "observed" not in gcall(obs_guest, "press_control", index=1)   # no frontmost app: nothing added


def test_guest_cuts_a_long_control_list(obs_guest):
    ObsKit.pad = native_tools.OBSERVE_AFTER_MAX_CHARS * 2
    r = gcall(obs_guest, "press_control", index=1)
    assert len(r["result"]) < native_tools.OBSERVE_AFTER_MAX_CHARS + 400 and "role= or label=" in r["result"]


def test_append_text_keeps_content_blocks():
    assert gs.append_text("done", "more") == "done\n\nmore"
    blocks = [{"type": "image", "base64": "AA=="}]
    assert gs.append_text(blocks, "more") == [*blocks, {"type": "text", "text": "more"}]


def test_notice_reaches_the_agent_through_the_native_wrapper(installed, native_served):
    b, g = native_served
    b.max_steps, b.cap = 0, br.UNLIMITED
    g.call = lambda body: {"ok": False, "cost": 1, "error": "no such index"}
    b.fail_notice_after = 1
    with pytest.raises(Exception) as e:
        asyncio.run(macarena_native._post("tools:click", "click", {"index": 3}, {}, False))
    assert "The last 1 actions failed (click)" in str(e.value)


def test_host_sleep_turns_a_timeout_into_a_retried_infra_error():
    run = runpy_                     # loaded once above: loading it again puts its folder back on sys.path
    rec = run.mark_host_sleep({"agent_status": "timeout", "host_slept_s": 964.0, "status": "ok"})
    assert rec["status"] == "infra_error" and "slept 964 s" in rec["error"]
    kept = run.mark_host_sleep({"agent_status": "done", "host_slept_s": 964.0, "status": "ok"})
    assert kept["status"] == "ok"                                     # finished on its own: the score stands
    short = run.mark_host_sleep({"agent_status": "timeout", "host_slept_s": 20.0, "status": "ok"})
    assert short["status"] == "ok"
    assert isinstance(run.awake_clock(), float)


def test_bench_run_keeps_the_mac_awake():
    text = (MA / "bench.sh").read_text()
    run_block = text[text.index("  run) shift; tokens"):text.index("  report)")]
    assert 'awake="caffeinate -i -s"' in run_block and "exec $awake " in run_block
    assert "MACARENA_ALLOW_SLEEP" in run_block


def test_native_prompt_has_the_paths_and_observation_rules():
    from backend.agent_library import _MACARENA_DESKTOP_AGENT_PROMPT as P
    assert "Controls now" in P and "never a full `/Users/<name>/...` path" in P and "vm_shell" not in P


def test_vm_shell_is_gone_everywhere():
    from backend.builtin_mcps.macarena_vm import server as vm
    from backend.action_policy import catalog
    assert not hasattr(vm, "vm_shell") and not hasattr(br.Bridge, "shell")
    assert "vm_shell" not in macarena_native.AGENT_TOOLS
    assert ("macarena-vm", "vm_shell") not in catalog._SERVER_TOOLS


def test_report_refuses_to_pool_different_harness_settings_and_labels_them():
    base = {"id": "t", "source": "macarena", "domain": "d", "run": 1, "infeasible": False, "score": 1.0,
            "status": "ok", "steps": 3, "track": "native", "max_steps": 0}
    on = {**base, "observe_after": True, "fail_notice_after": 4}
    off = {**base, "id": "u", "observe_after": False, "fail_notice_after": 0}
    with pytest.raises(ValueError, match="harness settings"):
        report.summarize([on, off])
    s = report.summarize([on, {**on, "id": "v", "host_slept_s": 900.0}])
    text = report.render(s)
    assert "observe_after=on" in text and "failure notice after 4" in text and "host slept" in text
    assert "observe_after=off" in report.render(report.summarize([off]))   # old rows: shown as off


def test_benchmark_prompt_leaves_step_counting_to_the_task_message():
    from backend import agent_library as lib
    p = lib._MACARENA_DESKTOP_AGENT_PROMPT
    assert p.startswith(lib._macos_agent_prompt(scripts=True, files=False))   # shared macOS prompt untouched
    assert "free but limited" not in p and "says how steps are counted" in p
    assert "Unless the task message says that some tasks cannot be done" in p
    assert "No task here is meant to be impossible" in lib._MACARENA_AGENT_PROMPT  # standard track unchanged


def test_desktop_prompt_asks_for_a_picture_of_the_screen_only_for_the_native_track():
    from backend import agent_library as lib
    d, std = lib._MACARENA_DESKTOP_AGENT_PROMPT, lib._MACARENA_AGENT_PROMPT
    assert "Look first" in d and "capture_app_screenshot" in d
    assert "observe()" in d and "usually already open" in d and "app_guide(app)" in d
    assert "Look first" not in std and "capture_app_screenshot" not in std   # the standard track gets a screenshot per call


class DeltaKit(ObsKit):
    """A toolkit whose scan cache holds the listing, so the guest can send only what changed."""
    scan: list = []
    dialog = "Dialog: none"
    extra_tools: list = []

    def __init__(self, cfg):
        super().__init__(cfg)
        self.tools.append(FakeTool("observe", lambda a: f"App: Notes\n{DeltaKit.dialog}"))
        self.tools.extend(DeltaKit.extra_tools)

    def _cache_get(self, app):
        return [dict(c) for c in DeltaKit.scan], {}


@pytest.fixture()
def delta_guest():
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    ObsKit.front, ObsKit.pad = "Notes", 0
    DeltaKit.scan = [{"index": 1, "role": "B", "label": "Save"}, {"index": 2, "role": "TF", "label": "Title"},
                     {"role": "window", "label": "w", "_structural": True}]
    DeltaKit.dialog = "Dialog: none"
    DeltaKit.extra_tools = []
    yield gs.GuestTools(DeltaKit, parse_batch(), loop)
    loop.call_soon_threadsafe(loop.stop)


def test_guest_sends_only_what_changed_while_indices_hold(delta_guest):
    first = gcall(delta_guest, "press_control", index=1)["result"]
    assert "[1] B'Save' in Notes" in first                              # nothing seen before: the full list
    same = gcall(delta_guest, "press_control", index=1)["result"]
    assert "Same 2 controls as the last read" in same and "B'Save' in Notes" not in same
    DeltaKit.scan = DeltaKit.scan + [{"index": 3, "role": "ST", "label": "Saved"}]
    DeltaKit.dialog = "Dialog: sheet on 'Untitled' (buttons: OK)"
    grown = gcall(delta_guest, "press_control", index=1)["result"]
    assert "+ [3] ST 'Saved'" in grown and "Dialog: sheet on 'Untitled'" in grown
    DeltaKit.scan = [{"index": 1, "role": "ST", "label": "New"}, {"index": 2, "role": "B", "label": "Save"}]
    shifted = gcall(delta_guest, "press_control", index=1)["result"]
    assert "[1] B'Save' in Notes" in shifted                            # an index moved: the full list again


def test_deploy_swaps_macarena_ok_clicker_for_one_that_skips_app_dialogs(monkeypatch, tmp_path):
    """MacArena's allow_permission.py clicked OK in every app dialog (Format Cells closed in 0.25 s)."""
    assert "background only is true" in pc.SCRIPT and '"OK"' in pc.SCRIPT
    sent: list = []

    class R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"trusted": True, "automation": True}

    fake = types.SimpleNamespace(post=lambda url, **kw: (sent.append((url, kw.get("json") or kw.get("data"))), R())[1],
                                 get=lambda url, **kw: R(), RequestException=Exception)
    monkeypatch.setitem(sys.modules, "requests", fake)
    env = types.SimpleNamespace(controller=types.SimpleNamespace(http_server="http://vm:5000"))
    bundle = tmp_path / "b.tgz"
    bundle.write_bytes(b"x")
    g = br.GuestTools(env, bundle, "/py")
    monkeypatch.setattr(g, "_execute", lambda cmd: sent.append(("execute", cmd)) or {})
    g.deploy()
    kills = [c for u, c in sent if u == "execute" and "allow_permissio[n].py" in c[-1] and "dialog foun[d]" in c[-1]]
    launches = [c["command"] for u, c in sent if u.endswith("/setup/launch")]
    assert kills and "permission_clicker.py" in launches[0] and "otto_tools_server.py" in launches[1]


def test_guest_shows_the_dialog_instead_of_the_window_while_one_is_open(delta_guest):
    """Live: after fill_dialog the full listing of the sheet behind Format Cells (400 cells) buried the dialog."""
    DeltaKit.dialog = "Dialog: window 'Format Cells' (in front of the document) (buttons: OK)"
    DeltaKit.extra_tools = [FakeTool("read_dialog", lambda a: "Open: window 'Format Cells'\nButtons: [9]B 'OK'")]
    out = gcall(delta_guest, "press_control", index=1)["result"]
    assert "Dialog now (Notes), read after the action:\nOpen: window 'Format Cells'" in out
    assert "B'Save' in Notes" not in out
