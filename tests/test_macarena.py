"""Hermetic tests for the MacArena harness (scripts/macarena) and Otto's side of it.

No VM, no UTM, no MacArena checkout and no model: the environment is ``bridge.FakeEnv``. These tests check
the harness's own logic (action validation, step and observation caps, auth, scoring math, the tool guard,
and the MCP server talking to a live bridge). They do NOT show that Otto can operate a real macOS VM or
that coordinates line up on a real VM screen; see scripts/macarena/README.md "Status".
"""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MA = ROOT / "scripts" / "macarena"
sys.path.insert(0, str(MA))

import actions  # noqa: E402
import bridge as br  # noqa: E402
import preflight  # noqa: E402
import report  # noqa: E402

# Imported; now take scripts/macarena off sys.path so its run.py / report.py cannot shadow same-named modules
# of other suites (tests/test_swebench_docker_env.py imports scripts/swebench's ``run``).
sys.path[:] = [p for p in sys.path if p != str(MA)]


# ── actions ─────────────────────────────────────────────────────────────────

SIZE = (1280, 720)


def cmd(**a):
    return actions.to_command(a, SIZE)


def test_click_family_maps_to_pyautogui():
    assert cmd(type="click", x=10, y=20) == "pyautogui.click(10, 20, button='left')"
    assert cmd(type="click", x=10, y=20, button="right") == "pyautogui.click(10, 20, button='right')"
    assert cmd(type="right_click", x=1, y=2) == "pyautogui.click(1, 2, button='right')"
    assert cmd(type="double_click", x=5, y=6) == "pyautogui.doubleClick(5, 6)"
    assert cmd(type="move_to", x=5, y=6) == "pyautogui.moveTo(5, 6)"
    assert "dragTo(30, 40" in cmd(type="drag_to", from_x=1, from_y=2, x=30, y=40)


def test_terminal_actions():
    assert cmd(type="done") == "DONE" and cmd(type="fail") == "FAIL" and cmd(type="wait") == "WAIT"


def test_scroll_signs_and_position():
    assert cmd(type="scroll", amount=-3) == "pyautogui.scroll(-3)"
    assert cmd(type="scroll", amount=2, x=100, y=100) == "pyautogui.moveTo(100, 100); pyautogui.scroll(2)"


def test_coordinates_are_scaled_to_the_vm_screen():
    got = actions.to_command({"type": "click", "x": 1000, "y": 400}, (2000, 1000), (0.5, 0.5))
    assert got == "pyautogui.click(500, 200, button='left')"


@pytest.mark.parametrize("bad", [
    {"type": "click", "x": -1, "y": 5}, {"type": "click", "x": 1280, "y": 5}, {"type": "click", "x": "a", "y": 1},
    {"type": "click", "x": float("nan"), "y": 1}, {"type": "click", "x": 1}, {"type": "click", "x": 1, "y": 1, "button": "x"},
    {"type": "run", "command": "ls"}, {"type": "shell"}, {}, {"type": "type", "text": ""}, {"type": "type", "text": 5},
    {"type": "type", "text": "a" * 2001}, {"type": "press", "key": "launchmail"}, {"type": "press", "key": "ab"},
    {"type": "hotkey", "keys": []}, {"type": "hotkey", "keys": ["a"] * 5}, {"type": "scroll", "amount": 999},
])
def test_invalid_actions_are_refused(bad):
    with pytest.raises(actions.ActionError):
        actions.to_command(bad, SIZE)


def test_non_dict_refused():
    with pytest.raises(actions.ActionError):
        actions.to_command("rm -rf /", SIZE)


def test_typed_text_cannot_break_out_of_the_string():
    text = "x'); import os; os.system('id'); print('"
    c = cmd(type="type", text=text)
    # the whole payload must sit inside one valid string literal: evaluating the call's argument gives the text back
    assert c.startswith("pyautogui.typewrite(") and c.endswith(")")
    arg = c[len("pyautogui.typewrite("):-1]
    assert eval(arg, {"__builtins__": {}}) == text            # noqa: S307 — literal from repr(), builtins stripped


def test_typing_has_the_exact_shape_macarenas_less_than_fix_matches():
    """MacArena's env rewrites '<' (pyautogui types '>') only for `pyautogui.typewrite('...')`, with no other
    arguments. Using write() or interval= would silently skip that fix and type '>' instead of '<'."""
    import re
    for text in ("a<b", "plain", "it's", 'say "hi"'):
        c = cmd(type="type", text=text)
        assert re.fullmatch(r"pyautogui\.typewrite\((['\"])(.*?)\1\)", c), c


def test_hotkey_and_press_normalise_case():
    assert cmd(type="hotkey", keys=["Command", "C"]) == "pyautogui.hotkey('command', 'c')"
    assert cmd(type="press", key="Enter") == "pyautogui.press('enter')"


# ── bridge ──────────────────────────────────────────────────────────────────

TASK = {"id": "t1", "instruction": "do it", "evaluator": []}


def make_bridge(tmp_path, **kw) -> br.Bridge:
    kw.setdefault("settle", 0.0)
    kw.setdefault("pause", 0.0)
    b = br.Bridge(br.FakeEnv(), max_steps=kw.pop("max_steps", 3), **kw)
    b.reset(TASK, tmp_path / "run_1")
    return b


def test_step_budget_is_enforced_and_extra_actions_are_not_executed(tmp_path):
    b = make_bridge(tmp_path, max_steps=3)
    for i in range(3):
        r = b.act({"type": "click", "x": 1, "y": 1})
        assert r["ok"] and r["steps_used"] == i + 1
    assert b.finished == "budget"
    r = b.act({"type": "click", "x": 1, "y": 1})
    assert not r["ok"] and "run is over" in r["error"]
    assert len(b.env.actions) == 3                            # the 4th never reached the VM


def test_done_and_fail_end_the_run(tmp_path):
    b = make_bridge(tmp_path)
    assert b.act({"type": "done"})["finished"] == "done"
    assert not b.act({"type": "click", "x": 1, "y": 1})["ok"]
    b2 = make_bridge(tmp_path / "b")
    assert b2.act({"type": "fail"})["finished"] == "fail"


def test_invalid_action_uses_no_step_but_repeated_abuse_ends_the_run(tmp_path):
    b = make_bridge(tmp_path, max_steps=3)
    for _ in range(2):
        r = b.act({"type": "shell", "command": "ls"})
        assert not r["ok"] and r["steps_used"] == 0 and b.env.actions == []
    b.act({"type": "nope"})
    assert b.finished == "budget" and b.steps == 0           # a loop of bad calls cannot run forever


def test_observation_cap_and_free_reads(tmp_path):
    b = make_bridge(tmp_path, max_steps=2, max_obs=2)
    assert b.observe("screenshot")["ok"] and b.observe("screenshot")["ok"]
    assert b.steps == 0                                      # reads are free
    r = b.observe("screenshot")
    assert not r["ok"] and "limit" in r["error"]


def test_a11y_tree_off_by_default(tmp_path):
    b = make_bridge(tmp_path)
    assert not b.observe("tree")["ok"]
    b2 = make_bridge(tmp_path / "x", use_a11y=True)
    assert b2.observe("tree")["tree"]


def test_not_armed_refuses_everything(tmp_path):
    b = br.Bridge(br.FakeEnv(), settle=0, pause=0)
    assert not b.act({"type": "done"})["ok"] and not b.observe("screenshot")["ok"]


def test_screenshot_after_final_step_is_not_returned(tmp_path):
    b = make_bridge(tmp_path, max_steps=1)
    assert "screenshot" not in b.act({"type": "click", "x": 1, "y": 1})


def test_trajectory_is_written_in_macarena_layout(tmp_path):
    b = make_bridge(tmp_path, record=True)
    b.act({"type": "click", "x": 3, "y": 4})
    rows = [json.loads(l) for l in (tmp_path / "run_1" / "traj.jsonl").read_text().splitlines()]
    assert rows[0]["step_num"] == 1 and rows[0]["action"] == "pyautogui.click(3, 4, button='left')"
    assert list((tmp_path / "run_1").glob("step_1_*.png"))
    b.finish(0.5)
    assert (tmp_path / "run_1" / "result.txt").read_text().strip() == "0.5"


def test_coordinate_scale_from_vm_and_override(tmp_path):
    env = br.FakeEnv(size=(1280, 720))
    env.controller.get_vm_screen_size = lambda: {"width": 640, "height": 360}
    b = br.Bridge(env, settle=0, pause=0)
    b.reset(TASK, tmp_path / "r")
    assert b.scale == (0.5, 0.5)
    assert b.act({"type": "click", "x": 100, "y": 100})["action"] == "pyautogui.click(50, 50, button='left')"
    b2 = br.Bridge(br.FakeEnv(), settle=0, pause=0, coord_scale=(2.0, 2.0))
    b2.reset(TASK, tmp_path / "r2")
    assert b2.scale == (2.0, 2.0)


def test_macosworld_evaluator_error_tuple_is_not_a_score(tmp_path):
    b = make_bridge(tmp_path)
    b.env.evaluate = lambda: (0, '{"status": "error"}')
    r = b.evaluate()
    assert r["score"] == 0.0 and "error" in r["eval_error"]


def test_step_failure_still_counts(tmp_path):
    b = make_bridge(tmp_path, max_steps=2)
    b.env.step = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("vm gone"))
    r = b.act({"type": "click", "x": 1, "y": 1})
    assert not r["ok"] and "vm gone" in r["error"] and b.steps == 1


def test_reset_without_fresh_vm_closes_apps_instead(tmp_path):
    env = br.FakeEnv()
    calls = []
    env.controller.run_bash_script = lambda script, **k: calls.append(script) or {}
    b = br.Bridge(env, settle=0, pause=0)
    b.reset(TASK, tmp_path / "a", fresh=True)
    b.reset(TASK, tmp_path / "b", fresh=False)
    assert env.resets == 1 and calls and "killall" in calls[0]


# ── bridge over HTTP: auth and the MCP server ───────────────────────────────

@pytest.fixture()
def served(tmp_path):
    taskfile = tmp_path / "task.json"
    taskfile.write_text(json.dumps(TASK))
    b = br.Bridge(br.FakeEnv(), max_steps=3, settle=0, pause=0, vm_name="osworld")
    handler = br.make_handler(b, "AGENT", "CONTROL", lambda p: json.loads(Path(p).read_text()))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", b, taskfile, tmp_path
    srv.shutdown()


def _http(url, path, token, body=None):
    import httpx
    h = {"Authorization": f"Bearer {token}"}
    r = httpx.post(url + path, json=body, headers=h) if body is not None else httpx.get(url + path, headers=h)
    return r.status_code, r.json()


def test_tokens_are_separate(served):
    url, b, taskfile, tmp = served
    assert _http(url, "/ctl/state", "AGENT")[0] == 401                   # the agent cannot reach control
    assert _http(url, "/agent/observe", "CONTROL", {})[0] == 401         # and the controller is not the agent
    assert _http(url, "/ctl/state", "wrong")[0] == 401
    assert _http(url, "/ctl/state", "CONTROL")[1]["vm"] == "osworld"


def test_full_run_over_http(served):
    url, b, taskfile, tmp = served
    s, st = _http(url, "/ctl/reset", "CONTROL", {"task_path": str(taskfile), "run_dir": str(tmp / "run"), "fresh": True})
    assert s == 200 and st["armed"]
    s, r = _http(url, "/agent/act", "AGENT", {"action": {"type": "click", "x": 5, "y": 5}})
    assert r["ok"] and r["steps_used"] == 1 and base64.b64decode(r["screenshot"])[:4] == b"\x89PNG"
    assert _http(url, "/agent/act", "AGENT", {"action": {"type": "shell"}})[1]["ok"] is False
    assert _http(url, "/ctl/evaluate", "CONTROL", {})[1]["score"] == 0.0
    assert _http(url, "/ctl/finish", "CONTROL", {"score": 0.0})[0] == 200
    assert not _http(url, "/agent/act", "AGENT", {"action": {"type": "click", "x": 5, "y": 5}})[1]["ok"]   # disarmed


@pytest.fixture()
def mcp_server(served, monkeypatch):
    pytest.importorskip("mcp")
    url, b, taskfile, tmp = served
    monkeypatch.setenv("MACARENA_BRIDGE_URL", url)
    monkeypatch.setenv("MACARENA_AGENT_TOKEN", "AGENT")
    spec = importlib.util.spec_from_file_location("macarena_vm_server", ROOT / "backend" / "builtin_mcps" / "macarena_vm" / "server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _http(url, "/ctl/reset", "CONTROL", {"task_path": str(taskfile), "run_dir": str(tmp / "run"), "fresh": True})
    return mod, b


def test_mcp_tools_drive_the_bridge(mcp_server):
    mod, b = mcp_server
    shot = mod.vm_screenshot()
    assert "1280x720" in shot[0] and shot[1].data[:4] == b"\x89PNG"
    out = mod.vm_click(10, 20)
    assert "steps left 2" in out[0] and b.env.actions == ["pyautogui.click(10, 20, button='left')"]
    assert mod.vm_type("hi")[0].startswith("Typed 2")
    mod.vm_hotkey(["command", "c"])
    assert b.env.actions[-1] == "pyautogui.hotkey('command', 'c')"
    over = mod.vm_click(1, 1)
    assert isinstance(over, list) and "run is over" in over[0].lower()
    assert "Marked done" not in mod.vm_done()                                   # the run is already over: an error, not a step


def test_mcp_reports_a_missing_token_and_an_unreachable_bridge(monkeypatch):
    pytest.importorskip("mcp")
    spec = importlib.util.spec_from_file_location("macarena_vm_server2", ROOT / "backend" / "builtin_mcps" / "macarena_vm" / "server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.delenv("MACARENA_AGENT_TOKEN", raising=False)
    assert "token is not configured" in mod.vm_click(1, 1)[0]
    monkeypatch.setenv("MACARENA_AGENT_TOKEN", "x")
    monkeypatch.setenv("MACARENA_BRIDGE_URL", "http://127.0.0.1:9")
    assert "cannot reach" in mod.vm_click(1, 1)[0]


# ── guard middleware and Otto registration ──────────────────────────────────

def test_guard_allows_only_vm_tools_for_the_macarena_agent():
    from backend.macarena_guard_middleware import VM_TOOLS, refusal
    for t in ("execute", "read_file", "write_file", "ls", "glob", "grep", "task", "view_image", "edit_file"):
        assert refusal("macarena-agent", t) and "Blocked" in refusal("macarena-agent", t)
    assert all(refusal("macarena-agent", t) is None for t in VM_TOOLS)
    assert refusal("some-other-agent", "execute") is None                         # no effect on other agents
    assert refusal(None, "execute") is None


def test_guard_tool_list_matches_mcp_server_allowlist_and_catalog():
    from backend.action_policy import catalog
    from backend.macarena_guard_middleware import VM_TOOLS
    from backend.session_manager import _AGENT_TOOL_ALLOWLIST
    served = {n for (srv, n) in catalog._SERVER_TOOLS if srv == "macarena-vm"}
    assert served == set(VM_TOOLS) == set(_AGENT_TOOL_ALLOWLIST["macarena-agent"])
    src = (ROOT / "backend" / "builtin_mcps" / "macarena_vm" / "server.py").read_text()
    import re
    defined = set(re.findall(r"^def (vm_\w+)\(", src, re.M))
    assert defined == set(VM_TOOLS)


def test_registered_and_off_by_default():
    from backend.builtin_mcps.registry import BUILTIN_MCPS
    spec = next(m for m in BUILTIN_MCPS if m.id == "macarena-vm")
    assert spec.enabled is False and (ROOT / "backend" / "builtin_mcps" / spec.source_dir_name / "server.py").is_file()


def test_agent_is_seeded_headless_with_no_memory_and_no_subagents():
    from backend import agent_library, session_manager
    assert "macarena-agent" in agent_library._BUILTIN_AGENT_VERSIONS
    assert "macarena-agent" in session_manager._AGENTS_WITHOUT_MEMORY
    assert "macarena-agent" in session_manager._AGENTS_WITHOUT_SUBAGENTS
    assert "never ask" in agent_library._MACARENA_AGENT_PROMPT.lower()


# ── report ──────────────────────────────────────────────────────────────────

def row(i, run, score, status="ok", **kw):
    return {"id": i, "run": run, "score": score, "status": status, "source": "macarena", "domain": "d", "steps": 5, **kw}


def test_report_three_protocols_differ_as_expected():
    rows = [row("a", 1, 0.0), row("a", 2, 1.0), row("b", 1, 1.0), row("b", 2, 0.0), row("c", 1, 0.0), row("c", 2, 0.0)]
    o = report.summarize(rows)["overall"]
    assert o["best_of_runs"]["solved"] == round(2 / 3, 4)
    assert o["mean_of_runs"]["mean_score"] == round(2 / 6, 4)
    assert o["run_1"]["solved"] == round(1 / 3, 4)


def test_report_excludes_infra_errors_from_rates_and_says_so():
    rows = [row("a", 1, 1.0), row("a", 2, 0.0, status="infra_error"), row("b", 1, 0.0, status="infra_error"),
            row("b", 2, 0.0, status="infra_error")]
    s = report.summarize(rows)
    assert s["infra_error_runs"] == 3 and s["tasks_with_no_valid_run"] == 1
    assert s["overall"]["mean_of_runs"]["n"] == 1 and s["overall"]["mean_of_runs"]["mean_score"] == 1.0
    assert "WARNING" in report.render(s)


def test_report_rerun_replaces_the_old_row(tmp_path):
    p = tmp_path / "results.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [row("a", 1, 0.0, status="infra_error"), row("a", 1, 1.0)]) + "\n")
    assert len(report.load(p)) == 1 and report.load(p)[0]["score"] == 1.0


def test_report_partial_scores_are_means_not_solves():
    s = report.summarize([row("a", 1, 0.5)])
    assert s["overall"]["run_1"] == {"n": 1, "mean_score": 0.5, "solved": 0.0}


# ── run.py task selection ───────────────────────────────────────────────────

def _run_module():
    pytest.importorskip("websockets")
    spec = importlib.util.spec_from_file_location("macarena_run", MA / "run.py")
    mod = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    try:
        spec.loader.exec_module(mod)             # run.py puts its own folder on sys.path; undo that
    finally:
        sys.path[:] = saved
    return mod


def _fake_checkout(tmp_path):
    base = tmp_path / "evaluation_examples"
    for src, dom, n in (("macarena", "productivity", 6), ("macarena", "system_apps", 2), ("osworld", "chrome", 3)):
        for i in range(n):
            f = base / src / dom / f"{src}-{dom}-{i}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({"id": f"{src}-{dom}-{i}", "instruction": "x",
                                     "evaluator": {"func": "infeasible"} if i == 0 and src == "osworld" else []}))
    (base / "macarena" / "productivity" / "README.json").write_text('{"not": "a task"}')
    return tmp_path


def test_find_tasks_skips_non_tasks_and_maps_vm_and_flags_infeasible(tmp_path):
    run = _run_module()
    ts = run.find_tasks(_fake_checkout(tmp_path), ["macarena", "osworld"])
    assert len(ts) == 11
    assert {t["vm"] for t in ts if t["source"] == "macarena"} == {"macarena"}
    assert {t["vm"] for t in ts if t["source"] == "osworld"} == {"osworld"}
    assert [t["infeasible"] for t in ts if t["source"] == "osworld"].count(True) == 1


def test_sample_is_stratified_deterministic_and_bounded(tmp_path):
    run = _run_module()
    ts = run.find_tasks(_fake_checkout(tmp_path), ["macarena"])
    a = run.select(ts, None, None, 4, seed=1)
    assert a == run.select(ts, None, None, 4, seed=1)
    assert len(a) <= 4 and {t["domain"] for t in a} == {"productivity", "system_apps"}


def test_per_group_takes_n_from_every_group_and_repeats_with_the_seed():
    run = _run_module()
    ts = [{"id": f"{d}{i}", "source": "macosworld", "domain": d} for d, n in (("big", 9), ("mid", 3), ("tiny", 1))
          for i in range(n)]
    a = run.select(ts, None, None, None, 0, per_group=2)
    count = {d: sum(t["domain"] == d for t in a) for d in ("big", "mid", "tiny")}
    assert count == {"big": 2, "mid": 2, "tiny": 1}                    # a group smaller than N gives all it has
    assert a == run.select(list(reversed(ts)), None, None, None, 0, per_group=2)   # order found does not matter
    assert any(run.select(ts, None, None, None, s, per_group=2) != a for s in range(1, 6))   # the seed changes it
    assert [t["id"] for t in run.select(ts, None, "mid", None, 0, per_group=5)] == ["mid0", "mid1", "mid2"]


def test_prompt_contains_instruction_and_budget_only():
    run = _run_module()
    p = run.prompt_for({"instruction": "Open Notes", "id": "secret-id", "path": "/x/evaluator.json"}, 15)
    assert "Open Notes" in p and "15" in p and "secret-id" not in p and "evaluator" not in p


def test_mcp_server_serves_images_over_real_stdio(served):
    """The in-process tests above missed that FastMCP could not serialise the Image return over stdio."""
    pytest.importorskip("mcp")
    import asyncio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    url, b, taskfile, tmp = served
    _http(url, "/ctl/reset", "CONTROL", {"task_path": str(taskfile), "run_dir": str(tmp / "run"), "fresh": True})
    params = StdioServerParameters(
        command=sys.executable, args=[str(ROOT / "backend" / "builtin_mcps" / "macarena_vm" / "server.py")],
        env={"MACARENA_BRIDGE_URL": url, "MACARENA_AGENT_TOKEN": "AGENT", "PATH": "/usr/bin:/bin"})

    async def go():
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                names = {t.name for t in (await s.list_tools()).tools}
                shot = await s.call_tool("vm_screenshot", {})
                click = await s.call_tool("vm_click", {"x": 5, "y": 5})
                bad = await s.call_tool("vm_click", {"x": 99999, "y": 5})
                return names, shot, click, bad

    names, shot, click, bad = asyncio.run(go())
    from backend.macarena_guard_middleware import VM_TOOLS
    assert names == set(VM_TOOLS)
    for res in (shot, click):
        assert not res.isError and [c.type for c in res.content] == ["text", "image"]
    assert "outside the screen" in bad.content[0].text and b.steps == 1


def test_mcp_reads_bridge_address_and_token_from_the_profile_file(served, monkeypatch, tmp_path):
    """Otto resets a built-in MCP's env on every config load, so the per-run token travels in a file."""
    pytest.importorskip("mcp")
    url, b, taskfile, tmp = served
    for k in ("MACARENA_BRIDGE_URL", "MACARENA_AGENT_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    f = tmp_path / "bridge.json"
    f.write_text(json.dumps({"url": url, "token": "AGENT"}))
    monkeypatch.setenv("MACARENA_BRIDGE_FILE", str(f))
    spec = importlib.util.spec_from_file_location("macarena_vm_server3", ROOT / "backend" / "builtin_mcps" / "macarena_vm" / "server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _http(url, "/ctl/reset", "CONTROL", {"task_path": str(taskfile), "run_dir": str(tmp / "run"), "fresh": True})
    assert "Clicked" in mod.vm_click(1, 1)[0]
    monkeypatch.setenv("MACARENA_AGENT_TOKEN", "wrong")                       # env wins over the file
    assert "unauthorized" in mod.vm_click(1, 1)[0].lower()
    f.write_text("not json")
    monkeypatch.delenv("MACARENA_AGENT_TOKEN")
    assert "not configured" in mod.vm_click(1, 1)[0]


def test_screenshots_and_recording_are_off_by_default(tmp_path):
    b = make_bridge(tmp_path)
    b.act({"type": "click", "x": 3, "y": 4})
    b.finish(0.5)
    run = tmp_path / "run_1"
    assert not list(run.glob("*.png")) and not (run / "recording.mp4").exists()
    row = json.loads((run / "traj.jsonl").read_text().splitlines()[0])
    assert row["screenshot_file"] is None and (run / "result.txt").exists()


def test_record_flag_saves_recording(tmp_path):
    b = make_bridge(tmp_path, record=True)
    b.act({"type": "click", "x": 1, "y": 1})
    b.finish(1.0)
    assert (tmp_path / "run_1" / "recording.mp4").exists()


def test_exclude_apps_and_ids_apply_before_sampling():
    run = _run_module()
    ts = [{"id": f"t{i}", "source": "macarena", "domain": "d", "apps": a}
          for i, a in enumerate([["Keynote"], ["Pages", "Safari"], ["Safari"], [], ["Numbers"], ["Notes"]])]
    assert [t["id"] for t in run.excluded(ts, "keynote, Pages")] == ["t0", "t1"]          # case-insensitive
    assert [t["id"] for t in run.excluded(ts, None, "t3")] == ["t3"]
    out = run.select(ts, None, None, None, 0, exclude_apps="Keynote,Pages,Numbers", exclude_ids="t3")
    assert [t["id"] for t in out] == ["t2", "t5"]
    picked = run.select(ts, None, None, 3, 1, exclude_apps="Keynote,Pages,Numbers")
    assert len(picked) == 3 and not {t["id"] for t in picked} & {"t0", "t1", "t4"}
    assert run.select(ts, None, None, None, 0) == ts                                       # off by default


def test_a_busy_agent_is_stopped_once_the_run_is_over():
    """An agent that keeps calling tools after vm_done never leaves the socket quiet; it must still be stopped."""
    import asyncio
    run = _run_module()
    calls = []

    class Api:
        def ws_url(self, sid): return "ws://x"
        async def call(self, method, path, *a, **k): calls.append((method, path)); return {}

    class Bridge:
        async def call(self, path, body=None): return {"finished": "done"}

    class Ws:
        async def send(self, data): pass
        async def recv(self):
            await asyncio.sleep(0.01)
            return json.dumps({"type": "tool_call", "content": "vm_done"})

    class Conn:
        async def __aenter__(self): return Ws()
        async def __aexit__(self, *a): return False

    run.websockets.connect = lambda *a, **k: Conn()
    run.STATE_CHECK_EVERY, run.OVER_GRACE = 0.05, 0.1   # the test shrinks the real 2 s / 4 s
    out = asyncio.run(asyncio.wait_for(
        run.drive(Api(), Bridge(), "sid", "task", time.time() + 30, Path("/dev/null"), track="native"), 10))
    assert out["status"] == "run_over" and ("POST", "/api/sessions/sid/stop") in calls


def test_local_network_block_is_detected_and_other_errors_are_not():
    calls = []

    def blocked(url):
        calls.append(url)
        raise OSError("HTTPConnection: Failed to establish a new connection: [Errno 65] No route to host")

    def refused(url):
        raise OSError("[Errno 61] Connection refused")                # VM still starting: MacArena retries

    assert "Errno 65" in br.local_network_error("http://vm/screenshot", attempts=3, wait=0, get=blocked)
    assert len(calls) == 3                                             # retried before giving up
    assert br.local_network_error("http://vm/screenshot", wait=0, get=refused) is None
    assert br.local_network_error("http://vm/screenshot", wait=0, get=lambda u: None) is None


# ── no step limit (max_steps 0) ─────────────────────────────────────────────

def test_no_step_limit_never_ends_on_budget_and_reports_no_steps_left(tmp_path):
    b = make_bridge(tmp_path, max_steps=0)
    for _ in range(40):                                                # far past the paper's 15
        r = b.act({"type": "click", "x": 1, "y": 1})
        assert r["ok"] and r["steps_left"] is None and r["finished"] is None
    assert b.state()["max_steps"] == 0 and b.state()["steps"] == 40 and b.max_obs == br.UNLIMITED_READS
    assert b.act({"type": "done"})["finished"] == "done"               # vm_done still ends the run


def test_a_limit_still_applies_when_set(tmp_path):
    b = make_bridge(tmp_path, max_steps=3)
    for _ in range(3):
        b.act({"type": "click", "x": 1, "y": 1})
    assert b.state()["finished"] == "budget" and b._budget()["steps_left"] == 0


def test_prompt_says_whether_there_is_a_step_limit():
    run = _run_module()
    t = {"instruction": "do it"}
    assert "Step budget: 15" in run.prompt_for(t, 15, "native")
    for track in ("native", "standard"):
        p = run.prompt_for(t, 0, track)
        assert "no step limit" in p and "Step budget" not in p


def test_report_labels_the_step_limit_and_refuses_to_mix_caps():
    rows = [dict(row("a", 1, 1.0), max_steps=0), dict(row("b", 1, 0.0), max_steps=0)]
    text = report.render(report.summarize(rows))
    assert "step limit: none" in text and "NOT comparable" in text
    assert "step limit: 15" in report.render(report.summarize([row("a", 1, 1.0)]))     # older rows count as 15
    with pytest.raises(ValueError):
        report.summarize([dict(row("a", 1, 1.0), max_steps=0), dict(row("b", 1, 1.0), max_steps=15)])


def test_a_failed_grader_is_an_infra_error_not_a_zero():
    run = _run_module()
    assert run.final_status("done", False, '{"status": "error"}') == "infra_error"
    assert run.final_status("done", False, None) == "ok"
    assert run.final_status("timeout", False, None) == "timeout"
    assert run.final_status("agent_error", False, None) == "agent_error"
    assert run.final_status("done", True, None) == "noop"
    assert run.final_status("run_over", False, "") == "ok"                     # empty eval_error: graded fine


# ── permission pre-flight ───────────────────────────────────────────────────

def write_task(tmp_path, name, pre="", evaluator=""):
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps({"id": name, "instruction": "x", "pre_command": pre, "evaluator": [[evaluator, 100]]}))
    return {"id": name, "domain": "d", "path": str(p)}


def test_apps_are_found_in_setup_and_grading_scripts(tmp_path):
    t = write_task(tmp_path, "t",
                   pre="osascript -e 'tell application \"Reminders\" to quit' && osascript -e 'tell application \"System Events\" to x'",
                   evaluator="osascript -e \"tell application \\\"Contacts\\\" to try\" ; echo tell application id \"com.x\"")
    assert preflight.apps_in_task(t["path"]) == {"Reminders", "System Events", "Contacts", "com.x"}
    assert preflight.apps_in_task(tmp_path / "missing.json") == set()
    assert preflight.apps_in_task(write_task(tmp_path, "plain", pre="rm -f x", evaluator="[ -f x ] && echo True")["path"]) == set()


def test_probe_results_are_classified():
    c = preflight.classify
    assert c(1, "31:46: execution error: Not authorized to send Apple events to Contacts. (-1743)")["status"] == "not_authorized"
    assert c(1, "execution error: Can\u2019t get application \"Keynote\". (-10814)")["status"] == "not_installed"
    assert c(0, "Reminders")["status"] == "ok"
    assert c(1, "some other failure")["status"] == "error"
    assert c(0, "")["status"] == "ok"
    assert c(-1, "Script execution timed out after 40 seconds")["status"] == "prompt_pending"   # the Contacts hang


def test_tasks_with_a_blocked_app_are_left_out_with_the_reason(tmp_path):
    a = write_task(tmp_path, "a", pre="osascript -e 'tell application \"Contacts\" to quit'")
    b = write_task(tmp_path, "b", pre="osascript -e 'tell application \"Reminders\" to quit'")
    c = write_task(tmp_path, "c", pre="touch x")
    results = {"Contacts": {"status": "not_authorized", "detail": "-1743"}, "Reminders": {"status": "ok", "detail": ""}}
    runnable, blocked = preflight.split_tasks([a, b, c], results)
    assert [t["id"] for t in runnable] == ["b", "c"]
    assert blocked == [{"id": "a", "domain": "d", "reason": "Contacts: not_authorized"}]
    assert "Contacts" in preflight.render(results) and "grant-apps" in preflight.FIX_HINT


def test_bridge_preflight_probes_each_app_and_forces_a_clean_vm(tmp_path):
    env = br.FakeEnv()
    seen = []
    answers = {"Contacts": {"returncode": 1, "output": "execution error: Not authorized (-1743)"},
               "Notes": {"returncode": 0, "output": "Notes"}}

    def run_bash(script, timeout=30, working_dir=None):
        seen.append(script)
        return next(v for k, v in answers.items() if f'"{k}"' in script)

    env.controller.run_bash_script = run_bash
    b = br.Bridge(env, settle=0, pause=0)
    out = b.preflight(["Notes", "Contacts", "Notes"])
    assert out["Contacts"]["status"] == "not_authorized" and out["Notes"]["status"] == "ok"
    assert len(seen) == 2 and all("osascript" in s and " get " in s for s in seen)
    assert env.is_environment_used is True              # MacArena reverts to a clean copy at the next reset
    b.reset(TASK, tmp_path / "r")
    with pytest.raises(RuntimeError):
        b.preflight(["Notes"])                            # not once a run is armed


def test_a_run_cut_at_the_tool_call_cap_is_labelled_and_counted():
    run = _run_module()
    assert run.final_status("done", False, None, capped=True) == "agent_limit"
    assert run.final_status("timeout", False, None, capped=True) == "timeout"      # the clock wins
    assert run.final_status("done", False, '{"x": 1}', capped=True) == "infra_error"   # a broken grader wins
    assert run.final_status("done", False, None, capped=False) == "ok"
    rows = [dict(row("a", 1, 1.0), max_steps=0, tool_call_cap=100, status="agent_limit"),
            dict(row("b", 1, 0.0), max_steps=0, tool_call_cap=100)]
    s = report.summarize(rows)
    assert s["agent_limit_runs"] == 1 and s["tool_call_cap"] == 100
    text = report.render(s)
    assert "tool-call cap: 100" in text and "cut at the tool-call cap 1" in text


def test_data_apps_are_asked_a_real_question_and_other_apps_for_their_name():
    assert "count of people" in preflight.probe_command("Contacts")
    assert "count of lists" in preflight.probe_command("Reminders")
    assert "get name" in preflight.probe_command("Safari")
    assert "'" not in preflight.probe_command("Evil'App").split("-e '")[1].split("' 2>&1")[0].replace("tell application", "")


def test_an_unreachable_vm_is_an_error_not_a_pending_dialog():
    assert preflight.classify(-1, "VM unreachable: ConnectionError")["status"] == "error"
    assert preflight.classify(-1, "Script execution timed out")["status"] == "prompt_pending"


def test_preflight_apps_cli_lists_scripted_apps(tmp_path, capsys):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "t.json").write_text('{"cmd": "tell application \\"Contacts\\" to get name"}')
    (tmp_path / "b.json").write_text('{"cmd": "tell application \\"Notes\\" to get name"}')
    assert preflight._main(["apps", str(tmp_path)]) == 0
    assert capsys.readouterr().out.split() == ["Contacts", "Notes"]
    assert preflight._main(["bogus"]) == 2


def test_bridge_finish_marks_the_vm_used_so_the_next_reset_reverts(tmp_path):
    """A task with no setup step and no env.step() (native tools, or an infeasible task) never marked the VM used in
    MacArena, so the next task started on its leftovers."""
    env = br.FakeEnv()
    env.is_environment_used = False
    b = br.Bridge(env, settle=0, pause=0)
    b.reset(TASK, tmp_path / "r")
    env.is_environment_used = False                      # what MacArena left it at
    b.finish(0.0)
    assert env.is_environment_used is True


# ── step unit: "turns" counts one step per agent turn, not per action ───────────────────────────────────────────

class _Guest:
    """Stands in for the in-VM tool server: every call works and costs one step."""
    def deploy(self):
        return {"ok": True}

    def call(self, body):
        return {"ok": True, "cost": 0 if body["tool"] in br.native_tools.OBSERVATION else 1, "result": "ok"}


def _native_bridge(tmp_path, unit, cap=2):
    b = br.Bridge(br.FakeEnv(), max_steps=cap, settle=0, pause=0, guest=_Guest(), step_unit=unit)
    b.reset(TASK, tmp_path / "r")
    b.armed = True
    return b


def test_turns_unit_counts_one_step_per_turn(tmp_path):
    b = _native_bridge(tmp_path, "turns")
    for tool in ("click", "type_text", "hotkey"):                         # three calls of turn t1
        assert b.native({"tool": tool, "args": {}, "turn": "t1"})["ok"]
    assert b.steps == 1 and b.finished is None
    assert b.native({"tool": "get_screen_controls", "args": {}, "turn": "t2"})["ok"]   # a read-only turn counts
    assert b.steps == 2 and b.finished is None                           # the last turn the budget allows
    assert b.native({"tool": "click", "args": {}, "turn": "t2"})["ok"]    # more calls in that turn still run
    res = b.native({"tool": "click", "args": {}, "turn": "t3"})           # a third turn does not
    assert not res["ok"] and "no turns left" in res["error"] and b.finished == "budget"
    assert b.state()["step_unit"] == "turns"


def test_actions_unit_still_counts_every_action_and_ignores_the_turn(tmp_path):
    b = _native_bridge(tmp_path, "actions")
    assert b.native({"tool": "click", "args": {}, "turn": "t1"})["ok"]
    assert b.native({"tool": "click", "args": {}, "turn": "t1"})["ok"]
    assert b.steps == 2 and b.finished == "budget"


def test_turns_unit_without_a_turn_id_falls_back_to_actions(tmp_path):
    b = _native_bridge(tmp_path, "turns")
    assert b.native({"tool": "click", "args": {}})["ok"] and b.native({"tool": "click", "args": {}})["ok"]
    assert b.steps == 2 and b.finished == "budget"


def test_standard_track_ignores_the_turns_unit(tmp_path):
    assert br.Bridge(br.FakeEnv(), step_unit="turns").step_unit == "actions"


def test_guard_turn_id_is_the_issuing_model_response():
    from types import SimpleNamespace
    from backend import macarena_guard_middleware as g
    ai1 = SimpleNamespace(id="m1", tool_calls=[{"id": "c1"}, {"id": "c2"}])
    ai2 = SimpleNamespace(id="m2", tool_calls=[{"id": "c3"}])
    req = lambda cid: SimpleNamespace(tool_call={"id": cid}, state={"messages": [ai1, SimpleNamespace(), ai2]})
    assert g.turn_id(req("c1")) == g.turn_id(req("c2")) == "m1" and g.turn_id(req("c3")) == "m2"
    assert g.turn_id(req("zzz")) is None and g.turn_id(SimpleNamespace(tool_call={}, state=None)) is None


# ── bookkeeping ticks: model calls, dialog messages, done / fail inside the turn budget ────────────────────────────

def test_tick_counts_model_calls_and_dialog_messages(tmp_path):
    b = _native_bridge(tmp_path, "turns", cap=15)
    for _ in range(3):
        assert b.tick({"kind": "model"})["ok"]
    for _ in range(4):
        assert b.tick({"kind": "tool", "tool": "click", "turn": "t1"})["ok"]
    st = b.state()
    assert (st["model_calls"], st["tool_messages"], st["dialog_messages"]) == (3, 4, 7)
    assert b.steps == 0                                   # a tick for an ordinary tool never uses a step
    assert not b.tick({"kind": "nonsense"})["ok"]


def test_tick_is_ignored_when_no_task_is_armed(tmp_path):
    b = _native_bridge(tmp_path, "turns")
    b.armed = False
    assert not b.tick({"kind": "model"})["ok"] and b.model_calls == 0


def test_done_in_a_new_turn_uses_a_step_and_is_refused_past_the_budget(tmp_path):
    b = _native_bridge(tmp_path, "turns", cap=2)
    assert b.native({"tool": "click", "args": {}, "turn": "t1"})["ok"]
    assert b.tick({"kind": "tool", "tool": "vm_done", "turn": "t2"})["ok"]          # its own turn, inside the budget
    assert b.steps == 2 and b.finished is None
    res = b.tick({"kind": "tool", "tool": "vm_fail", "turn": "t3"})                # a third turn: refused
    assert not res["ok"] and "no turns left" in res["error"] and b.finished == "budget"


def test_done_in_the_last_allowed_turn_is_accepted(tmp_path):
    b = _native_bridge(tmp_path, "turns", cap=1)
    assert b.native({"tool": "click", "args": {}, "turn": "t1"})["ok"]
    assert b.tick({"kind": "tool", "tool": "vm_done", "turn": "t1"})["ok"]          # same turn as the last action
    assert b.steps == 1 and b.finished is None


def test_done_tick_does_not_count_steps_in_actions_mode(tmp_path):
    b = _native_bridge(tmp_path, "actions", cap=1)
    assert b.tick({"kind": "tool", "tool": "vm_done", "turn": "t9"})["ok"] and b.steps == 0


def _guard_env(monkeypatch, agent, answer=None):
    """Patch the scope and the bridge ticks; return the list of ticks sent."""
    from types import SimpleNamespace
    from backend import macarena_native
    import backend.action_policy.runtime as rt
    sent = []
    monkeypatch.setattr(rt, "current_scope", lambda: SimpleNamespace(agent_name=agent))
    monkeypatch.setattr(macarena_native, "tick", lambda kind, tool=None, turn=None: sent.append((kind, tool, turn)) or answer)
    return sent


def _tool_request(name, call_id="c1", msg_id="m1"):
    from types import SimpleNamespace
    ai = SimpleNamespace(id=msg_id, tool_calls=[{"id": call_id}])
    return SimpleNamespace(tool_call={"id": call_id, "name": name}, tool=None, state={"messages": [ai]})


def test_guard_ticks_only_for_the_benchmark_agent(monkeypatch):
    """Otto's other agents pass through this middleware too: no ticks, no turn tag, the handler runs as before."""
    from backend import macarena_guard_middleware as g, macarena_native
    sent = _guard_env(monkeypatch, "general-purpose")
    token = macarena_native.TURN.set(None)
    try:
        mw = g.MacarenaGuardMiddleware("general-purpose")
        assert mw.wrap_tool_call(_tool_request("ls"), lambda r: "ran") == "ran"
        assert sent == [] and macarena_native.TURN.get() is None
    finally:
        macarena_native.TURN.reset(token)


def test_guard_ticks_and_tags_turns_for_the_benchmark_agent(monkeypatch):
    from backend import macarena_guard_middleware as g, macarena_native
    sent = _guard_env(monkeypatch, g.NATIVE_AGENT, answer={"ok": True})
    token = macarena_native.TURN.set(None)
    try:
        mw = g.MacarenaGuardMiddleware(g.NATIVE_AGENT)
        assert mw.wrap_tool_call(_tool_request("write_todos"), lambda r: "ran") == "ran"
        assert sent == [("tool", "write_todos", "m1")] and macarena_native.TURN.get() == "m1"
        out = mw.wrap_tool_call(_tool_request("click", "c2", "m2"), lambda r: "ran")   # a host click: still refused,
        assert out.status == "error" and sent[-1] == ("tool", "click", "m2")        # and still one dialog message
    finally:
        macarena_native.TURN.reset(token)


def test_guard_answers_a_refused_done_without_running_it(monkeypatch):
    from backend import macarena_guard_middleware as g
    _guard_env(monkeypatch, g.NATIVE_AGENT, answer={"ok": False, "error": "no turns left (15); vm_done was not accepted"})
    out = g.MacarenaGuardMiddleware(g.NATIVE_AGENT).wrap_tool_call(
        _tool_request("vm_done"), lambda r: pytest.fail("vm_done must not run past the budget"))
    assert out.status == "error" and "no turns left" in out.content


def test_guard_runs_done_when_the_bridge_cannot_be_reached(monkeypatch):
    from backend import macarena_guard_middleware as g
    _guard_env(monkeypatch, g.NATIVE_AGENT, answer=None)      # bookkeeping failure never blocks the agent
    assert g.MacarenaGuardMiddleware(g.NATIVE_AGENT).wrap_tool_call(_tool_request("vm_done"), lambda r: "ran") == "ran"


def test_report_shows_model_calls_and_dialog_messages():
    import report
    rows = [{"id": "a", "run": 1, "source": "macosworld", "domain": "d", "score": 1.0, "status": "ok", "steps": 15,
             "max_steps": 15, "track": "native", "step_unit": "turns", "model_calls": 16, "tool_messages": 17,
             "dialog_messages": 33, "observe_after": True, "fail_notice_after": 4, "infeasible": False},
            {"id": "b", "run": 1, "source": "macosworld", "domain": "d", "score": 0.0, "status": "ok", "steps": 5,
             "max_steps": 15, "track": "native", "step_unit": "turns", "model_calls": 6, "tool_messages": 5,
             "dialog_messages": 11, "observe_after": True, "fail_notice_after": 4, "infeasible": False}]
    s = report.summarize(rows)
    assert s["dialog"] == {"runs": 2, "model_calls_max": 16, "model_calls_mean": 11, "dialog_messages_max": 33,
                           "dialog_messages_mean": 22, "runs_over_step_cap": 0, "runs_over_twice_cap_messages": 1}
    assert "model calls max 16" in report.render(s) and "15 turns" in report.render(s)


def test_guard_async_paths_tick_model_and_tool_calls_for_the_benchmark_agent_only(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from backend import macarena_guard_middleware as g, macarena_native
    import backend.action_policy.runtime as rt
    sent = []

    async def atick(kind, tool=None, turn=None):
        sent.append((kind, tool, turn))
        return {"ok": True}

    monkeypatch.setattr(macarena_native, "atick", atick)
    model_req = SimpleNamespace(tools=[], messages=[], system_message=None, override=lambda **kw: model_req)

    async def handler(r):
        return "ran"

    for agent, ticks in (("general-purpose", []), (g.NATIVE_AGENT, [("model", None, None), ("tool", "write_todos", "m1")])):
        sent.clear()
        monkeypatch.setattr(rt, "current_scope", lambda a=agent: SimpleNamespace(agent_name=a))
        mw = g.MacarenaGuardMiddleware(agent)
        monkeypatch.setattr(mw, "_offer", lambda r: r)
        token = macarena_native.TURN.set(None)
        try:
            assert asyncio.run(mw.awrap_model_call(model_req, handler)) == "ran"
            assert asyncio.run(mw.awrap_tool_call(_tool_request("write_todos"), handler)) == "ran"
        finally:
            macarena_native.TURN.reset(token)
        assert sent == ticks


def test_skip_known_hosts_adds_the_option_once():
    opts = ["-o", "StrictHostKeyChecking=no"]
    br.skip_known_hosts(opts)
    br.skip_known_hosts(opts)
    assert opts.count("UserKnownHostsFile=/dev/null") == 1 and opts[:2] == ["-o", "StrictHostKeyChecking=no"]


def test_macos_launch_translates_only_the_linux_office_and_gimp_commands():
    assert br.macos_launch(["soffice", "--calc", "/Users/admin/a.xlsx"]) == ["open", "-a", "LibreOffice", "/Users/admin/a.xlsx"]
    assert br.macos_launch(["soffice", "--writer", "/Users/admin/b.docx"]) == ["open", "-a", "LibreOffice", "/Users/admin/b.docx"]
    assert br.macos_launch(["gimp"]) == ["open", "-a", "GIMP"]
    assert br.macos_launch(["gimp", "/Users/admin/Desktop/x.png"]) == ["open", "-a", "GIMP", "/Users/admin/Desktop/x.png"]
    for same in (["google-chrome", "--remote-debugging-port=1337"], ["socat", "a", "b"], ["open", "-a", "Safari"]):
        assert br.macos_launch(same) == same
    assert br.macos_launch(["gimp"], shell=True) == ["gimp"] and br.macos_launch("gimp") == "gimp"


def test_translated_launch_brings_the_app_forward_and_other_launches_do_not(monkeypatch):
    import sys, types
    calls = []

    class SC:
        def _launch_setup(self, command, shell=False):
            calls.append(("launch", command))

        def _execute_setup(self, command, **kw):
            calls.append(("exec", command))

    mod = types.ModuleType("desktop_env.controllers.setup")
    mod.SetupController = SC
    for name in ("desktop_env", "desktop_env.controllers"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "desktop_env.controllers.setup", mod)
    monkeypatch.setattr(br, "ACTIVATE_WAIT", 0)
    br.translate_launch_setup()
    br.translate_launch_setup()  # idempotent
    SC()._launch_setup(["soffice", "--calc", "/Users/admin/a.xlsx"])
    assert calls == [("launch", ["open", "-a", "LibreOffice", "/Users/admin/a.xlsx"]),
                     ("exec", ["osascript", "-e", 'tell application "LibreOffice" to activate'])]
    calls.clear()
    SC()._launch_setup(["google-chrome", "--remote-debugging-port=1337"])
    assert calls == [("launch", ["google-chrome", "--remote-debugging-port=1337"])]


def test_native_prompt_states_the_step_rule_the_bridge_uses():
    run = _run_module()
    t = {"instruction": "do it", "source": "macarena"}
    acts = run.prompt_for(t, 15, "native", "actions")
    turns = run.prompt_for(t, 30, "native", "turns")
    assert "Step budget: 15" in acts and "reading tools such as list_apps and get_screen_controls are free" in acts
    assert "Step budget: 30" in turns and "one reply of yours that calls tools" in turns and "are free" not in turns
    assert run.prompt_for(t, 15, "native") == acts                       # default unit is unchanged
    assert "vm_screenshot" in run.prompt_for(t, 15, "standard", "turns")  # standard track ignores the unit


def test_infeasible_note_only_for_sources_that_have_infeasible_tasks():
    run = _run_module()
    for src, has in (("osworld", True), ("macosworld", True), ("macarena", False), (None, False)):
        p = run.prompt_for({"instruction": "x", "source": src}, 15, "native", "turns")
        assert ("call vm_fail with the reason" in p) is has, src
    assert "vm_fail" not in run.prompt_for({"instruction": "x", "source": "osworld"}, 15, "standard")


def test_run_stops_after_consecutive_infra_errors_and_resumes_cleanly(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location("macarena_run_guard", MA / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    tasks = [{"id": f"t{i}", "source": "osworld", "domain": "d", "vm": "osworld", "apps": [], "infeasible": False,
              "instruction": "x"} for i in range(6)]

    class FakeBridge:
        def __init__(self, *a): pass
        async def call(self, path, *a, **k):
            return {"vm": "osworld", "step_unit": "turns", "max_steps": 30, "track": "native"}

    outcomes = iter(["ok", "infra_error", "infra_error", "infra_error", "ok", "ok"])
    ran = []

    async def fake_run_one(api, bridge, t, k, args, out):
        ran.append(t["id"])
        st = next(outcomes)
        return {"id": t["id"], "run": k, "status": st, "score": 0.0, "steps": 0, "track": "native",
                "error": "boom" if st == "infra_error" else ""}

    monkeypatch.setattr(run, "find_tasks", lambda *a: tasks)
    monkeypatch.setattr(run, "select", lambda all_tasks, *a: all_tasks)
    monkeypatch.setattr(run, "excluded", lambda *a: [])
    monkeypatch.setattr(run, "Bridge", FakeBridge)
    monkeypatch.setattr(run, "Api", lambda *a: None)
    monkeypatch.setattr(run, "run_one", fake_run_one)
    args = SimpleNamespace(out=str(tmp_path), macarena_dir=str(tmp_path), sources="osworld", ids=None, domain=None,
                           sample=None, seed=0, exclude_apps=None, exclude_ids=None, runs=1, step_unit="turns",
                           max_steps=30, track="native", bridge="x", control_token="t", api="x", token=None,
                           preflight_only=False, no_preflight=True, dry_run=False, noop=True, paper_protocol=False,
                           max_infra_streak=3)
    with pytest.raises(SystemExit, match="3 infra errors in a row"):
        asyncio.run(run.main_async(args))
    assert ran == ["t0", "t1", "t2", "t3"]            # stopped before t4
    ran.clear()
    args.max_infra_streak = 0                          # 0 = never stop
    outcomes = iter(["ok"] * 5)
    asyncio.run(run.main_async(args))
    assert ran == ["t1", "t2", "t3", "t4", "t5"]      # t0 done; the infra rows are retried


# ── starting state and refused script calls (native track) ──────────────────

_START_OUT = """##APPS
ASN:0x0-0xaa0aa-"Google Chrome": ASN:0x0-0x11011-"Finder":
##WINDOWS
[["universalAccessAuthWarn","Accessibility Access"],["Google Chrome","Rentalcars.com - Car hire"],["Finder",""]]
##TABS
[{"type":"page","title":"Rentalcars.com - Car hire","url":"https://www.rentalcars.com/"},
 {"type":"page","title":"New Tab","url":"chrome://newtab/"},
 {"type":"other","title":"x","url":"devtools://x"}]
##END
"""


def test_start_state_lists_apps_front_to_back_windows_and_chrome_tabs():
    text = br.parse_start_state(_START_OUT)
    assert text.startswith("What is on the test Mac's screen as the task starts:")
    assert '- Frontmost app: Google Chrome; windows: "Rentalcars.com - Car hire"' in text
    assert "- Also open: Finder" in text
    assert '"Accessibility Access" (universalAccessAuthWarn)' in text           # a dialog on screen is a fact too
    assert '"Rentalcars.com - Car hire" https://www.rentalcars.com/ (showing)' in text
    assert '"New Tab" chrome://newtab/' in text and "(showing)" not in text.split("New Tab")[1].splitlines()[0]
    assert "devtools" not in text


def test_start_state_uses_the_window_list_spelling_of_app_names():
    out = _START_OUT.replace('"Google Chrome": ASN', '"Google_Chrome": ASN')
    text = br.parse_start_state(out)
    assert "- Frontmost app: Google Chrome; windows:" in text and "Google_Chrome" not in text
    assert "Other windows on screen: \"Rentalcars" not in text


def test_start_state_is_empty_when_the_vm_says_nothing_or_garbage():
    assert br.parse_start_state("") == ""
    assert br.parse_start_state("##APPS\n##WINDOWS\nnot json\n##TABS\n<html>\n##END\n") == ""


def test_script_refusal_blocks_harness_ports_everywhere_and_fetches_only_on_chrome_tasks():
    chrome, calc = ("osworld", "chrome"), ("osworld", "libreoffice_calc")
    run = lambda s: {"script": f'do shell script "{s}"'}
    for kind in (chrome, calc, None):
        assert br.script_refusal("run_osascript", run("curl -s http://localhost:5000/"), kind)
        assert br.script_refusal("run_osascript", run("curl 127.0.0.1:9222/json"), kind)
    assert "browser" in br.script_refusal("run_osascript", run("curl -s https://www.nfl.com/api"), chrome)
    assert br.script_refusal("run_osascript", run("wget https://x.org/a.pdf"), chrome)
    assert br.script_refusal("run_osascript", run("curl -o ~/a.pdf https://x.org/a.pdf"), calc) is None
    assert br.script_refusal("run_osascript", run("ls ~/Desktop"), chrome) is None
    assert br.script_refusal("run_osascript", run("echo curly"), chrome) is None
    assert br.script_refusal("click", {"script": "curl x"}, chrome) is None


def test_bridge_refuses_a_chrome_task_fetch_without_running_it(tmp_path):
    b = _native_bridge(tmp_path, "turns", cap=5)
    b.kind = ("osworld", "chrome")
    calls_before = len(b.guest.calls) if hasattr(b.guest, "calls") else None
    res = b.native({"tool": "run_osascript", "args": {"script": 'do shell script "curl https://x.org"'}, "turn": "t1"})
    assert not res["ok"] and "browser" in res["error"] and b.invalid == 1
    if calls_before is not None:
        assert len(b.guest.calls) == calls_before                              # never reached the VM
    assert b.native({"tool": "run_osascript", "args": {"script": "return 1"}, "turn": "t2"})["ok"]


def test_reset_returns_the_start_state_on_the_native_track_only(tmp_path, monkeypatch):
    monkeypatch.setattr(br.Bridge, "_run_script_once", lambda self, s, t: {"output": _START_OUT})
    nb = br.Bridge(br.FakeEnv(), max_steps=3, settle=0, pause=0, guest=_Guest(), step_unit="turns")
    st = nb.reset(TASK, tmp_path / "n", kind=("osworld", "chrome"))
    assert "Frontmost app: Google Chrome" in st["start_state"]
    assert (tmp_path / "n" / "start_state.txt").read_text().startswith("What is on")
    sb = br.Bridge(br.FakeEnv(), max_steps=3, settle=0, pause=0)
    assert sb.reset(TASK, tmp_path / "s")["start_state"] == ""


def test_native_task_message_carries_the_start_state_before_the_budget():
    run = _run_module()
    task = {"instruction": "Find a car.", "source": "osworld"}
    msg = run.prompt_for(task, 30, "native", "turns", start_state="What is on the test Mac's screen as the task starts:\n- Frontmost app: Google Chrome")
    assert msg.index("Find a car.") < msg.index("Frontmost app: Google Chrome") < msg.index("Step budget: 30")
    assert "Frontmost app" not in run.prompt_for(task, 30, "native", "turns")
    assert "Frontmost app" not in run.prompt_for(task, 30, "standard", "actions", start_state="- Frontmost app: X")
