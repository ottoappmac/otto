"""Task setup run and checked by the bridge, the adjusted score, and the harness-folder refusal (2026-11-24)."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "scripts" / "macarena"
FINDER_SETUP = ("open /Users/admin/Desktop; osascript -e 'tell application \"Finder\" to close windows' && "
                "rm -rf /Users/admin/Documents/benchmark_files/ && cp -r /Users/admin/Benchmark_Backup/benchmark_files "
                "/Users/admin/Documents && open /Users/admin/Documents/benchmark_files/NCE/Languages")
KEYNOTE_SETUP = ("osascript -e 'tell application \"Keynote\" to quit without saving' && sleep 5 && osascript -e "
                 "'tell application \"Keynote\" to make new document' && osascript -e 'tell application \"Numbers\" "
                 "to set value of cell \"A1\" of table 1 of sheet 1 of document 1 to \"Alice\"'")


def _mod(name):
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    return importlib.import_module(name)


class ScriptedCtl:
    """FakeEnv's controller, answering each script from a list of rules (substring -> result) and recording it."""

    def __init__(self, env, rules):
        self.env, self.rules, self.scripts = env, rules, []

    def get_screenshot(self):
        return self.env.png

    def get_vm_screen_size(self):
        return {"width": self.env.size[0], "height": self.env.size[1]}

    def run_bash_script(self, script, timeout=30, working_dir=None):
        self.scripts.append(script)
        for key, results in self.rules:
            if key in script:
                return results.pop(0) if isinstance(results, list) and len(results) > 1 else (
                    results[0] if isinstance(results, list) else results)
        return {"returncode": 0, "output": ""}


def _bridge(rules, **kw):
    br = _mod("bridge")
    env = br.FakeEnv()
    env.controller = ScriptedCtl(env, rules)
    seen = []
    orig = env.reset
    env.reset = lambda task_config=None, **k: (seen.append(task_config), orig(task_config))[1]
    return br, br.Bridge(env, settle=0.0, **kw), env, seen


def test_setup_targets_are_the_folders_the_command_opens():
    br = _mod("bridge")
    assert br.setup_targets(FINDER_SETUP) == ["/Users/admin/Desktop", "/Users/admin/Documents/benchmark_files/NCE/Languages"]
    assert br.setup_targets("open -a Safari; open '/Users/admin/My Files'") == ["/Users/admin/My Files"]
    assert br.setup_targets(KEYNOTE_SETUP) == []


def test_only_setups_that_close_or_quit_are_run_twice():
    br = _mod("bridge")
    assert br.setup_rerunnable(FINDER_SETUP)
    assert not br.setup_rerunnable(KEYNOTE_SETUP)                  # a second run would make a second document
    assert br.setup_rerunnable("rm -rf ~/x && cp -r ~/y ~/x")        # no AppleScript at all


def test_sync_setup_runs_the_command_itself_waits_and_records_it(tmp_path):
    br, bridge, env, seen = _bridge([])
    st = bridge.reset({"id": "t1", "pre_command": FINDER_SETUP, "instruction": "x"}, tmp_path)
    assert "pre_command" not in seen[-1]                              # MacArena no longer only starts it
    assert FINDER_SETUP in env.controller.scripts
    assert env.controller.scripts.index(br.FINDER_READY_SCRIPT) < env.controller.scripts.index(FINDER_SETUP)
    assert st["setup"]["status"] == "ok" and st["setup_mode"] == "sync" and env.is_environment_used
    rec = json.loads((tmp_path / "setup.json").read_text())
    assert rec["command"] == FINDER_SETUP and rec["attempts"][0]["returncode"] == 0


def test_a_failed_setup_is_retried_once_when_safe_and_then_reported(tmp_path):
    fail = {"returncode": 1, "output": "", "error": "execution error: Finder got an error (-600)"}
    missing = {"returncode": 0, "output": "/Users/admin/Documents/benchmark_files/NCE/Languages\n"}
    br, bridge, env, _ = _bridge([(FINDER_SETUP, [fail, fail]), ("[ -e", missing)])
    st = bridge.reset({"id": "t1", "pre_command": FINDER_SETUP}, tmp_path)
    assert env.controller.scripts.count(FINDER_SETUP) == 2
    s = st["setup"]
    assert s["status"] == "failed" and s["attempts"] == 2 and "Languages" in s["reason"]
    assert "-600" in json.loads((tmp_path / "setup.json").read_text())["attempts"][0]["error"]


def test_a_setup_that_makes_documents_is_not_run_twice(tmp_path):
    _, bridge, env, _ = _bridge([(KEYNOTE_SETUP, {"returncode": 1, "output": "error"})])
    st = bridge.reset({"id": "t2", "pre_command": KEYNOTE_SETUP}, tmp_path)
    assert env.controller.scripts.count(KEYNOTE_SETUP) == 1
    assert st["setup"]["status"] == "failed" and st["setup"]["reason"] == "exit code 1"


def test_retry_recovers_when_the_second_run_works(tmp_path):
    fail = {"returncode": 1, "output": "err"}
    _, bridge, env, _ = _bridge([(FINDER_SETUP, [fail, {"returncode": 0, "output": ""}])])
    assert bridge.reset({"id": "t1", "pre_command": FINDER_SETUP}, tmp_path)["setup"]["status"] == "ok"


def test_launch_mode_keeps_macarenas_way(tmp_path):
    _, bridge, env, seen = _bridge([], setup_mode="launch")
    st = bridge.reset({"id": "t1", "pre_command": FINDER_SETUP}, tmp_path)
    assert seen[-1]["pre_command"] == FINDER_SETUP and FINDER_SETUP not in env.controller.scripts
    assert st["setup"] == {"mode": "launch", "status": "launched"}


def test_harness_folders_are_refused_in_every_tool_and_task_folders_are_not():
    br = _mod("bridge")
    for args in ({"shell_command": "ls -la ~/vm_files/server"}, {"script": "cat /Users/admin/Benchmark_Backup/a"},
                 {"path": "/tmp/otto-bench/x"}):
        assert "harness" in (br.script_refusal("run_osascript", args, None) or "")
    assert br.script_refusal("type_text", {"text": "rm ~/Benchmark_Backup/x"}, None)
    assert br.script_refusal("run_osascript", {"shell_command": "ls ~/Documents/benchmark_files"}, None) is None
    msg = br.harness_path_refusal({"shell_command": "ls ~/vm_files"})
    assert "vm_fail" not in msg                             # says nothing that would steer an infeasible-task answer


def test_no_task_instruction_names_a_harness_folder():
    root = Path(__file__).resolve().parents[2] / "MacArena" / "evaluation_examples"
    files = list(root.glob("*/*/*.json"))
    if not files:
        pytest.skip("no MacArena checkout next to this repo")
    br = _mod("bridge")
    import re
    named = [f.name for f in files if re.search(br.HARNESS_PATH_RE, json.loads(f.read_text()).get("instruction", ""))]
    assert named == []


def test_runner_records_the_setup_and_old_bridges_read_as_launch():
    run = _mod("run")
    assert run.setup_fields({"setup_mode": "sync", "setup": {"status": "failed", "reason": "exit code 1",
                                                              "attempts": 2}}) == {
        "setup_mode": "sync", "setup_status": "failed", "setup_error": "exit code 1", "setup_attempts": 2}
    assert run.setup_fields({}) == {"setup_mode": "launch", "setup_status": "unknown"}


def _row(i, score, **kw):
    return {"id": i, "source": "macosworld", "domain": "d", "run": 1, "score": score, "status": "ok", "steps": 3,
            "track": "native", "max_steps": 100, "step_unit": "turns", **kw}


def test_report_gives_the_measured_and_the_adjusted_score_side_by_side():
    rep = _mod("report")
    bad = {"bad1": {"status": "confirmed"}, "sus1": {"status": "suspected"}}
    rows = [_row("a", 1.0, setup_mode="sync", setup_status="ok"),
            _row("b", 0.0, setup_mode="sync", setup_status="failed", setup_error="exit code 1"),
            _row("bad1", 0.0, setup_mode="sync", setup_status="ok"),
            _row("sus1", 0.0, setup_mode="sync", setup_status="ok")]
    keep, left = rep.adjusted_rows(rows, bad)
    assert [r["id"] for r in keep] == ["a", "sus1"]                     # suspected graders are NOT left out
    assert left["setup_failed_runs"] == 1 and left["bad_grader_tasks"] == ["bad1"]
    assert left["suspected_grader_tasks"] == ["sus1"]
    s = rep.summarize(rows)
    # summarize reads the real known_bad.json (none of these ids): only the failed setup is left out there
    assert s["overall"]["mean_of_runs"]["mean_score"] == 0.25 and s["adjusted"]["mean_of_runs"]["n"] == 3
    text = rep.render(s)
    assert "mean-of-runs (per attempt)   : mean score 25.0%" in text and "adjusted (" in text


def test_report_refuses_to_pool_setup_modes():
    rep = _mod("report")
    with pytest.raises(ValueError, match="setup modes"):
        rep.summarize([_row("a", 1.0, setup_mode="sync"), _row("b", 0.0)])


def test_known_bad_list_only_confirms_what_has_evidence():
    rep = _mod("report")
    bad = rep.known_bad()
    assert bad["77be0051-e4d5-139b-04d4-a760344aa282"]["status"] == "confirmed"
    assert bad["8887a199-469d-c544-09d4-bb32d5b285b4"]["status"] == "suspected"
    assert all(e.get("reason") and e.get("evidence") for e in bad.values())


def test_report_will_not_score_a_same_task_smoke_run():
    rep = _mod("report")
    rows = [_row("a", 1.0, setup_mode="sync", memory_mode="snapshot:s", snapshot_id="s", memory_same_task=True,
                 memory_leak=["ep_1:r0"])]
    with pytest.raises(ValueError, match="same-task recall smoke"):
        rep.summarize(rows)
    with pytest.raises(ValueError, match="different procedural memory"):
        rep.summarize([_row("a", 1.0, setup_mode="sync", memory_mode="snapshot:s", snapshot_id="s"),
                       _row("b", 1.0, setup_mode="sync", memory_mode="snapshot:s", snapshot_id="s",
                            memory_same_task=True)])
