"""Procedural memory (backend/procedural_memory): compaction, admission, retrieval, the tool, and the benchmark
bookkeeping. No model: embeddings use HashEmbedder, enrichment its deterministic fallback or a stub LLM."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from backend.procedural_memory import compact, enrich, judge, policy, retrieve, service, tool
from backend.procedural_memory.embed import HashEmbedder
from backend.procedural_memory.store import ProcedureStore

ROOT = Path(__file__).resolve().parents[1]
MA = ROOT / "scripts" / "macarena"


def run(coro):
    return asyncio.run(coro)


# ── transcripts ─────────────────────────────────────────────────────────────────────────────────────────────

def call(tool_name, args, result, cid, tag=None):
    meta_c = {"args": args, "tool_call_id": cid}
    meta_r = {"name": tool_name, "tool_call_id": cid}
    if tag:
        meta_c["subagent"] = meta_r["subagent"] = tag
    return [{"type": "tool_call", "content": tool_name, "metadata": meta_c},
            {"type": "tool_result", "content": result, "metadata": meta_r}]


AUTOMATOR = "Automator"
DIALOG = "Automator  window 'Untitled'  [95]B'Quick Action' [127]B'Choose'"


def automator_rows(tag=None, *, fail_tail=True):
    rows = [{"type": "user", "content": 'Task: Make a Quick Action named "Hello" that echoes input.\n\n'
                                        "What is on the screen: Finder\n\nStep budget: 100."}]
    n = iter(range(1000))
    c = lambda t, a, r: call(t, a, r, f"c{next(n)}", tag)  # noqa: E731
    rows += c("launch_app", {"app_name": AUTOMATOR}, DIALOG)
    rows += c("press_control", {"index": 95}, "Pressed control #95 (Button 'Quick Action')  Dialog now (Automator)")
    rows += c("press_control", {"index": 127}, "Pressed control #127  Controls now (Automator) window 'Untitled (Quick Action)'")
    for _ in range(3):
        rows += c("click", {"index": 9}, "Clicked control #9 (Button 'search') at (488, 91)  Controls now (Automator), "
                                         "read after the action: Same 112 controls as the last read")
    rows += c("double_click", {"index": 45}, "Double-clicked control #45 (TextField 'Run Shell Script')  Controls added "
                                             "or gone since the last read: [99]TA'cat'")
    rows += c("type_into_control", {"index": 99, "text": 'echo "$@"'}, "Typed 'echo \"$@\"' into control #99  The field now reads 'echo'")
    rows += c("menu_select", {"app_name": AUTOMATOR, "path": "File > Save…"}, "Chose File ▸ Save… in Automator. Now open: sheet")
    rows += c("run_osascript", {"shell_command": "ls ~/Library/Services"}, '{"ok": true, "stdout": "Hello.workflow"}')
    if fail_tail:
        rows += c("run_osascript", {"shell_command": "rm -rf ~/Library/Services/Hello.workflow"}, '{"ok": true}')
        rows += c("press_control", {"index": 83}, "Error: control #83 is not available")
    rows += c("vm_done", {}, "Recorded: the run is done.")
    return rows


def test_split_runs_maps_task_calls_to_tagged_subagent_rows():
    rows = [{"type": "user", "content": "find jobs"}]
    rows += [{"type": "tool_call", "content": "task", "metadata": {"tool_call_id": "t1", "args": {
        "description": "Search LinkedIn for roles", "subagent_type": "browser-agent"}}}]
    rows += [{"type": "tool_call", "content": "task", "metadata": {"tool_call_id": "t2", "args": {
        "description": "Search Wellfound", "subagent_type": "browser-agent"}}}]
    rows += call("browser_navigate", {"url": "https://www.linkedin.com/jobs"}, "ok", "a", "browser-agent #1")
    rows += call("browser_navigate", {"url": "https://wellfound.com"}, "ok", "b", "browser-agent #2")
    rows += call("web_research", {"query": "x"}, "ok", "c")                   # the orchestrator's own call
    rows += [{"type": "tool_result", "content": "done 1", "metadata": {"name": "task", "tool_call_id": "t1"}}]
    runs = compact.split_runs(rows, "s")
    assert [(r.subagent_tag, r.task_text) for r in runs] == [("browser-agent #1", "Search LinkedIn for roles"),
                                                            ("browser-agent #2", "Search Wellfound")]
    assert runs[0].final_text == "done 1" and len(runs[0].calls) == 1
    assert compact.compact(runs[0]).apps == ["linkedin.com"]


def test_direct_agent_run_uses_the_instruction_not_the_harness_prompt():
    runs = compact.split_runs(automator_rows(), "s", direct_agent="macarena-desktop-agent")
    assert len(runs) == 1 and runs[0].task_text == 'Make a Quick Action named "Hello" that echoes input.'
    assert compact.split_runs(automator_rows(), "s", direct_agent="coding-agent") == []


def test_compaction_resolves_labels_slots_values_and_finds_noops():
    ep = compact.compact(compact.split_runs(automator_rows(), "s", direct_agent="macarena-desktop-agent")[0])
    lines = [s.line() for s in ep.steps if s.keep]
    assert "press_control('Quick Action')" in lines and "press_control('Choose')" in lines
    assert "double_click('Run Shell Script')" in lines
    assert "type_into_control('cat', text='<text>')" in lines              # typed text is a slot, never stored
    assert not any("search" in ln for ln in lines)                          # the no-op clicks are not a recipe
    assert any("'search' changed nothing (3 tries" in f.text for f in ep.facts)
    ls = next(s for s in ep.steps if s.raw_args.get("shell_command", "").startswith("ls"))
    assert ls.kind == "read" and not ls.keep                                # a looking shell command is a read
    assert [a.body for a in ep.artifacts] == ["rm -rf ~/Library/Services/Hello.workflow"]
    assert ep.artifacts[0].writes_or_sends
    assert ep.self_report == "success" and not ep.infra_error and ep.apps == [AUTOMATOR]


def test_strip_values_removes_names_files_numbers_and_urls():
    t = compact.strip_values('Rename "Q3 report.xlsx" to final.csv and email bob@x.com at https://a.b/c by 12/05')
    assert "Q3" not in t and "bob@" not in t and "https" not in t and "12/05" not in t


# ── policy ──────────────────────────────────────────────────────────────────────────────────────────────────

def lab(source, value, at=1.0):
    return {"source": source, "value": value, "at": at}


def test_self_report_and_uncalibrated_judge_admit_nothing():
    p = policy.Policy()
    assert policy.decide([lab("self_report", "success")], actions=5, infra_error=False, policy=p).status == "pending"
    assert policy.decide([lab("judge", "success")], actions=5, infra_error=False, policy=p).status == "pending"
    trusted = policy.Policy(judge_success_trusted=True)
    assert policy.decide([lab("judge", "success")], actions=5, infra_error=False, policy=trusted).status == "admitted"
    assert policy.decide([lab("judge", "failure")], actions=5, infra_error=False, policy=trusted).status == "pending"


def test_most_trusted_label_wins_and_infra_or_tiny_runs_are_excluded():
    labels = [lab("benchmark", "failure", 1), lab("user", "success", 0), lab("judge", "success", 9)]
    assert policy.effective_label(labels)[:2] == ("success", "user")
    p = policy.Policy()
    assert policy.decide(labels, actions=5, infra_error=True, policy=p).status == "excluded"
    assert policy.decide(labels, actions=1, infra_error=False, policy=p).status == "excluded"


def test_failed_run_contributes_confirmed_segments_but_never_its_last():
    segs = [{"ord": 1, "label": "success", "recipe": ["a"]}, {"ord": 2, "label": "failure", "recipe": ["b"]},
            {"ord": 3, "label": "success", "recipe": ["c"]}]
    got = policy.admitted_segments(segs, "failure", policy.Policy())
    assert [s["ord"] for s in got] == [1] and got[0]["seg_source"] == "segment_evidence"
    assert policy.admitted_segments(segs, "failure", policy.Policy(segments_from_failed_runs=False)) == []
    assert [s["ord"] for s in policy.admitted_segments(segs, "success", policy.Policy())] == [1, 2, 3]


# ── store, admission, retrieval ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mem(tmp_path, monkeypatch):
    path = tmp_path / "procedures.db"
    rt = service.Runtime(mode="app", serve_path=path, write_path=path, floor=0.2, max_tokens=800,
                         policy=policy.Policy(), enrich_with_llm=False, serve_tool=True)
    monkeypatch.setattr(service, "runtime", lambda: rt)
    service.set_embedder(HashEmbedder())
    yield rt, service.store(path)
    service.set_embedder(None)


def save(rows, sid="s1", agent="macarena-desktop-agent", label=None, task_id=None):
    return run(service.save_session(sid, rows=rows, meta={"agent_name": agent}, label=label,
                                    label_source="benchmark" if label else None, task_id=task_id))


def test_saved_runs_are_pending_without_vectors_until_labelled(mem):
    _, st = mem
    [eid] = save(automator_rows())
    ep = st.episode(eid)
    assert ep["status"] == "pending" and st.items_for_episode(eid) == []
    assert [x["source"] for x in st.labels(eid)] == ["self_report"]
    assert save(automator_rows()) == [eid]                                  # saving again is idempotent
    run(service.label_session("s1", "success", "user"))
    ep = st.episode(eid)
    assert ep["status"] == "admitted" and ep["label_source"] == "user"
    kinds = sorted({i.kind for i in st.items_for_episode(eid)})
    assert kinds == ["fact", "recipe"]


def test_relabelling_a_success_as_failure_removes_its_whole_run_recipe(mem):
    _, st = mem
    [eid] = save(automator_rows(), label="success")
    assert f"{eid}:r0" in {i.id for i in st.items_for_episode(eid)}
    run(service.label_session("s1", "failure", "user"))
    ids = {i.id for i in st.items_for_episode(eid)}
    assert f"{eid}:r0" not in ids and any(":f" in i for i in ids)


def test_recall_returns_one_recipe_with_app_facts_under_the_cap(mem):
    rt, st = mem
    save(automator_rows(), label="success", task_id="T1")
    res = run(retrieve.recall(st, service.get_embedder(), query="Automator: Quick Action, search for Run Shell Script",
                              family="desktop", app=AUTOMATOR, floor=0.05, max_tokens=800))
    recipes = [i for i in res.returned if i.kind == "recipe"]
    assert len(recipes) == 1 and not res.abstained
    assert "Known in this app" in res.text and "search" in res.text
    assert res.tokens <= 800 and "never copy values" in res.text
    assert "Hello" not in res.text                                          # no values from the other task


def test_recall_abstains_below_floor_and_filters_app_task_and_near_duplicates(mem):
    _, st = mem
    save(automator_rows(), label="success", task_id="T1")
    emb = service.get_embedder()
    high = run(retrieve.recall(st, emb, query="Automator: Quick Action", family="desktop", app=AUTOMATOR, floor=0.99))
    assert high.abstained and "No similar procedure" in high.text
    other_app = run(retrieve.recall(st, emb, query="Automator: Quick Action", family="desktop", app="Numbers", floor=0.0))
    assert other_app.abstained
    browser = run(retrieve.recall(st, emb, query="Automator: Quick Action", family="browser", app=None, floor=0.0))
    assert browser.abstained
    same_task = run(retrieve.recall(st, emb, query="Automator: Quick Action", family="desktop", app=AUTOMATOR,
                                    floor=0.0, exclude_task_ids={"T1"}))
    assert same_task.abstained
    twin = run(retrieve.recall(st, emb, query="Automator: Quick Action", family="desktop", app=AUTOMATOR, floor=0.0,
                               current_task_text='Make a Quick Action named "Bye" that echoes input.'))
    assert twin.abstained                                                   # same template, other value


def test_render_drops_recipe_lines_to_fit_the_token_cap(mem):
    _, st = mem
    [eid] = save(automator_rows(), label="success")
    it = st.item(f"{eid}:r0")
    it.body["lines"] = [f"press_control('control number {k} with a long label')" for k in range(60)]
    text = retrieve.render(it, [], max_tokens=120)
    assert len(text) / 4 <= 125 and text.startswith(f"[proc {eid}:r0]")


def test_script_detail_flags_writes(mem):
    _, st = mem
    [eid] = save(automator_rows(), label="success")
    text = retrieve.detail(st, f"{eid}:r0", "script")
    assert "rm -rf" in text and "(writes or sends)" in text


def test_enrichment_uses_llm_intents_and_lessons_and_falls_back_on_garbage(mem):
    rt, st = mem
    [eid] = save(automator_rows())
    st.add_label(eid, "benchmark", "failure", 1.0)

    async def llm(prompt):
        assert "Outcome: failure" in prompt and "Hello" not in prompt.split("Steps")[0]
        return json.dumps({"intent": "Automator: create a Quick Action running a shell script",
                           "segments": [{"ord": 1, "intent": "Automator: start a Quick Action"}],
                           "lessons": [{"title": "Check Pass input", "sentence": "s", "content": "Read the pop-up."}]})

    run(service.reconsider(eid, st=st, rt=rt, llm=llm))
    items = {i.kind: i for i in st.items_for_episode(eid)}
    assert st.episode(eid)["intent"].startswith("Automator: create")
    assert items["lesson"].from_outcome == "failure" and items["lesson"].body["title"] == "Check Pass input"

    async def bad(prompt):
        return "no json here"

    out = run(enrich.enrich({"segments": [], "task_text": "x", "apps": ["A"]}, "success", "user", bad))
    assert out["source"] == "fallback" and out["lessons"] == []


def test_embedder_mismatch_is_refused(mem):
    _, st = mem
    save(automator_rows(), label="success")
    service.set_embedder(HashEmbedder(128))
    with pytest.raises(RuntimeError, match="embedded with"):
        run(service.label_session("s1", "success", "user"))


# ── the tool ────────────────────────────────────────────────────────────────────────────────────────────────

def test_tool_only_for_learning_agents_and_only_when_serving(mem, monkeypatch):
    rt, _ = mem
    assert [t.name for t in tool.build_procedure_tools("macos-desktop-agent", "s")] == ["recall_procedure"]
    assert tool.build_procedure_tools("coding-agent", "s") == []
    monkeypatch.setattr(service, "runtime", lambda: service.Runtime(**{**rt.__dict__, "serve_tool": False}))
    assert tool.build_procedure_tools("macos-desktop-agent", "s") == []


def test_tool_records_usage_and_outcome(mem):
    _, st = mem
    save(automator_rows(), sid="old", label="success")
    text, rec = run(tool.run_recall(agent="macos-desktop-agent", session_id="new",
                                    task="Automator: Quick Action Run Shell Script", app=AUTOMATOR))
    assert rec["used"] and not rec["abstained"] and rec["candidates"]
    assert st.usage_for_session("new")
    assert service.record_outcome("new", score=1.0, outcome="success") == len(rec["used"])
    stats = st.usage_stats()[rec["used"][0]["item"]]
    assert stats == {"uses": 1, "known": 1, "passes": 1}


def test_benchmark_runtime_serves_a_read_only_snapshot(tmp_path, monkeypatch):
    snap = tmp_path / "snap.db"
    service.set_embedder(HashEmbedder())
    try:
        st = ProcedureStore(snap)
        ids = service.save_rows(automator_rows(), session_id="x", direct_agent="macarena-desktop-agent",
                                track="benchmark", st=st, task_id="T1")
        st.add_label(ids[0], "benchmark", "success", 1.0)
        rt = service.Runtime(mode="build", serve_path=None, write_path=snap, floor=0.0, max_tokens=800,
                             policy=policy.Policy(), enrich_with_llm=False, serve_tool=False)
        run(service.reconsider(ids[0], st=st, rt=rt))
        monkeypatch.setenv(service.ENV_MODE, "serve")
        monkeypatch.setenv(service.ENV_SERVE, str(snap))
        monkeypatch.delenv(service.ENV_WRITE, raising=False)
        brt = service.runtime()
        assert brt.benchmark and brt.serve_tool and brt.write_path is None
        ro = service.serve_store(brt)
        assert ro.read_only
        with pytest.raises(Exception):
            ro.record_usage(["x"], "s")
        with pytest.raises(PermissionError):
            from backend.config import get_app_data_dir
            service.set_write_path(get_app_data_dir() / "procedures.db")
        monkeypatch.setenv(service.ENV_MODE, "serve")
        monkeypatch.delenv(service.ENV_SERVE)
        with pytest.raises(ValueError):
            service.runtime()
        monkeypatch.setenv(service.ENV_MODE, "off")
        assert not service.runtime().serve_tool
    finally:
        service._runtime_override.clear()
        service.set_embedder(None)


# ── judge statistics ────────────────────────────────────────────────────────────────────────────────────────

def test_wilson_gate_needs_the_lower_bound_over_the_bar():
    good = judge.agreement([("success", "success")] * 59 + [("failure", "failure")] * 10).as_dict()
    assert judge.gate(good)["success_trusted"]                              # 59/59: lower bound ~0.94
    meh = judge.agreement([("success", "success")] * 18 + [("success", "failure")] * 2).as_dict()
    assert meh["success_precision"] == 0.9 and not judge.gate(meh)["success_trusted"]
    assert judge._verdict('<think>x</think>{"verdict": "failure"}') == "failure"


def test_judge_is_unanimous_and_never_sees_the_final_message():
    r = compact.split_runs(automator_rows(), "s", direct_agent="macarena-desktop-agent")[0]
    r.final_text = "I completed the task successfully!"
    seen = []

    async def llm(prompt):
        seen.append(prompt)
        return '{"verdict": "success"}' if len(seen) < 3 else '{"verdict": "failure"}'

    out = run(judge.judge(r, llm, samples=3))
    assert out["verdict"] == "unknown" and out["votes"] == ["success", "success", "failure"]
    assert "completed the task successfully" not in seen[0]


# ── benchmark bookkeeping ───────────────────────────────────────────────────────────────────────────────────

def _load(name, path):
    sys.path.insert(0, str(MA))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(MA))


br = _load("macarena_bridge_procmem", MA / "bridge.py")
report = _load("macarena_report_procmem", MA / "report.py")


class _Guest:
    def deploy(self):
        return {"ok": True}

    def call(self, body):
        return {"ok": True, "cost": 0 if body["tool"] in br.native_tools.OBSERVATION else 1, "result": "ok"}


def _bridge(tmp_path, unit, cap):
    b = br.Bridge(br.FakeEnv(), max_steps=cap, settle=0, pause=0, guest=_Guest(), step_unit=unit)
    b.reset({"id": "t1", "instruction": "x", "evaluator": []}, tmp_path / "r")
    b.armed = True
    return b


def test_memory_lookup_costs_a_step_by_actions_and_is_refused_past_the_budget(tmp_path):
    b = _bridge(tmp_path, "actions", 2)
    assert b.tick({"kind": "tool", "tool": "recall_procedure"})["ok"] and b.steps == 1
    assert b.native({"tool": "click", "args": {}})["ok"] and b.steps == 2
    res = b.tick({"kind": "tool", "tool": "recall_procedure"})
    assert not res["ok"] and b.memory_calls == 1


def test_memory_lookup_shares_its_turn_by_turns(tmp_path):
    b = _bridge(tmp_path, "turns", 3)
    assert b.tick({"kind": "tool", "tool": "recall_procedure", "turn": "t1"})["ok"] and b.steps == 1
    assert b.native({"tool": "click", "args": {}, "turn": "t1"})["ok"] and b.steps == 1      # same reply: free
    assert b.native({"tool": "click", "args": {}, "turn": "t2"})["ok"] and b.steps == 2
    assert b.tick({"kind": "tool", "tool": "recall_procedure", "turn": "t2"})["ok"] and b.steps == 2
    st = b.state()
    assert st["memory_calls"] == 2 and st["memory_turns"] == 2


def test_memory_record_reaches_state_and_the_run_folder(tmp_path):
    b = _bridge(tmp_path, "actions", 5)
    rec = {"query": "Automator: x", "used": [{"item": "ep_1:s1", "source_task": "t0"}], "abstained": False,
           "tokens": 120, "candidates": [{"item": "ep_1:s1", "cos": 0.6}]}
    assert b.tick({"kind": "memory", "record": rec})["ok"]
    assert b.tick({"kind": "memory", "record": {"used": [], "abstained": True, "tokens": 30}})["ok"]
    st = b.state()
    assert [u["item"] for u in st["memory_used"]] == ["ep_1:s1"]
    assert st["memory_abstained"] == 1 and st["memory_tokens"] == 150
    lines = (tmp_path / "r" / "memory_calls.jsonl").read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["candidates"][0]["cos"] == 0.6
    b.reset({"id": "t2", "instruction": "x", "evaluator": []}, tmp_path / "r2")
    assert b.state()["memory_used"] == []                                    # per task


def _row(i, **kw):
    return {"id": f"t{i}", "run": 1, "score": 1.0, "status": "ok", "source": "osworld", "domain": "d", "steps": 3,
            "track": "native", **kw}


def test_report_keeps_memory_arms_apart_and_rejects_leaks():
    with pytest.raises(ValueError, match="procedural memory"):
        report.summarize([_row(1, memory_mode="snapshot:a", snapshot_id="a"), _row(2)])
    with pytest.raises(ValueError, match="memory_leak"):
        report.summarize([_row(1, memory_leak=["x"])])
    s = report.summarize([_row(1, memory_mode="snapshot:a", snapshot_id="a", memory_calls=1,
                               memory_used=[{"item": "i1"}]),
                          _row(2, score=0.0, memory_mode="snapshot:a", snapshot_id="a")])
    m = s["memory"]
    assert m["runs_returned"] == 1 and m["pass_rate_returned"] == 1.0 and m["pass_rate_not_returned"] == 0.0
    assert "procedural memory: snapshot:a" in report.render(s)
    assert report.summarize([_row(1)])["memory"] is None


def test_guard_allows_only_the_real_memory_tool():
    from types import SimpleNamespace
    from backend import macarena_guard_middleware as g, macarena_native
    assert "recall_procedure" in macarena_native.AGENT_TOOLS
    ok = SimpleNamespace(metadata={"procedure_memory": True})
    fake = SimpleNamespace(metadata={})
    assert g.refusal(g.NATIVE_AGENT, "recall_procedure", ok) is None
    assert "Blocked" in g.refusal(g.NATIVE_AGENT, "recall_procedure", fake)


def test_benchmark_floor_override_and_snapshot_embedder(tmp_path, monkeypatch):
    snap = tmp_path / "snap.db"
    service.set_embedder(HashEmbedder(128))
    st = ProcedureStore(snap)
    ids = service.save_rows(automator_rows(), session_id="x", direct_agent="macarena-desktop-agent",
                            track="benchmark", st=st, task_id="T1")
    st.add_label(ids[0], "benchmark", "success", 1.0)
    rt = service.Runtime(mode="build", serve_path=None, write_path=snap, floor=0.0, max_tokens=800,
                         policy=policy.Policy(), enrich_with_llm=False, serve_tool=False)
    run(service.reconsider(ids[0], st=st, rt=rt))
    service.set_embedder(None)
    try:
        monkeypatch.setenv(service.ENV_MODE, "serve")
        monkeypatch.setenv(service.ENV_SERVE, str(snap))
        monkeypatch.delenv(service.ENV_EMBEDDER, raising=False)
        monkeypatch.setenv(service.ENV_FLOOR, "0.2")
        s = service.status()
        assert s["floor"] == 0.2 and s["embedder"] == "hash-128"       # queries use the snapshot's embedder
        monkeypatch.setenv(service.ENV_FLOOR, "low")
        with pytest.raises(ValueError):
            service.runtime()
    finally:
        service.set_embedder(None)


def test_the_same_fact_from_two_runs_is_said_once(mem):
    _, st = mem
    save(automator_rows(), sid="a", label="success", task_id="T1")
    save(automator_rows(), sid="b", label="success", task_id="T2")
    res = run(retrieve.recall(st, service.get_embedder(), query="Automator: Quick Action, search for Run Shell Script",
                              family="desktop", app=AUTOMATOR, floor=0.05))
    assert res.text.count("changed nothing") == 1
    assert any(c.cut == "duplicate" for c in res.candidates)


def test_report_keeps_floors_apart():
    with pytest.raises(ValueError, match="procedural memory"):
        report.summarize([_row(1, memory_mode="snapshot:a", snapshot_id="a", memory_floor=0.3),
                          _row(2, memory_mode="snapshot:a", snapshot_id="a", memory_floor=0.45)])


def test_memory_instruction_is_step_zero_of_the_benchmark_workflow_and_only_when_bound(mem):
    from backend.agent_library import _MACARENA_DESKTOP_AGENT_PROMPT as base
    bound = tool.build_procedure_tools("macarena-desktop-agent", "s")
    out = tool.apply_prompt(base, bound, agent="macarena-desktop-agent")
    assert out.count("recall_procedure") >= 1 and "### Workflow\n0. **Search memory first.**" in out
    assert out.index("0. **Search memory first.**") < out.index("1. **Look first.**")
    assert out.replace(tool.STEP_ZERO, "") == base                        # nothing else changes
    assert tool.apply_prompt(base, [], agent="x") == base                 # memory off: the prompt as it was
    assert tool.apply_prompt("plain", bound).endswith(tool.PROMPT_HINT)   # no numbered workflow: a section
    from types import SimpleNamespace
    assert tool.apply_prompt(base, [SimpleNamespace(name="recall_procedure", metadata={})]) == base


# ── success and failure together ────────────────────────────────────────────────────────────────────────────

def failed_run_rows():
    return automator_rows(fail_tail=True)


def test_a_recipe_comes_with_what_went_wrong_before(mem):
    rt, st = mem
    save(automator_rows(), sid="ok", label="success", task_id="T1")
    [bad] = save(failed_run_rows(), sid="bad", label="failure", task_id="T2")

    async def llm(prompt):
        return json.dumps({"intent": "Automator: Quick Action with a shell script", "segments": [],
                           "lessons": [{"title": "Check Pass input", "sentence": "s",
                                        "content": "Set Pass input to 'as arguments' before running."}]})

    run(service.reconsider(bad, st=st, rt=rt, llm=llm))
    res = run(retrieve.recall(st, service.get_embedder(), query="Automator: Quick Action, search for Run Shell Script",
                              family="desktop", app=AUTOMATOR, floor=0.05))
    assert any(i.kind == "recipe" for i in res.returned)
    assert any(i.kind == "lesson" and i.from_outcome == "failure" for i in res.returned)
    assert "Check: Check Pass input" in res.text and "went wrong" in res.text


def test_a_missing_embedding_model_keeps_the_run_and_its_labels(mem):
    rt, st = mem

    class Broken:
        name = "minilm"

        async def embed(self, texts):
            raise RuntimeError("sentence-transformers is not installed")

    service.set_embedder(Broken())
    [eid] = save(automator_rows(), label="success", task_id="T1")       # no exception
    ep = st.episode(eid)
    assert ep["status"] == "pending" and ep["label"] == "success"
    assert {x["source"] for x in st.labels(eid)} == {"self_report", "benchmark"}
    assert st.items_for_episode(eid) == [] and not st.get_meta().get("embedder")
    service.set_embedder(HashEmbedder())                                   # later, with a model: admitted
    run(service.reconsider(eid, st=st, rt=rt))
    assert st.episode(eid)["status"] == "admitted" and st.items_for_episode(eid)


def test_first_message_asks_for_the_tool_only_when_it_is_offered(mem, monkeypatch):
    rt, _ = mem
    q = "Task: Fill all the blank cells with the value in the cell above it"
    out = tool.first_message("macarena-desktop-agent", q)
    assert out.startswith(q) and out.endswith(tool.FIRST_MESSAGE_NUDGE)
    assert tool.first_message("coding-agent", q) == q
    monkeypatch.setattr(service, "runtime", lambda: service.Runtime(**{**rt.__dict__, "serve_tool": False}))
    assert tool.first_message("macarena-desktop-agent", q) == q                 # memory off: unchanged


def test_the_app_the_agent_names_is_matched_to_the_stored_name():
    known = ["Google Chrome", "LibreOffice", "LibreOffice Writer", "QuickTime Player", "Finder", "Preview"]
    assert tool.resolve_app("Chrome", known) == "Google Chrome"
    assert tool.resolve_app("google chrome", known) == "Google Chrome"
    assert tool.resolve_app("LibreOffice Calc", known) == "LibreOffice"
    assert tool.resolve_app("LibreOffice Writer", known) == "LibreOffice Writer"
    assert tool.resolve_app("QuickTime", known) == "QuickTime Player"
    assert tool.resolve_app("Keynote", known) == "Keynote"                      # nothing stored: unchanged
    assert tool.resolve_app(None, known) is None


def test_recall_finds_items_when_the_agent_uses_a_short_app_name(mem):
    save(automator_rows(), sid="old", label="success", task_id="T1")
    text, rec = run(tool.run_recall(agent="macos-desktop-agent", session_id="new",
                                    task="Automator: Quick Action, search for Run Shell Script", app="automator"))
    assert rec["used"] and rec["app_resolved"] == "Automator" and rec["text"] == text


def test_same_task_recall_is_off_by_default_and_on_only_when_asked(mem, monkeypatch):
    """MACARENA_MEMORY_SAME_TASK=1 (smoke tests): a rerun of a stored task gets its own run back; normally it does not."""
    rt, st = mem
    save(automator_rows(), sid="old", label="success", task_id="T1")
    with st.connect() as db:
        stored_text = db.execute("select task_text from episodes").fetchone()[0]
    assert stored_text

    def recall(same, mode="benchmark"):
        monkeypatch.setattr(service, "runtime",
                            lambda: service.Runtime(**{**rt.__dict__, "same_task": same, "mode": mode}))
        return run(tool.run_recall(agent="macos-desktop-agent", session_id="new", app=AUTOMATOR,
                                   task="Automator: Quick Action Run Shell Script", current_task_text=stored_text))

    assert recall(False)[1]["abstained"]                       # benchmark: the identical task is left out
    _, rec = recall(True)
    assert not rec["abstained"] and rec["used"]
    # App: a task the user asks for again gets its saved run back (it used to be hidden, so "list the emails"
    # twice found nothing the second time).
    _, rec = recall(False, mode="app")
    assert not rec["abstained"] and rec["used"]


def test_same_task_switch_needs_memory_served(tmp_path, monkeypatch):
    snap = tmp_path / "snap.db"
    ProcedureStore(snap)
    monkeypatch.setenv(service.ENV_SAME_TASK, "1")
    monkeypatch.setenv(service.ENV_MODE, "serve")
    monkeypatch.setenv(service.ENV_SERVE, str(snap))
    assert service.runtime().same_task
    monkeypatch.setenv(service.ENV_MODE, "off")
    assert not service.runtime().same_task
    monkeypatch.setenv(service.ENV_MODE, "serve")
    monkeypatch.delenv(service.ENV_SAME_TASK)
    assert not service.runtime().same_task


# ── app API: saved runs stay visible with memory off ─────────────────────────────────────────────────────────

def test_episodes_route_reads_the_app_store_when_memory_is_off(tmp_path, monkeypatch):
    from backend.routes import procedures as routes
    import backend.config as config

    monkeypatch.delenv(service.ENV_MODE, raising=False)
    monkeypatch.setattr(config, "get_app_data_dir", lambda: tmp_path)
    off = service.Runtime(mode="off", serve_path=None, write_path=None, floor=0.45, max_tokens=800,
                          policy=policy.Policy(), enrich_with_llm=False, serve_tool=False)
    monkeypatch.setattr(service, "runtime", lambda: off)

    # Nothing saved yet: an empty list, not an error.
    assert run(routes.episodes()) == {"episodes": [], "counts": {}}
    assert tool.build_procedure_tools("macos-desktop-agent", "s") == []

    st = service.store(tmp_path / "procedures.db")
    eid, _ = st.upsert_episode(session_id="s1", subagent_tag="", agent="macos-desktop-agent", family="desktop",
                               track="app", task_id=None, task_text="Numbers: export to CSV", apps=["Numbers"],
                               self_report="unknown", data={})
    out = run(routes.episodes())
    assert [e["id"] for e in out["episodes"]] == [eid]
    assert out["counts"].get("episodes_pending") == 1
    assert run(routes.episode(eid))["episode"]["task_text"] == "Numbers: export to CSV"
    run(routes.delete(eid))
    assert run(routes.episodes())["episodes"] == []


# ── saved-runs limit ─────────────────────────────────────────────────────────────────────────────────────────

def _ep(st, sid, status="pending"):
    eid, _ = st.upsert_episode(session_id=sid, subagent_tag="", agent="macos-desktop-agent", family="desktop",
                               track="app", task_id=None, task_text=f"task {sid}", apps=["Notes"],
                               self_report="unknown", data={})
    if status != "pending":
        st.set_status(eid, status)
    return eid


def test_prune_drops_least_useful_oldest_first_and_spares_kept(tmp_path):
    from backend.procedural_memory.store import ProcedureStore
    st = ProcedureStore(tmp_path / "p.db")
    old_admitted = _ep(st, "a", "admitted")
    excluded = _ep(st, "b", "excluded")
    pending_old = _ep(st, "c")
    pending_new = _ep(st, "d")
    just_saved = _ep(st, "e")
    assert st.prune(0) == []                                     # 0 = no limit
    assert st.prune(10) == []                                    # under the limit
    assert st.prune(3, keep=[just_saved]) == [excluded, pending_old]
    left = {e["id"] for e in st.list_episodes()}
    assert left == {old_admitted, pending_new, just_saved}
    assert st.prune(1, keep=[just_saved]) == [pending_new, old_admitted]   # admitted go last


def test_save_session_applies_the_app_limit(mem, monkeypatch):
    rt, st = mem
    monkeypatch.setattr(service, "runtime", lambda: service.Runtime(**{**rt.__dict__, "max_saved_runs": 2}))
    for sid in ("s1", "s2"):
        _ep(st, sid)
    kept = list(st.list_episodes())
    seen = {}

    def fake_prune(max_runs, *, keep=()):
        seen.update(max_runs=max_runs, keep=list(keep))
        return []
    monkeypatch.setattr(st, "prune", fake_prune)
    monkeypatch.setattr(service, "save_rows", lambda *a, **k: ["ep_new"])
    async def no_reconsider(*a, **k):
        return None
    monkeypatch.setattr(service, "reconsider", no_reconsider)
    run(service.save_session("s3", rows=[], meta={"agent_name": "macos-desktop-agent"}))
    assert seen == {"max_runs": 2, "keep": ["ep_new"]} and len(kept) == 2


def test_config_limit_default_and_bounds():
    from backend.config import ProceduralMemoryConfig
    import pydantic
    assert ProceduralMemoryConfig().max_saved_runs == 500
    with pytest.raises(pydantic.ValidationError):
        ProceduralMemoryConfig(max_saved_runs=-1)


# ── chat thumbs up / down ────────────────────────────────────────────────────────────────────────────────────

def _desktop_ep(st, sid, tag, agent="macos-desktop-agent"):
    eid, _ = st.upsert_episode(session_id=sid, subagent_tag=tag, agent=agent,
                               family="browser" if agent == "browser-agent" else "desktop", track="app",
                               task_id=None, task_text=f"Notes: task {tag}", apps=["Notes"], self_report="unknown",
                               data={})
    return eid


def test_feedback_state_asks_only_for_unreviewed_desktop_runs(mem):
    rt, st = mem
    assert service.session_feedback_state("chat")["ask"] is False            # nothing saved yet
    _desktop_ep(st, "chat", "b", agent="browser-agent")
    assert service.session_feedback_state("chat")["ask"] is False            # browser runs are not asked about
    eid = _desktop_ep(st, "chat", "d")
    state = service.session_feedback_state("chat")
    assert state["ask"] is True and [e["id"] for e in state["episodes"]] == [eid]


def test_feedback_off_never_asks(mem, monkeypatch):
    rt, st = mem
    _desktop_ep(st, "chat", "d")
    monkeypatch.setattr(service, "runtime", lambda: service.Runtime(**{**rt.__dict__, "mode": "off",
                                                                        "write_path": None, "serve_path": None}))
    assert service.session_feedback_state("chat") == {"enabled": False, "ask": False, "answer": None, "episodes": []}
    with pytest.raises(ValueError):
        run(service.record_session_feedback("chat", True))


def test_thumbs_up_labels_and_admits(mem, monkeypatch):
    rt, st = mem
    eid = _desktop_ep(st, "chat", "d")
    seen = []

    async def fake_reconsider(episode_id, *, st, rt=None, **_):
        seen.append(episode_id)
        st.set_status(episode_id, "admitted", label="success", label_source="user")
    monkeypatch.setattr(service, "reconsider", fake_reconsider)
    out = run(service.record_session_feedback("chat", True))
    assert out == {"updated": [eid], "saved": 1, "still_pending": 0} and seen == [eid]
    assert ("user", "success") in {(lab["source"], lab["value"]) for lab in st.labels(eid)}
    state = service.session_feedback_state("chat")
    assert state["ask"] is False and state["answer"] == "success"
    assert run(service.record_session_feedback("chat", True))["updated"] == []    # answered once only


def test_thumbs_down_excludes_and_saves_nothing(mem):
    rt, st = mem
    eid = _desktop_ep(st, "chat", "d")
    out = run(service.record_session_feedback("chat", False))
    assert out["updated"] == [eid] and out["saved"] == 0
    assert st.episode(eid)["status"] == "excluded" and st.items_for_episode(eid) == []
    assert service.session_feedback_state("chat")["answer"] == "failure"


def test_procedures_page_didnt_work_excludes_and_worked_brings_back(mem, monkeypatch):
    from backend.routes import procedures as routes
    rt, st = mem
    eid = _desktop_ep(st, "chat2", "d")

    async def fake_reconsider(episode_id, *, st, rt=None, **_):
        st.set_status(episode_id, "admitted", label="success", label_source="user")
    monkeypatch.setattr(service, "reconsider", fake_reconsider)
    assert run(routes.episode_feedback(eid, {"worked": False})) == {"status": "excluded"}
    assert st.items_for_episode(eid) == []
    assert run(routes.episode_feedback(eid, {"worked": True})) == {"status": "admitted"}
    with pytest.raises(routes.HTTPException):
        run(routes.episode_feedback(eid, {"worked": "yes"}))


def test_similarity_threshold_bounds():
    from backend.config import ProceduralMemoryConfig
    import pydantic
    assert ProceduralMemoryConfig().similarity_floor == 0.45
    for bad in (-0.1, 1.5):
        with pytest.raises(pydantic.ValidationError):
            ProceduralMemoryConfig(similarity_floor=bad)
