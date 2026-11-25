"""Hermetic tests for the cad-tools MCP and the CAD agents.

No Docker or cadquery needed: the sandbox call is stubbed. A real-sandbox run is
covered by ``scripts/benchcad/bench.sh check``.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("mcp")

SERVER = Path(__file__).resolve().parents[1] / "backend" / "builtin_mcps" / "cad_tools" / "server.py"


@pytest.fixture()
def srv(tmp_path, monkeypatch):
    monkeypatch.setenv("CAD_WORK_ROOT", str(tmp_path))
    spec = importlib.util.spec_from_file_location("cad_tools_server", SERVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GOOD = {"ok": True, "metrics": {
    "bbox_min": [0, 0, 0], "bbox_max": [10, 10, 5], "size": [10, 10, 5], "volume": 500.0,
    "area": 400.0, "solids": 1, "faces": 6, "edges": 12, "valid": True, "center_of_mass": [5, 5, 2.5]}}


def _graph():
    return {"components": [{"id": "R1", "terminals": ["R1.1", "R1.2"]}],
            "nets": [{"id": "GND"}, {"id": "N1"}],
            "incidences": [["R1.1", "N1"], ["R1.2", "GND"]]}


def test_graph_valid(srv):
    assert srv.check_graph(_graph()) == []


@pytest.mark.parametrize("mutate, needle", [
    (lambda g: g["incidences"].append(["R1.1", "GND"]), "two nets"),
    (lambda g: g["incidences"].append(["R9.1", "GND"]), "no component declares"),
    (lambda g: g["incidences"].append(["R1.1", "NOPE"]), "not in nets"),
    (lambda g: g["nets"].append({"id": "GND"}), "duplicate net"),
    (lambda g: g["components"].append({"id": "R1", "terminals": ["X"]}), "duplicate component"),
])
def test_graph_problems(srv, mutate, needle):
    g = _graph(); mutate(g)
    assert any(needle in p for p in srv.check_graph(g))


def test_netlist_validate_tool(srv, tmp_path):
    (tmp_path / "pred_graph.json").write_text(json.dumps(_graph()))
    assert srv.netlist_validate(str(tmp_path)).startswith("VALID")
    (tmp_path / "pred_graph.json").write_text("{nope")
    assert "not JSON" in srv.netlist_validate(str(tmp_path))


def test_paths_must_stay_inside_workdir(srv, tmp_path):
    assert "outside the workdir" in srv.cad_run(str(tmp_path), code_file="../x.py")
    assert "does not exist" in srv.cad_run(str(tmp_path / "missing"))


def test_work_root_restriction(srv, tmp_path, monkeypatch):
    root = tmp_path / "root"; root.mkdir()
    other = tmp_path / "other"; other.mkdir()
    monkeypatch.setenv("CAD_WORK_ROOT", str(root))
    assert "outside CAD_WORK_ROOT" in srv.cad_run(str(other))


def test_cad_run_formats_result(srv, tmp_path, monkeypatch):
    (tmp_path / "answer.py").write_text("result = None")
    monkeypatch.setattr(srv, "_exec_probe", lambda *a, **k: GOOD)
    out = srv.cad_run(str(tmp_path))
    assert "OK" in out and "volume: 500.0" in out and "size (x,y,z): [10, 10, 5]" in out


def test_cad_run_reports_error(srv, tmp_path, monkeypatch):
    (tmp_path / "answer.py").write_text("boom")
    monkeypatch.setattr(srv, "_exec_probe", lambda *a, **k: {"ok": False, "error": "NameError: boom"})
    out = srv.cad_run(str(tmp_path))
    assert out.startswith("NOT OK") and "NameError" in out


def test_checkpoint_keeps_only_good_answers(srv, tmp_path, monkeypatch):
    code = tmp_path / "answer.py"
    results = iter([GOOD, {"ok": False, "error": "bad"}])
    monkeypatch.setattr(srv, "_exec_probe", lambda *a, **k: next(results))

    code.write_text("# v1")
    assert srv.answer_checkpoint(str(tmp_path), "save").startswith("Saved")
    code.write_text("# v2 broken")
    assert srv.answer_checkpoint(str(tmp_path), "save").startswith("NOT saved")
    assert (tmp_path / ".cad" / "answer.best.py").read_text() == "# v1"

    assert "Restored" in srv.answer_checkpoint(str(tmp_path), "restore")
    assert code.read_text() == "# v1"
    assert "Checkpoint exists" in srv.answer_checkpoint(str(tmp_path), "status")


def test_checkpoint_restore_without_save(srv, tmp_path):
    assert "no checkpoint" in srv.answer_checkpoint(str(tmp_path), "restore")


def test_budget_status_offline(srv, monkeypatch, tmp_path):
    monkeypatch.setenv("METER_URL", "http://127.0.0.1:9")
    assert "No meter reachable" in srv.budget_status(str(tmp_path))


def test_exec_mode(srv, monkeypatch):
    monkeypatch.setenv("CAD_EXEC", "docker")
    assert srv._exec_mode() == "docker"
    monkeypatch.setenv("CAD_EXEC", "nope")
    with pytest.raises(srv.CadError):
        srv._exec_mode()
    monkeypatch.setenv("CAD_EXEC", "auto")
    monkeypatch.setattr(srv.sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", lambda _n: None)
    with pytest.raises(srv.CadError, match="No sandbox"):
        srv._exec_mode()


def test_exec_probe_local_parses_last_json_line(srv, tmp_path, monkeypatch):
    fake = tmp_path / "probe.py"
    fake.write_text("print('noise'); print('{\"ok\": true}')")
    monkeypatch.setenv("CAD_EXEC", "local")
    monkeypatch.setattr(srv, "PROBE", fake)
    assert srv._exec_probe(tmp_path, [], 20) == {"ok": True}


def test_seatbelt_profile(srv, tmp_path):
    prof = srv.seatbelt_profile(tmp_path, [tmp_path / "venv"])
    work = str(tmp_path.resolve())
    assert prof.startswith("(version 1)\n(deny default)")
    assert prof.rstrip().endswith("(deny network*)")
    # the work folder is re-opened after the sensitive denies (it may sit under Otto's data folder)
    # read and write are allowed separately, after the denies (`allow file*` would not override them)
    assert prof.rindex(f'(allow file-read* (subpath "{work}"))') > prof.rindex("(deny file-read*")
    assert prof.rindex(f'(allow file-write* (subpath "{work}"))') > prof.rindex("(deny file-write*")
    assert ".ssh" in prof and "Keychains" in prof


# ── registry, policy, agents ────────────────────────────────────────────────

def test_registered_and_classified():
    from backend.action_policy.catalog import KNOWN_SERVER_IDS
    from backend.builtin_mcps.registry import BUILTIN_MCPS
    mcp = next(m for m in BUILTIN_MCPS if m.id == "cad-tools")
    assert mcp.enabled is False
    assert mcp.python == "3.12" and mcp.no_deps == ("nlopt==2.11.0",)
    req = (SERVER.parent / "requirements.txt").read_text()
    assert "cadquery==2.3.0" in req and "numpy==1.26.4" in req
    src = SERVER.parents[1] / mcp.source_dir_name
    assert (src / "server.py").is_file() and (src / "_probe.py").is_file()
    assert "cad-tools" in KNOWN_SERVER_IDS


def test_cad_agents_are_wired():
    from backend.agent_library import _BUILTIN_AGENT_VERSIONS
    from backend.session_manager import (
        _AGENT_SUBAGENTS, _AGENT_TOOL_ALLOWLIST, _AGENTS_WITHOUT_MEMORY,
    )
    for parent in ("cad-agent", "benchcad-agent"):
        for n in (parent, *_AGENT_SUBAGENTS[parent]):
            assert n in _BUILTIN_AGENT_VERSIONS, n
    for n in ("drawing-reader", "cad-critic"):
        assert n in _AGENT_TOOL_ALLOWLIST and n in _AGENTS_WITHOUT_MEMORY, n
    # trimmed: the benchmark agent keeps only the drawing reader; the general one adds the critic
    assert _AGENT_SUBAGENTS["benchcad-agent"] == ("drawing-reader",)
    assert _AGENT_SUBAGENTS["cad-agent"] == ("drawing-reader", "cad-critic")
    from backend.agent_library import _RENAMED_AGENTS
    for gone in ("assembly-planner", "pcb-tracer"):
        assert gone not in _BUILTIN_AGENT_VERSIONS and gone in _RENAMED_AGENTS
    # the benchmark agent is narrow and memoryless; the general one is neither
    assert "benchcad-agent" in _AGENTS_WITHOUT_MEMORY and "cad-agent" not in _AGENTS_WITHOUT_MEMORY
    assert "cad-agent" not in _AGENT_TOOL_ALLOWLIST
    # delegates must be a subset of what the benchmark orchestrator itself may use
    for sub in _AGENT_SUBAGENTS["benchcad-agent"]:
        assert _AGENT_TOOL_ALLOWLIST[sub] <= _AGENT_TOOL_ALLOWLIST["benchcad-agent"], sub


def test_general_prompt_has_no_benchmark_terms():
    from backend.agent_library import _CAD_AGENT_PROMPT, _BENCHCAD_AGENT_PROMPT
    for term in ("BenchCAD", "scores 0", "hidden reference", "tools.py"):
        assert term not in _CAD_AGENT_PROMPT, term
    assert "answer.py" in _BENCHCAD_AGENT_PROMPT


def test_subagent_tools_exist_on_server(srv):
    from backend.session_manager import _AGENT_TOOL_ALLOWLIST
    server_tools = {"cad_run", "cad_render", "cad_compare", "step_inspect", "answer_checkpoint",
                    "budget_status", "assembly_status", "cad_autoplace", "netlist_validate"}
    for n in ("benchcad-agent", "drawing-reader", "cad-critic"):
        for t in _AGENT_TOOL_ALLOWLIST[n]:
            assert t in server_tools or t in {"view_image", "grep_large_results"}, (n, t)
    for t in server_tools:
        assert callable(getattr(srv, t))


def test_cad_export_validates_and_reports(srv, tmp_path, monkeypatch):
    (tmp_path / "part.py").write_text("result=None")
    assert "unsupported format" in srv.cad_export(str(tmp_path), formats="step,dwg")
    assert "name must be" in srv.cad_export(str(tmp_path), name="../x")
    seen = {}

    def fake(work, args, timeout):
        seen["args"] = args
        return {"ok": True, "files": ["exports/part.step", "exports/part.obj"],
                "step_roundtrip_volume_diff_pct": 0.0, "metrics": GOOD["metrics"]}
    monkeypatch.setattr(srv, "_exec_probe", fake)
    out = srv.cad_export(str(tmp_path), formats="step")
    assert "part.step" in out and "0.0%" in out and "part.obj" not in out
    monkeypatch.setattr(srv, "_find_blender", lambda: None)
    out = srv.cad_export(str(tmp_path), formats="blend")
    assert "Blender not found" in out and "obj" in seen["args"][seen["args"].index("--formats") + 1]


# ── backend confinement to the session folder ───────────────────────────────

def test_resolve_workdir(tmp_path):
    from backend.cad_workdir_middleware import resolve_workdir
    root = tmp_path.resolve()
    (root / "sub").mkdir()
    assert resolve_workdir(root, ".") == root
    assert resolve_workdir(root, "") == root
    assert resolve_workdir(root, "sub") == root / "sub"
    assert resolve_workdir(root, str(root / "sub")) == root / "sub"
    for bad in ("..", "/etc", str(root.parent), "sub/../.."):
        assert isinstance(resolve_workdir(root, bad), str), bad


def _request(name, args):
    from types import SimpleNamespace
    class Req(SimpleNamespace):
        def override(self, **kw):
            return Req(**{**self.__dict__, **kw})
    return Req(tool_call={"name": name, "args": args, "id": "1"})


def test_middleware_fills_and_blocks(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from langchain_core.messages import ToolMessage
    import backend.cad_workdir_middleware as m
    mw = m.CadWorkdirMiddleware()
    monkeypatch.setattr(mw, "_root", lambda: tmp_path.resolve())
    seen = {}
    ok = mw.wrap_tool_call(_request("cad_run", {"workdir": "."}), lambda r: seen.update(r.tool_call["args"]) or "ran")
    assert ok == "ran" and seen["workdir"] == str(tmp_path.resolve())
    blocked = mw.wrap_tool_call(_request("cad_run", {"workdir": "/etc"}), lambda r: "ran")
    assert isinstance(blocked, ToolMessage) and "outside this session" in blocked.content
    # other tools are untouched
    assert mw.wrap_tool_call(_request("execute", {"workdir": "/etc"}), lambda r: "ran") == "ran"


def test_root_prefers_mapped_folder_then_session_files(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import backend.cad_workdir_middleware as m
    import backend.session_manager as sm
    import backend.action_policy.runtime as rt
    monkeypatch.setattr(rt, "current_scope", lambda: SimpleNamespace(session_id="s1"))
    files = tmp_path / "files"
    monkeypatch.setattr(sm, "_session_files_dir", lambda sid: files)
    mw = m.CadWorkdirMiddleware()
    monkeypatch.setattr(sm, "_read_workspace_meta", lambda sid: None)
    assert mw._root() == files.resolve() and files.is_dir()
    proj = tmp_path / "proj"; proj.mkdir()
    monkeypatch.setattr(sm, "_read_workspace_meta", lambda sid: SimpleNamespace(host_path=str(proj)))
    assert mw._root() == proj.resolve()


def test_middleware_covers_every_server_tool(srv):
    from backend.cad_workdir_middleware import CAD_TOOLS
    takes_workdir = {n for n in dir(srv) if callable(getattr(srv, n)) and not n.startswith("_")
                     and "workdir" in getattr(getattr(srv, n), "__code__", type("x", (), {"co_varnames": ()})).co_varnames[:1]}
    assert takes_workdir <= CAD_TOOLS, takes_workdir - CAD_TOOLS


def test_invented_absolute_workdir_maps_into_root(tmp_path):
    from backend.cad_workdir_middleware import resolve_workdir
    root = tmp_path.resolve()
    got = resolve_workdir(root, "/output_nonexistent_xyz")
    assert got == root / "output_nonexistent_xyz"
    assert isinstance(resolve_workdir(root, "/etc"), str)  # real outside path still blocked


def test_seatbelt_denies_extra_read_paths(tmp_path, monkeypatch):
    import backend.builtin_mcps.cad_tools.server as srv
    monkeypatch.setenv("CAD_DENY_READ", str(tmp_path / "answers"))
    prof = srv.seatbelt_profile(tmp_path / "w", [])
    assert f'(deny file-read* (subpath "{(tmp_path / "answers").resolve()}"))' in prof


def test_elapsed_note_only_when_limit_set(tmp_path, monkeypatch):
    import backend.builtin_mcps.cad_tools.server as srv
    monkeypatch.delenv("CAD_TIME_LIMIT", raising=False)
    assert srv._elapsed(tmp_path) == ""
    monkeypatch.setenv("CAD_TIME_LIMIT", "600")
    (tmp_path / "TASK.md").write_text("x")
    assert "of 10 min used" in srv._elapsed(tmp_path)


def test_benchcad_agent_has_no_shell():
    from backend.cad_workdir_middleware import NO_TOOLS_FOR_AGENT
    assert "execute" in NO_TOOLS_FOR_AGENT["benchcad-agent"]


def test_benchcad_prompts_use_the_compare_loop():
    from backend import agent_library as al
    assert "cad_compare" in al._BENCHCAD_AGENT_PROMPT and "interior" in al._BENCHCAD_AGENT_PROMPT
    assert "proxy" in al._BENCHCAD_AGENT_PROMPT
    assert "cad_compare" in al._CAD_CRITIC_PROMPT and "cad_compare" in al._CAD_AGENT_PROMPT


def test_graph_warnings_flag_shorts_and_merged_rails(srv):
    g = {"components": [{"id": "C1", "terminals": ["C1.1", "C1.2"]}, {"id": "R1", "terminals": ["R1.1", "R1.2"]}],
         "nets": [{"id": "GND"}, {"id": "VCC"}, {"id": "GND_VCC"}],
         "incidences": [["C1.1", "GND"], ["C1.2", "GND"], ["R1.1", "VCC"]]}
    w = "\n".join(srv.graph_warnings(g))
    assert "SHORT" in w and "C1" in w and "R1" not in w.split("SHORT")[1].split("\n")[0]
    assert "MERGED RAILS: net 'GND_VCC'" in w
    assert "OPEN?" in w and "R1.2" in w
    one_rail = {"components": [], "nets": [{"id": "n1", "meta": {"name": "VCC_5V"}}], "incidences": []}
    assert srv.graph_warnings(one_rail) == []


def test_netlist_validate_appends_warnings(srv, tmp_path):
    g = _graph()
    (tmp_path / "pred_graph.json").write_text(json.dumps(g))
    out = srv.netlist_validate(str(tmp_path))
    assert out.startswith("VALID") and ("Warnings" in out) == bool(srv.graph_warnings(g))


def test_topology_line_reports_counts_and_fill(srv):
    m = {"topology": [1, 1, 0], "size": [10, 10, 10], "volume": 100.0}
    line = srv._topo_line(m)
    assert "1 through-hole" in line and "fill" in line and "thin shell" in line
    assert "could not be read" in srv._topo_line({"topology": None, "size": [1, 1, 1], "volume": 1.0})


def test_cad_compare_rejects_bad_shape(srv, tmp_path):
    (tmp_path / "answer.py").write_text("result = 1")
    (tmp_path / "views.png").write_bytes(b"x")
    assert "shape must be" in srv.cad_compare(str(tmp_path), shape="part:../x")


def test_cad_compare_passes_shape_to_probe(srv, tmp_path, monkeypatch):
    (tmp_path / "answer.py").write_text("result = 1")
    (tmp_path / "views.png").write_bytes(b"x")
    seen = {}
    monkeypatch.setattr(srv, "_exec_probe", lambda w, args, t: seen.setdefault("args", args) and {"ok": False, "error": "x"})
    srv.cad_compare(str(tmp_path), shape="part:body")
    a = seen["args"]
    assert a[a.index("--shape") + 1] == "part:body"


def test_prompts_mention_topology_assembly_and_power():
    from backend import agent_library as al
    assert "topology" in al._BENCHCAD_AGENT_PROMPT and 'shape="assembly"' in al._BENCHCAD_AGENT_PROMPT
    assert "SHORT" in al._BENCHCAD_AGENT_PROMPT and "GND" in al._BENCHCAD_AGENT_PROMPT
    assert "pcb-tracer" not in al._BENCHCAD_AGENT_PROMPT and "assembly-planner" not in al._BENCHCAD_AGENT_PROMPT
    assert "cad-critic" not in al._BENCHCAD_AGENT_PROMPT and "STALLED" in al._BENCHCAD_AGENT_PROMPT
    assert "topology" in al._CAD_AGENT_PROMPT


def test_compare_tracks_best_and_restores_on_stall(srv, tmp_path):
    code = tmp_path / "answer.py"
    def step(text, score):
        code.write_text(text)
        return " ".join(srv._track_best(tmp_path, code, "views.png|auto", score))
    assert "best so far" in step("v1", 0.60)
    assert "best so far" in step("v2", 0.65)
    for i, t in enumerate(("v3", "v4", "v5")):
        assert "no gain" in step(t, 0.62) and "STALLED" not in step.__name__
    out = step("v6", 0.64)
    assert "STALLED" in out and code.read_text() == "v2"
    assert (tmp_path / ".cad" / "best" / "last_try_answer.py").read_text() == "v6"
    # a different target is tracked on its own
    assert "best so far" in " ".join(srv._track_best(tmp_path, code, "parts/a.png|part:a", 0.1))


def test_compare_floor_and_proportions(srv):
    res = {"combined": 0.50, "standins": {"box": 0.40, "cylinder along z": 0.49},
           "views": [{"view": "t1", "aspect_target": 1.0, "aspect_candidate": 1.3},
                     {"view": "iso", "aspect_target": 1.0, "aspect_candidate": 0.7},
                     {"view": "t3", "aspect_target": 1.0, "aspect_candidate": 1.0}]}
    out = "\n".join(srv._floor_lines(res))
    assert "NO BETTER THAN A PLAIN CYLINDER ALONG Z" in out and "PROPORTIONS" in out and "t1, iso" in out
    res["combined"] = 0.8
    assert "NO BETTER" not in "\n".join(srv._floor_lines(res))


def test_no_subagents_switch_blocks_task(monkeypatch):
    from backend.cad_workdir_middleware import blocked_tools
    monkeypatch.delenv("BENCHCAD_NO_SUBAGENTS", raising=False)
    assert "task" not in blocked_tools("benchcad-agent") and "execute" in blocked_tools("benchcad-agent")
    monkeypatch.setenv("BENCHCAD_NO_SUBAGENTS", "1")
    assert "task" in blocked_tools("benchcad-agent") and "task" not in blocked_tools("cad-agent")


def test_placed_compare_shape_option_and_report(srv, tmp_path, monkeypatch):
    (tmp_path / "answer.py").write_text("result = 1")
    (tmp_path / "x.png").write_bytes(b"x")
    assert "shape must be" in srv.cad_compare(str(tmp_path), target="x.png", shape="placed:../a")
    (tmp_path / ".cad").mkdir()
    (tmp_path / ".cad" / "c.png").write_bytes(b"x")
    res = {"ok": True, "placed": "jaw_1", "n_instances": 1, "combined": 0.05, "compare_png": ".cad/c.png",
           "views": [{"view": "t1", "iou": 0.08, "yours_vs_target": [-0.2, 0.1], "size_ratio": 1.1},
                     {"view": "iso", "iou": 0.9, "yours_vs_target": [0.0, 0.0], "size_ratio": 1.0}]}
    out = srv._placed_report(tmp_path, tmp_path / "answer.py", "x.png", "placed:jaw_1", res)[0]
    assert "WRONG PLACE" in out and "0.20 left and 0.10 below" in out
    res["combined"] = 0.8
    assert "WRONG PLACE" not in srv._placed_report(tmp_path, tmp_path / "answer.py", "x.png", "placed:jaw_1", res)[0]


def test_assembly_prompt_uses_placed_compare():
    from backend import agent_library as al
    assert 'shape="placed:<id>"' in al._BENCHCAD_AGENT_PROMPT and "_in_assembly.png" in al._BENCHCAD_AGENT_PROMPT


# ── assembly_status, the placed gate and the view_image cap ──────────────────

def _asm_workdir(tmp_path, built=(), instances=(), placed=None):
    import json
    (tmp_path / "parts").mkdir(exist_ok=True)
    (tmp_path / "submission" / "parts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "submission" / "assembly").mkdir(parents=True, exist_ok=True)
    (tmp_path / "bom.json").write_text(json.dumps({"items": [
        {"part_id": "body", "quantity": 1}, {"part_id": "pinion", "quantity": 3}]}))
    for pid in ("body", "pinion"):
        (tmp_path / "parts" / f"{pid}_in_assembly.png").write_bytes(b"x")
    for pid in built:
        (tmp_path / "submission" / "parts" / f"{pid}.step").write_text("x")
    (tmp_path / "submission" / "assembly" / "instances.json").write_text(
        json.dumps([{"part_id": p, "transform": []} for p in instances]))
    if placed:
        (tmp_path / ".cad").mkdir(exist_ok=True)
        (tmp_path / ".cad" / "progress.json").write_text(json.dumps({"placed": placed}))
    return tmp_path


def test_assembly_status_next_steps(tmp_path):
    from backend.builtin_mcps.cad_tools import server as S
    w = _asm_workdir(tmp_path)
    assert "NEXT: build body" in S.assembly_status(str(w))
    w = _asm_workdir(tmp_path, built=("body", "pinion"), instances=("body", "pinion"))
    assert "NEXT: place every part in one go: cad_autoplace()" in S.assembly_status(str(w))
    (w / ".cad").mkdir(exist_ok=True); (w / ".cad" / "autoplace.json").write_text("{}")       # placed once: now the details
    out = S.assembly_status(str(w))
    assert "WRONG COUNT" in out and "pinion has 3 (it has 1)" in out
    w = _asm_workdir(tmp_path, built=("body", "pinion"), instances=("body", "pinion", "pinion", "pinion"))
    assert 'shape="placed:body"' in S.assembly_status(str(w))
    w = _asm_workdir(tmp_path, built=("body", "pinion"), instances=("body", "pinion", "pinion", "pinion"),
                     placed={"body": 0.9, "pinion": 0.2})
    assert "fix the placement of pinion" in S.assembly_status(str(w))
    w = _asm_workdir(tmp_path, built=("body", "pinion"), instances=("body", "pinion", "pinion", "pinion"),
                     placed={"body": 0.9, "pinion": 0.8})
    assert "every part is built" in S.assembly_status(str(w))


def test_assembly_compare_gate(tmp_path):
    import json
    from backend.builtin_mcps.cad_tools import server as S
    w = _asm_workdir(tmp_path, built=("body", "pinion"), instances=("body", "pinion"))
    (w / "views.png").write_bytes(b"x"); (w / "answer.py").write_text("result=1")
    (w / ".cad").mkdir(exist_ok=True)
    (w / ".cad" / "progress.json").write_text(json.dumps({"asm_compares": 2}))
    out = S.cad_compare(str(w), shape="assembly")
    assert "refused" in out and "body, pinion" in out
    # a recorded placed check for every part lifts the gate (it then goes on to run, and fails on our stub)
    (w / ".cad" / "progress.json").write_text(json.dumps({"asm_compares": 2, "placed": {"body": 1, "pinion": 1}}))
    assert "refused" not in str(S.cad_compare(str(w), shape="assembly"))


def test_view_image_cap(monkeypatch):
    from types import SimpleNamespace
    from backend import cad_workdir_middleware as M
    from backend.action_policy import runtime
    monkeypatch.setenv("BENCHCAD_MAX_VIEW_IMAGES", "2")
    monkeypatch.setattr(runtime, "current_scope", lambda: SimpleNamespace(session_id="s1", agent_name="benchcad-agent"))
    M._VIEWS.clear()
    mw = M.CadWorkdirMiddleware()
    req = SimpleNamespace(tool_call={"name": "view_image", "args": {}, "id": "1"})
    assert mw._prepare(req) is req and mw._prepare(req) is req
    blocked = mw._prepare(req)
    assert getattr(blocked, "status", "") == "error" and "limit for this run is 2" in blocked.content


def test_duplicate_write_blocked(monkeypatch):
    from types import SimpleNamespace
    from backend import cad_workdir_middleware as M
    from backend.action_policy import runtime
    monkeypatch.setattr(runtime, "current_scope", lambda: SimpleNamespace(session_id="s2", agent_name="benchcad-agent"))
    M._WRITES.clear()
    mw = M.CadWorkdirMiddleware()
    def req(path, text):
        return SimpleNamespace(tool_call={"name": "write_file", "args": {"file_path": path, "content": text}, "id": "1"})
    prog = "x = 1\n" * 60
    a = req("a.py", prog); assert mw._prepare(a) is a
    blocked = mw._prepare(req("b.py", prog))
    assert getattr(blocked, "status", "") == "error" and "already wrote" in blocked.content
    c = req("c.py", prog + "y = 2\n"); assert mw._prepare(c) is c


# ── the shape check at one shared scale (gain over a plain block) ─────────────

def _chk(gain, shared=0.80, plain=0.85, stretched=None, stretch=(1.0, 1.0, 1.5)):
    return {"gain": gain, "shared_iou": shared, "plain_block": plain, "plain_block_name": "box",
            "stretched_iou": shared if stretched is None else stretched, "stretch": list(stretch)}


def test_shared_lines_flag_plain_blocks_blind_outlines_and_stretch(srv):
    out = "\n".join(srv._shared_lines(_chk(0.0, stretched=0.86), "jaw_1", "parts/jaw_1_alone.png"))
    assert "gain 0.0" in out and "NOT DONE: jaw_1" in out and "PROPORTIONS" in out and "z 1.5x" in out
    out = "\n".join(srv._shared_lines(_chk(0.6, shared=0.95), "jaw_1", "parts/jaw_1_alone.png"))
    assert "NOT DONE" not in out and "PROPORTIONS" not in out
    out = "\n".join(srv._shared_lines(_chk(0.0, shared=0.94, plain=0.97), "body", "parts/body_alone.png"))
    assert "cannot judge body" in out and "NOT DONE" not in out


def test_status_asks_to_improve_plain_block_parts_and_join_pieces(tmp_path):
    import json
    from backend.builtin_mcps.cad_tools import server as S
    w = _asm_workdir(tmp_path, built=("body", "pinion"), instances=("body", "pinion", "pinion", "pinion"),
                     placed={"body": 0.9, "pinion": 0.8})
    (w / ".cad" / "autoplace.json").write_text("{}")
    for pid in ("body", "pinion"):
        (w / "parts" / f"{pid}_alone.png").write_bytes(b"x")
    # not scored yet: score the shapes first
    assert 'NEXT: score the shape of body' in S.assembly_status(str(w))
    S._note_shape(w, "body", _chk(0.0, shared=0.95, plain=0.97), {"topology": [1, 2, 0]})   # outline-blind: not asked
    S._note_shape(w, "pinion", _chk(0.02), {"topology": [1, 0, 0]})
    out = S.assembly_status(str(w))
    assert "NEXT: improve the shape of pinion" in out and "outline-blind" in out and "(PLAIN BLOCK)" in out
    S._note_shape(w, "body", None, {"topology": [2, 2, 0]})
    assert "NEXT: join body: it is 2 separate pieces" in S.assembly_status(str(w))
    S._note_shape(w, "body", None, {"topology": [1, 2, 0]})
    S._note_shape(w, "pinion", _chk(0.5), {"topology": [1, 0, 0]})
    assert "every part is built" in S.assembly_status(str(w))
    # after SHAPE_MAX_TRIES compares the checklist stops asking
    S._note_shape(w, "pinion", _chk(0.0), {"topology": [1, 0, 0]})
    assert "improve the shape of pinion" in S.assembly_status(str(w))
    d = json.loads((w / ".cad" / "progress.json").read_text()); d["shape_tries"]["pinion"] = S.SHAPE_MAX_TRIES
    (w / ".cad" / "progress.json").write_text(json.dumps(d))
    assert "every part is built" in S.assembly_status(str(w))


def test_benchcad_prompt_explains_shape_gain():
    from backend import agent_library as al
    p = al._BENCHCAD_AGENT_PROMPT
    assert "SHAPE `gain`" in p and "plain box or cylinder earns nothing" in p and "use a simple stand-in" not in p
