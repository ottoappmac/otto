"""The design spec (spec.json): validation, the checks against a built solid, and how cad_run / cad_export
report them. Hermetic: the sandbox call is stubbed."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("mcp")

TOOLS = Path(__file__).resolve().parents[1] / "backend" / "builtin_mcps" / "cad_tools"


@pytest.fixture()
def srv(tmp_path, monkeypatch):
    monkeypatch.setenv("CAD_WORK_ROOT", str(tmp_path))
    spec = importlib.util.spec_from_file_location("cad_tools_server_spec", TOOLS / "server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def cs(srv):
    return srv.cad_spec


def _m(size=(80, 50, 30), volume=42000.0, topo=(1, 0, 0), solids=1):
    return {"bbox_min": [0, 0, 0], "bbox_max": list(size), "size": list(size), "volume": volume, "area": 1.0,
            "solids": solids, "faces": 6, "edges": 12, "valid": True, "center_of_mass": [0, 0, 0],
            "topology": list(topo) if topo else None}


def _spec(**part):
    p = {"id": "stand", "file": "part.py"}
    p.update(part)
    return json.dumps({"parts": [p], "assumptions": ["lip is 5 mm"]})


def test_parse_accepts_a_full_spec(cs):
    s = cs.parse(_spec(size=[80, 50, None], holes=2, pieces=1, voids=0, volume=[1000, 90000], axes="any"))
    p = s["parts"][0]
    assert p["size"] == [80.0, 50.0, None] and p["holes"] == 2 and p["axes"] == "any" and p["size_tol"] == 0.05
    assert s["assumptions"] == ["lip is 5 mm"]


@pytest.mark.parametrize("text, needle", [
    ("{nope", "not valid JSON"),
    ("[]", "top level"),
    (json.dumps({"parts": []}), "non-empty"),
    (_spec(size=[1, 2]), "[x, y, z]"),
    (_spec(size=[None, None, None]), "all null"),
    (_spec(size=[80, -1, 3]), "above 0"),
    (_spec(size=["80", 1, 3]), "must be a number"),
    (_spec(size=[1, 2, 3], size_tol=0), "size_tol"),
    (_spec(size=[1, 2, 3], axes="y-up"), "axes"),
    (_spec(holes=-1), "whole number"),
    (_spec(holes=1.5), "whole number"),
    (_spec(pieces=0), "at least 1"),
    (_spec(volume=[5, 1]), "min <= max"),
    (_spec(), "state at least one"),
    (json.dumps({"parts": [{"size": [1, 2, 3]}]}), "needs an \"id\""),
    (json.dumps({"parts": [{"id": "a", "holes": 1}, {"id": "a", "holes": 1}]}), "two parts are called"),
])
def test_parse_rejects_bad_specs(cs, text, needle):
    with pytest.raises(cs.SpecError, match=needle.replace("[", r"\[").replace("]", r"\]")
                       if needle.startswith("[") else needle):
        cs.parse(text)


def test_matching_result_has_no_differences(cs):
    p = cs.parse(_spec(size=[80, 50, 30], holes=0, pieces=1))["parts"][0]
    assert cs.check(p, _m()) == []
    assert cs.check(p, _m(size=(81.5, 49, 30.4))) == []            # inside 5 %


def test_size_mismatch_names_the_axis_the_amount_and_direction(cs):
    p = cs.parse(_spec(size=[37, 37, 45]))["parts"][0]
    bad = cs.check(p, _m(size=(36, 36, 11)))
    assert len(bad) == 1 and "size z: 11 mm, spec says 45" in bad[0] and "-76%" in bad[0] and "too short" in bad[0]


def test_small_parts_get_an_absolute_floor(cs):
    p = cs.parse(_spec(size=[2, 40, 40]))["parts"][0]
    assert cs.check(p, _m(size=(2.08, 40, 40))) == []              # 4 % of 2 mm is under the 0.1 mm floor
    assert cs.check(p, _m(size=(2.3, 40, 40)))


def test_swapped_axes_are_called_out_and_axes_any_accepts_them(cs):
    p = cs.parse(_spec(size=[80, 50, 30]))["parts"][0]
    bad = cs.check(p, _m(size=(50, 80, 30)))
    assert any("axis is swapped" in b for b in bad)
    q = cs.parse(_spec(size=[80, 50, 30], axes="any"))["parts"][0]
    assert cs.check(q, _m(size=(50, 80, 30))) == []
    assert cs.check(q, _m(size=(50, 80, 12)))                       # a real difference still shows


def test_null_sizes_are_not_checked(cs):
    p = cs.parse(_spec(size=[80, None, None]))["parts"][0]
    assert cs.check(p, _m(size=(80, 7, 999))) == []


def test_hole_counts_pieces_voids_and_volume(cs):
    p = cs.parse(_spec(size=[80, 50, 30], holes=4, voids=0, volume=[30000, 50000]))["parts"][0]
    bad = cs.check(p, _m(topo=(1, 0, 0)))
    assert any("0 through-hole(s), spec says 4 (a hole is missing)" in b for b in bad)
    bad = cs.check(p, _m(topo=(1, 5, 0)))
    assert any("5 through-hole(s), spec says 4 (one too many)" in b for b in bad)
    bad = cs.check(p, _m(topo=(2, 4, 0), solids=2))
    assert any("2 separate piece(s), spec says 1; join them" in b for b in bad)
    bad = cs.check(p, _m(topo=(1, 4, 1)))
    assert any("1 sealed void(s), spec says 0" in b for b in bad)
    assert any("volume" in b for b in cs.check(p, _m(topo=(1, 4, 0), volume=99000.0)))
    assert cs.check(p, _m(topo=(1, 4, 0))) == []


def test_unreadable_topology_is_said_not_assumed(cs):
    p = cs.parse(_spec(holes=2))["parts"][0]
    assert any("cannot read the through-hole(s) count" in b for b in cs.check(p, _m(topo=None)))


def test_only_stated_figures_are_checked(cs):
    p = cs.parse(_spec(volume=[1, 1e9]))["parts"][0]
    assert cs.check(p, _m(topo=(3, 9, 9), solids=3)) == []         # no size/holes/pieces stated


def test_cad_run_appends_the_spec_verdict(srv, tmp_path, monkeypatch):
    (tmp_path / "part.py").write_text("result = 1\n")
    monkeypatch.setattr(srv, "_exec_probe", lambda *a, **k: {"ok": True, "metrics": _m(size=(36, 36, 11))})
    assert "SPEC" not in srv.cad_run(str(tmp_path), code_file="part.py")           # no spec.json: silent
    (tmp_path / "spec.json").write_text(_spec(size=[37, 37, 45], holes=0))
    out = srv.cad_run(str(tmp_path), code_file="part.py")
    assert "SPEC MISMATCH (stand)" in out and "size z: 11 mm, spec says 45" in out and "Do not report the part as done" in out
    monkeypatch.setattr(srv, "_exec_probe", lambda *a, **k: {"ok": True, "metrics": _m(size=(37, 37, 45))})
    assert "SPEC OK (stand): matches spec.json on size, holes" in srv.cad_run(str(tmp_path), code_file="part.py")


def test_cad_run_reports_a_broken_spec_without_failing(srv, tmp_path, monkeypatch):
    (tmp_path / "part.py").write_text("result = 1\n")
    monkeypatch.setattr(srv, "_exec_probe", lambda *a, **k: {"ok": True, "metrics": _m()})
    (tmp_path / "spec.json").write_text("{nope")
    out = srv.cad_run(str(tmp_path), code_file="part.py")
    assert out.startswith("OK:") and "SPEC PROBLEM: spec.json not valid JSON" in out


def test_failed_run_is_not_judged_against_the_spec(srv, tmp_path, monkeypatch):
    (tmp_path / "part.py").write_text("boom\n")
    (tmp_path / "spec.json").write_text(_spec(size=[1, 2, 3]))
    monkeypatch.setattr(srv, "_exec_probe", lambda *a, **k: {"ok": False, "error": "NameError: boom"})
    out = srv.cad_run(str(tmp_path), code_file="part.py")
    assert "NameError" in out and "SPEC" not in out


def test_several_parts_match_by_file(srv, tmp_path, cs, monkeypatch):
    two = json.dumps({"parts": [{"id": "base", "file": "base.py", "size": [100, 100, 10]},
                                {"id": "lid", "file": "lid.py", "size": [100, 100, 3]}]})
    s = cs.parse(two)
    assert cs.part_for(s, "lid.py")[0]["id"] == "lid"
    part, why = cs.part_for(s, "other.py")
    assert part is None and "add \"file\"" in why
    (tmp_path / "lid.py").write_text("result = 1\n")
    (tmp_path / "spec.json").write_text(two)
    monkeypatch.setattr(srv, "_exec_probe", lambda *a, **k: {"ok": True, "metrics": _m(size=(100, 100, 10))})
    assert "SPEC MISMATCH (lid)" in srv.cad_run(str(tmp_path), code_file="lid.py")
    (tmp_path / "other.py").write_text("result = 1\n")
    assert "SPEC: other.py is not tied to a part" in srv.cad_run(str(tmp_path), code_file="other.py")


def test_one_part_without_file_applies_to_any_program(cs, tmp_path):
    (tmp_path / "spec.json").write_text(json.dumps({"parts": [{"id": "x", "size": [10, 10, 10]}]}))
    assert cs.lines(tmp_path, "whatever.py", _m(size=(10, 10, 10)))[0].startswith("SPEC OK (x)")


def test_cad_export_reports_the_check_and_the_assumptions(srv, tmp_path, monkeypatch):
    (tmp_path / "part.py").write_text("result = 1\n")
    (tmp_path / "spec.json").write_text(_spec(size=[80, 50, 30], holes=2))
    res = {"ok": True, "files": ["exports/part.step"], "metrics": _m(topo=(1, 0, 0))}
    monkeypatch.setattr(srv, "_exec_probe", lambda *a, **k: res)
    out = srv.cad_export(str(tmp_path), code_file="part.py", formats="step")
    assert "SPEC MISMATCH (stand)" in out and "0 through-hole(s), spec says 2" in out
    assert "assumptions in spec.json (tell the user): lip is 5 mm" in out


def test_spec_file_cannot_be_huge(cs, tmp_path):
    (tmp_path / "spec.json").write_text(" " * (cs.MAX_BYTES + 1))
    with pytest.raises(cs.SpecError, match="over"):
        cs.load(tmp_path)


def test_shipped_general_prompt_has_no_spec_step_but_the_eval_arm_does():
    from backend import agent_library as al
    shipped, with_spec = al._CAD_AGENT_PROMPT, al._CAD_AGENT_PROMPT_WITH_SPEC
    assert "spec.json" not in shipped and "SPEC" not in shipped
    assert "spec.json" in with_spec and "Write the spec before any CAD code" in with_spec and "SPEC MISMATCH" in with_spec
    assert with_spec.index("spec.json") < with_spec.index("Write a first working program early")
    for term in ("BenchCAD", "scores 0", "hidden reference", "tools.py"):
        assert term not in shipped and term not in with_spec, term
    assert "spec.json" not in al._BENCHCAD_AGENT_PROMPT        # the benchmark agent is unchanged
    assert al._BUILTIN_AGENT_VERSIONS["cad-agent"] >= 10        # installed copies at v9 (with the step) get replaced
