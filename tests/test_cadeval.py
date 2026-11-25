"""The CAD eval set: its cases are well formed, the control prompt is the shipped one minus the spec step, and
the report reads results correctly. Hermetic: no model, no CadQuery."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

EVAL = Path(__file__).resolve().parents[1] / "scripts" / "cadeval"
TOOLS = Path(__file__).resolve().parents[1] / "backend" / "builtin_mcps" / "cad_tools"
sys.path.insert(0, str(TOOLS))


@pytest.fixture(scope="module")
def ev():
    sys.path.insert(0, str(EVAL))
    spec = importlib.util.spec_from_file_location("cadeval_run", EVAL / "evalrun.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_case_is_well_formed(ev):
    from cases import CASES
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids)) >= 12
    assert {c["category"] for c in CASES} <= {"part", "assembly", "open", "hard"}
    for c in CASES:
        part = ev.expect_part(c)                      # parses as a spec
        assert part.get("size") and any(k in part for k in ("holes", "solids", "pieces")), c["id"]
        assert "result" in c["reference"] and "import cadquery" in c["reference"], c["id"]
        assert len(c["prompt"]) > 40


def test_stated_dimensions_appear_in_the_prompts(ev):
    """The figures must come from the prompt, so a case cannot demand something the user never said."""
    from cases import CASES
    for c in CASES:
        if c["category"] != "part":
            continue
        for v in c["expect"]["size"]:
            if v is not None and abs(v - round(v)) < 1e-9:
                assert f"{int(round(v))} mm" in c["prompt"] or f"{int(round(v))} x" in c["prompt"] \
                    or f"{int(round(v))}," in c["prompt"] or f"{int(round(v))}." in c["prompt"] \
                    or f"{int(round(v))} " in c["prompt"], (c["id"], v)


def test_judge_uses_the_cases_own_figures(ev):
    from cases import by_id
    c = by_id(["flange"])[0]
    good = {"size": [80, 80, 8], "volume": 32100.0, "solids": 1, "topology": [1, 7, 0]}
    assert ev.judge(c, good) == []
    assert any("through-hole" in d for d in ev.judge(c, {**good, "topology": [1, 1, 0]}))
    assert any("size z" in d for d in ev.judge(c, {**good, "size": [80, 80, 4]}))
    box = by_id(["box_with_lid"])[0]
    assert ev.judge(box, {"size": [60, 40, 22], "volume": 16500.0, "solids": 2, "topology": [1, 0, 0]}) == []
    assert any("solid body" in d for d in ev.judge(box, {"size": [60, 40, 22], "volume": 16500.0, "solids": 1,
                                                         "topology": [1, 0, 0]}))


def test_spec_gap_reports_disagreements_with_the_case(ev, tmp_path):
    from cases import by_id
    c = by_id(["flange"])[0]
    assert ev.spec_gap(c, tmp_path) == {"spec_written": False}
    (tmp_path / "spec.json").write_text(json.dumps({"parts": [{"id": "f", "size": [80, 80, 8], "holes": 6}]}))
    g = ev.spec_gap(c, tmp_path)
    assert g["spec_valid"] and g["spec_gap"] == ["holes 6 vs 7"]
    (tmp_path / "spec.json").write_text("{nope")
    assert ev.spec_gap(c, tmp_path)["spec_valid"] is False


def test_control_prompt_is_the_shipped_prompt_and_the_spec_arm_adds_one_step():
    from backend import agent_library as al
    shipped, control = al._CAD_AGENT_PROMPT_WITH_SPEC, al._CAD_AGENT_PROMPT
    assert control == al.cad_agent_prompt_without_spec()
    assert "spec.json" in shipped and "spec.json" not in control and "SPEC" not in control
    steps = [l[:2] for l in control.split("## Loop")[1].split("## Rules")[0].splitlines() if l[:1].isdigit()]
    assert steps == ["1.", "2.", "3.", "4.", "5."]
    assert "move on to step 5" in control and "step 6" not in control
    # nothing else changed: dropping the spec lines from both leaves the same rules
    assert control.split("## Rules")[1] == shipped.split("## Rules")[1]


def test_report_counts_passes_per_arm_and_category(ev):
    rows = [{"id": "a", "category": "part", "arm": "spec", "rep": 0, "pass": True, "reasons": [], "spec_written": True,
             "spec_mismatch_seen": 1, "agent_seconds": 120},
            {"id": "a", "category": "part", "arm": "control", "rep": 0, "pass": False,
             "reasons": ["size z: 11 mm, spec says 45"], "agent_seconds": 60},
            {"id": "b", "category": "hard", "arm": "spec", "rep": 0, "pass": False, "reasons": ["no STEP exported"],
             "spec_written": False, "agent_seconds": 300}]
    t = ev.table(rows)
    assert "spec: 1/2 pass" in t and "control: 0/1 pass" in t and "hard 0/1" in t
    assert "spec.json written 1/2" in t and "mismatch seen in 1/2" in t
    assert "small set" in t
