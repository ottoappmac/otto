"""cad_autoplace: the search helpers, the answer.py block that carries the found poses, the tool's
keep/revert logic (probe stubbed), and, when the CAD venv is installed, a real recovery test."""
import importlib.util
import json
import math
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("mcp")

HERE = Path(__file__).resolve().parent.parent / "backend" / "builtin_mcps" / "cad_tools"


@pytest.fixture(scope="module")
def probe():
    spec = importlib.util.spec_from_file_location("cad_probe", HERE / "_probe.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture()
def srv():
    from backend.builtin_mcps.cad_tools import server
    return server


# ── pure helpers ─────────────────────────────────────────────────────────────

def test_right_angle_rotations_are_the_24_proper_ones(probe):
    rots = probe._right_angle_rotations()
    assert len(rots) == 24
    assert len({tuple(R.flatten()) for R in rots}) == 24
    for R in rots:
        assert abs(np.linalg.det(R) - 1) < 1e-12 and np.allclose(R @ R.T, np.eye(3))


def test_orthonormal_is_exact_and_proper(probe):
    R = probe._orthonormal(probe._rot_z(37.0) + 1e-5)
    assert abs(R @ R.T - np.eye(3)).max() < 1e-14 and abs(np.linalg.det(R) - 1) < 1e-14
    flipped = probe._orthonormal(np.diag([1.0, 1.0, -1.0]))      # a reflection becomes a rotation
    assert np.linalg.det(flipped) > 0


def test_box_blur_keeps_mass_and_spreads(probe):
    m = np.zeros((20, 20)); m[8:12, 8:12] = 1
    assert np.allclose(probe._box_blur(m, 0), m)
    b = probe._box_blur(m, 2)
    assert abs(b.sum() - m.sum()) < 1e-9 and b.max() < 1


def test_shift_moves_whole_pixels_and_drops_what_leaves(probe):
    m = np.zeros((probe.FS, probe.FS), bool); m[10:14, 10:14] = True
    s = probe._shift(m, 5, -3)
    assert s.sum() == m.sum() and s[7:11, 15:19].all()
    assert probe._shift(m, probe.FS, 0).sum() == 0 and probe._shift(m, 0, -12).sum() == 8      # two rows left the top


def test_closed_fills_pinholes_but_keeps_shape(probe):
    m = np.zeros((probe.FS, probe.FS), bool); m[40:100, 40:100] = True
    holey = m.copy(); holey[60, 60] = holey[61, 75] = False
    assert (probe._closed(holey) == m).all()
    assert abs(int(probe._closed(m).sum()) - int(m.sum())) < 40          # no growth


def test_rotvec_is_a_proper_rotation_about_its_axis(probe):
    R = probe._rotvec([0, 0, math.pi / 2])
    assert np.allclose(R, probe._rot_z(90.0)) and np.allclose(probe._rotvec([0, 0, 0]), np.eye(3))
    R = probe._rotvec([0.3, -0.2, 0.5]); assert np.allclose(R @ R.T, np.eye(3)) and abs(np.linalg.det(R) - 1) < 1e-12


def test_position_follows_from_outline_centres(probe):
    """Draw an L-shaped cloud at a known place; the closed-form position finds it back from the outlines."""
    rng = np.random.default_rng(1)
    pts = np.vstack([rng.uniform([-6, -6, -15], [6, 6, 15], (4000, 3)), rng.uniform([0, -4, -15], [24, 4, -7], (4000, 3))])
    pts -= (pts.min(0) + pts.max(0)) / 2
    names = ["t1", "iso", "t3", "t4"]
    frame = {"sc": 1.6, "w": [np.zeros(3)] * 4}
    axes = probe._frame_axes(names, frame)
    truth = np.array([14.0, -9.0, 20.0]); R = probe._rot_z(35.0)
    tgt = [probe._splat(pts @ R.T + truth, ax, frame["sc"]) for ax in axes]
    part = {"pts": pts, "tgt": tgt, "tg": [probe._centroid(m) for m in tgt]}
    t = probe._position(part, R, frame, axes)
    assert np.linalg.norm(t - truth) < 1.5
    assert probe._pose_iou(part, R, t, frame, axes) > 0.9
    assert probe._pose_iou(part, probe._rot_z(215.0), probe._position(part, probe._rot_z(215.0), frame, axes), frame, axes) < 0.9
    part["tg"] = [part["tg"][0], None, None, None]                      # one view is not enough to place a part
    assert probe._position(part, R, frame, axes) is None


def test_group_poses_turn_one_instance_about_the_vertical(probe):
    poses = probe._group_poses(np.eye(3), np.array([10.0, 0.0, 5.0]), 3)
    assert len(poses) == 3 and np.allclose(poses[0][1], [10, 0, 5])
    assert np.allclose(poses[1][1], [10 * math.cos(math.radians(120)), 10 * math.sin(math.radians(120)), 5])
    assert np.allclose(poses[2][0], probe._rot_z(240.0))


def test_pattern_search_finds_a_peak_and_respects_limits(probe):
    import time
    f = lambda x: -((x[0] - 3.0) ** 2 + (x[1] + 2.0) ** 2) - 0.01 * abs(x[2])
    x, fx, evals = probe._pattern_search(f, [0, 0, 5], [4, 4, 4], [0.01, 0.01, 0.01], time.time() + 5, 2000)
    assert abs(x[0] - 3) < 0.05 and abs(x[1] + 2) < 0.05 and evals < 2000
    _, _, e = probe._pattern_search(f, [0, 0, 0], [4, 4, 4], [1e-9] * 3, time.time() + 5, 25)
    assert e <= 25                                              # the evaluation budget is a hard limit
    _, _, e = probe._pattern_search(f, [0, 0, 0], [4, 4, 4], [1e-9] * 3, time.time() - 1, 1000)
    assert e == 1                                               # an expired deadline stops at once


# ── apply_fix and the answer.py block ────────────────────────────────────────

def _T(x):
    return [[1, 0, 0, x], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def test_apply_fix_replaces_drops_adds_and_leaves_others(srv):
    recs = [{"part_id": "a", "instance_id": "a_i1", "transform": _T(0)},
            {"part_id": "p", "instance_id": "p_i1", "T": _T(1)},
            {"part_id": "p", "instance_id": "p_i2", "transform": _T(2)},
            {"part_id": "b", "instance_id": "b_i1", "transform": _T(9)}]
    out = srv.apply_fix(recs, {"p": [_T(10)]})                    # fewer: the second p goes
    assert [r["part_id"] for r in out] == ["a", "p", "b"] and out[1]["transform"] == _T(10) and "T" not in out[1]
    assert out[0] == recs[0] and out[2] == recs[3]
    out = srv.apply_fix(recs, {"p": [_T(10), _T(11), _T(12)]})   # more: one is added, with a fresh id
    ps = [r for r in out if r["part_id"] == "p"]
    assert [r["transform"] for r in ps] == [_T(10), _T(11), _T(12)]
    assert len({r["instance_id"] for r in out}) == len(out)
    assert srv.apply_fix([], {"p": [_T(1)]})[0]["instance_id"] == "p_i1"
    assert recs[1]["T"] == _T(1)                                   # the input is not modified


def _run_block(tmp_path, monkeypatch, srv, fix, instances, stub):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "submission" / "assembly" / "instances.json"
    if instances is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(instances))
    calls = []
    tools = types.ModuleType("tools")
    tools.SUBMISSION = tmp_path / "submission"
    tools.submit_assembly = stub(calls)
    monkeypatch.setitem(sys.modules, "tools", tools)
    exec(compile(srv.autoplace_block(fix), "answer.py", "exec"), {"__name__": "__main__"})
    return calls


def test_block_rewrites_the_submitted_instances(tmp_path, monkeypatch, srv):
    inst = [{"part_id": "p", "instance_id": "p_i1", "transform": _T(0)}, {"part_id": "q", "instance_id": "q_i1", "transform": _T(5)}]
    calls = _run_block(tmp_path, monkeypatch, srv, {"p": [_T(7)]}, inst, lambda c: c.append)
    assert len(calls) == 1 and calls[0][0]["transform"] == _T(7) and calls[0][1]["transform"] == _T(5)


def test_block_never_breaks_a_working_answer(tmp_path, monkeypatch, srv, capsys):
    # a clean run (no instances.json yet, answer.py never submitted): the placed parts are still submitted
    got = _run_block(tmp_path, monkeypatch, srv, {"p": [_T(7)]}, None, lambda c: c.append)
    assert [(r["part_id"], r["transform"]) for r in got[0]] == [("p", _T(7))]
    def boom(calls):
        def f(recs):
            raise ValueError("bad")
        return f
    _run_block(tmp_path, monkeypatch, srv, {"p": [_T(7)]}, [{"part_id": "p", "transform": _T(0)}], boom)
    assert "placements not applied" in capsys.readouterr().err                                      # reported, not raised


def test_with_block_is_idempotent_and_round_trips(srv):
    text = "import tools\nresult = 1\n"
    once = srv._with_block(text, {"p": [_T(1)]})
    twice = srv._with_block(once, {"p": [_T(1)], "q": [_T(2)]})
    assert twice.count(srv.AUTOPLACE_BEGIN) == 1 and twice.count(srv.AUTOPLACE_END) == 1
    assert twice.startswith(text.rstrip("\n")) and srv._fix_in(twice) == {"p": [_T(1)], "q": [_T(2)]}
    compile(twice, "answer.py", "exec")


def test_autoplace_note_warns_when_the_block_is_gone(tmp_path, srv):
    code = tmp_path / "answer.py"; code.write_text("x = 1\n")
    assert srv._autoplace_note(tmp_path, code) == ""
    (tmp_path / ".cad").mkdir(); (tmp_path / ".cad" / "autoplace.json").write_text(json.dumps({"fix": {"p": []}}))
    assert "NOT in the scored answer" in srv._autoplace_note(tmp_path, code)
    code.write_text(srv._with_block("x = 1\n", {"p": []}))
    assert srv._autoplace_note(tmp_path, code) == ""


# ── the tool: keep or revert (probe stubbed) ─────────────────────────────────

def _work(tmp_path, n_p=1):
    (tmp_path / "parts").mkdir(); (tmp_path / "submission" / "parts").mkdir(parents=True)
    (tmp_path / "submission" / "assembly").mkdir(parents=True)
    (tmp_path / "parts" / "p_in_assembly.png").write_bytes(b"x")
    (tmp_path / "submission" / "parts" / "p.step").write_text("x")
    (tmp_path / "bom.json").write_text(json.dumps({"items": [{"part_id": "p", "quantity": n_p}]}))
    recs = [{"part_id": "p", "instance_id": f"p_i{i + 1}", "transform": _T(0)} for i in range(n_p)]
    (tmp_path / "submission" / "assembly" / "instances.json").write_text(json.dumps(recs))
    (tmp_path / "answer.py").write_text("import tools\nresult = 1\n")
    return tmp_path


def _probe_stub(placed_scores, found):
    scores = iter(placed_scores)
    calls = []
    def fake(work, args, timeout):
        if args[0] == "placed":
            return {"ok": True, "combined": next(scores)}
        assert args[0] == "autoplace"
        calls.append(args)
        return found
    fake.calls = calls
    return fake


def _found(n=1):
    return {"ok": True, "frame": {"sc": 0.5, "w": [[0, 0, 0]] * 4}, "seconds": 1.0, "worst": "p",
            "parts": {"p": {"n_instances": n, "iou": 0.7, "transforms": [_T(3 + i) for i in range(n)]}},
            "compare_png": ".cad/render/compare.png"}


def test_tool_keeps_a_clear_improvement(tmp_path, srv, monkeypatch):
    w = _work(tmp_path)
    monkeypatch.setattr(srv, "_exec_probe", _probe_stub([0.1, 0.6], _found()))
    out = srv.cad_autoplace(str(w))
    assert "KEPT" in out[0]
    assert json.loads((w / "submission/assembly/instances.json").read_text())[0]["transform"] == _T(3)
    assert srv.AUTOPLACE_BEGIN in (w / "answer.py").read_text()
    assert json.loads((w / ".cad/autoplace.json").read_text())["fix"]["p"] == [_T(3)]
    assert json.loads((w / ".cad/frame.json").read_text())["sc"] == 0.5          # kept for single-part calls
    assert srv._load_progress(w)["placed"]["p"] == 0.6                    # counts as a placement check for the gate
    assert (w / ".cad/instances.before_autoplace.json").is_file()


def test_tool_reverts_when_the_official_score_does_not_improve(tmp_path, srv, monkeypatch):
    w = _work(tmp_path)
    inst, code = (w / "submission/assembly/instances.json").read_text(), (w / "answer.py").read_text()
    monkeypatch.setattr(srv, "_exec_probe", _probe_stub([0.30, 0.31], _found()))     # its own estimate was optimistic
    out = srv.cad_autoplace(str(w))
    assert "NOT KEPT" in out[0]
    assert (w / "submission/assembly/instances.json").read_text() == inst and (w / "answer.py").read_text() == code
    assert not (w / ".cad/autoplace.json").exists() and not (w / ".cad/frame.json").exists()


def test_tool_places_every_instance_and_works_without_an_instances_file(tmp_path, srv, monkeypatch):
    w = _work(tmp_path, n_p=1)
    (w / "submission/assembly/instances.json").unlink()
    stub = _probe_stub([0.38], _found(3))                                  # only the "after" check: nothing to score before
    monkeypatch.setattr(srv, "_exec_probe", stub)
    assert "KEPT" in srv.cad_autoplace(str(w))[0]
    assert len(json.loads((w / "submission/assembly/instances.json").read_text())) == 3
    assert stub.calls[0][stub.calls[0].index("--part") + 1] == "all"
    # and a rejected result leaves no half-made instances file behind
    (w / "submission/assembly/instances.json").unlink(); (w / ".cad/autoplace.json").unlink()
    monkeypatch.setattr(srv, "_exec_probe", _probe_stub([-1.0], _found(3)))
    assert "NOT KEPT" in srv.cad_autoplace(str(w))[0]
    assert not (w / "submission/assembly/instances.json").exists()


def test_tool_places_one_part_in_the_saved_frame(tmp_path, srv, monkeypatch):
    w = _work(tmp_path)
    assert "place all parts first" in srv.cad_autoplace(str(w), "p")
    monkeypatch.setattr(srv, "_exec_probe", _probe_stub([0.1, 0.6], _found()))
    srv.cad_autoplace(str(w))
    stub = _probe_stub([0.2, 0.7], _found())
    monkeypatch.setattr(srv, "_exec_probe", stub)
    assert "KEPT" in srv.cad_autoplace(str(w), "p")[0]
    a = stub.calls[0]
    assert a[a.index("--part") + 1] == "p" and a[a.index("--frame") + 1].endswith("frame.json")


def test_tool_reapply_restores_the_block_without_searching(tmp_path, srv, monkeypatch):
    w = _work(tmp_path)
    monkeypatch.setattr(srv, "_exec_probe", _probe_stub([0.1, 0.6], _found()))
    srv.cad_autoplace(str(w))
    (w / "answer.py").write_text("import tools\nresult = 2\n")          # the agent rewrote the file
    monkeypatch.setattr(srv, "_exec_probe", lambda *a, **k: pytest.fail("reapply must not run the probe"))
    assert "back into answer.py" in srv.cad_autoplace(str(w), reapply=True)[0]
    assert srv._fix_in((w / "answer.py").read_text()) == {"p": [_T(3)]}
    assert "no saved placement" in srv.cad_autoplace(str(w), "other", reapply=True)


def test_tool_refuses_without_its_inputs(tmp_path, srv):
    out = srv.cad_autoplace(str(tmp_path))
    assert out.startswith("ERROR") and "answer.py" in out
    (tmp_path / "w").mkdir(); w = _work(tmp_path / "w"); (w / "submission/parts/p.step").unlink()
    assert "exported part" in srv.cad_autoplace(str(w))
    assert srv.cad_autoplace(str(w), "../x").startswith("ERROR")


def test_tool_rejects_a_matrix_that_is_not_a_rotation(tmp_path, srv, monkeypatch):
    w = _work(tmp_path)
    bad = _found(); bad["parts"]["p"]["transforms"] = [[[1.0001, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]]
    monkeypatch.setattr(srv, "_exec_probe", _probe_stub([0.1, 0.9], bad))
    assert srv.cad_autoplace(str(w)).startswith("ERROR")
    assert json.loads((w / "submission/assembly/instances.json").read_text())[0]["transform"] == _T(0)


# ── the real thing: recover a known pose from rendered views (needs the CAD venv) ──

CAD_PY = Path.home() / "Library/Application Support/Otto/mcp_server/cad-tools/.venv/bin/python"


SCENE = r'''
import json, math, os, sys, numpy as np, cadquery as cq
from PIL import Image
sys.path.insert(0, %(here)r)
import _probe as P
body = cq.Workplane("XY").box(120, 120, 40).val()
part = (cq.Workplane("XY").box(12, 12, 24).union(cq.Workplane("XY").box(22, 8, 8).translate((6, 0, -8)))).val()
for d in ("submission/parts", "submission/assembly", "parts"):
    os.makedirs(d, exist_ok=True)
cq.exporters.export(cq.Workplane(obj=body), "submission/parts/body.step"); cq.exporters.export(cq.Workplane(obj=part), "submission/parts/p.step")
def mat(yaw, x, y, z):
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    return [[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]
truth = %(truth)s
json.dump({"items": [{"part_id": "body", "quantity": 1}, {"part_id": "p", "quantity": len(truth)}]}, open("bom.json", "w"))
Vb, Tb = P._mesh(body); Vp, Tp = P._mesh(part)
Pb = P._surface_points(Vb, Tb); Pp = P._surface_points(Vp, Tp)
world_p = [Pp @ np.array(T)[:3, :3].T + np.array(T)[:3, 3] for T in truth]
allp = np.vstack([Pb] + world_p); C = (allp.min(0) + allp.max(0)) / 2      # the images are centred on the assembly's centre
P.FS = 300
sc = 1.5
for pid, mine in (("body", [Pb]), ("p", world_p)):
    canvas = np.full((600, 600, 3), 255, np.uint8)
    for name, (r, c) in zip(["t1", "iso", "t3", "t4"], [(0, 0), (0, 1), (1, 0), (1, 1)]):
        _, right, up = P._basis(P.VIEWS[name]); ax = (right, up)
        whole = P._splat(allp - C, ax, sc); red = P._splat(np.vstack(mine) - C, ax, sc)
        q = np.full((300, 300, 3), 255, np.uint8); q[whole] = (205, 205, 205); q[red] = (225, 40, 40)
        canvas[r * 300:(r + 1) * 300, c * 300:(c + 1) * 300] = q
    Image.fromarray(canvas).save("parts/%%s_in_assembly.png" %% pid)
json.dump([{"part_id": "body", "instance_id": "b", "transform": mat(0, 0, 0, 0)}] +
          [{"part_id": "p", "instance_id": "p_i%%d" %% (i + 1), "transform": T} for i, T in enumerate(truth)],
          open("submission/assembly/instances.json", "w"))
json.dump({"truth": [list(map(list, np.array(T) - np.array([[0, 0, 0, C[0]], [0, 0, 0, C[1]], [0, 0, 0, C[2]], [0, 0, 0, 0]]))) for T in truth]},
          open("truth.json", "w"))
'''


def _run_scene(tmp_path, truth, seconds):
    script = tmp_path / "scene.py"
    script.write_text(SCENE % {"here": str(HERE), "truth": truth})
    env = {"PATH": "/usr/bin:/bin", "PYTHONWARNINGS": "ignore", "HOME": str(tmp_path), "MPLCONFIGDIR": str(tmp_path / "mpl")}
    cp = subprocess.run([str(CAD_PY), str(script)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=240)
    assert cp.returncode == 0, cp.stderr[-800:]
    cp = subprocess.run([str(CAD_PY), str(HERE / "_probe.py"), "autoplace", "--part", "all", "--seconds", str(seconds),
                         "--outdir", ".cad/render"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=500)
    assert cp.returncode == 0, cp.stderr[-800:]
    res = json.loads([l for l in cp.stdout.splitlines() if l.strip()][-1])
    truth = [np.array(T) for T in json.loads((tmp_path / "truth.json").read_text())["truth"]]
    return res, truth


@pytest.mark.timeout(600)
@pytest.mark.skipif(not CAD_PY.exists(), reason="cad-tools venv not installed")
def test_autoplace_recovers_a_known_pose(tmp_path):
    res, truth = _run_scene(tmp_path, "[mat(70, 20, -15, 30)]", 90)
    got = np.array(res["parts"]["p"]["transforms"][0]); t = truth[0]
    assert res["parts"]["p"]["iou"] > 0.8 and res["frame"]["sc"] == pytest.approx(1.5 * 176 / 300, rel=0.1)   # the probe works on a 176 px grid
    assert np.linalg.norm(got[:3, 3] - t[:3, 3]) < 6.0, (got[:3, 3], t[:3, 3])          # 5% of the 120 mm body
    ang = np.degrees(np.arccos(np.clip((np.trace(got[:3, :3].T @ t[:3, :3]) - 1) / 2, -1, 1)))
    assert ang < 12.0, ang
    assert abs(got[:3, :3] @ got[:3, :3].T - np.eye(3)).max() < 1e-12
    # the `placed:` check still works next to it: the true poses (written by the scene) fit their own picture
    env = {"PATH": "/usr/bin:/bin", "PYTHONWARNINGS": "ignore", "HOME": str(tmp_path), "MPLCONFIGDIR": str(tmp_path / "mpl")}
    cp = subprocess.run([str(CAD_PY), str(HERE / "_probe.py"), "placed", "--part", "p", "--target", "parts/p_in_assembly.png",
                         "--outdir", ".cad/render"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=240)
    assert cp.returncode == 0, cp.stderr[-800:]
    assert json.loads([l for l in cp.stdout.splitlines() if l.strip()][-1])["combined"] > 0.7


@pytest.mark.timeout(600)
@pytest.mark.skipif(not CAD_PY.exists(), reason="cad-tools venv not installed")
def test_autoplace_recovers_three_instances_around_an_axis(tmp_path):
    truth = "[mat(a, 32 * math.cos(math.radians(a)), 32 * math.sin(math.radians(a)), 22) for a in (20, 140, 260)]"
    res, truth = _run_scene(tmp_path, truth, 120)
    got = [np.array(T) for T in res["parts"]["p"]["transforms"]]
    assert res["parts"]["p"]["n_instances"] == 3 and res["parts"]["p"]["iou"] > 0.75
    left = list(range(len(got)))
    for T in truth:                                        # every true instance is matched by a distinct found one
        d = [np.linalg.norm(got[i][:3, 3] - T[:3, 3]) for i in left]
        i = left[int(np.argmin(d))]
        assert min(d) < 8.0, (min(d), [g[:3, 3] for g in got], T[:3, 3])
        left.remove(i)


# ── the shape check at one shared scale (needs the CAD venv) ──

SHAPE_SCENE = r'''
import json, os, sys, numpy as np, cadquery as cq
from PIL import Image
sys.path.insert(0, %(here)r)
import _probe as P
# a stepped jaw-like part: long base, a raised block at one end, three teeth underneath
part = (cq.Workplane("XY").box(60, 20, 10)
        .union(cq.Workplane("XY").box(20, 20, 24).translate((20, 0, 17)))
        .union(cq.Workplane("XY").box(4, 20, 6).translate((-20, 0, -8)))
        .union(cq.Workplane("XY").box(4, 20, 6).translate((-8, 0, -8)))
        .union(cq.Workplane("XY").box(4, 20, 6).translate((4, 0, -8)))).val()
V, T = P._mesh(part); pts = P._surface_points(V, T); pts = pts - (pts.min(0) + pts.max(0)) / 2
os.makedirs("parts", exist_ok=True)
canvas = np.full((2 * P.FS, 2 * P.FS, 3), 255, np.uint8)
for name, (r, c) in zip(["t1", "iso", "t3", "t4"], [(0, 0), (0, 1), (1, 0), (1, 1)]):
    _, right, up = P._basis(P.VIEWS[name])
    m = P._splat(pts, (right, up), 1.6)
    q = np.full((P.FS, P.FS, 3), 255, np.uint8); q[m] = (80, 140, 140)
    canvas[r * P.FS:(r + 1) * P.FS, c * P.FS:(c + 1) * P.FS] = q
Image.fromarray(canvas).save("parts/p_alone.png")
bb = part.BoundingBox()
out = {}
for who, shp in (("exact", part), ("box", cq.Workplane("XY").box(bb.xlen, bb.ylen, bb.zlen).val()),
                 ("squat", part.scale(1.0).transformGeometry(cq.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0.5, 0]])))):
    out[who] = P._shared_check(shp, "parts/p_alone.png", ["t1", "iso", "t3", "t4"], seconds=12)
print(json.dumps(out))
'''


@pytest.mark.timeout(400)
@pytest.mark.skipif(not CAD_PY.exists(), reason="cad-tools venv not installed")
def test_shared_check_separates_a_real_part_from_a_plain_block(tmp_path):
    script = tmp_path / "shape.py"
    script.write_text(SHAPE_SCENE % {"here": str(HERE)})
    env = {"PATH": "/usr/bin:/bin", "PYTHONWARNINGS": "ignore", "HOME": str(tmp_path), "MPLCONFIGDIR": str(tmp_path / "mpl")}
    cp = subprocess.run([str(CAD_PY), str(script)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=380)
    assert cp.returncode == 0, cp.stderr[-800:]
    r = json.loads([l for l in cp.stdout.splitlines() if l.strip()][-1])
    assert r["exact"]["gain"] > 0.4, r                       # the right shape clearly beats a plain block
    assert r["box"]["gain"] < 0.15, r                        # its own bounding box does not
    assert r["squat"]["gain"] < r["exact"]["gain"] - 0.2, r  # wrong proportions cost most of the gain
    s = r["squat"]["stretch"]
    assert r["squat"]["stretched_iou"] > r["squat"]["shared_iou"] and s[2] > max(s[0], s[1]), r   # says: make it taller
    assert (tmp_path / ".cad" / "shape_floor.json").is_file()
