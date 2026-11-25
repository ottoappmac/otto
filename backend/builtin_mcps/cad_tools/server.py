#!/usr/bin/env python3
"""Built-in MCP server: CAD tools (CadQuery sandbox, renders, STEP inspection).

Tools for an agent that writes CadQuery programs, e.g. the BenchCAD tasks:

* ``cad_run``            run a program, report errors + size/volume/topology
* ``cad_render``         run it and render views (same tetrahedral cameras as the
                         BenchCAD T3/T4 input images) so the agent can compare
* ``cad_export``         write STEP (universal CAD), STL, OBJ, or a Blender .blend file
* ``step_inspect``       bounding box, volume and cylinder axes/radii of a STEP file
* ``answer_checkpoint``  keep the last answer that ran; restore it later
* ``budget_status``      read the token meter (benchmark runs only)
* ``netlist_validate``   check a PCB graph the way the BenchCAD scorer would

Programs never run in this process. ``CAD_EXEC`` picks where they run:
  auto      (default) Seatbelt on macOS, else Docker
  seatbelt  this server's own Python (CadQuery is in its venv) under sandbox-exec: no network,
            writes only to the work folder and temp, sensitive home folders unreadable
  docker    the BenchCAD sandbox image (``benchcad-sandbox:<arch>`` or ``CADENV_DOCKER_IMAGE``)
  local     this server's Python with no isolation (debugging only)
Otto confines ``workdir`` to the session's mapped folder (backend/cad_workdir_middleware.py);
``CAD_WORK_ROOT`` is an optional extra limit.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP, Image

if str(Path(__file__).resolve().parent) not in sys.path:      # sibling modules, however the file was loaded
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import cad_spec  # noqa: E402

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("otto.mcp.cad_tools")

mcp = FastMCP("CAD Tools")

HERE = Path(__file__).resolve().parent
PROBE = HERE / "_probe.py"
STATE_DIR = ".cad"
MAX_TIMEOUT = 900


class CadError(RuntimeError):
    pass


# ── plumbing ────────────────────────────────────────────────────────────────

def _image() -> str:
    env = os.environ.get("CADENV_DOCKER_IMAGE", "").strip()
    if env:
        return env
    return "benchcad-sandbox:" + ("arm64" if platform.machine().lower() in ("arm64", "aarch64") else "amd64")


def _workdir(workdir: str) -> Path:
    p = Path(workdir).expanduser().resolve()
    if not p.is_dir():
        raise CadError(f"workdir does not exist or is not a directory: {p}")
    # Otto's backend (backend/cad_workdir_middleware.py) confines `workdir` to the session's mapped
    # folder before a call reaches this server. CAD_WORK_ROOT, when set, is a second, fixed limit.
    root = os.environ.get("CAD_WORK_ROOT", "").strip()
    if root:
        r = Path(root).expanduser().resolve()
        if p != r and r not in p.parents:
            raise CadError(f"workdir {p} is outside CAD_WORK_ROOT {r}")
    return p


def _inside(work: Path, rel: str) -> Path:
    p = (work / rel).resolve()
    if p != work and work not in p.parents:
        raise CadError(f"{rel!r} is outside the workdir")
    return p


def _exec_mode() -> str:
    mode = os.environ.get("CAD_EXEC", "auto").strip().lower() or "auto"
    if mode not in ("auto", "seatbelt", "docker", "local"):
        raise CadError(f"CAD_EXEC must be auto, seatbelt, docker or local, not {mode!r}")
    if mode != "auto":
        return mode
    if sys.platform == "darwin" and Path(SANDBOX_EXEC).is_file():
        return "seatbelt"
    if shutil.which("docker"):
        return "docker"
    raise CadError("No sandbox available: not macOS (Seatbelt) and no Docker. Install Docker, or set "
                   "CAD_EXEC=local to run programs unsandboxed with this server's Python.")


SANDBOX_EXEC = "/usr/bin/sandbox-exec"
_HOME = Path.home()
# Never readable by model-written code, same idea as Otto's shell sandbox.
_SENSITIVE = [_HOME / ".ssh", _HOME / ".aws", _HOME / ".gnupg", _HOME / ".config",
              _HOME / "Library" / "Keychains", _HOME / "Library" / "Application Support" / "Otto"]


def _sb(p: Path) -> str:
    return str(p.resolve()).replace("\\", "\\\\").replace('"', '\\"')


def seatbelt_profile(work: Path, readable: list[Path]) -> str:
    """Deny-default: read most files, write only the work folder and temp, no network."""
    lines = [
        "(version 1)", "(deny default)", '(import "system.sb")',
        "(allow process-fork)", "(allow process-exec)", "(allow signal (target self))",
        "(allow sysctl-read)", "(allow file-read*)", "(allow file-ioctl)", "(allow ipc-posix-shm)",
        "(allow mach-lookup)", "(deny appleevent-send)",
        '(allow file-write* (subpath "/private/tmp"))', '(allow file-write* (subpath "/private/var/folders"))',
        f'(allow file* (subpath "{_sb(work)}"))',
    ]
    for p in _SENSITIVE:
        lines.append(f'(deny file-read* (subpath "{_sb(p)}"))')
        lines.append(f'(deny file-write* (subpath "{_sb(p)}"))')
    for extra in os.environ.get("CAD_DENY_READ", "").split(os.pathsep):   # e.g. the BenchCAD checkout (answers)
        if extra.strip():
            lines.append(f'(deny file-read* (subpath "{_sb(Path(extra.strip()).expanduser())}"))')
    for p in readable:                       # this server's venv and code live under the Otto folder
        lines.append(f'(allow file-read* (subpath "{_sb(p)}"))')
    # `allow file*` does not override an earlier `deny file-read*`: spell read and write out, after the denies.
    lines.append(f'(allow file-read* (subpath "{_sb(work)}"))')
    lines.append(f'(allow file-write* (subpath "{_sb(work)}"))')
    lines.append("(deny network*)")
    return "\n".join(lines) + "\n"


def _exec_probe(work: Path, args: list[str], timeout: int) -> dict[str, Any]:
    timeout = max(5, min(int(timeout), MAX_TIMEOUT))
    mode = _exec_mode()
    name = None
    env = {"PATH": "/usr/bin:/bin", "HOME": str(work), "PYTHONPATH": str(work),
           "PYTHONWARNINGS": "ignore", "MPLCONFIGDIR": tempfile.gettempdir() + "/cadtools-mpl",
           "PYTHONDONTWRITEBYTECODE": "1", "XDG_CACHE_HOME": tempfile.gettempdir() + "/cadtools-cache"}
    if mode == "docker":
        name = "cadtools-" + uuid.uuid4().hex[:10]
        cmd = ["docker", "run", "--rm", "--name", name, "--network", "none",
               "--memory", "6g", "--cpus", "2",
               "-v", f"{work}:/work", "-v", f"{HERE}:/cadtools:ro", "-w", "/work",
               "-e", "PYTHONPATH=/work", "-e", "MPLCONFIGDIR=/tmp/mpl", "-e", "PYTHONWARNINGS=ignore",
               _image(), "python", "/cadtools/_probe.py", *args]
        env = None
    elif mode == "seatbelt":
        venv_root = Path(sys.prefix).resolve()
        readable = [venv_root, HERE, Path(sys.base_prefix).resolve()]
        cmd = [SANDBOX_EXEC, "-p", seatbelt_profile(work, readable), sys.executable, str(PROBE), *args]
    else:  # local: no isolation
        cmd = [sys.executable, str(PROBE), *args]
    try:
        cp = subprocess.run(cmd, cwd=str(work), env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        if name:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        return {"ok": False, "error": f"timed out after {timeout}s (program too slow; simplify booleans/fillets)"}
    lines = [l for l in cp.stdout.splitlines() if l.strip()]
    if lines:
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError:
            pass
    err = (cp.stderr or cp.stdout or "")[-1500:]
    if mode == "docker" and ("Unable to find image" in err or "No such image" in err):
        err += f"\nBuild it once: docker build -t {_image()} <benchcad repo>/sandbox/"
    return {"ok": False, "error": f"sandbox ({mode}) failed (exit {cp.returncode}): {err}"}


def _elapsed(work: Path) -> str:
    """With CAD_TIME_LIMIT set (benchmark runs), tell the agent how much of its time is gone."""
    try:
        limit = float(os.environ.get("CAD_TIME_LIMIT", "0") or 0)
    except ValueError:
        return ""
    if limit <= 0:
        return ""
    stamp = work / STATE_DIR / "started"
    try:
        if not stamp.exists():
            task = work / "TASK.md"
            start = task.stat().st_mtime if task.is_file() else time.time()
            stamp.parent.mkdir(parents=True, exist_ok=True)
            stamp.write_text(str(start))
        used = time.time() - float(stamp.read_text())
    except (OSError, ValueError):
        return ""
    note = f"time: {used / 60:.0f} of {limit / 60:.0f} min used"
    if used > 0.8 * limit:
        note += ". OVER 80%: save the best answer.py now and finish."
    return note


def _topo_line(m: dict[str, Any]) -> str:
    """Topology and fill, the two things a matching outline hides. BenchCAD's part score multiplies the
    match of each count, so one wrong count costs most of the 20% topology share."""
    t = m.get("topology")
    sz = m.get("size") or [0, 0, 0]
    box = sz[0] * sz[1] * sz[2]
    fill = f"  fill (volume / bounding box): {m['volume'] / box:.2f}" if box > 0 else ""
    if fill and m["volume"] / box < 0.15:
        fill += " -- very low: a thin shell? check the inputs really show a cavity"
    if not t:
        return "topology: could not be read" + fill
    return (f"topology: {t[0]} piece(s), {t[1]} through-hole(s), {t[2]} sealed void(s)" + fill
            + "\n  check each count against the inputs: a tube or ring open at both ends is a through-hole, "
            "a cup or blind bore is not")


def _fmt(res: dict[str, Any]) -> str:
    lines = ["OK: program ran and left a valid solid." if res.get("ok") else "NOT OK."]
    for k in ("error", "warning", "note"):
        if res.get(k):
            lines.append(f"{k}: {res[k]}")
    m = res.get("metrics")
    if m:
        lines += [f"size (x,y,z): {m['size']}", f"bbox min: {m['bbox_min']}  max: {m['bbox_max']}",
                  f"volume: {m['volume']}  area: {m['area']}",
                  f"solids: {m['solids']}  faces: {m['faces']}  edges: {m['edges']}  valid: {m['valid']}",
                  f"center of mass: {m['center_of_mass']}", _topo_line(m)]
    if res.get("files_in_submission"):
        lines.append("submission files: " + ", ".join(res["files_in_submission"][:20]))
    if res.get("output_tail"):
        lines.append("--- program output (tail) ---\n" + res["output_tail"])
    return "\n".join(lines)


def _guard(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except CadError as exc:
            return f"ERROR: {exc}"
    return wrapper


# ── tools ───────────────────────────────────────────────────────────────────

@mcp.tool()
@_guard
def cad_run(workdir: str, code_file: str = "answer.py", timeout: int = 120) -> str:
    """Run a CadQuery program in the sandbox and report what it produced.

    The program must leave the solid in a variable named ``result`` (single-part tasks).
    Returns the traceback if it failed, otherwise bounding box, volume, area and counts
    of solids/faces/edges. Use it after every edit; a program that errors or leaves
    nothing scores 0. When the work folder has a ``spec.json`` (the design spec: sizes, through-holes,
    pieces, volume the part must have) the result is checked against it: "SPEC OK", or "SPEC MISMATCH"
    with each difference in numbers.

    Args:
        workdir: The working directory; pass "." for the session's mapped folder.
        code_file: Program path relative to workdir.
        timeout: Seconds before the run is killed (5-900).
    """
    work = _workdir(workdir)
    code = _inside(work, code_file)
    if not code.is_file():
        raise CadError(f"{code_file} not found in {work}")
    note = ""
    if (work / "TASK.md").is_file() and code.resolve() != (work / "answer.py").resolve():
        note = (f"NOTE: only answer.py is scored; {code_file} is not. When this version is the best, "
                "copy it over answer.py (and run answer.py once to confirm it works).")
    res = _exec_probe(work, ["run", "--code", str(code.relative_to(work))], timeout)
    spec = cad_spec.lines(work, str(code.relative_to(work)), res.get("metrics")) if res.get("ok") else []
    return "\n".join(x for x in (_fmt(res), *spec, note, _autoplace_note(work, code), _elapsed(work)) if x)


@mcp.tool()
@_guard
def cad_render(workdir: str, code_file: str = "answer.py", views: str = "t1,iso,t3,t4",
               size: int = 512, timeout: int = 180) -> list:
    """Run the program and render its solid from several viewpoints (parallel projection).

    Default views ``t1,iso,t3,t4`` are the four tetrahedral cameras BenchCAD uses for its
    four-view input images, in 2x2 order: (-1,-1,1), (1,1,1), (-1,1,-1), (1,-1,-1).
    Only ``iso`` is exact in the real inputs; the others are rotated 3-8 degrees there.
    Other views: front back right left top bottom (axis-aligned). Compare the image with
    the input views and fix differences. The shading differs from the benchmark's, so
    compare shape and proportions, not colours.

    Args:
        workdir: The working directory; pass "." for the session's mapped folder.
        code_file: Program path relative to workdir.
        views: Comma-separated view names.
        size: Pixels per view (128-1024).
        timeout: Seconds before the run is killed.
    """
    work = _workdir(workdir)
    code = _inside(work, code_file)
    if not code.is_file():
        raise CadError(f"{code_file} not found in {work}")
    size = max(128, min(int(size), 1024))
    res = _exec_probe(work, ["run", "--code", str(code.relative_to(work)), "--render", views,
                             "--size", str(size), "--outdir", f"{STATE_DIR}/render"], timeout)
    out: list[Any] = ["\n".join(x for x in (_fmt(res), _elapsed(work)) if x)]
    png = res.get("render_png")
    if png:
        p = _inside(work, png)
        if p.is_file():
            out.append(Image(data=p.read_bytes(), format="png"))
            out.append(f"render saved at {p}")
    return out


BLENDER_PATHS = ("/Applications/Blender.app/Contents/MacOS/Blender", "/usr/bin/blender", "/usr/local/bin/blender")


def _find_blender() -> str | None:
    env = os.environ.get("BLENDER_BIN", "").strip()
    for c in ([env] if env else []) + [shutil.which("blender") or "", *BLENDER_PATHS]:
        if c and Path(c).is_file():
            return c
    return None


@mcp.tool()
@_guard
def cad_compare(workdir: str, target: str = "views.png", code_file: str = "answer.py",
                views: str = "t1,iso,t3,t4", shape: str = "auto", timeout: int = 240) -> list:
    """Score the program's shape against a 2x2 four-view input image (BenchCAD tasks 3 and 4 style).

    What is compared (`shape`): "auto" = the program's `result`, or, when it leaves none, the assembly it
    submitted (submission/parts x instances.json, placed as the scorer places them); "assembly" = that
    assembly; "part:<id>" = submission/parts/<id>.step; "placed:<id>" = where that part's instances sit
    in your assembly. For an assembly from views: score each part's shape with
    `target="parts/<id>_alone.png", shape="part:<id>"`; check where each part goes with
    `target="parts/<id>_in_assembly.png", shape="placed:<id>"` (that image shows the part in red inside
    the whole assembly: your instances are drawn in the same frame and compared with the red, and the
    result says how far off and how much too big or small they are, per view); the whole with
    `target="views.png", shape="assembly"`. The whole-assembly score cannot see a jaw or pinion in the
    wrong spot, `placed:` can: do it for every part type, quantity included. When a placement is
    ambiguous, write both versions as two programs and keep the one that scores higher.

    Renders your solid from the four input cameras (it tries small tilts, because the real views are
    rotated 3-8 degrees) and compares it with the input two ways, per view and on average:
    * outline overlap (IoU, 1.0 = same outline), and
    * interior match (F1 of inner edges: holes, steps, ribs, slots, hollows; 1.0 = same).
    `combined` is their mean: raise it. Each view is cropped to its shape and rescaled, so overall
    size is not tested, but proportions, features and orientation are.
    Picture: grey = both outlines, RED = your model only (remove material), BLUE = the input only
    (add material); dark lines are inner edges (black = both, dark blue = only in the input, dark
    red = only in yours). A high outline score with a low interior score means a feature is missing
    or wrong inside; do not stop on the outline alone.
    It also scores plain stand-ins (a box and cylinders the size of your model) and says so when your
    model does no better. For "part:<id>" it adds the SHAPE check the benchmark's part score resembles:
    all four views drawn at ONE scale (so wrong proportions show), compared with the best plain box or
    cylinder of any proportions; `gain` 0 means no better than a plain block (the benchmark gives that 0),
    a correct part gets about 0.5-0.8. Raise the gain, not just `combined`. It keeps your best-scoring program in .cad/best/ and, after 4 compares in a
    row without a gain, puts that version back and says STALLED: then finish.

    Args:
        workdir: The working directory; pass "." for the session's mapped folder.
        target: The 2x2 views image, relative to workdir.
        code_file: Program path relative to workdir.
        views: Comma-separated camera names, in the 2x2 order of the input.
        shape: "auto", "assembly" or "part:<id>" (see above).
        timeout: Seconds before the run is killed.
    """
    if shape not in ("auto", "assembly") and not re.fullmatch(r"(part|placed):[A-Za-z0-9_.-]+", shape):
        raise CadError('shape must be "auto", "assembly", "part:<id>" or "placed:<id>"')
    work = _workdir(workdir)
    code = _inside(work, code_file)
    tgt = _inside(work, target)
    if not code.is_file():
        raise CadError(f"{code_file} not found in {work}")
    if not tgt.is_file():
        raise CadError(f"{target} not found in {work}")
    if shape == "assembly":
        prog = _load_progress(work)
        missing = _missing_placed(work)
        if prog.get("asm_compares", 0) >= FREE_ASSEMBLY_COMPARES and missing:
            raise CadError(
                "whole-assembly compare refused: it cannot see wrong placements and you have used your "
                f"{FREE_ASSEMBLY_COMPARES} free ones. First run the placement check for: {', '.join(missing)} "
                '(cad_compare(target="parts/<id>_in_assembly.png", shape="placed:<id>")). '
                "Call assembly_status for the checklist.")
        prog["asm_compares"] = prog.get("asm_compares", 0) + 1
        _save_progress(work, prog)
    res = _exec_probe(work, ["compare", "--code", str(code.relative_to(work)), "--target",
                             str(tgt.relative_to(work)), "--render", views,
                             "--shape", shape, "--outdir", f"{STATE_DIR}/render"], timeout)
    if not res.get("ok"):
        return [_fmt(res)]
    if res.get("placed"):
        _record_progress(work, "placed", str(res["placed"]), float(res["combined"]))
        return _placed_report(work, code, target, shape, res)
    sh_chk = res.get("shared") if isinstance(res.get("shared"), dict) and "gain" in res.get("shared", {}) else None
    m = res.get("metrics") or {}
    track = float(res["combined"])
    if shape.startswith("part:"):
        pid = shape[5:]
        _record_progress(work, "part", pid, float(res["combined"]))
        _note_shape(work, pid, sh_chk, m)
        if sh_chk:
            track = round(SHAPE_GAIN_WEIGHT * sh_chk["gain"] + (1 - SHAPE_GAIN_WEIGHT) * track, 4)
    lines = [f"combined: {res['combined']}  (outline IoU {res['mean_iou']}, interior edges F1 {res['interior_f1']})"]
    lines += _shared_lines(sh_chk, shape[5:], target) if sh_chk else _floor_lines(res)
    lines += _track_best(work, code, f"{target}|{shape}", track)
    for v in res["views"]:
        lines.append(f"  {v['view']}: IoU {v['iou']}, interior {v['interior_f1']}, camera tilt {v['camera_tilt_deg']} deg; "
                     f"width/height target {v['aspect_target']} vs yours {v['aspect_candidate']}")
    if m:
        lines.append(f"size (x,y,z): {m['size']}  volume: {m['volume']}")
        topo = _topo_line(m)
        if shape == "assembly" or (shape == "auto" and m.get("solids", 1) > 1 and "result" not in code.read_text()):
            topo = topo.split(" -- very low")[0].split("\n")[0] + "  (an assembly: fill is naturally low)"
        lines.append(topo)
        t = m.get("topology")
        if shape.startswith("part:") and t and int(t[0]) > 1:
            lines.append(f"ONE PART, {t[0]} PIECES: a part must be one solid; union the pieces or make them touch "
                         "(the benchmark counts pieces, through-holes and voids)")
    note = _autoplace_note(work, code)
    if note:
        lines.append(note)
    el = _elapsed(work)
    if el:
        lines.append(el)
    out: list[Any] = ["\n".join(lines)]
    p = _inside(work, res["compare_png"])
    if p.is_file():
        out.append(Image(data=p.read_bytes(), format="png"))
    return out


def _placed_report(work: Path, code: Path, target: str, shape: str, res: dict[str, Any]) -> list:
    """Text and picture for `placed:<id>`: how well the part's instances sit where the input shows them."""
    pid = res["placed"]
    lines = [f"placement of {pid} ({res['n_instances']} instance(s) drawn): {res['combined']}  "
             "(overlap of your instances with the red part in the input, mean over the 4 views; 1.0 = same place)"]
    bad = []
    for v in res["views"]:
        if v.get("note"):
            lines.append(f"  {v['view']}: {v['note']}"); continue
        dx, dy = v["yours_vs_target"]
        lines.append(f"  {v['view']}: overlap {v['iou']}, yours is {abs(dx):.2f} {'right' if dx > 0 else 'left'} and "
                     f"{abs(dy):.2f} {'below' if dy > 0 else 'above'} the target (fraction of the view), "
                     f"{v['size_ratio']}x the target's area")
        if v["iou"] < 0.5:
            bad.append(v["view"])
    scored = [v for v in res["views"] if not v.get("note")]
    if res["combined"] < 0.5 and scored:
        lines.append("WRONG PLACE: this part type is not where the input shows it. Look at the red part in "
                     f"{target} in the views where overlap is lowest, work out its position (radius, height, "
                     "angle) and which way it faces, then change its transforms. Two instances at the same "
                     "position are a mistake unless the part is symmetric about it.")
    elif bad:
        lines.append(f"Low in {', '.join(bad)}: often only a view where the part is hidden behind others "
                     "(the input shows it faint); trust the views where it is clearly red.")
    n = res.get("n_instances")
    lines += _track_best(work, code, f"{target}|{shape}", float(res["combined"]))
    el = _elapsed(work)
    if el:
        lines.append(el)
    out: list[Any] = ["\n".join(lines)]
    p = _inside(work, res["compare_png"])
    if p.is_file():
        out.append(Image(data=p.read_bytes(), format="png"))
    return out


# ── assembly progress (what each part type has had done) ────────────────────

FREE_ASSEMBLY_COMPARES = 2      # whole-assembly compares allowed before every part needs a `placed:` check


def _state_file(work: Path) -> Path:
    return work / STATE_DIR / "progress.json"


def _load_progress(work: Path) -> dict[str, Any]:
    try:
        d = json.loads(_state_file(work).read_text())
        if isinstance(d, dict):
            return d
    except (OSError, ValueError):
        pass
    return {}


def _save_progress(work: Path, d: dict[str, Any]) -> None:
    try:
        _state_file(work).parent.mkdir(parents=True, exist_ok=True)
        _state_file(work).write_text(json.dumps(d))
    except OSError:
        pass


def _record_progress(work: Path, kind: str, pid: str, score: float) -> None:
    d = _load_progress(work)
    d.setdefault(kind, {})[pid] = score
    _save_progress(work, d)


def _expected_parts(work: Path) -> dict[str, int]:
    """Part types of an assembly task and their quantities: bom.json, else the parts/<id>_in_assembly.png files."""
    out: dict[str, int] = {}
    try:
        for it in json.loads((work / "bom.json").read_text()).get("items", []):
            out[str(it["part_id"])] = int(it.get("quantity", 1))
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if not out and (work / "parts").is_dir():
        for f in sorted((work / "parts").glob("*_in_assembly.png")):
            out[f.name[: -len("_in_assembly.png")]] = 1
    return out


def _instance_counts(work: Path) -> dict[str, int]:
    try:
        recs = json.loads((work / "submission" / "assembly" / "instances.json").read_text())
    except (OSError, ValueError):
        return {}
    if isinstance(recs, dict):
        recs = recs.get("instances", [])
    out: dict[str, int] = {}
    for r in recs if isinstance(recs, list) else []:
        if isinstance(r, dict) and r.get("part_id"):
            out[str(r["part_id"])] = out.get(str(r["part_id"]), 0) + 1
    return out


def _missing_placed(work: Path) -> list[str]:
    done = _load_progress(work).get("placed", {})
    return [p for p in _expected_parts(work)
            if (work / "parts" / f"{p}_in_assembly.png").is_file() and p not in done]


@mcp.tool()
@_guard
def assembly_status(workdir: str = ".") -> str:
    """Checklist for an assembly task: every part type with its quantity, whether it is built and
    placed, its last scores, and the ONE next thing to do. Call it when unsure what to do next, and
    before finishing.

    Args:
        workdir: The working directory; pass ".".
    """
    work = _workdir(workdir)
    want = _expected_parts(work)
    if not want:
        return "No assembly task here (no bom.json or parts/<id>_in_assembly.png)."
    prog = _load_progress(work)
    shapes, placed = prog.get("part", {}), prog.get("placed", {})
    gains, blind = prog.get("shape_gain", {}), prog.get("shape_blind", {})
    tries, pieces = prog.get("shape_tries", {}), prog.get("pieces", {})
    inst = _instance_counts(work)
    rows, todo = [], []
    for pid, qty in want.items():
        built = (work / "submission" / "parts" / f"{pid}.step").is_file()
        n = inst.get(pid, 0)
        g = gains.get(pid)
        gtxt = "-" if g is None else ("outline-blind" if blind.get(pid) else f"{g}{' (PLAIN BLOCK)' if g < SHAPE_GAIN_DONE else ''}")
        rows.append(f"  {pid:14} qty {qty}  built {'yes' if built else 'NO '}  instances {n}"
                    f"{'' if n == qty else ' (WRONG COUNT)'}  shape gain {gtxt}  placed {placed.get(pid, '-')}")
        if not built:
            todo.append(f"build {pid}: write its program, export it with tools.export_part, "
                        f"check with cad_compare(target=\"parts/{pid}_alone.png\", shape=\"part:{pid}\")")
        elif int(pieces.get(pid, 1)) > 1:
            todo.append(f"join {pid}: it is {pieces[pid]} separate pieces; a part must be ONE solid (union the "
                        "pieces or make them touch), then compare it again")
    if not todo and not (work / STATE_DIR / "autoplace.json").is_file():
        todo.append("place every part in one go: cad_autoplace() (a few minutes; it estimates the shared scale and "
                    "camera tilts and finds each part's position and facing, and keeps them in answer.py)")
    if not todo:
        for pid in want:
            if pid not in gains and (work / "parts" / f"{pid}_alone.png").is_file() \
                    and int(tries.get(pid, 0)) < SHAPE_MAX_TRIES:
                todo.append(f"score the shape of {pid}: cad_compare(target=\"parts/{pid}_alone.png\", shape=\"part:{pid}\")")
        for pid in want:
            g = gains.get(pid)
            if g is not None and g < SHAPE_GAIN_DONE and not blind.get(pid) and int(tries.get(pid, 0)) < SHAPE_MAX_TRIES:
                todo.append(f"improve the shape of {pid} (shape gain {g}: no better than a plain block, which the "
                            f"benchmark scores 0): match the proportions and main features of parts/{pid}_alone.png, "
                            f"run answer.py, cad_compare(target=\"parts/{pid}_alone.png\", shape=\"part:{pid}\"), "
                            f"then cad_autoplace(part_id=\"{pid}\")")
    for pid, qty in want.items():
        if (work / "submission" / "parts" / f"{pid}.step").is_file() and inst.get(pid, 0) != qty:
            todo.append(f"add instances so {pid} has {qty} (it has {inst.get(pid, 0)}) in submission/assembly/instances.json")
    for pid in want:
        if (work / "parts" / f"{pid}_in_assembly.png").is_file() and pid in inst and pid not in placed:
            todo.append(f"check where {pid} goes: cad_compare(target=\"parts/{pid}_in_assembly.png\", shape=\"placed:{pid}\")")
    weak = sorted((v, k) for k, v in placed.items() if isinstance(v, (int, float)) and v < 0.5 and k in want)
    if weak and not todo:
        todo.append(f"fix the placement of {weak[0][1]} (score {weak[0][0]}): cad_autoplace(part_id=\"{weak[0][1]}\") places it again "
                    "(run answer.py first if you changed the part); if it stays low the part's shape is wrong")
    if not todo:
        todo.append("every part is built, counted and placed. Run cad_compare(target=\"views.png\", shape=\"assembly\") "
                    "once, save with answer_checkpoint, and finish.")
    out = ["parts:"] + rows + ["", "NEXT: " + todo[0]]
    if len(todo) > 1:
        out.append(f"({len(todo) - 1} more after that)")
    el = _elapsed(work)
    if el:
        out.append(el)
    return "\n".join(out)



MIN_PLACEMENT_GAIN = 0.02       # an auto-placement that does not beat the current one by this much is not kept
AUTOPLACE_BEGIN = "# >>> cad_autoplace (generated; run cad_autoplace again to change it, do not edit) >>>"
AUTOPLACE_END = "# <<< cad_autoplace <<<"


def apply_fix(recs, fix):
    """New instance list: the records of each part id in `fix` get that part's matrices in order (surplus
    records are dropped, missing ones are added); other records are untouched. Pure and self-contained:
    its source is copied into answer.py, so the file server and the scored program use the same code."""
    out, seen = [], {}
    for r in recs:
        pid = r.get("part_id")
        if pid in fix:
            k = seen.get(pid, 0)
            seen[pid] = k + 1
            if k >= len(fix[pid]):
                continue
            r = {key: v for key, v in r.items() if key not in ("T", "transform")}
            r["transform"] = fix[pid][k]
        out.append(r)
    used = {str(r.get("instance_id")) for r in out}
    for pid, mats in fix.items():
        for k in range(seen.get(pid, 0), len(mats)):
            j = 1
            while f"{pid}_i{j}" in used:
                j += 1
            used.add(f"{pid}_i{j}")
            out.append({"part_id": pid, "instance_id": f"{pid}_i{j}", "transform": mats[k]})
    return out


def autoplace_block(fix: dict) -> str:
    """The code appended to answer.py so the found poses survive when the scorer runs answer.py from scratch
    (it regenerates submission/ and ignores edits made to it afterwards). It rewrites the instances
    the program just submitted, and never turns a working answer into a crash."""
    import inspect
    return "\n".join([
        AUTOPLACE_BEGIN,
        "_CAD_AUTOPLACE_FIX = " + json.dumps(fix, separators=(",", ":")),
        inspect.getsource(apply_fix).rstrip(),
        "",
        "",
        "def _cad_autoplace_run():",
        "    import json, sys",
        "    try:",
        "        import tools",
        '        path = tools.SUBMISSION / "assembly" / "instances.json"',
        "        if path.is_file():",
        "            recs = json.loads(path.read_text())",
        "        else:  # a clean run: use the list answer.py built, if any",
        '            recs = globals().get("instances") or []',
        "        if isinstance(recs, dict):",
        '            recs = recs.get("instances", [])',
        "        tools.submit_assembly(apply_fix(recs, _CAD_AUTOPLACE_FIX))",
        "    except Exception as exc:  # keep the answer that was already written",
        '        print("cad_autoplace: placements not applied:", exc, file=sys.stderr)',
        "",
        "",
        "_cad_autoplace_run()",
        AUTOPLACE_END, ""])


def _fix_in(text: str) -> dict:
    m = re.search(r"^_CAD_AUTOPLACE_FIX = (.*)$", text, re.M)
    try:
        return json.loads(m.group(1)) if m else {}
    except ValueError:
        return {}


def _with_block(text: str, fix: dict) -> str:
    """`text` (answer.py) with the autoplace block for `fix` appended, replacing an older block."""
    i, j = text.find(AUTOPLACE_BEGIN), text.find(AUTOPLACE_END)
    if i != -1 and j > i:
        text = text[:i].rstrip("\n") + "\n" + text[j + len(AUTOPLACE_END):].lstrip("\n")
    return text.rstrip("\n") + "\n\n\n" + autoplace_block(fix)


def _check_rotations(transforms) -> None:
    for T in transforms:
        R = [row[:3] for row in T[:3]]
        dev = max(abs(sum(R[i][k] * R[j][k] for k in range(3)) - (1.0 if i == j else 0.0))
                  for i in range(3) for j in range(3))
        if dev > 1e-9 or abs(_det3(R) - 1.0) > 1e-9:
            raise CadError(f"auto-placement produced a matrix that is not a rotation (off by {dev:.1e}); not applied")


def _det3(R: list[list[float]]) -> float:
    return (R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
            + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]))


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _placed_score(work: Path, part_id: str, target: str, views: str, timeout: int) -> float:
    """The same placement score `cad_compare(shape="placed:<id>")` gives, for what instances.json holds now."""
    res = _exec_probe(work, ["placed", "--part", part_id, "--target", target, "--render", views,
                             "--outdir", f"{STATE_DIR}/render"], timeout)
    if not res.get("ok"):
        raise CadError("could not score the placement: " + str(res.get("error", res))[:300])
    return float(res["combined"])


def _autoplace_note(work: Path, code: Path) -> str:
    """After answer.py was edited, the block that carries the found poses may be gone: say so."""
    try:
        fix = json.loads((work / STATE_DIR / "autoplace.json").read_text()).get("fix", {})
        if fix and AUTOPLACE_BEGIN not in code.read_text():
            return (f"WARNING: {code.name} no longer has the cad_autoplace block, so the placements found for "
                    f"{', '.join(sorted(fix))} are NOT in the scored answer. Restore them with "
                    "cad_autoplace(part_id=<id>, reapply=true).")
    except (OSError, ValueError, AttributeError):
        pass
    return ""


@mcp.tool()
@_guard
def cad_autoplace(workdir: str, part_id: str = "all", count: int = 0, code_file: str = "answer.py",
                  views: str = "t1,iso,t3,t4", reapply: bool = False,
                  seconds: int = 240, timeout: int = 600) -> list:
    """Place the parts of an assembly from the red outlines in the input images, and keep it in answer.py.

    For an assembly task whose input shows each part in red inside the whole assembly
    (`parts/<id>_in_assembly.png`). Run answer.py first so the parts are exported
    (`submission/parts/<id>.step`). The four views share one frame (one scale, small camera tilts), which the
    tool estimates from a few parts; each part's position then follows from where its outline sits and only
    its orientation is searched. Use part_id="all" (the default) to place every part in one go: it takes a
    few minutes. A later call with one part_id places only that part (for example after you fixed its
    shape) in the frame found earlier. The result is scored with the same check as `placed:<id>` and kept
    only if that score improves clearly. The scorer re-runs answer.py from scratch, so the poses are written
    into a marked block at the end of answer.py (do not edit it; do not delete it: if you rewrite answer.py,
    run this tool again with reapply=true). Rigid motion only: it cannot fix a wrong SHAPE. A part that still
    overlaps badly afterwards has the wrong shape or size: fix the part, re-run answer.py, then run this
    again for that part.

    Args:
        workdir: The working directory; pass "." for the session's mapped folder.
        part_id: "all" or one part type, e.g. "pinion".
        count: With one part_id: how many instances it should have (0 = the bill of materials quantity).
        code_file: The program that writes the submission, relative to workdir.
        views: Comma-separated camera names, in the 2x2 order of the input.
        reapply: Do not search; put the poses saved earlier back into answer.py.
        seconds: Search time budget.
        timeout: Seconds before the run is killed.
    """
    if part_id != "all" and not re.fullmatch(r"[A-Za-z0-9_.-]+", part_id or ""):
        raise CadError("part_id must be \"all\" or a part name such as 'pinion'")
    work = _workdir(workdir)
    code = _inside(work, code_file)
    if not code.is_file():
        raise CadError(f"{code_file} not found in {work}")
    state_path = work / STATE_DIR / "autoplace.json"
    frame_path = work / STATE_DIR / "frame.json"
    try:
        saved = json.loads(state_path.read_text()).get("fix", {})
    except (OSError, ValueError, AttributeError):
        saved = {}
    if reapply:                                  # only the block in the program: the next run rebuilds the rest
        if part_id != "all" and part_id not in saved:
            raise CadError(f"no saved placement for {part_id}; run cad_autoplace(part_id=\"{part_id}\") first")
        if not saved:
            raise CadError("no saved placement; run cad_autoplace() first")
        _write_atomic(code, _with_block(code.read_text(), saved))
        return [f"put the saved placements of {', '.join(sorted(saved))} back into {code_file}; "
                "run it (cad_run) and check with `placed:<id>`."]
    if part_id != "all" and not frame_path.is_file():
        raise CadError("place all parts first: cad_autoplace() estimates the shared scale and camera tilts "
                       "that a single part is placed in")
    if not list((work / "submission" / "parts").glob("*.step")):
        raise CadError("submission/parts has no exported part: run answer.py (tools.export_part) first")
    if part_id != "all":
        for rel, why in ((f"parts/{part_id}_in_assembly.png", "the input image showing this part in red"),
                         (f"submission/parts/{part_id}.step", "the exported part (run answer.py: tools.export_part)")):
            if not (work / rel).is_file():
                raise CadError(f"{rel} not found: autoplace needs {why}")
    tmo = max(timeout, seconds + 240)
    inst_path = work / "submission" / "assembly" / "instances.json"
    old_code = code.read_text()
    old_inst = inst_path.read_text() if inst_path.is_file() else None
    found = _exec_probe(work, ["autoplace", "--part", part_id, "--render", views,
                               *(["--frame", str(frame_path)] if part_id != "all" else []),
                               "--count", str(max(0, int(count))), "--seconds", str(max(20, min(int(seconds), 900))),
                               "--outdir", f"{STATE_DIR}/render"], tmo)
    if not found.get("ok"):
        return [_fmt(found)]
    new = {pid: r["transforms"] for pid, r in found["parts"].items()}
    for mats in new.values():
        _check_rotations(mats)
    targets = sorted(new)

    def target(pid):
        return f"parts/{pid}_in_assembly.png"

    def scores():
        return {pid: _placed_score(work, pid, target(pid), views, tmo) for pid in targets}

    before = scores() if old_inst is not None else {pid: 0.0 for pid in targets}
    doc = json.loads(old_inst) if old_inst is not None else []
    recs = doc.get("instances", []) if isinstance(doc, dict) else doc
    new_recs = apply_fix(recs, new)
    new_doc = {**doc, "instances": new_recs} if isinstance(doc, dict) else new_recs
    fix = {**saved, **new}
    backup = work / STATE_DIR / "instances.before_autoplace.json"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if old_inst is not None:
        backup.write_text(old_inst)

    def restore():
        _write_atomic(code, old_code)
        if old_inst is None:
            inst_path.unlink(missing_ok=True)
        else:
            _write_atomic(inst_path, old_inst)

    try:
        _write_atomic(inst_path, json.dumps(new_doc, indent=1))
        _write_atomic(code, _with_block(old_code, fix))
        after = scores()
    except BaseException:                        # never leave a half-applied placement behind
        restore()
        raise
    mean_b = sum(before.values()) / len(targets); mean_a = sum(after.values()) / len(targets)
    keep = mean_a >= mean_b + MIN_PLACEMENT_GAIN
    lines = [f"search: frame scale {found['frame']['sc']:.3f}, {found['seconds']} s; own fit per part: "
             + ", ".join(f"{p} {found['parts'][p]['iou']:.2f}" for p in targets)]
    lines.append("placement score (as `placed:` measures it), before -> after: "
                 + ", ".join(f"{p} {before[p]:.2f}->{after[p]:.2f}" for p in targets))
    if keep:
        _write_atomic(state_path, json.dumps({"fix": fix}))
        if part_id == "all":
            _write_atomic(frame_path, json.dumps(found["frame"]))
        for pid in targets:
            _record_progress(work, "placed", pid, after[pid])
        lines.append(f"KEPT: {', '.join(targets)} placed (instance counts: "
                     + ", ".join(f"{p} {found['parts'][p]['n_instances']}" for p in targets)
                     + f"), written into a block at the end of {code_file}.")
        low = [p for p in targets if after[p] < 0.5]
        if low:
            lines.append(f"Still low for {', '.join(low)}: placement alone cannot fix it. The shape or size is "
                         "probably wrong (compare with parts/<id>_alone.png), or the part is hidden in some views.")
    else:
        restore()
        lines.append("NOT KEPT: no clearly better placement was found; instances.json and the program are as before. "
                     "Check the shapes and sizes of the parts against parts/<id>_alone.png.")
    el = _elapsed(work)
    if el:
        lines.append(el)
    out: list[Any] = ["\n".join(lines)]
    p_ = _inside(work, found["compare_png"])
    if p_.is_file():
        out.append(Image(data=p_.read_bytes(), format="png"))
    return out


STALL_AFTER = 4          # compares in a row without beating the best before cad_compare restores it
MIN_GAIN = 0.002         # smaller changes are noise in the score


def _track_best(work: Path, code: Path, key: str, score: float) -> list[str]:
    """Keep the best-scoring version of the program per (program, target, shape) in .cad/best/, and after
    STALL_AFTER compares without a gain put it back as the program. Models hill-climbing on their own drift
    for an hour and hand in a worse version than their best, so the tool keeps the best, not the model."""
    state_file = work / STATE_DIR / "compare_best.json"
    best_dir = work / STATE_DIR / "best"
    try:
        state = json.loads(state_file.read_text())
    except (OSError, ValueError):
        state = {}
    k = f"{code.relative_to(work)}|{key}"
    st = state.get(k) or {"best": -1.0, "since": 0, "n": 0}
    st["n"] += 1
    best_dir.mkdir(parents=True, exist_ok=True)
    saved = best_dir / code.name
    out: list[str] = []
    if score > st["best"] + MIN_GAIN:
        st.update(best=score, since=0)
        shutil.copyfile(code, saved)
        out.append(f"best so far ({score}); saved as {saved.relative_to(work)}")
    else:
        st["since"] += 1
        if st["since"] >= STALL_AFTER and saved.is_file():
            if code.read_bytes() != saved.read_bytes():
                shutil.copyfile(code, best_dir / f"last_try_{code.name}")
                shutil.copyfile(saved, code)
                restored = f"Your best version (combined {st['best']}) is back in {code.name}; this try is in " \
                           f"{(best_dir / f'last_try_{code.name}').relative_to(work)}."
            else:
                restored = f"{code.name} already holds your best version (combined {st['best']})."
            out.append(f"STALLED: {st['since']} compares in a row without beating {st['best']}. {restored} "
                       "Stop hill-climbing: make the last feature check and finish. More edits of the same kind "
                       "will not help; only a different main shape would.")
        else:
            out.append(f"no gain: best is still {st['best']} ({st['since']} of {STALL_AFTER} tries without a gain)")
    state[k] = st
    state_file.write_text(json.dumps(state))
    return out


SHAPE_GAIN_DONE = 0.15          # below this a part's shape is no better than a plain block (benchmark: ~0)
SHAPE_OUTLINE_BLIND = 0.95      # a plain block this close to the outline: outlines cannot judge the part
SHAPE_GAIN_WEIGHT = 0.7         # share of the shape gain in the score the best-program tracker keeps
SHAPE_MAX_TRIES = 6             # part compares after which the checklist stops asking for more on a part
STRETCH_HINT_MIN = 0.04         # outline gain from stretching before the stretch is suggested


def _shared_lines(c: dict[str, Any], pid: str, target: str) -> list[str]:
    """Report of the shape check at one shared scale (the probe's `_shared_check`)."""
    out = [f"SHAPE (all 4 views at one scale, as the benchmark judges a part): gain {c['gain']} "
           f"(your outline {c['shared_iou']} vs the best plain {c['plain_block_name']} {c['plain_block']}; "
           "0 = no better than a plain block, which the benchmark scores 0; a correct part gets 0.5-0.8)"]
    if c["plain_block"] >= SHAPE_OUTLINE_BLIND:
        out.append(f"the outline alone cannot judge {pid}: a plain {c['plain_block_name']} already matches it. "
                   "Judge it by the interior edges and the hole count, and model what is inside it (bores, "
                   "recesses, slots) as the picture shows.")
    elif c["gain"] < SHAPE_GAIN_DONE:
        out.append(f"NOT DONE: {pid} is no better than a plain block. Rebuild its overall proportions and the "
                   f"main features {target} shows (steps, teeth, flanges, bosses, holes), largest first.")
    if c["stretched_iou"] - c["shared_iou"] >= STRETCH_HINT_MIN:
        s = c["stretch"]
        out.append(f"PROPORTIONS: it fits better stretched along your part's own x {s[0]}x, y {s[1]}x, z {s[2]}x "
                   f"(relative; outline {c['shared_iou']} -> {c['stretched_iou']}). The direction is reliable, "
                   "the amount only roughly: measure the picture before changing dimensions.")
    return out


def _note_shape(work: Path, pid: str, c: dict[str, Any] | None, m: dict[str, Any]) -> None:
    """Remember a part's shape gain, whether outlines can judge it, its piece count and the compares spent."""
    d = _load_progress(work)
    d.setdefault("shape_tries", {})[pid] = int(d.get("shape_tries", {}).get(pid, 0)) + 1
    if c:
        d.setdefault("shape_gain", {})[pid] = c["gain"]
        d.setdefault("shape_blind", {})[pid] = bool(c["plain_block"] >= SHAPE_OUTLINE_BLIND)
    t = (m or {}).get("topology")
    if t:
        d.setdefault("pieces", {})[pid] = int(t[0])
    _save_progress(work, d)


def _floor_lines(res: dict[str, Any]) -> list[str]:
    """Compare with plain stand-ins (box, cylinders) the size of the model's bounding box."""
    f = res.get("standins") or {}
    if not f or "error" in f:
        return []
    name, val = max(f.items(), key=lambda kv: kv[1])
    c = float(res["combined"])
    line = f"plain stand-in for comparison: {name} scores {val}"
    if c <= val + 0.02:
        line += (f". YOUR MODEL ({c}) IS NO BETTER THAN A PLAIN {name.upper()}: the main body or its proportions "
                 "are wrong. Rebuild the overall shape first (check the width/height ratios per view below), "
                 "then add features.")
    off = [v["view"] for v in res.get("views", [])
           if v.get("aspect_target") and abs(v["aspect_candidate"] / v["aspect_target"] - 1) > 0.15]
    out = [line]
    if len(off) >= 2:
        out.append(f"PROPORTIONS: width/height is more than 15% off in {', '.join(off)}; fix the overall "
                   "dimensions before details")
    return out


@mcp.tool()
@_guard
def cad_export(workdir: str, code_file: str = "part.py", formats: str = "step,stl", name: str = "part",
               tolerance: float = 0.01, timeout: int = 180) -> str:
    """Export the program's solid to files other software can open. Writes into <workdir>/exports/.

    Formats (comma-separated):
      step  the universal CAD exchange format (ISO 10303). Opens in FreeCAD, Fusion, SolidWorks, Onshape,
            Inventor, CATIA, etc. with exact geometry. Re-read after writing; the volume difference is reported.
      stl   triangle mesh for 3D printing and slicers.
      obj   triangle mesh for Blender and other DCC tools.
      blend a Blender file, made by importing the OBJ in a headless Blender. Needs Blender installed
            (or BLENDER_BIN). Units: the mesh keeps CAD millimetres as Blender units; scale 0.001 for metres.

    Meshes are lossy (curves become facets); ``tolerance`` is the max chord deviation in mm.
    Use step whenever the receiver is CAD software.

    Args:
        workdir: The working directory; pass "." for the session's mapped folder.
        code_file: Program leaving the solid in ``result``.
        formats: Any of step, stl, obj, blend.
        name: Base file name.
        tolerance: Mesh chord tolerance in mm.
    """
    work = _workdir(workdir)
    code = _inside(work, code_file)
    if not code.is_file():
        raise CadError(f"{code_file} not found in {work}")
    wanted = [f.strip().lower() for f in formats.split(",") if f.strip()]
    bad = [f for f in wanted if f not in ("step", "stl", "obj", "blend")]
    if bad:
        raise CadError(f"unsupported format(s) {bad}; use step, stl, obj, blend")
    if not name.replace("_", "").replace("-", "").isalnum():
        raise CadError("name must be letters, digits, - or _")
    probe_fmts = [f for f in wanted if f != "blend"] + (["obj"] if "blend" in wanted and "obj" not in wanted else [])
    res = _exec_probe(work, ["export", "--code", str(code.relative_to(work)), "--formats", ",".join(probe_fmts),
                             "--outdir", "exports", "--name", name, "--tolerance", str(tolerance)], timeout)
    if not res.get("ok"):
        return f"NOT exported: {res.get('error', res)}"
    lines = ["Exported:"] + [f"  {work / f}" for f in res["files"] if Path(f).suffix[1:] in wanted]
    if "step_roundtrip_volume_diff_pct" in res:
        lines.append(f"STEP re-read check: volume differs by {res['step_roundtrip_volume_diff_pct']}%")
    m = res["metrics"]
    lines.append(f"size mm: {m['size']}  volume mm3: {m['volume']}")
    lines += cad_spec.lines(work, str(code.relative_to(work)), m)
    stated = cad_spec.assumptions(work)
    if stated:
        lines.append("assumptions in spec.json (tell the user): " + "; ".join(stated))
    if "blend" in wanted:
        blender = _find_blender()
        obj = work / "exports" / f"{name}.obj"
        blend = work / "exports" / f"{name}.blend"
        if not blender:
            lines.append("blend NOT made: Blender not found (install it or set BLENDER_BIN). The OBJ can be imported by hand.")
        else:
            script = ("import bpy,sys\n"
                      "bpy.ops.wm.read_factory_settings(use_empty=True)\n"
                      f"bpy.ops.wm.obj_import(filepath={str(obj)!r})\n"
                      f"bpy.ops.wm.save_as_mainfile(filepath={str(blend)!r})\n")
            cp = subprocess.run([blender, "-b", "--factory-startup", "--python-expr", script],
                                capture_output=True, text=True, timeout=120)
            if blend.is_file():
                lines.append(f"  {blend}")
            else:
                lines.append("blend NOT made: " + (cp.stderr or cp.stdout)[-400:])
    return "\n".join(lines)


@mcp.tool()
@_guard
def step_inspect(workdir: str, step_path: str, timeout: int = 120) -> str:
    """Describe a STEP file: bounding box, volume, face counts, cylinder radii and axes.

    Use for supplied parts in assembly tasks, to place them by numbers instead of guessing.
    Cylinder axes and radii give hole/shaft positions; matching radii across parts shows
    what mates with what.

    Args:
        workdir: The working directory; pass "." for the session's mapped folder.
        step_path: STEP file path relative to workdir.
    """
    work = _workdir(workdir)
    p = _inside(work, step_path)
    if not p.is_file():
        raise CadError(f"{step_path} not found in {work}")
    res = _exec_probe(work, ["step", "--path", str(p.relative_to(work))], timeout)
    if "metrics" not in res:
        return f"ERROR: {res.get('error', res)}"
    m = res["metrics"]
    lines = [f"size: {m['size']}  bbox min: {m['bbox_min']}  max: {m['bbox_max']}",
             f"volume: {m['volume']}  solids: {m['solids']}  faces: {m['faces']}  valid: {m['valid']}",
             f"center of mass: {m['center_of_mass']}", _topo_line(m),
             f"planar faces: {res['n_planar']}  cylindrical: {res['n_cylindrical']}  other: {res['n_other']}"]
    for c in res["cylindrical_faces"]:
        lines.append(f"  cylinder r={c['radius']} axis={c['axis']} through {c['axis_point']}")
    return "\n".join(lines)


@mcp.tool()
@_guard
def answer_checkpoint(workdir: str, action: str = "save", code_file: str = "answer.py") -> str:
    """Keep the last good answer so a later bad edit can be undone.

    Only an answer that runs and leaves a valid result (or writes submission/) is kept.
    Whatever answer file exists when the run ends is what gets scored, so save a rough
    working answer early, then improve it.

    Args:
        workdir: The working directory; pass "." for the session's mapped folder.
        action: ``save`` (run it, keep it if good), ``restore`` (copy the kept one back), ``status``.
        code_file: Answer path relative to workdir.
    """
    work = _workdir(workdir)
    code = _inside(work, code_file)
    state = work / STATE_DIR
    best, meta = state / "answer.best.py", state / "answer.best.json"
    if action == "status":
        if not best.is_file():
            return "No checkpoint yet."
        return "Checkpoint exists.\n" + (meta.read_text() if meta.is_file() else "")
    if action == "restore":
        if not best.is_file():
            return "ERROR: no checkpoint to restore."
        shutil.copyfile(best, code)
        return f"Restored {code_file} from the checkpoint."
    if action != "save":
        return "ERROR: action must be save, restore or status."
    if not code.is_file():
        raise CadError(f"{code_file} not found in {work}")
    res = _exec_probe(work, ["run", "--code", str(code.relative_to(work))], 180)
    if not res.get("ok"):
        return "NOT saved: this answer does not run cleanly, the previous checkpoint is unchanged.\n" + _fmt(res)
    state.mkdir(exist_ok=True)
    shutil.copyfile(code, best)
    meta.write_text(json.dumps({"metrics": res.get("metrics"), "files": res.get("files_in_submission")}, indent=1))
    return "Saved checkpoint.\n" + _fmt(res)


@mcp.tool()
@_guard
def budget_status(workdir: str = ".") -> str:
    """Show time used and output tokens left, when running under the BenchCAD token meter.

    Args:
        workdir: The working directory; pass ".".

    Reads ``$METER_URL/_meter/status`` (default http://meter:8080). Outside a metered
    run it says so. Check it before starting a big refinement.
    """
    import urllib.request
    t = _elapsed(_workdir(workdir))
    base = os.environ.get("METER_URL", "http://meter:8080").rstrip("/")
    try:
        with urllib.request.urlopen(base + "/_meter/status", timeout=5) as r:
            return (t + "\n" if t else "") + r.read().decode()[:2000]
    except Exception as exc:  # noqa: BLE001
        return (t + "\n" if t else "") + f"No meter reachable at {base} ({exc}). Not a metered run; token budget unknown."


@mcp.tool()
@_guard
def netlist_validate(workdir: str, graph_path: str = "pred_graph.json") -> str:
    """Check a PCB terminal-net graph (task 6) the way the scorer would.

    A graph that fails these checks scores 0. In answer.py leave the dict in ``result`` and call
    ``tools.export(result, "pred_graph.json")`` to write the file this checks. Format: ``{"components": [{"id", "terminals": [..]}],
    "nets": [{"id"}], "incidences": [[terminal, net], ..]}``.

    Args:
        workdir: The working directory; pass "." for the session's mapped folder.
        graph_path: JSON file path relative to workdir.
    """
    work = _workdir(workdir)
    p = _inside(work, graph_path)
    if not p.is_file():
        raise CadError(f"{graph_path} not found in {work}")
    try:
        g = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        return f"INVALID: not JSON ({exc})"
    problems = check_graph(g)
    if problems:
        return "INVALID, scorer would give 0:\n  " + "\n  ".join(problems[:15])
    head = (f"VALID: {len(g['components'])} components, {len(g['nets'])} nets, "
            f"{len(g['incidences'])} incidences.")
    warn = graph_warnings(g)
    return head + ("\nWarnings (valid, but likely to cost points):\n  " + "\n  ".join(warn) if warn else "")


def check_graph(g: Any) -> list[str]:
    """Mirror of the BenchCAD sandbox `_check_graph`; returns problems (empty = ok)."""
    if not isinstance(g, dict):
        return ["the graph must be a dict with components, nets and incidences"]
    comps, nets, inc = g.get("components"), g.get("nets"), g.get("incidences")
    if not all(isinstance(x, list) for x in (comps, nets, inc)):
        return ["the graph needs three lists: components, nets, incidences"]
    problems: list[str] = []
    terminals: set = set(); seen: set = set()
    for c in comps:
        if not isinstance(c, dict) or not c.get("id") or not isinstance(c.get("terminals"), list):
            problems.append(f"component without id/terminals: {c!r}"[:120]); continue
        if c["id"] in seen:
            problems.append(f"duplicate component id {c['id']!r}")
        seen.add(c["id"])
        for t in c["terminals"]:
            if t in terminals:
                problems.append(f"duplicate terminal {t!r}")
            terminals.add(t)
    net_ids = [n.get("id") if isinstance(n, dict) else None for n in nets]
    if any(i is None for i in net_ids):
        problems.append("a net without an id")
    if len(set(net_ids)) != len(net_ids):
        problems.append("duplicate net ids")
    on_net: dict = {}
    for pair in inc:
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            problems.append(f"incidence is not [terminal, net]: {pair!r}"[:120]); continue
        t, n = pair
        if t not in terminals:
            problems.append(f"incidence names terminal {t!r}, which no component declares")
        if n not in set(net_ids):
            problems.append(f"incidence names net {n!r}, which is not in nets")
        if t in on_net and on_net[t] != n:
            problems.append(f"terminal {t!r} is on two nets ({on_net[t]!r} and {n!r})")
        on_net[t] = n
    return problems


def graph_warnings(g: dict) -> list[str]:
    """Likely shorts and opens in a valid graph. Connections score worst on power rails, and a merged rail
    or a shorted part costs far more than a terminal left on its own."""
    comps = {c["id"]: c for c in g["components"]}
    on_net = {t: n for t, n in g["incidences"]}
    members: dict = {}
    for t, n in g["incidences"]:
        members.setdefault(n, []).append(t)
    out: list[str] = []
    shorted = [cid for cid, c in comps.items()
               if len(c["terminals"]) == 2 and c["terminals"][0] in on_net
               and on_net[c["terminals"][0]] == on_net.get(c["terminals"][1])]
    if shorted:
        out.append(f"SHORT: both terminals of {', '.join(shorted[:10])} are on one net; a two-terminal part "
                   "(resistor, capacitor, diode, LED) across a single net does nothing, so two nets were merged")
    total = sum(len(c["terminals"]) for c in comps.values())
    for n, ts in sorted(members.items(), key=lambda kv: -len(kv[1])):
        if len(ts) >= 8 and len(ts) > 0.4 * total:
            out.append(f"POSSIBLE MERGED RAILS: net {n!r} holds {len(ts)} of {total} terminals; check it is one "
                       "rail (GND, or one supply), not GND and a supply, or two supplies, joined")
    names = {}
    for n in g["nets"]:
        label = " ".join(str(x) for x in (n.get("id"), (n.get("meta") or {}).get("name")) if x).upper()
        ground = [r for r in ("GND", "VSS") if r in label]
        supply = [r for r in ("VCC", "VDD", "3V3", "3.3V", "5V", "12V", "VBAT", "VIN") if r in label]
        if ground and supply:                           # "VCC_5V" is one rail; "GND/VCC" is a short
            names[n.get("id")] = ground[:1] + supply[:1]
    for nid, rails in names.items():
        out.append(f"MERGED RAILS: net {nid!r} is named after {' and '.join(rails)}; ground and a supply are separate nets")
    single = [n for n, ts in members.items() if len(ts) == 1]
    if single:
        out.append(f"OPEN?: {len(single)} net(s) connect only one terminal ({', '.join(map(str, single[:8]))}); "
                   "fine for an unused pin, otherwise a trace was missed")
    loose = [t for c in comps.values() for t in c["terminals"] if t not in on_net]
    if loose:
        out.append(f"OPEN?: {len(loose)} terminal(s) are on no net ({', '.join(loose[:8])}); check each is really unconnected")
    return out


if __name__ == "__main__":
    mcp.run()
