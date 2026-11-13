#!/usr/bin/env python
"""BenchCAD-side helper: list, stage and score cases with BenchCAD's own code.

Run it with BenchCAD's environment (it needs PyMuPDF, cadquery, vtk for scoring):

    uv run --project <benchcad-2-main> python scripts/benchcad/cases.py list  <benchcad-2-main> <cases-root>
    uv run --project <benchcad-2-main> python scripts/benchcad/cases.py stage <benchcad-2-main> <case> <workdir>
    uv run --project <benchcad-2-main> python scripts/benchcad/cases.py score <benchcad-2-main> <case> <workdir> [out.json]

stage  copies the case inputs into <workdir> exactly as BenchCAD's sandbox does (PDFs rendered to
       sheets and tiles, tools.py, BOM) and writes TASK.md, the prompt for the agent.
score  stages a FRESH copy of the inputs next to <workdir>, copies in only the agent's own .py files,
       runs answer.py the way BenchCAD's harness does (finish() appended, Docker, no network) and
       scores the result with BenchCAD's scorer. Prints one JSON object on the last line.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ANSWER = "answer.py"
SKIP = {"tools.py", "sitecustomize.py", "_run.py"}


def _boot(bench: str) -> None:
    sys.path.insert(0, str(Path(bench).resolve()))


def cmd_list(bench: str, root: str) -> None:
    _boot(bench)
    cases = []
    for cj in sorted(Path(root).rglob("case.json")):
        d = cj.parent
        if not (d / "input").is_dir():
            continue
        task = next((p for p in d.parts if p.startswith("task")), "")
        cases.append({"case": str(d), "task": task, "id": f"{task}/{d.name}" if task else d.name})
    print(json.dumps(cases))


def _stage(bench: str, case: Path, work: Path):
    _boot(bench)
    os.environ.setdefault("CADENV_LOCAL", "1")          # staging only; nothing runs here
    from envs.common.sandbox import Sandbox
    if work.exists():
        shutil.rmtree(work)
    return Sandbox(case, work)


def _compare_note(work: Path, kind: str) -> str:
    if kind != "part" or not (work / "views.png").is_file():
        return ""
    return (
        "\n## Scoring tool for this case\n\n"
        "`views.png` is four views of the part. `cad_compare` (workdir `.`, target `views.png`) renders your solid from the same "
        "cameras and returns `combined`: outline overlap and interior-edge match. Raise it step by step; keep a change only if "
        "it goes up. It is a proxy: an outline can match while the interior is wrong, so check holes, hollows and steps too.\n"
    )


def _prompt(case: Path, work: Path) -> str:
    from envs.common.episode import _listing, _submit_what, _task_brief, tools_help
    from envs.common.episode import _kind
    brief = _task_brief(case)
    extra = _compare_note(work, _kind(case))
    listing = "\n".join(_listing(work))
    return f"""{brief}

## Your working directory

`/links/{work.name}/` for the file tools (`read_file`, `write_file`, `edit_file`, `view_image`, `ls`), and `.` for the CAD
tools. These are the same folder. Write `answer.py` and every other file you create there, for example
`/links/{work.name}/answer.py`. Do not write to `/work`, `/output` or any other path: the CAD tools and the scorer
cannot see them. Its contents:

{listing}

## Python helpers (import tools)

{tools_help(case)}

## How to answer

Write your answer as `answer.py` in the working directory: {_submit_what(case)}
The scorer runs `answer.py` once, in a clean sandbox with no network and a 600 s limit, and compares
the geometry with a hidden reference. No `answer.py`, or one that fails, scores 0. Use `cad_run` to run it,
`cad_render` to look at it, and `answer_checkpoint` to keep a working version. The CAD tools take `workdir`:
pass `.` (the working folder above). Nobody can answer questions. Finish when the result matches the inputs.
{extra}"""


def cmd_stage(bench: str, case_s: str, work_s: str) -> None:
    case, work = Path(case_s).resolve(), Path(work_s).resolve()
    _stage(bench, case, work)
    from envs.common.episode import _kind
    (work / "TASK.md").write_text(_prompt(case, work))
    print(json.dumps({"work": str(work), "kind": _kind(case)}))


def cmd_score(bench: str, case_s: str, work_s: str, out: str | None) -> None:
    case, work = Path(case_s).resolve(), Path(work_s).resolve()
    res: dict = {"case": str(case), "score": 0.0}
    answer = work / ANSWER
    if not answer.is_file():
        res["error"] = "no answer.py"
        return _emit(res, out)
    fresh = work.parent / (work.name + "_score")
    box = _stage(bench, case, fresh)
    for p in work.glob("*.py"):                      # the agent's own modules, never tools/sitecustomize
        if p.name not in SKIP:
            shutil.copyfile(p, fresh / p.name)
    from envs.common.episode import FINALISE, _artifact
    code = answer.read_text() + FINALISE
    r = box.run(code, timeout=600)
    res.update(returncode=r.returncode, stderr_tail=(r.stderr or "")[-1500:])
    art = _artifact(box)
    if art is None:
        res["error"] = "program ran but produced no answer file"
        return _emit(res, out)
    try:
        from envs.common.score_case import score_case
        s = score_case(case, art)
        res["score"] = float(s.get("score") or 0.0)
        res["detail"] = s
    except Exception as exc:  # noqa: BLE001
        res["error"] = f"scorer failed: {exc!r}"
    _emit(res, out)


def _emit(res: dict, out: str | None) -> None:
    if out:
        Path(out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, default=str))


def main() -> None:
    a = sys.argv[1:]
    if len(a) < 3:
        sys.exit(__doc__)
    cmd = a[0]
    if cmd == "list":
        cmd_list(a[1], a[2])
    elif cmd == "stage" and len(a) == 4:
        cmd_stage(a[1], a[2], a[3])
    elif cmd == "score" and len(a) in (4, 5):
        cmd_score(a[1], a[2], a[3], a[4] if len(a) == 5 else None)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
