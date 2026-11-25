#!/usr/bin/env python
"""A BenchCAD harness built around the problem, not a general agent loop (single-part cases).

The harness, not the model, decides what happens next:

  1. stage the case (BenchCAD's own staging), then
  2. ask the model for N candidate programs in parallel,
  3. run each, and score it with `cad_compare` (outline overlap with the input views) when the
     case has a 2x2 `views.png`, else by "it runs and is valid",
  4. repair the best K candidates for R rounds: the model gets its code, the error or the
     overlap numbers and the red/blue difference picture, and returns a new version,
  5. keep the best-scoring valid program as `answer.py` after every step (a timeout still scores),
  6. score it with BenchCAD's scorer.

It talks to any OpenAI-compatible chat endpoint (OpenAI, Anthropic's compat endpoint, Gemini's,
or a local oMLX/vLLM server). Run it with the CAD tools' Python (it has CadQuery, numpy, httpx):

  "$HOME/Library/Application Support/Otto/mcp_server/cad-tools/.venv/bin/python" \\
      scripts/benchcad/harness.py --bench ../benchcad-2-main --case <case dir> --work <dir> \\
      --base-url https://api.openai.com/v1 --model <model> --api-key $KEY
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parents[1] / "backend" / "builtin_mcps" / "cad_tools"
PROBE = TOOLS / "_probe.py"

SYSTEM = """You write CadQuery (2.3.0) programs that rebuild a part from images. Reply with a short \
analysis (overall size, main features, dimensions you can read) and then ONE ```python block: a complete \
program that leaves the solid in a variable named `result`. Units mm.

CadQuery facts that matter:
- `import cadquery as cq`. Keep every part a `cq.Workplane`; wrap a bare solid with `cq.Workplane(obj=solid)`; \
join with `.union()`, cut with `.cut()`.
- `cq.Workplane("XY").box(l, w, h)` is centred. `.circle(r).extrude(h)` starts on the plane. `.faces(">Z").workplane()` \
then `.pushPoints([(x, y)]).hole(d)` for holes (`.hole` takes no position).
- `.fillet(r)` / `.chamfer(r)` round every selected edge; select the ones you mean.
- Revolve: `cq.Workplane("XZ").polyline(pts).close().revolve(360, (0,0,0), (0,1,0))`.
- These do not exist: Face.sweep, Wire.valueAt, a zDir argument on Plane.
- Keep it fast (a run over 120 s is killed): no spiral sweeps, no huge boolean cuts; use simple stand-ins.
- The image shows four views of the same part; the top-right one is the exact isometric (+X +Y +Z). Z is up.
Get overall proportions and the biggest features right first; small details matter less than the main body.
"""


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


# ── model ──────────────────────────────────────────────────────────────────

class Model:
    def __init__(self, base, model, key, temperature=0.7, max_tokens=8000):
        self.base, self.model, self.t, self.max = base.rstrip("/"), model, temperature, max_tokens
        self.h = {"Authorization": f"Bearer {key}"} if key else {}
        self.calls = 0

    def chat(self, content: list[dict], system: str = SYSTEM, temperature: float | None = None) -> str:
        body = {"model": self.model, "max_tokens": self.max,
                "temperature": self.t if temperature is None else temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]}
        for attempt in range(3):
            try:
                r = httpx.post(self.base + "/chat/completions", json=body, headers=self.h, timeout=600)
                r.raise_for_status()
                self.calls += 1
                return r.json()["choices"][0]["message"]["content"] or ""
            except Exception as exc:  # noqa: BLE001
                log("model call failed:", repr(exc)[:200])
                time.sleep(5 * (attempt + 1))
        return ""


def img(path: Path, max_side=1024) -> dict:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if max(im.size) > max_side:
        k = max_side / max(im.size)
        im = im.resize((round(im.width * k), round(im.height * k)))
    buf = io.BytesIO(); im.save(buf, "PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}


def code_of(text: str) -> str:
    m = re.findall(r"```(?:python|py|cadquery)?[ \t]*\r?\n(.*?)(?:```|\Z)", text, re.S | re.I)
    return (m[-1] if m else "").strip()


# ── running and scoring candidates ─────────────────────────────────────────

def probe(work: Path, *args: str, timeout=150) -> dict:
    """Run the probe the way cad_run does (Seatbelt on macOS) via the tool server's own helper."""
    sys.path.insert(0, str(TOOLS))
    import importlib.util
    spec = importlib.util.spec_from_file_location("cad_tools_server", TOOLS / "server.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod._exec_probe(work, list(args), timeout)


class Cand:
    def __init__(self, name):
        self.name, self.code, self.score, self.detail, self.png, self.hist = name, "", -1.0, "", None, []


def evaluate(work: Path, c: Cand, target: str | None) -> None:
    f = f"cand_{c.name}.py"
    (work / f).write_text(c.code)
    if target:
        r = probe(work, "compare", "--code", f, "--target", target, "--outdir", f".cad/cmp_{c.name}")
        if r.get("ok"):
            c.score = float(r["combined"])
            c.detail = (f"combined {r['combined']} = mean of outline overlap {r['mean_iou']} and interior-edge match "
                        f"{r['interior_f1']}. Per view (outline / interior): "
                        + ", ".join(f"{v['view']} {v['iou']}/{v['interior_f1']}" for v in r["views"])
                        + f"; size {r['metrics']['size']}")
            c.png = work / r["compare_png"]
            return
    else:
        r = probe(work, "run", "--code", f)
        if r.get("ok") and r.get("metrics"):
            c.score, c.detail, c.png = 0.01, f"runs; size {r['metrics']['size']}", None
            return
    c.score, c.png = 0.0, None
    c.detail = "ERROR: " + (r.get("error") or r.get("warning") or "no result")[-1200:]


def save_best(work: Path, cands: list[Cand]) -> Cand | None:
    best = max((c for c in cands if c.score > 0), key=lambda c: c.score, default=None)
    if best:
        (work / "answer.py").write_text(best.code)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--base-url", default=os.environ.get("BENCHCAD_BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--model", default=os.environ.get("BENCHCAD_MODEL"))
    ap.add_argument("--api-key", default=os.environ.get("BENCHCAD_API_KEY", ""))
    ap.add_argument("--candidates", type=int, default=3)
    ap.add_argument("--keep", type=int, default=2, help="candidates to keep repairing")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seconds", type=int, default=3000)
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--numbers-only", action="store_true", help="send the model scores only, not the difference picture (cheaper)")
    ap.add_argument("--good-enough", type=float, default=0.93, help="stop repairing when the best combined score reaches this")
    a = ap.parse_args()
    if not a.model:
        sys.exit("--model (or BENCHCAD_MODEL) is required")
    t0 = time.time()
    left = lambda: a.seconds - (time.time() - t0)   # noqa: E731
    work = Path(a.work).resolve()

    cases = HERE / "cases.py"
    run = ["uv", "run", "--project", a.bench, "python", str(cases)]
    info = json.loads(subprocess.run([*run, "stage", a.bench, a.case, str(work)], capture_output=True,
                                     text=True, check=True).stdout.strip().splitlines()[-1])
    if info.get("kind") != "part":
        sys.exit(f"this harness handles single-part cases; this one is {info.get('kind')!r}")
    brief = (work / "TASK.md").read_text()
    views = work / "views.png"
    target = "views.png" if views.is_file() else None
    images = [img(views)] if views.is_file() else [img(p) for p in sorted(work.glob("drawing*.png"))[:1]]
    if not images:
        sys.exit("no input image found to show the model")
    m = Model(a.base_url, a.model, a.api_key)
    note = ("Ignore any mention of tools or files in the task text below; the harness runs your program. "
            "Reply with your analysis and one python block.\n\n")
    log(f"stage ok: {info}; target={target}; {a.candidates} candidates, {a.rounds} repair rounds")

    # 1. candidates, in parallel
    def first(i: int) -> Cand:
        c = Cand(f"c{i}")
        out = m.chat([{"type": "text", "text": note + brief}, *images], temperature=0.5 + 0.2 * i)
        c.code = code_of(out)
        evaluate(work, c, target)
        log(f"{c.name}: {c.score:.3f} {c.detail[:110]}")
        return c

    with ThreadPoolExecutor(max_workers=a.candidates) as ex:
        cands = list(ex.map(first, range(a.candidates)))
    save_best(work, cands)

    # 2. repair the best few
    live = sorted(cands, key=lambda c: -c.score)[:a.keep]
    for rnd in range(a.rounds):
        if left() < 240:
            log("out of time"); break
        def fix(c: Cand) -> Cand:
            msg = [{"type": "text", "text": note + brief + "\n\n## Your current program\n```python\n" + c.code + "\n```\n\n"
                    f"## Result\n{c.detail}\n\n" + (
                    ("In the picture: grey = both outlines, RED = your model only (too much material), BLUE = the input "
                     "only (missing material); dark lines are inner edges (black = both, dark blue = only in the input, "
                     "dark red = only in yours). Fix the biggest differences first; keep what already matches. A good "
                     "outline with a low interior score means a feature is missing or wrong inside.\n"
                     if not a.numbers_only else "Fix the biggest weakness the numbers show; keep what already matches.\n") if c.png else
                    "Fix the error. Keep what works.\n") + "Reply with a short list of changes and the full new program."},
                   *images]
            if c.png and not a.numbers_only:
                msg.append(img(c.png, 800))
            n = Cand(c.name)
            n.code = code_of(m.chat(msg, temperature=0.3))
            if not n.code:
                return c
            evaluate(work, n, target)
            log(f"round {rnd + 1} {c.name}: {c.score:.3f} -> {n.score:.3f}")
            return n if n.score > c.score else c
        with ThreadPoolExecutor(max_workers=len(live)) as ex:
            live = list(ex.map(fix, live))
        best = save_best(work, live + cands)
        log(f"round {rnd + 1} best {best.score if best else 0:.3f}; model calls {m.calls}")
        if target and best and best.score >= a.good_enough:
            log("good enough (outline and interior both match); stopping"); break

    best = save_best(work, live + cands)
    result = {"case": a.case, "best_overlap": best.score if best else 0.0, "model_calls": m.calls,
              "seconds": round(time.time() - t0)}
    if best is None:
        result["error"] = "no valid program"
    elif not a.no_score:
        sc = subprocess.run([*run, "score", a.bench, a.case, str(work)], capture_output=True, text=True)
        try:
            result.update(json.loads(sc.stdout.strip().splitlines()[-1]))
        except Exception:  # noqa: BLE001
            result["score_error"] = (sc.stderr or sc.stdout)[-500:]
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
