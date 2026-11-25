#!/usr/bin/env python
"""Run the CAD eval set: selftest the cases, or drive an agent through them and score the results.

    eval.sh selftest                       references pass their own check, broken ones fail (no model, no backend)
    eval.sh run --arm spec|control|both    run the prompts through Otto's backend and score each result
    eval.sh report OUTDIR                  print the table again

Both arms are agents made fresh in the benchmark backend from the same prompt, tools and settings; the only
difference is the spec step ("spec": the prompt with the spec step, "control": the shipped prompt, which has none).
Scoring never trusts the agent's own spec: the deliverable (exports/*.step) is measured and checked against
the figures in cases.py. Results are pass/fail per case; a set this small shows large effects only.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend" / "builtin_mcps" / "cad_tools"))

import cad_spec  # noqa: E402
from cases import by_id  # noqa: E402

NO_USER = "No user is available. Make reasonable assumptions and finish; do not ask questions."
ARMS = {"spec": "cadeval-spec", "control": "cadeval-control"}


def cad_python() -> str:
    cands = [os.environ.get("CADEVAL_PYTHON", ""),
             str(Path.home() / "Library/Application Support/Otto/mcp_server/cad-tools/.venv/bin/python"),
             str(ROOT.parent / "otto-benchcad-runs/home/Library/Application Support/Otto/mcp_server/cad-tools/.venv/bin/python")]
    for c in cands:
        if c and Path(c).is_file():
            return c
    raise SystemExit("no CAD Tools Python found: open Otto, Settings > MCP, Connect CAD Tools once, "
                     "or set CADEVAL_PYTHON")


def measure(mode: str, path: Path, timeout: int = 180) -> dict:
    try:
        cp = subprocess.run([cad_python(), str(HERE / "measure.py"), mode, str(path)],
                            capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timed out"}
    lines = [l for l in cp.stdout.splitlines() if l.startswith("{")]
    return json.loads(lines[-1]) if lines else {"ok": False, "error": (cp.stderr or cp.stdout)[-300:]}


def expect_part(case: dict) -> dict:
    spec = cad_spec.parse(json.dumps({"parts": [{"id": case["id"], **case["expect"]}]}))
    return spec["parts"][0]


def judge(case: dict, metrics: dict) -> list[str]:
    return cad_spec.check(expect_part(case), metrics)


# ── selftest ────────────────────────────────────────────────────────────────

CUT_TOP_HALF = """
_bb = result.val().BoundingBox()
result = result.intersect(cq.Workplane("XY").box(1e5, 1e5, _bb.zlen / 2 + 1, centered=(True, True, False))
                          .translate((0, 0, _bb.zmin - 1)))
"""


def selftest(args) -> int:
    bad = 0
    with tempfile.TemporaryDirectory() as td:
        for case in by_id(args.ids.split(",") if args.ids else None):
            row = [case["id"].ljust(20)]
            variants = [("reference", case["reference"], True)]
            if not case.get("loose"):
                variants.append(("top half cut off", case["reference"] + CUT_TOP_HALF, False))
            for name, code, want_pass in variants:
                f = Path(td) / f"{case['id']}_{name[:3]}.py"
                f.write_text(code)
                m = measure("code", f)
                if not m.get("ok"):
                    row.append(f"{name}: DID NOT RUN {m.get('error')}"); bad += 1; continue
                diffs = judge(case, m["metrics"])
                ok = (not diffs) == want_pass
                bad += not ok
                row.append(f"{name}: {'ok' if ok else 'WRONG'}" + (f" ({'; '.join(diffs)[:110]})" if not ok or not want_pass else ""))
            print(" | ".join(row[:1] + [r.split(" (")[0] for r in row[1:]]), flush=True)
            for r in row[1:]:
                if "WRONG" in r or "DID NOT RUN" in r:
                    print("      ", r)
    print("selftest:", "all cases consistent" if not bad else f"{bad} problem(s)")
    return 1 if bad else 0


# ── run ─────────────────────────────────────────────────────────────────────

def load_benchcad_run():
    spec = importlib.util.spec_from_file_location("benchcad_run", ROOT / "scripts" / "benchcad" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["benchcad_run"] = mod
    spec.loader.exec_module(mod)
    return mod


def agent_prompt(arm: str) -> str:
    from backend import agent_library as al
    return al._CAD_AGENT_PROMPT_WITH_SPEC if arm == "spec" else al._CAD_AGENT_PROMPT


def spec_gap(case: dict, work: Path) -> dict:
    """What the agent wrote in spec.json, against the case's own figures (informational, never scored)."""
    f = work / "spec.json"
    if not f.is_file():
        return {"spec_written": False}
    try:
        spec = cad_spec.load(work)
    except cad_spec.SpecError as exc:
        return {"spec_written": True, "spec_valid": False, "spec_error": str(exc)}
    want, notes = case["expect"], []
    got = spec["parts"][0] if spec["parts"] else {}
    for k in ("holes", "solids"):
        if k in want and got.get(k) is not None and got[k] != want[k]:
            notes.append(f"{k} {got[k]} vs {want[k]}")
        elif k in want and got.get(k) is None:
            notes.append(f"{k} not stated")
    gs, ws = got.get("size"), want.get("size")
    if ws and gs:
        for ax, w, g in zip("xyz", ws, gs):
            if w is not None and (g is None or abs(g - w) > max(0.05 * w, 0.1)):
                notes.append(f"size {ax} {g} vs {w:g}")
    elif ws:
        notes.append("size not stated")
    return {"spec_written": True, "spec_valid": True, "spec_gap": notes}


def tool_text(messages: list) -> str:
    return json.dumps(messages, default=str)


async def run_case(api, mod, case: dict, arm: str, rep: int, out: Path, sem, timeout: int) -> dict:
    cid = f"{case['id']}.{arm}.{rep}"
    work = out / "work" / cid
    work.mkdir(parents=True, exist_ok=True)
    traj = out / "trajectories"
    traj.mkdir(parents=True, exist_ok=True)
    rec = {"id": case["id"], "category": case["category"], "arm": arm, "rep": rep, "pass": False,
           "status": "harness_error", "reasons": []}
    t0 = time.time()
    try:
        async with sem:
            grant = await api.call("POST", "/api/workspace-grants", json={"path": str(work), "mode": "write"})
            await api.call("PATCH", f"/api/workspace-grants/{grant['id']}", json={"network": False})
            sess = await api.call("POST", "/api/sessions", json={"agent_name": ARMS[arm]})
            sid = sess["id"]
            await api.call("POST", f"/api/sessions/{sid}/workspace", json={"grant_id": grant["id"]})
            events = traj / f"{cid}.events.jsonl"
            events.write_text("")
            t_agent = time.time()
            stats = await mod.drive(api, sid, f"{case['prompt']}\n\n{NO_USER}", t_agent + timeout, events)
            rec.update(stats, session_id=sid, agent_seconds=round(time.time() - t_agent, 1))
            msgs = []
            try:
                msgs = await api.call("GET", f"/api/sessions/{sid}/messages")
                (traj / f"{cid}.messages.json").write_text(json.dumps(msgs, default=str))
            except Exception as exc:  # noqa: BLE001
                rec["messages_error"] = str(exc)
            try:
                await api.call("DELETE", f"/api/workspace-grants/{grant['id']}")
            except Exception:  # noqa: BLE001
                pass
        text = tool_text(msgs)
        rec["spec_mismatch_seen"] = text.count("SPEC MISMATCH")
        rec["spec_ok_seen"] = text.count("SPEC OK")
        rec.update(spec_gap(case, work))
        steps = sorted((work / "exports").glob("*.step"), key=lambda p: p.stat().st_mtime) if (work / "exports").is_dir() else []
        if not steps:
            rec["reasons"] = ["no STEP exported"]
        else:
            m = measure("step", steps[-1])
            if not m.get("ok"):
                rec["reasons"] = [f"STEP unreadable: {m.get('error')}"]
            else:
                rec["metrics"] = m["metrics"]
                rec["reasons"] = judge(case, m["metrics"])
                rec["pass"] = not rec["reasons"]
        rec["status"] = rec.get("status") if rec.get("status") != "harness_error" else "done"
    except Exception as exc:  # noqa: BLE001
        rec["error"] = repr(exc)
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def table(rows: list[dict]) -> str:
    out = []
    arms = sorted({r["arm"] for r in rows})
    cases = []
    for r in rows:
        if r["id"] not in cases:
            cases.append(r["id"])
    out.append("case".ljust(22) + "".join(a.ljust(48) for a in arms))
    for cid in cases:
        line = cid.ljust(22)
        for a in arms:
            rs = [r for r in rows if r["id"] == cid and r["arm"] == a]
            if not rs:
                line += "-".ljust(48); continue
            cell = f"{sum(r['pass'] for r in rs)}/{len(rs)} pass"
            why = next((r["reasons"][0] for r in rs if r["reasons"]), "")
            line += (cell + (f"  {why[:34]}" if why else "")).ljust(48)
        out.append(line)
    out.append("")
    for a in arms:
        rs = [r for r in rows if r["arm"] == a]
        n = len(rs)
        by = {}
        for r in rs:
            by.setdefault(r["category"], []).append(r["pass"])
        out.append(f"{a}: {sum(r['pass'] for r in rs)}/{n} pass   "
                   + "  ".join(f"{c} {sum(v)}/{len(v)}" for c, v in sorted(by.items()))
                   + f"   spec.json written {sum(bool(r.get('spec_written')) for r in rs)}/{n}"
                   + f"   mismatch seen in {sum(bool(r.get('spec_mismatch_seen')) for r in rs)}/{n}"
                   + f"   mean {sum(r.get('agent_seconds', 0) for r in rs) / max(n, 1) / 60:.1f} min")
    out.append("DEV EVAL: a small set; read it as pass counts per case, not as a score.")
    return "\n".join(out)


async def run(args) -> None:
    mod = load_benchcad_run()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    res = out / "results.jsonl"
    cases = by_id(args.ids.split(",") if args.ids else None)
    arms = list(ARMS) if args.arm == "both" else [args.arm]
    done = {(r["id"], r["arm"], r["rep"]) for r in map(json.loads, res.read_text().splitlines())} if res.exists() else set()
    jobs = [(c, a, i) for i in range(args.repeat) for c in cases for a in arms if (c["id"], a, i) not in done]
    print(f"{len(jobs)} run(s) ({len(done)} already done)")
    if args.dry_run:
        for c, a, i in jobs:
            print("  ", c["id"], a, i)
        return
    api = mod.Api(args.api, args.token)
    for arm in arms:       # same name each time: the prompt is replaced, so a changed prompt is always picked up
        body = {"name": ARMS[arm], "description": f"cadeval arm: {arm}", "system_prompt": agent_prompt(arm),
                "tools": ["cad-tools"], "skills": []}
        try:
            await api.call("POST", "/api/agents", json=body)
        except Exception:  # noqa: BLE001
            await api.call("PUT", f"/api/agents/{ARMS[arm]}", json=body)
    sem, lock = asyncio.Semaphore(args.workers), asyncio.Lock()

    async def one(c, a, i):
        print("start", c["id"], a, i, flush=True)
        rec = await run_case(api, mod, c, a, i, out, sem, args.timeout)
        async with lock:
            with res.open("a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        print(f"end   {c['id']} {a} {i}: {'PASS' if rec['pass'] else 'FAIL'} {'; '.join(rec['reasons'])[:100]} "
              f"{rec.get('error') or ''}", flush=True)

    await asyncio.gather(*(one(*j) for j in jobs))
    report(out)


def report(out: Path) -> None:
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines() if l]
    text = table(rows)
    (out / "summary.txt").write_text(text + "\n")
    print(text)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selftest"); s.add_argument("--ids")
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=["spec", "control", "both"], default="spec")
    r.add_argument("--ids", help="comma-separated case ids or categories (part, assembly, open, hard)")
    r.add_argument("--repeat", type=int, default=1)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--timeout", type=int, default=900, help="seconds per case for the agent")
    r.add_argument("--api", default="http://127.0.0.1:18092")
    r.add_argument("--token", default=os.environ.get("OTTO_API_TOKEN"))
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--out", default=str(Path(os.environ.get("BENCHCAD_RUNS") or ROOT.parent / "otto-benchcad-runs")
                                        / "cadeval" / time.strftime("%Y%m%d%H%M")))
    rp = sub.add_parser("report"); rp.add_argument("outdir")
    args = p.parse_args()
    if args.cmd == "selftest":
        sys.exit(selftest(args))
    elif args.cmd == "run":
        asyncio.run(run(args))
    else:
        report(Path(args.outdir))


if __name__ == "__main__":
    main()
