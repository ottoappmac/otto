#!/usr/bin/env python
"""How well did the agent do? A plain summary of a benchmark run folder. Reads files only.

    python3 scripts/swebench/report.py --out ../otto-swebench-runs/out/sample10

Counts what the scorer recorded (score_ledger.jsonl), splits the failures into causes
(wrong fix, patch would not apply, no patch because of a timeout, no patch because the model
server stalled), breaks results down by repo, and gives a 95% range for the success rate.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path

STALL_SECONDS = 600     # a silent gap this long inside a run is a stalled model, not slow thinking


def ledger(path: Path) -> dict[str, dict]:
    state: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                state[e["instance_id"]] = e
    return state


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% range for a success rate k/n."""
    if n == 0:
        return 0.0, 0.0
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def stalled(traj: Path, iid: str) -> bool:
    f = traj / f"{iid}.events.jsonl"
    if not f.exists():
        return False
    ts = [json.loads(l)["ts"] for l in f.read_text().splitlines() if l.strip()]
    return any(b - a > STALL_SECONDS for a, b in zip(ts, ts[1:]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out).expanduser()
    traj = out / "trajectories"
    preds = {}
    for line in (out / "preds.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            preds[r["instance_id"]] = r
    score, gold = ledger(out / "score_ledger.jsonl"), ledger(out / "gold" / "score_ledger.jsonl")

    rows = []
    for iid, r in preds.items():
        patch = r["model_patch"].strip()
        ev = score.get(iid)
        # A ledger entry only counts if it was made for the patch that is in preds.jsonl now.
        import hashlib
        current = hashlib.sha1(r["model_patch"].encode()).hexdigest()[:10]
        status = ev["status"] if ev and ev.get("patch_sha") == current else "not scored yet"
        t = {}
        tf = traj / f"{iid}.json"
        if tf.exists():
            t = json.loads(tf.read_text())
        calls = sum(1 for x in (t.get("messages") or []) if x.get("type") == "tool_call")
        if not patch:
            cause = "no patch: model stalled" if stalled(traj, iid) else "no patch: ran out of time"
        elif status == "resolved":
            cause = "solved"
        elif status == "patch_failed":
            cause = "patch would not apply"
        elif status == "unresolved":
            cause = "wrong or incomplete fix"
        else:
            cause = status
        rows.append({"id": iid, "repo": iid.split("__")[0], "cause": cause, "status": t.get("status"),
                     "minutes": (t.get("agent_seconds") or 0) / 60, "calls": calls,
                     "gold": gold.get(iid, {}).get("status")})

    n = len(rows)
    solved = [r for r in rows if r["cause"] == "solved"]
    pending = [r for r in rows if r["cause"] == "not scored yet"]
    with_patch = [r for r in rows if not r["cause"].startswith("no patch")]
    scored = [r for r in rows if r["cause"] not in ("not scored yet",)]
    stall = [r for r in rows if r["cause"] == "no patch: model stalled"]

    def pct(k, d):
        return f"{100 * k / d:5.1f}%" if d else "  n/a"

    lo, hi = wilson(len(solved), n - len(pending))
    print(f"run folder: {out}\n")
    print(f"HEADLINE: {len(solved)} of {n - len(pending)} scored instances solved = {pct(len(solved), n - len(pending))}"
          f"   (95% range {100 * lo:.0f}% to {100 * hi:.0f}%)")
    if pending:
        print(f"          {len(pending)} more have a patch that is not scored yet")
    print(f"          of the instances where the agent produced a patch: {len(solved)} of "
          f"{len(with_patch) - len([r for r in pending])} = {pct(len(solved), len(with_patch) - len(pending))}")
    fair = n - len(pending) - len(stall)
    if stall:
        print(f"          leaving out {len(stall)} that lost their time to a model-server stall: "
              f"{len(solved)} of {fair} = {pct(len(solved), fair)}")
    gold_ok = [r for r in rows if r["gold"] == "resolved"]
    if gold_ok:
        k = sum(1 for r in gold_ok if r["cause"] == "solved")
        print(f"          among the {len(gold_ok)} whose environment is proven sound (gold patch passes): {k} solved = {pct(k, len(gold_ok))}")
    else:
        print("          (no gold check recorded: can't separate environment problems from agent failures)")

    print("\nWHAT HAPPENED TO EACH INSTANCE")
    for cause, k in collections.Counter(r["cause"] for r in rows).most_common():
        print(f"  {k:4d}  {cause}")

    print("\nBY PROJECT                solved / attempted")
    by = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if r["cause"] == "not scored yet":
            continue
        by[r["repo"]][1] += 1
        by[r["repo"]][0] += r["cause"] == "solved"
    for repo, (k, d) in sorted(by.items(), key=lambda kv: -kv[1][1]):
        print(f"  {repo:14s} {k:3d} / {d:3d}   {pct(k, d)}")

    def med(vals):
        vals = sorted(vals)
        return vals[len(vals) // 2] if vals else 0
    print("\nEFFORT (median)")
    for label, group in (("solved", solved), ("wrong fix", [r for r in rows if r["cause"] == "wrong or incomplete fix"]),
                         ("no patch, timed out", [r for r in rows if r["cause"] == "no patch: ran out of time"])):
        if group:
            print(f"  {label:20s} {med([r['minutes'] for r in group]):5.1f} min, {med([r['calls'] for r in group]):3d} tool calls   (n={len(group)})")
    timeouts = sum(1 for r in rows if r["status"] == "timeout")
    print(f"\n  hit the time cap: {timeouts} of {n}")


if __name__ == "__main__":
    main()
