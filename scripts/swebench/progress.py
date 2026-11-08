#!/usr/bin/env python
"""Where is the benchmark run? Works from any terminal; reads files only.

    python3 scripts/swebench/progress.py            # the most recently active run
    python3 scripts/swebench/progress.py --out DIR  # a specific run folder
    python3 scripts/swebench/progress.py --watch    # refresh every 15 seconds

Shows which instances have finished (and how), which are running (tool calls so far,
time since the last step), and, if scoring has started, the score so far.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import time
from pathlib import Path

RUNS = Path(os.environ.get("SWEBENCH_RUNS") or Path(__file__).resolve().parents[2].parent / "otto-swebench-runs")


def newest_run() -> Path | None:
    runs = [p for p in (RUNS / "out").glob("*/") if (p / "trajectories").exists()]
    return max(runs, key=lambda p: max((f.stat().st_mtime for f in (p / "trajectories").glob("*")), default=0), default=None)


def processes() -> str:
    out = subprocess.run(["ps", "-eo", "etime,command"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "scripts/swebench/bench.sh" in line and "grep" not in line and "progress.py" not in line:
            return line.strip().split(None, 1)[0]
    return ""


def show(run: Path) -> None:
    traj = run / "trajectories"
    finished = {Path(f).name[:-5] for f in glob.glob(str(traj / "*.json")) if not f.endswith(".events.jsonl")}
    preds = {}
    if (run / "preds.jsonl").exists():
        for line in (run / "preds.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                preds[r["instance_id"]] = r
    running = []
    for f in sorted(traj.glob("*.events.jsonl"), key=lambda p: p.stat().st_mtime):
        iid = f.name[: -len(".events.jsonl")]
        if iid in finished:
            continue
        ev = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        if not ev:
            continue
        calls = sum(1 for e in ev if e.get("type") == "tool_call")
        running.append((iid, calls, int((time.time() - ev[0]["ts"]) // 60), int(time.time() - ev[-1]["ts"])))

    alive = processes()
    print(f"run folder : {run}")
    print(f"bench.sh   : {'running for ' + alive if alive else 'not running'}   ({time.strftime('%H:%M:%S')})")
    print(f"finished   : {len(finished)}    running: {len(running)}\n")
    for iid in sorted(finished):
        d = json.loads((traj / f"{iid}.json").read_text())
        calls = sum(1 for x in (d.get("messages") or []) if x.get("type") == "tool_call")
        mins = (d.get("agent_seconds") or d.get("seconds") or 0) / 60
        print(f"  done     {iid:36s} {str(d.get('status')):12s} {mins:5.1f} min  {calls:3d} calls  patch {d.get('patch_chars')} chars")
    for iid, calls, started, quiet in running:
        note = "  <- long quiet spell (probably a long test run)" if quiet > 300 else ""
        print(f"  running  {iid:36s} {calls:3d} calls, started {started} min ago, last step {quiet}s ago{note}")
    res = run / "results.json"
    if res.exists():
        r = json.loads(res.read_text())
        print(f"\nscoring    : {r.get('scored')}/{r.get('submitted')} scored, {r.get('resolved')} resolved")
        if "gold_valid_ids" in r:
            print(f"gold check : {len(r['gold_valid_ids'])} valid, {len(r['gold_failed_ids'])} fail even with the correct patch")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out")
    p.add_argument("--watch", action="store_true")
    args = p.parse_args()
    while True:
        run = Path(args.out).expanduser() if args.out else newest_run()
        if run is None:
            raise SystemExit(f"no runs found under {RUNS / 'out'}")
        if args.watch:
            print("\033[2J\033[H", end="")
        show(run)
        if not args.watch:
            return
        time.sleep(15)


if __name__ == "__main__":
    main()
