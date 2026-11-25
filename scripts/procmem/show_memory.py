#!/usr/bin/env python
"""What procedural memory did in a benchmark run: per task, whether memory was asked for, whether anything was
returned, which items (and from which task), and the exact text the agent got.

    .venv/bin/python scripts/procmem/show_memory.py ../otto-macarena-runs/out/<run> [--text]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--text", action="store_true", help="print what the agent got too")
    a = p.parse_args()
    res = a.run / "results.jsonl"
    if not res.exists():
        raise SystemExit(f"no results.jsonl in {a.run}")
    rows = [json.loads(line) for line in res.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit("no finished tasks yet")
    r0 = rows[0]
    print(f"memory: {r0.get('memory_mode', 'off')}  "
          f"floor={r0.get('memory_floor')}  items={r0.get('snapshot_size', 0)}")
    if r0.get("memory_mode", "off") == "off":
        print("memory was off for this run: nothing was retrieved or inserted")
        return
    got = 0
    for r in rows:
        logs = list((a.run / "tasks").glob(f"*/*/{r['id']}/run_{r.get('run', 1)}/memory_calls.jsonl"))
        calls = [json.loads(line) for f in logs for line in f.read_text().splitlines() if line.strip()]
        used = r.get("memory_used") or []
        got += bool(used)
        state = "RETRIEVED" if used else "asked, nothing similar" if calls else "never asked"
        print(f"\n{r['id'][:8]} score={r.get('score')} steps={r.get('steps')}  {state}"
              + ("  LEAK" if r.get("memory_leak") else ""))
        for c in calls:
            best = max((x.get("cos", 0) for x in c.get("candidates") or []), default=None)
            print(f"  {c.get('mode', 'tool')}: query={str(c.get('query'))[:70]!r} best_cos={best} "
                  f"used={[u.get('item') for u in c.get('used') or []]} tokens={c.get('tokens')}")
            for u in c.get("used") or []:
                print(f"    - {u.get('item')} ({u.get('kind')}, from task {str(u.get('source_task'))[:8]})")
            if a.text and c.get("text"):
                print("    --- returned to the agent ---\n    " + c["text"].replace("\n", "\n    "))
    print(f"\n{got}/{len(rows)} task(s) got memory")


if __name__ == "__main__":
    main()
