#!/usr/bin/env python
"""Build a frozen procedural-memory snapshot from finished MacArena runs, holding out one fold.

    .venv/bin/python scripts/procmem/build_snapshot.py --runs ../otto-macarena-runs/out/* \\
        --folds 5 --fold 1 --out ../otto-macarena-runs/procmem/f1of5.db

Writes the snapshot (serve it with ``bench.sh up --native --memory <db>``) and ``<db>.test-ids`` (comma-separated
task ids of the held-out fold: pass them to ``bench.sh run --ids``). Folds group near-duplicate tasks (same template,
other values) so a task's twin is never in its own snapshot; ``--folds 1`` builds from everything and holds nothing
out (for testing on a different task set, e.g. built from osworld, run on macosworld with ``--source``).

Labels are the evaluator's scores. Runs that used memory are skipped. ``--llm`` writes intents and lessons with the
memory model Otto is configured with (one call per run); without it intents are deterministic and there are no
lessons. ``--embedder`` must be the one the benchmark backend will use (default: Otto's configured model).
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AGENT, clusters, fold_of, load_runs  # noqa: E402

from backend.procedural_memory import policy, service  # noqa: E402
from backend.procedural_memory.embed import default_embedder  # noqa: E402
from backend.procedural_memory.store import ProcedureStore  # noqa: E402


async def build(args) -> None:
    out = Path(args.out).expanduser().resolve()
    if out.exists():
        raise SystemExit(f"{out} exists; snapshots are immutable, pick a new name")
    runs = load_runs([Path(p) for p in args.runs])
    if args.source:
        keep = set(args.source.split(","))
        runs = [r for r in runs if r.source in keep]
    if not runs:
        raise SystemExit("no scored native-track runs with trajectories found")
    groups = clusters(runs)
    test = sorted({r.task_id for r in runs if args.folds > 1 and fold_of(groups[r.task_id], args.folds) == args.fold})
    train = [r for r in runs if r.task_id not in set(test)]
    embedder = default_embedder(args.embedder)
    service.set_embedder(embedder)
    st = ProcedureStore(out)
    rt = service.Runtime(mode="build", serve_path=None, write_path=out, floor=0.0, max_tokens=800,
                         policy=policy.Policy(), enrich_with_llm=args.llm, serve_tool=False)
    admitted = 0
    for k, r in enumerate(train, 1):
        ids = service.save_rows(r.rows, session_id=r.session_id, direct_agent=AGENT, track="benchmark", st=st,
                                task_id=r.task_id)
        for eid in ids:
            st.add_label(eid, "benchmark", r.label, 1.0)
            d = await service.reconsider(eid, st=st, rt=rt, use_llm=args.llm)
            admitted += d.status == "admitted"
        print(f"[{k}/{len(train)}] {r.task_id} run{r.run} {r.label}", flush=True)
    digest = hashlib.sha256(",".join(sorted(f"{r.task_id}:{r.run}" for r in train)).encode()).hexdigest()[:8]
    sid = f"procmem-{datetime.date.today():%Y%m%d}-f{args.fold}of{args.folds}-{digest}"
    st.set_meta(snapshot_id=sid, embedder=embedder.name, folds=str(args.folds), fold=str(args.fold),
                built_from=[str(Path(p).resolve()) for p in args.runs], train_runs=str(len(train)),
                test_tasks=str(len(test)), llm=str(bool(args.llm)), built_at=datetime.datetime.now().isoformat())
    (out.parent / f"{out.name}.test-ids").write_text(",".join(test) + "\n")
    print(f"snapshot {sid}: {len(train)} runs, {admitted} admitted, items {st.counts()}; "
          f"held out {len(test)} task(s) -> {out.name}.test-ids")
    if os.environ.get("OTTO_PROCEDURE_EMBEDDER") and os.environ["OTTO_PROCEDURE_EMBEDDER"] != embedder.name:
        print("WARNING: OTTO_PROCEDURE_EMBEDDER differs from this snapshot's embedder", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="run folders (each with results.jsonl, trajectories/)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold", type=int, default=1, help="the held-out fold (1..folds)")
    p.add_argument("--source", help="only runs from these sources (comma-separated), e.g. osworld")
    p.add_argument("--out", required=True, help="new snapshot file (.db)")
    p.add_argument("--llm", action="store_true", help="LLM intents and lessons (Otto's memory model)")
    p.add_argument("--embedder", help="embedding model (default: Otto's configured one; 'hash-256' for no model)")
    args = p.parse_args()
    if not 1 <= args.fold <= max(args.folds, 1):
        p.error("--fold must be within 1..--folds")
    asyncio.run(build(args))


if __name__ == "__main__":
    main()
