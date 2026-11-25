#!/usr/bin/env python
"""Phase 0 of procedural memory (docs/procedural-memory-plan.md): measure before trusting anything.

    .venv/bin/python scripts/procmem/calibrate.py --runs ../otto-macarena-runs/out/* --out procmem-phase0 \\
        [--judge --samples 3] [--embedder hash-256]

1. Labels. On every scored benchmark run, compare with the evaluator's score:
   * the agent's own claim (``vm_done`` / ``vm_fail`` / final message);
   * the trace evidence (does the last segment show a confirmed change?);
   * with ``--judge``: the trajectory judge (final message excluded, key points, unanimous over ``--samples``).
   Reports precision with a 95% Wilson interval for "success" and for "failure", and the gate: a label source is
   trusted for a class only if the *lower* bound clears 0.90.
2. Retrieval. Builds a throwaway store from all runs (deterministic intents) and, for each run, retrieves the top 3
   from *other* tasks (same-task and near-duplicate excluded), using the run's own intent as the query. Writes
   ``retrieval.jsonl`` for labelling (does the neighbour share an app and a sub-skill?) and summarises cosine and
   same-app rates, to set the similarity floor.
3. Compaction. Token sizes of the rendered recipes (budget: 600-800 tokens).

Writes everything to ``--out`` (summary.json, labels.jsonl, retrieval.jsonl). Changes nothing else.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AGENT, load_runs  # noqa: E402

from backend.procedural_memory import compact, judge, policy, retrieve, service  # noqa: E402
from backend.procedural_memory.embed import default_embedder  # noqa: E402
from backend.procedural_memory.store import ProcedureStore, tokens  # noqa: E402


def trace_label(ep: compact.Episode) -> str:
    if not ep.segments:
        return "unknown"
    last = ep.segments[-1]
    return "success" if last.label == "success" else "failure"


async def main_async(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs([Path(p) for p in args.runs])
    if not runs:
        raise SystemExit("no scored native-track runs found")
    llm = None
    if args.judge:
        from backend.procedural_memory.enrich import default_llm
        llm = await default_llm()
        if llm is None:
            raise SystemExit("--judge: no model could be built from Otto's memory model settings")

    # 1. labels
    rows, pairs = [], {"self_report": [], "trace": [], "judge": []}
    episodes: list[tuple] = []
    for r in runs:
        split = compact.split_runs(r.rows, r.session_id, direct_agent=AGENT)
        if not split:
            continue
        run = split[0]
        ep = compact.compact(run)
        episodes.append((r, run, ep))
        rec = {"task": r.task_id, "run": r.run, "source": r.source, "actual": r.label, "self_report": ep.self_report,
               "trace": trace_label(ep), "actions": ep.actions, "apps": ep.apps}
        pairs["self_report"].append((ep.self_report, r.label))
        pairs["trace"].append((rec["trace"], r.label))
        if llm is not None:
            j = await judge.judge(run, llm, samples=args.samples)
            rec["judge"], rec["judge_votes"] = j["verdict"], j["votes"]
            pairs["judge"].append((j["verdict"], r.label))
            print(f"judge {r.task_id} run{r.run}: {j['verdict']} (actual {r.label})", flush=True)
        rows.append(rec)
    (out / "labels.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    label_stats = {}
    for src, ps in pairs.items():
        if not ps:
            continue
        stats = judge.agreement(ps).as_dict()
        stats["gate"] = judge.gate(stats, min_precision=args.min_precision)
        label_stats[src] = stats

    # 2. retrieval (deterministic intents: no LLM cost; rerun with LLM intents via build_snapshot for the real floor)
    embedder = default_embedder(args.embedder)
    service.set_embedder(embedder)
    tmp = Path(tempfile.mkdtemp(prefix="procmem-"))
    st = ProcedureStore(tmp / "all.db")
    rt = service.Runtime(mode="build", serve_path=None, write_path=tmp / "all.db", floor=0.0, max_tokens=800,
                         policy=policy.Policy(), enrich_with_llm=False, serve_tool=False)
    for r, _run, _ep in episodes:
        for eid in service.save_rows(r.rows, session_id=r.session_id, direct_agent=AGENT, track="benchmark", st=st,
                                     task_id=r.task_id):
            st.add_label(eid, "benchmark", r.label, 1.0)
            await service.reconsider(eid, st=st, rt=rt, use_llm=False)
    retr, top_cos, same_app, sizes = [], [], [], []
    seen_tasks = set()
    for r, run, ep in episodes:
        if r.task_id in seen_tasks:
            continue
        seen_tasks.add(r.task_id)
        app = ep.apps[0] if ep.apps else None
        query = compact.fallback_intent(app, run.task_text)
        res = await retrieve.recall(st, embedder, query=query, family="desktop", app=None, floor=-1.0,
                                    max_tokens=800, exclude_task_ids={r.task_id}, current_task_text=run.task_text)
        recipes = [c for c in res.candidates if c.item.kind == "recipe"][:3]
        if recipes:
            top_cos.append(recipes[0].cos)
            same_app.append(bool(app) and recipes[0].item.app == app)
        if res.returned:
            sizes.append(res.tokens)
        retr.append({"task": r.task_id, "query": query, "app": app,
                     "neighbours": [{"item": c.item.id, "intent": c.item.intent, "app": c.item.app,
                                     "source_task": c.item.task_id, "cos": round(c.cos, 4),
                                     "useful": None} for c in recipes]})
    (out / "retrieval.jsonl").write_text("".join(json.dumps(x) + "\n" for x in retr))

    def q(xs: list[float], p: float) -> float | None:
        return round(statistics.quantiles(xs, n=100)[int(p) - 1], 4) if len(xs) >= 4 else None

    summary = {
        "runs": len(episodes), "tasks": len(seen_tasks),
        "passed": sum(1 for r, _, _ in episodes if r.label == "success"),
        "labels": label_stats,
        "retrieval": {"embedder": embedder.name, "tasks_with_neighbour": len(top_cos),
                      "top1_cos_p25_p50_p75": [q(top_cos, 25), q(top_cos, 50), q(top_cos, 75)],
                      "top1_same_app_rate": round(sum(same_app) / len(same_app), 4) if same_app else None,
                      "note": "label retrieval.jsonl 'useful' (shares an app AND a sub-skill), then set the floor "
                              "where useful neighbours start"},
        "compaction": {"rendered_tokens_p50_max": [q(sizes, 50), max(sizes) if sizes else None],
                       "budget": 800},
        "store_counts": st.counts(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--judge", action="store_true", help="also run the trajectory judge (LLM calls: runs x samples)")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--min-precision", type=float, default=0.90)
    p.add_argument("--embedder", help="embedding model (default: Otto's configured one; 'hash-256' for no model)")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
