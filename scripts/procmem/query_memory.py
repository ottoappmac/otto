#!/usr/bin/env python
"""Ask a procedural-memory store what recall_procedure would return, without a VM, a backend or a run.

    .venv/bin/python scripts/procmem/query_memory.py <store.db> "Keep only the largest image" --app Finder

<store.db> is a snapshot or a run's memory/write.db; it is opened read-only. Prints what is stored (episodes and
items per app), then the candidates with their scores and the text the agent would get. Use the embedder the store
was built with (it is checked against the store's meta).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.procedural_memory.embed import default_embedder  # noqa: E402
from backend.procedural_memory.retrieve import recall  # noqa: E402
from backend.procedural_memory.store import ProcedureStore  # noqa: E402


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("store", type=Path)
    p.add_argument("query")
    p.add_argument("--app", help="app name as stored (e.g. Finder, 'Google Chrome'); omit to search all apps")
    p.add_argument("--family", default="desktop")
    p.add_argument("--floor", type=float, default=0.45, help="similarity floor (the benchmark's default is 0.45)")
    p.add_argument("--exclude-task", action="append", default=[], help="task id to leave out (as a held-out run would)")
    a = p.parse_args()
    if not a.store.exists():
        raise SystemExit(f"no store at {a.store}")
    store = ProcedureStore(a.store, read_only=True)
    with store.connect() as db:
        meta = dict(db.execute("select key, value from meta").fetchall())
        print(f"store: {a.store}  embedder: {meta.get('embedder')}")
        for task, label, apps, intent in db.execute(
                "select task_id, label, apps, intent from episodes order by created_at"):
            print(f"  episode  {str(task)[:8]}  {label:<8} {', '.join(json.loads(apps))}: {intent}")
        for app, kind, n in db.execute("select app, kind, count(*) from items group by app, kind"):
            print(f"  items    {app}: {n} {kind}")
    emb = default_embedder(meta.get("embedder"))
    r = await recall(store, emb, query=a.query, family=a.family, app=a.app, floor=a.floor,
                     exclude_task_ids=set(a.exclude_task))
    print(f"\nquery: {a.query!r}  app: {a.app}  floor: {a.floor}  ->  "
          f"{'NOTHING (abstained)' if r.abstained else f'{len(r.returned)} item(s), {r.tokens} tokens'}")
    for c in r.candidates:
        print(f"  {c.item.kind:<7} cos {c.cos:.2f}  lex {c.lexical:.2f}  score {c.score:.2f}  "
              f"{'cut:' + c.cut if c.cut else 'RETURNED'}  {c.item.app}  task {str(c.item.task_id)[:8]} "
              f"({c.item.from_outcome})")
    print("\n--- what the agent gets ---\n" + r.text)


if __name__ == "__main__":
    asyncio.run(main())
