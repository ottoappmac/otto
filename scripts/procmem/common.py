"""Shared helpers for the procedural-memory scripts: loading benchmark runs and splitting them into folds."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "src"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.procedural_memory import compact, retrieve  # noqa: E402

AGENT = "macarena-desktop-agent"


@dataclass
class BenchRun:
    task_id: str
    run: int
    source: str
    domain: str
    score: float
    folder: Path
    rows: list[dict]
    session_id: str

    @property
    def label(self) -> str:
        return "success" if self.score >= 1.0 else "failure"

    @property
    def instruction(self) -> str:
        for r in self.rows:
            if r.get("type") == "user":
                return compact.instruction(str(r.get("content") or ""))
        return ""


def load_runs(folders: list[Path]) -> list[BenchRun]:
    """Every scored native-track run with a saved trajectory. Infra errors and runs whose grader failed are skipped,
    and so are runs that used memory (their trajectories would carry the snapshot's content into a new one)."""
    out: list[BenchRun] = []
    for folder in folders:
        res = folder / "results.jsonl"
        if not res.exists():
            continue
        for line in res.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status") != "ok" or r.get("track") != "native" or r.get("eval_error"):
                continue
            if r.get("memory_used"):
                continue
            path = folder / "trajectories" / f"{r['id']}.run{r.get('run', 1)}.messages.json"
            if not path.exists():
                continue
            rows = compact.load_rows(path.read_text())
            out.append(BenchRun(task_id=r["id"], run=int(r.get("run", 1)), source=r.get("source", ""),
                                domain=r.get("domain", ""), score=float(r.get("score") or 0), folder=folder,
                                rows=rows, session_id=str(r.get("session_id") or f"{r['id']}.run{r.get('run', 1)}")))
    return out


def clusters(runs: list[BenchRun]) -> dict[str, str]:
    """task id -> cluster id. Tasks whose instructions are near-duplicates (same template, other values) share a
    cluster, so a fold never trains on one and tests on its twin. Same-app tasks are *not* grouped: sharing an app
    is what memory is for; sharing a task is leakage."""
    texts: dict[str, str] = {}
    for r in runs:
        texts.setdefault(r.task_id, r.instruction)
    ids = sorted(texts)
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a_i, a in enumerate(ids):
        for b in ids[a_i + 1:]:
            if retrieve.near_duplicate(texts[a], texts[b]):
                parent[find(a)] = find(b)
    return {i: find(i) for i in ids}


def fold_of(cluster_id: str, folds: int) -> int:
    """Stable fold (1..folds) for a cluster."""
    h = int.from_bytes(hashlib.sha256(cluster_id.encode()).digest()[:8], "big")
    return h % folds + 1
