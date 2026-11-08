"""Load the SWE-bench dataset from Hugging Face or from a local .jsonl file.

A local file means no network is needed: ``bench.sh warm`` exports it once, and
``--offline`` runs read it. The scorer (swebench 4.x) also accepts a .jsonl path
as its --dataset_name, so the same path works everywhere.

    python data.py export --dataset verified --out dataset/swe-bench_verified.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATASETS = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite": "princeton-nlp/SWE-bench_Lite",
    "full": "princeton-nlp/SWE-bench",
}


def resolve(name: str) -> str:
    """A short name, a Hugging Face id, or a path: returned in the form the loaders take."""
    return DATASETS.get(name, name)


def is_local(name: str) -> bool:
    p = Path(name).expanduser()
    return p.suffix in (".jsonl", ".json") and p.exists()


def load_rows(name: str, split: str = "test") -> list[dict]:
    """All rows of the dataset, as plain dicts."""
    name = resolve(name)
    if is_local(name):
        text = Path(name).expanduser().read_text()
        if name.endswith(".json"):
            return json.loads(text)
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    from datasets import load_dataset
    return [dict(r) for r in load_dataset(name, split=split)]


def export(name: str, out: Path) -> int:
    rows = load_rows(name)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(out)                       # never leave a half-written file behind
    return len(rows)


def warm_scorer_cache(name: str) -> int:
    """Build the scorer's spec for every instance once, online, so http_cache records the
    GitHub files it reads. Later offline runs answer from that record."""
    import http_cache
    from swebench.harness.test_spec.test_spec import make_test_spec
    http_cache.install()
    rows = load_rows(name)
    for r in rows:
        make_test_spec(r, namespace="logicstar")
    return len(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("warm-scorer-cache")
    w.add_argument("--dataset", required=True)
    e = sub.add_parser("export")
    e.add_argument("--dataset", default="verified")
    e.add_argument("--out", required=True)
    args = p.parse_args()
    if args.cmd == "warm-scorer-cache":
        print(f"prepared the scorer's specs for {warm_scorer_cache(args.dataset)} instances")
        return
    n = export(args.dataset, Path(args.out).expanduser())
    print(f"exported {n} rows to {args.out}")


if __name__ == "__main__":
    main()
