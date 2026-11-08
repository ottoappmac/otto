#!/usr/bin/env python
"""Write a preds.jsonl made of the dataset's own correct (gold) patches.

No model is involved. Use it to time and sanity-check the scoring side: every
instance should come back resolved, and any that does not points at a problem
with the environment or the scorer rather than the agent.

    uv run --with-requirements scripts/swebench/requirements.txt \
        python scripts/swebench/gold_preds.py --out <runs-dir>/out/gold-mixed \
        --repos scikit-learn,matplotlib,sympy,sphinx,astropy,django

Then:  scripts/swebench/bench.sh score --out <runs-dir>/out/gold-mixed --score-batch 6
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--dataset", default="verified")
    p.add_argument("--repos", default="scikit-learn,matplotlib,sympy,sphinx,astropy,django",
                   help="comma-separated substrings of repo names; one instance is taken per repo")
    p.add_argument("--per-repo", type=int, default=1)
    args = p.parse_args()

    from datasets import load_dataset
    rows = sorted(load_dataset(DATASETS.get(args.dataset, args.dataset), split="test"),
                  key=lambda r: r["instance_id"])
    picked = []
    for name in args.repos.split(","):
        # Lowest-numbered instances tend to be the oldest and cheapest to build.
        picked += [r for r in rows if name in r["repo"]][: args.per_repo]

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "preds.jsonl").open("w") as f:
        for r in picked:
            f.write(json.dumps({"instance_id": r["instance_id"],
                                "model_name_or_path": "gold",
                                "model_patch": r["patch"]}) + "\n")
    for r in picked:
        print(r["instance_id"])
    print(f"wrote {len(picked)} gold predictions to {out / 'preds.jsonl'}")


if __name__ == "__main__":
    main()
