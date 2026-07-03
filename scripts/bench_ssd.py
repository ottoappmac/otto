"""Benchmark the MLX turbo SSD cold-tier against in-memory cache baseline.

Why this script (and not just ``bench_turbo.py``):
The existing harness measures ``off / basic / cache`` within one process, so
it never exercises the ``_maybe_prime_from_ssd`` path — the in-memory
prefix cache always wins over any disk read within a single Python run.
To see the SSD tier's real value (skipping the system-prompt prefill on
a *cold* process) we need a protocol that simulates process restarts,
which we do here by calling ``_registry.evict_all()`` + ``mx.clear_cache()``
between each measurement.  This is strictly cheaper than spawning a new
Python process per trial (weight tensors stay in
``chat_models.mlx._shared._LOADED_MODELS``) and still exercises the same
code path, at the cost of a warm OS page cache — which only helps the
SSD tier look better than it would on a truly cold machine.  If you
need reboot-level numbers wrap this script in a shell loop and call
``sudo purge`` between invocations.

Protocol per trial:
    1. Fresh ``cache`` singleton, invoke prompt[i] — cold baseline.
    2. evict + clear_cache.
    3. Fresh ``ssd`` singleton, invoke prompt[i]:
         * trial 0 — disk is empty: MISS, writes on completion (seed).
         * trials 1..N — disk has prior seeds sharing the system
           prefix: HIT, primes the in-memory cache from disk before
           prefill.
    4. evict + clear_cache.

Each trial therefore pairs (``cache_cold_wall`` vs ``ssd_wall``) on the
same prompt.  Trial 0's SSD measurement is the pessimistic "first time
this machine ever saw the prefix" number; trials 1+ are the realistic
steady-state number once the SSD tier has been populated.

Run from the repo root with the project venv:

    PYTHONPATH=src .venv/bin/python scripts/bench_ssd.py

Optional knobs (CLI):

    --model         HF repo id (default: $HF_LLM_MODEL_ID from .env)
    --draft         Optional draft model id for speculative decoding
    --max-tokens    Generation cap per call (default 128)
    --trials        Paired trials to run (default 4)
    --ssd-dir       Override MLX_TURBO_SSD_DIR (default: a temp dir)
    --keep-cache    Skip the pre-run wipe of the SSD dir
    --out           Path to write JSON results (default bench_ssd.json)

The output JSON is the source of truth; the printed tables are just a
human-readable summary.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utilities.environment import Environment       # noqa: E402
from utilities.logger import init_logger            # noqa: E402

Environment.load()
init_logger()

from langchain_core.messages import HumanMessage, SystemMessage   # noqa: E402


# A non-trivial system prompt that's realistic for an agent: long enough
# that prefilling it shows up in wall time on any Apple Silicon Mac, but
# short enough that a small 2-4B model can handle it quickly.  The SSD
# win scales with this length — make it longer if you want a stronger
# signal on a faster machine.
SYSTEM_PROMPT = (
    "You are a careful, concise research assistant operating inside an "
    "agentic workflow. Adhere to the following conventions on every "
    "turn:\n"
    "- Never preamble. Begin with the answer.\n"
    "- Prefer bullet points for enumerable answers, prose for continuous "
    "reasoning.\n"
    "- Cite sources inline as (source: <short-label>). Never fabricate "
    "a source; if unsure, say so.\n"
    "- Keep responses under 200 tokens unless the user explicitly asks "
    "for more.\n"
    "- When asked to reason about code, describe behaviour in terms of "
    "inputs → observable outputs, not internal implementation details "
    "unless those details are the question.\n"
    "- For factual questions with multiple reasonable answers, surface "
    "the disagreement rather than picking one silently.\n"
    "- Do not hedge excessively. State confidence explicitly when "
    "relevant (high / medium / low) and move on.\n"
    "- If the question is underspecified, state the smallest clarifying "
    "question needed before answering; do not guess.\n"
    "- You have no internet access and no tool calls available in this "
    "benchmark; rely only on your internal knowledge."
)


# Distinct user prompts so the prefix-cache trim doesn't collapse the
# whole prompt to zero tokens on a rerun.  Each one shares the full
# SYSTEM_PROMPT prefix, which is what the SSD tier is meant to reuse.
USER_PROMPTS = [
    "Explain depth-first search in three sentences.",
    "List five differences between TCP and UDP.",
    "Why is the sky blue? Be brief.",
    "Summarise the Krebs cycle in 80 words.",
    "Give two examples of when a B-tree outperforms a hash table.",
    "What is the CAP theorem, in one sentence?",
    "Name three techniques to reduce transformer inference latency.",
    "Compare mutexes and semaphores in two sentences.",
]


def _reset_state() -> None:
    """Drop turbo singletons and free MLX scratch buffers.

    Weight tensors stay in ``chat_models.mlx._shared._LOADED_MODELS``
    so rebuilding a singleton is cheap — only per-instance state
    (prompt cache, SSD store handle, prefix tokens) is wiped.  This
    is what lets each trial's fresh singleton exercise the
    ``_maybe_prime_from_ssd`` path without paying a multi-GB weight
    reload between trials.
    """
    try:
        from chat_models.mlx_turbo import _registry as turbo_registry
        turbo_registry.evict_all()
    except Exception:
        pass
    try:
        import mlx.core as mx
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        else:
            mx.metal.clear_cache()
    except Exception:
        pass


def _build(
    level: str,
    *,
    model: str,
    draft: Optional[str],
    max_tokens: int,
    ssd_dir: str,
) -> Any:
    """Construct a TurboMLXChat for *level* using production defaults."""
    from chat_models.mlx_turbo import build_turbo_chat
    return build_turbo_chat(
        turbo_level=level,
        model_path=model,
        draft_model_path=draft or "",
        num_draft_tokens=Environment.get_mlx_num_draft_tokens(),
        max_tokens=max_tokens,
        thinking=Environment.get_mlx_thinking(),
        enable_prompt_cache=Environment.get_mlx_prompt_cache(),
        enable_system_prompt_cache=Environment.get_mlx_system_prompt_cache(),
        kv_bits=Environment.get_mlx_kv_bits(),
        kv_group_size=Environment.get_mlx_kv_group_size(),
        turbo_ssd_dir=ssd_dir,
        turbo_ssd_max_gb=Environment.get_mlx_turbo_ssd_max_gb(),
    )


def _run_once(
    level: str,
    user_prompt: str,
    *,
    model: str,
    draft: Optional[str],
    max_tokens: int,
    ssd_dir: str,
) -> Dict[str, Any]:
    """Build a fresh singleton, invoke once, return wall + metadata."""
    _reset_state()
    llm = _build(
        level,
        model=model,
        draft=draft,
        max_tokens=max_tokens,
        ssd_dir=ssd_dir,
    )
    t0 = time.perf_counter()
    ai = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])
    wall = time.perf_counter() - t0
    meta = dict(ai.response_metadata or {})
    return {
        "level": level,
        "wall_s": round(wall, 4),
        "prompt_tps": meta.get("prompt_tps"),
        "gen_tps": meta.get("generation_tps"),
        "gen_tokens": meta.get("generation_tokens"),
        "cache_hit": meta.get("cache_hit_ratio"),
        "tokens_from_cache": meta.get("tokens_from_cache"),
        "tokens_prefilled": meta.get("tokens_prefilled"),
        "peak_gb": meta.get("peak_memory_gb"),
    }


def _ssd_disk_usage(ssd_dir: str) -> Dict[str, Any]:
    """Snapshot of on-disk SSD cache size + file count for the report."""
    root = Path(ssd_dir)
    if not root.exists():
        return {"exists": False, "files": 0, "bytes": 0}
    total = 0
    files = 0
    for p in root.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
            files += 1
    return {"exists": True, "files": files, "bytes": total}


def _median(values: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return round(statistics.median(clean), 4) if clean else None


def _p95(values: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return clean[0] if clean else None
    return round(statistics.quantiles(clean, n=20)[-1], 4)


def _print_table(
    title: str,
    rows: List[Dict[str, Any]],
    cols: List[str],
) -> None:
    widths = {
        c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0))
        for c in cols
    }
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "-" * len(header)
    print(f"\n=== {title} ===")
    print(header)
    print(sep)
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("HF_LLM_MODEL_ID", ""))
    ap.add_argument(
        "--draft",
        default=os.environ.get("HF_DRAFT_LLM_MODEL_ID", "") or None,
    )
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument(
        "--ssd-dir",
        default="",
        help="Override MLX_TURBO_SSD_DIR. Empty ⇒ fresh temp dir per run.",
    )
    ap.add_argument(
        "--keep-cache",
        action="store_true",
        help="Skip the pre-run wipe of the SSD dir.",
    )
    ap.add_argument("--out", default="bench_ssd.json")
    args = ap.parse_args()

    if not args.model:
        sys.exit("HF_LLM_MODEL_ID is empty — set it in .env or pass --model.")

    # Isolate the cache to a dedicated dir so we never interfere with
    # the user's real ``<app_data>/kv_cache`` and so --keep-cache has
    # deterministic semantics (either the tmp dir persists for re-reads
    # or it doesn't).  Caller can override with --ssd-dir for a stable
    # location across invocations of this script.
    ssd_dir = args.ssd_dir or tempfile.mkdtemp(prefix="bench_ssd_")
    if not args.keep_cache and Path(ssd_dir).exists():
        shutil.rmtree(ssd_dir, ignore_errors=True)
    Path(ssd_dir).mkdir(parents=True, exist_ok=True)

    print(
        f"Model:      {args.model}"
        + (f"  draft: {args.draft}" if args.draft else "")
    )
    print(f"Max tokens: {args.max_tokens}  trials: {args.trials}")
    print(f"SSD dir:    {ssd_dir}  (keep_cache={args.keep_cache})\n")

    usage_before = _ssd_disk_usage(ssd_dir)
    print(
        f"SSD disk before: {usage_before['files']} file(s), "
        f"{usage_before['bytes'] / 1e6:.2f} MB\n"
    )

    # Warmup pass: MLX Metal kernels compile lazily on the first real
    # generation a process performs, and that compile cost is attributed
    # to whichever trial runs first.  Running one throwaway generation
    # up front amortises that cost out of the measurement window so
    # trial 0's ``cache_cold_wall`` reflects "cold prompt cache" — not
    # "cold prompt cache *plus* cold Metal cache" — and is directly
    # comparable to trial 1..N.  The warmup uses the "cache" level (no
    # SSD writes) with a throwaway prompt that won't share a prefix
    # with the benchmark prompts, so it doesn't pollute the SSD dir.
    print("Warmup: compiling Metal kernels with a throwaway generation…")
    warmup_t0 = time.perf_counter()
    _reset_state()
    warmup_llm = _build(
        "cache",
        model=args.model,
        draft=args.draft,
        max_tokens=args.max_tokens,
        ssd_dir=ssd_dir,
    )
    warmup_llm.invoke([
        SystemMessage(content="You are a warmup assistant. Reply tersely."),
        HumanMessage(content="Say 'ok' and nothing else."),
    ])
    _reset_state()
    print(f"Warmup complete in {time.perf_counter() - warmup_t0:.2f}s.\n")

    pairs: List[Dict[str, Any]] = []
    for i in range(args.trials):
        prompt = USER_PROMPTS[i % len(USER_PROMPTS)]

        # 1. cache baseline — every trial starts with a fresh singleton
        #    and no SSD lookup; this is the "no turbo help" number for
        #    a cold process.  We intentionally build ``cache`` (not
        #    ``off``) so both sides of the comparison use the same
        #    executor + singleton machinery and only the SSD layer
        #    differs.  That way any delta is attributable to SSD, not
        #    to turbo infrastructure overhead.
        print(f"--- trial {i} prompt={prompt!r} ---")
        cache_row = _run_once(
            "cache", prompt,
            model=args.model, draft=args.draft,
            max_tokens=args.max_tokens, ssd_dir=ssd_dir,
        )
        print(f"  cache_cold: {cache_row}")

        # 2. ssd measurement — trial 0 hits an empty disk and seeds it,
        #    trial 1+ is where the ``SSD prime: warmed cache`` log
        #    line should fire.  Both cases go through the same code
        #    path; what differs is just whether
        #    ``find_longest_match`` returns a hit.
        ssd_row = _run_once(
            "ssd", prompt,
            model=args.model, draft=args.draft,
            max_tokens=args.max_tokens, ssd_dir=ssd_dir,
        )
        print(f"  ssd       : {ssd_row}")

        pairs.append({
            "trial": i,
            "prompt": prompt,
            "cache_cold": cache_row,
            "ssd": ssd_row,
            "delta_wall_s": round(
                cache_row["wall_s"] - ssd_row["wall_s"], 4,
            ),
            "delta_cache_hit": round(
                (ssd_row["cache_hit"] or 0.0)
                - (cache_row["cache_hit"] or 0.0),
                3,
            ),
        })

    usage_after = _ssd_disk_usage(ssd_dir)
    print(
        f"\nSSD disk after:  {usage_after['files']} file(s), "
        f"{usage_after['bytes'] / 1e6:.2f} MB"
    )

    # Aggregate.  Trial 0 is the pessimistic "first time on this
    # machine" number (SSD miss → seed); we report it separately from
    # the steady-state median over trials 1..N so a reader can see
    # both numbers and decide whether they care about cold-bootstrap
    # cost.  This is also why we warn loudly if trial 1's
    # ``cache_hit`` didn't rise — that's the signal that the prime
    # path isn't actually firing.
    trial0 = pairs[0] if pairs else None
    steady = pairs[1:] if len(pairs) > 1 else []

    summary_rows = []
    if trial0:
        summary_rows.append({
            "scope": "trial 0 (cold disk, SSD seeds)",
            "cache_wall_s": trial0["cache_cold"]["wall_s"],
            "ssd_wall_s": trial0["ssd"]["wall_s"],
            "delta_wall_s": trial0["delta_wall_s"],
            "ssd_cache_hit": trial0["ssd"]["cache_hit"],
            "ssd_tokens_from_cache": trial0["ssd"]["tokens_from_cache"],
        })
    if steady:
        summary_rows.append({
            "scope": f"trials 1..{len(steady)} median (SSD primed)",
            "cache_wall_s": _median(
                [p["cache_cold"]["wall_s"] for p in steady]
            ),
            "ssd_wall_s": _median([p["ssd"]["wall_s"] for p in steady]),
            "delta_wall_s": _median(
                [p["delta_wall_s"] for p in steady]
            ),
            "ssd_cache_hit": _median(
                [p["ssd"]["cache_hit"] for p in steady]
            ),
            "ssd_tokens_from_cache": _median(
                [
                    (p["ssd"]["tokens_from_cache"] or 0)
                    for p in steady
                ]
            ),
        })
        summary_rows.append({
            "scope": f"trials 1..{len(steady)} p95",
            "cache_wall_s": _p95([p["cache_cold"]["wall_s"] for p in steady]),
            "ssd_wall_s": _p95([p["ssd"]["wall_s"] for p in steady]),
            "delta_wall_s": _p95([p["delta_wall_s"] for p in steady]),
            "ssd_cache_hit": None,
            "ssd_tokens_from_cache": None,
        })

    _print_table(
        "SSD benchmark summary (lower wall / higher cache_hit is better)",
        summary_rows,
        [
            "scope", "cache_wall_s", "ssd_wall_s", "delta_wall_s",
            "ssd_cache_hit", "ssd_tokens_from_cache",
        ],
    )

    if steady and not any(
        (p["ssd"]["cache_hit"] or 0.0) > 0.01 for p in steady
    ):
        print(
            "\nWARN: steady-state trials show cache_hit ≈ 0 for the SSD "
            "level. That means ``_maybe_prime_from_ssd`` is not finding "
            "a prior save. Check the logs for ``SSD prime: no saved "
            "prefix`` and verify ``_maybe_save_to_ssd`` actually ran on "
            "trial 0 (``SSD cache: saved prefix len=...``).",
        )

    results = {
        "model": args.model,
        "draft": args.draft,
        "max_tokens": args.max_tokens,
        "trials": args.trials,
        "ssd_dir": ssd_dir,
        "disk_usage_before": usage_before,
        "disk_usage_after": usage_after,
        "pairs": pairs,
        "summary": summary_rows,
    }
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nWrote raw results → {args.out}")


if __name__ == "__main__":
    main()
