"""Benchmark MLX turbo modes (off / basic / cache) against each other.

Run from the repo root with the project venv:

    PYTHONPATH=src .venv/bin/python scripts/bench_turbo.py

Optional knobs (CLI):

    --model       HF repo id (default: $HF_LLM_MODEL_ID from .env)
    --draft       Optional draft model id for speculative decoding
    --max-tokens  Generation cap per call (default 256)
    --warmup      Discarded iterations per level (default 1)
    --trials      Measured iterations per level (default 5)
    --sessions    Fresh sessions in the multi-session scenario (default 3)
    --levels      Comma-separated list (default off,basic,cache)
    --concurrent  Also run a concurrent-throughput scenario
    --workers     Concurrent worker threads (default 4)
    --jobs        Concurrent jobs total (default 8)
    --out         Path to write JSON results (default bench_turbo.json)

The output JSON is the source of truth; the printed tables are just a
human-readable summary.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utilities.environment import Environment       # noqa: E402
from utilities.logger import init_logger            # noqa: E402

Environment.load()
init_logger()

from langchain_core.messages import HumanMessage, SystemMessage   # noqa: E402

SYSTEM_PROMPT = (
    "You are a concise assistant. Answer in <= 200 tokens. "
    "Do not preamble. Respond directly to the user's question."
)

PROMPTS = [
    "Explain depth-first search in three sentences.",
    "List five differences between TCP and UDP.",
    "Write a haiku about the Pacific Ocean.",
    "Why is the sky blue? Be brief.",
    "Summarise the Krebs cycle in 80 words.",
]

# A throwaway prompt for warmup turns.  Must NOT appear in PROMPTS — when the
# system_prompt_cache is on (default in cache mode), reusing a prompt verbatim
# trims the KV cache to a 100% match, leaving an empty token list for
# mlx_lm.stream_generate which then raises ``Either input_embeddings or
# prompt (or both) must be provided.`` (see chat_mlx_text.py:506).
_WARMUP_PROMPT = "Reply with only the single word OK."


def build_llm(level: str, *, model: str, draft: str | None,
              max_tokens: int) -> Any:
    """Construct a chat model for *level* using the same knobs the app uses."""
    common = dict(
        max_tokens=max_tokens,
        num_draft_tokens=Environment.get_mlx_num_draft_tokens(),
        thinking=Environment.get_mlx_thinking(),
        enable_prompt_cache=Environment.get_mlx_prompt_cache(),
        enable_system_prompt_cache=Environment.get_mlx_system_prompt_cache(),
        kv_bits=Environment.get_mlx_kv_bits(),
        kv_group_size=Environment.get_mlx_kv_group_size(),
    )

    if level == "off":
        from chat_models.mlx import ChatMLXText
        return ChatMLXText(
            model_path=model,
            draft_model_path=draft or "",
            **common,
        )

    from chat_models.mlx_turbo import build_turbo_chat
    return build_turbo_chat(
        turbo_level=level,
        model_path=model,
        draft_model_path=draft or "",
        **common,
    )


def reset_state() -> None:
    """Drop turbo cache-mode singletons + free MLX scratch buffers between levels.

    Weight tensors stay in chat_models.mlx._shared._LOADED_MODELS so we don't
    pay reload cost; only per-instance / scratch state is wiped.
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


def _median(values: list[float | int | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(statistics.median(clean), 3) if clean else None


def bench_single_session(level: str, *, model: str, draft: str | None,
                         max_tokens: int, warmup: int, trials: int) -> dict:
    """Single-session micro-benchmark — measures kernel/lock overhead."""
    reset_state()
    llm = build_llm(level, model=model, draft=draft, max_tokens=max_tokens)

    for _ in range(warmup):
        llm.invoke([SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=_WARMUP_PROMPT)])

    rows: list[dict] = []
    for i in range(trials):
        prompt = PROMPTS[i % len(PROMPTS)]
        t0 = time.perf_counter()
        ai = llm.invoke([SystemMessage(content=SYSTEM_PROMPT),
                         HumanMessage(content=prompt)])
        wall = time.perf_counter() - t0
        m = dict(ai.response_metadata or {})
        rows.append({
            "i": i,
            "wall_s": round(wall, 4),
            "prompt_tps": m.get("prompt_tps"),
            "gen_tps": m.get("generation_tps"),
            "gen_tokens": m.get("generation_tokens"),
            "cache_hit": m.get("cache_hit_ratio"),
            "tokens_from_cache": m.get("tokens_from_cache"),
            "tokens_prefilled": m.get("tokens_prefilled"),
            "peak_gb": m.get("peak_memory_gb"),
        })

    return {
        "level": level,
        "scenario": "single_session",
        "rows": rows,
        "summary": {
            "median_wall_s": _median([r["wall_s"] for r in rows]),
            "median_prompt_tps": _median([r["prompt_tps"] for r in rows]),
            "median_gen_tps": _median([r["gen_tps"] for r in rows]),
            "median_cache_hit": _median([r["cache_hit"] for r in rows]),
            "max_peak_gb": max(
                (r["peak_gb"] for r in rows if r.get("peak_gb") is not None),
                default=None,
            ),
        },
    }


def bench_multi_session(level: str, *, model: str, draft: str | None,
                        max_tokens: int, sessions: int) -> dict:
    """Cross-session prefix benchmark — primary signal for turbo_level=cache.

    Builds *sessions* fresh chat-model instances, all sharing the same SYSTEM
    prompt.  In ``cache`` mode every session after the first should reuse the
    KV cache for the system prefix and therefore show a high ``cache_hit_ratio``
    and a much smaller wall-clock first turn.
    """
    reset_state()
    rows: list[dict] = []
    for s in range(sessions):
        llm = build_llm(level, model=model, draft=draft, max_tokens=max_tokens)
        # Per-session unique suffix so cache-mode's shared prefix covers the
        # SYSTEM prompt (the part we want to measure being reused) but NOT the
        # entire prompt — otherwise a 100% prefix match collapses the prompt
        # to zero tokens and mlx_lm raises ``Either input_embeddings or prompt
        # (or both) must be provided`` (see chat_mlx_text.py:506).
        user = f"Session {s}: please reply with one short sentence."
        t0 = time.perf_counter()
        ai = llm.invoke([SystemMessage(content=SYSTEM_PROMPT),
                         HumanMessage(content=user)])
        wall = time.perf_counter() - t0
        m = dict(ai.response_metadata or {})
        rows.append({
            "session": s,
            "wall_s": round(wall, 4),
            "prompt_tps": m.get("prompt_tps"),
            "gen_tps": m.get("generation_tps"),
            "cache_hit": m.get("cache_hit_ratio"),
            "tokens_from_cache": m.get("tokens_from_cache"),
            "tokens_prefilled": m.get("tokens_prefilled"),
        })
    return {
        "level": level,
        "scenario": "multi_session_first_turn",
        "rows": rows,
    }


def bench_concurrent(level: str, *, model: str, draft: str | None,
                     max_tokens: int, workers: int, jobs: int) -> dict:
    """Concurrent-throughput benchmark — stresses lock vs. executor scheduling.

    Classic ``off`` serialises through ``MLX_GEN_LOCK``; turbo serialises
    through the single executor thread (``mlx_turbo._executor``).  Total wall
    time is bounded by GPU throughput in both cases, but the per-call wall
    distribution should be cleaner under turbo because each call no longer
    fights for a Python lock.
    """
    reset_state()
    llm = build_llm(level, model=model, draft=draft, max_tokens=max_tokens)

    llm.invoke([SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=_WARMUP_PROMPT)])

    def one(prompt: str) -> tuple[float, float | None]:
        t0 = time.perf_counter()
        ai = llm.invoke([SystemMessage(content=SYSTEM_PROMPT),
                         HumanMessage(content=prompt)])
        return time.perf_counter() - t0, (ai.response_metadata or {}).get("generation_tps")

    # Make every job's prompt unique so the prefix-cache trim never reduces
    # the prompt to zero tokens (see _WARMUP_PROMPT comment).
    payload = [f"({i}) {PROMPTS[i % len(PROMPTS)]}" for i in range(jobs)]
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(one, payload))
    total = time.perf_counter() - t0

    walls = [r[0] for r in results]
    tps_vals = [r[1] for r in results if r[1] is not None]
    return {
        "level": level,
        "scenario": "concurrent",
        "workers": workers,
        "jobs": jobs,
        "total_wall_s": round(total, 3),
        "median_per_call_wall_s": round(statistics.median(walls), 3),
        "p95_per_call_wall_s": round(
            statistics.quantiles(walls, n=20)[-1] if len(walls) >= 2 else walls[0], 3,
        ),
        "median_gen_tps": round(statistics.median(tps_vals), 1) if tps_vals else None,
    }


def _print_table(title: str, rows: list[dict], cols: list[str]) -> None:
    widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0))
              for c in cols}
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
    ap.add_argument("--draft", default=os.environ.get("HF_DRAFT_LLM_MODEL_ID", "") or None)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--levels", default="off,basic,cache")
    ap.add_argument("--concurrent", action="store_true",
                    help="Also run the concurrent-throughput scenario.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default="bench_turbo.json")
    args = ap.parse_args()

    if not args.model:
        sys.exit("HF_LLM_MODEL_ID is empty — set it in .env or pass --model.")

    levels = [lvl.strip() for lvl in args.levels.split(",") if lvl.strip()]
    print(f"Model:      {args.model}"
          + (f"  draft: {args.draft}" if args.draft else ""))
    print(f"Max tokens: {args.max_tokens}  trials: {args.trials}  "
          f"sessions: {args.sessions}")
    print(f"Levels:     {levels}\n")

    results: dict[str, Any] = {
        "model": args.model,
        "draft": args.draft,
        "max_tokens": args.max_tokens,
        "warmup": args.warmup,
        "trials": args.trials,
        "sessions": args.sessions,
        "single_session": [],
        "multi_session": [],
        "concurrent": [],
    }

    single_summaries: list[dict] = []
    for lvl in levels:
        print(f"\n--- single_session  level={lvl} ---")
        r = bench_single_session(
            lvl, model=args.model, draft=args.draft,
            max_tokens=args.max_tokens, warmup=args.warmup, trials=args.trials,
        )
        results["single_session"].append(r)
        for row in r["rows"]:
            print(row)
        single_summaries.append({"level": lvl, **r["summary"]})

    multi_summaries: list[dict] = []
    for lvl in levels:
        print(f"\n--- multi_session  level={lvl} ---")
        r = bench_multi_session(
            lvl, model=args.model, draft=args.draft,
            max_tokens=args.max_tokens, sessions=args.sessions,
        )
        results["multi_session"].append(r)
        for row in r["rows"]:
            print(row)
        rows = r["rows"]
        first, last = rows[0], rows[-1]
        multi_summaries.append({
            "level": lvl,
            "first_wall_s": first["wall_s"],
            "last_wall_s": last["wall_s"],
            "first_cache_hit": first["cache_hit"],
            "last_cache_hit": last["cache_hit"],
            "first_prompt_tps": first["prompt_tps"],
            "last_prompt_tps": last["prompt_tps"],
        })

    concurrent_summaries: list[dict] = []
    if args.concurrent:
        for lvl in levels:
            print(f"\n--- concurrent  level={lvl}  workers={args.workers} jobs={args.jobs} ---")
            r = bench_concurrent(
                lvl, model=args.model, draft=args.draft,
                max_tokens=args.max_tokens,
                workers=args.workers, jobs=args.jobs,
            )
            results["concurrent"].append(r)
            print(r)
            concurrent_summaries.append(r)

    _print_table("single-session medians", single_summaries, [
        "level", "median_wall_s", "median_prompt_tps",
        "median_gen_tps", "median_cache_hit", "max_peak_gb",
    ])

    _print_table("multi-session first-turn (1st vs Nth fresh session)",
                 multi_summaries, [
                     "level", "first_wall_s", "last_wall_s",
                     "first_cache_hit", "last_cache_hit",
                     "first_prompt_tps", "last_prompt_tps",
                 ])

    if args.concurrent:
        _print_table("concurrent throughput", concurrent_summaries, [
            "level", "workers", "jobs", "total_wall_s",
            "median_per_call_wall_s", "p95_per_call_wall_s", "median_gen_tps",
        ])

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nWrote raw results → {args.out}")


if __name__ == "__main__":
    main()
