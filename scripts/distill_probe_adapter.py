#!/usr/bin/env python3
"""Phase 0: prove mlx_lm.load(..., adapter_path=) and the cache-key isolation.

    python scripts/distill_probe_adapter.py
    python scripts/distill_probe_adapter.py --adapter /path/to/lora --model mlx-community/Qwen3-8B-4bit
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def probe_api() -> dict:
    from mlx_lm import load

    sig = inspect.signature(load)
    return {
        "load_accepts_adapter_path": "adapter_path" in sig.parameters,
        "load_signature": str(sig),
    }


def probe_cache(model: str, adapter: str | None) -> dict:
    from chat_models.mlx._shared import (
        _LOADED_MODELS,
        cache_key,
        effective_adapter_path,
        evict_all_mlx_models,
    )

    evict_all_mlx_models()
    sample_adapter = adapter or "/tmp/example-lora"
    bare = cache_key(model, None, None)
    adapted = cache_key(model, None, sample_adapter)
    return {
        "bare_key": list(bare),
        "adapted_key": list(adapted),
        "keys_differ": bare != adapted,
        "effective_adapter": effective_adapter_path(model, adapter),
        "cache_size_before_load": len(_LOADED_MODELS),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe mlx_lm adapter loading")
    parser.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")
    parser.add_argument("--adapter", default=None)
    args = parser.parse_args(argv)

    report = {"api": probe_api(), "cache": probe_cache(args.model, args.adapter)}
    import json
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    if not report["api"]["load_accepts_adapter_path"]:
        print("mlx_lm.load does not accept adapter_path — distillation cannot proceed.", file=sys.stderr)
        return 1
    if args.adapter and not report["cache"]["keys_differ"]:
        print("Cache keys for bare vs adapter loads are identical — abort.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
