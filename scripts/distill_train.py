#!/usr/bin/env python3
"""Offline LoRA train.  Unloads nothing itself — stop chat sessions first.

    python scripts/distill_train.py --dataset path/to/sft.jsonl --model mlx-community/Qwen3-8B-4bit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from backend.config import AppConfig  # noqa: E402
from backend.distillation import DEFAULT_STUDENT_MODEL_ID, DEFAULT_TEACHER_MODEL_ID  # noqa: E402
from backend.distillation.train import train_lora  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    try:
        cfg = AppConfig.load()
        dist = cfg.distillation
    except Exception:
        dist = None

    parser = argparse.ArgumentParser(description="Train a LoRA adapter on an SFT JSONL dataset")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--model",
        default=(dist.student_model_id if dist else DEFAULT_STUDENT_MODEL_ID),
        help="Student model (LoRA base)",
    )
    parser.add_argument(
        "--teacher",
        default=(dist.teacher_model_id if dist else DEFAULT_TEACHER_MODEL_ID),
        help="Teacher id recorded in adapter_meta.json",
    )
    parser.add_argument("--out", type=Path, default=None, help="Adapter output directory")
    parser.add_argument("--name", default="activity", help="Adapter name under distillation/adapters/")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true", help="Prepare data + metadata only")
    parser.add_argument("--allow-family-mismatch", action="store_true")
    args = parser.parse_args(argv)

    result = train_lora(
        sft_jsonl=args.dataset,
        student_id=args.model,
        teacher_id=args.teacher,
        output_name=args.name,
        adapter_path=args.out,
        allow_family_mismatch=args.allow_family_mismatch,
        run=not args.dry_run,
        iters=args.iters,
    )
    json.dump({k: v for k, v in result.items() if k != "argv"} | {"argv": result["argv"]}, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
