"""Offline LoRA training via ``mlx_lm.lora``.  Never run inside a chat process."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from backend.distillation.adapter_meta import write_adapter_meta
from backend.distillation.model_pair import ModelPairError, validate_pair
from backend.distillation.paths import adapters_dir, runs_dir

logger = logging.getLogger(__name__)

DEFAULT_RANK = 32
DEFAULT_ITERS = 200
DEFAULT_MAX_SEQ = 2048
DEFAULT_NUM_LAYERS = 16

# Explicit targets.  mlx_lm otherwise LoRAs every Linear it finds.
# Left out on purpose:
#   mlp.gate            MoE router.  Backward through expert indices crashes MLX.
#   mlp.switch_mlp.*    routed experts.  Same gather-index backward.
#   linear_attn.*       Qwen3.5/3.6 Gated DeltaNet.  A LoRA on the decay gate
#                       (in_proj_a / in_proj_b) overflows on a long prompt and
#                       generation collapses to "!".
_SAFE_LORA_KEYS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "mlp.shared_expert.gate_proj",
    "mlp.shared_expert.up_proj",
    "mlp.shared_expert.down_proj",
)

_HYBRID_MARKERS = ("qwen3.5", "qwen3.6", "qwen3-next", "qwen3_5", "qwen3_next")
_LARGE_MOE_MARKERS = ("35b", "30b", "40b", "a3b", "a22b")

_LORA_CLI = Path(__file__).resolve().with_name("mlx_lora_cli.py")

# mlx_lm.lora prints ``Iter 12: Train loss …`` (and occasionally ``Iteration``).
_ITER_RE = re.compile(r"\bIter(?:ation)?\s+(\d+)\b", re.IGNORECASE)


def parse_iter_from_line(line: str) -> int | None:
    """Extract the current train iteration from an mlx_lm.lora log line."""
    m = _ITER_RE.search(line or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def looks_moe(repo_id: str) -> bool:
    """True when *repo_id* looks like a Mixture-of-Experts checkpoint."""
    slug = (repo_id or "").lower()
    return any(tok in slug for tok in ("moe", "a3b", "a22b", "mixtral", "switch"))


def looks_hybrid(repo_id: str) -> bool:
    """True for Qwen3.5 / Qwen3.6 style Gated DeltaNet checkpoints."""
    slug = (repo_id or "").lower()
    return any(tok in slug for tok in _HYBRID_MARKERS)


def student_profile(repo_id: str) -> dict[str, Any]:
    """Training limits that follow from the student architecture.

    Dense Qwen3-style models keep mlx_lm's own layer discovery.  Hybrid and
    MoE students get an explicit module list, and a large MoE student is
    capped below 2048 tokens so the backward pass fits in Metal.
    """
    hybrid = looks_hybrid(repo_id)
    moe = looks_moe(repo_id)
    slug = (repo_id or "").lower()
    large_moe = moe and any(tok in slug for tok in _LARGE_MOE_MARKERS)
    max_seq = 1024 if large_moe else DEFAULT_MAX_SEQ
    warnings: list[str] = []
    if hybrid:
        warnings.append(
            "Student is a Qwen3.5/3.6 hybrid (Gated DeltaNet). LoRA trains full "
            "attention and the feed-forward only. Linear-attention layers stay "
            "frozen — a LoRA there overflows the decay gate and generation "
            "collapses to '!'."
        )
    if moe:
        warnings.append(
            "Student is a Mixture-of-Experts model. LoRA skips the router and "
            "the routed experts, and routing indices are detached so MLX can backprop."
        )
    if large_moe:
        warnings.append(
            f"Sequence length is capped at {max_seq} tokens. A 2048-token "
            "backward through this student does not fit in the Metal working set."
        )
    return {
        "hybrid": hybrid,
        "moe": moe,
        "max_seq": max_seq,
        "use_safe_keys": hybrid or moe,
        "warnings": warnings,
    }


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough token count without loading a tokenizer (~3 chars / token)."""
    n = 0
    for msg in messages:
        content = msg.get("content")
        if content is None:
            text = ""
        elif isinstance(content, str):
            text = content
        else:
            text = json.dumps(content, default=str)
        n += max(1, len(text) // 3)
        for tc in msg.get("tool_calls") or []:
            n += max(8, len(json.dumps(tc, default=str)) // 3)
        n += 16
    return n


def _crop_content(msg: dict[str, Any], keep_chars: int) -> dict[str, Any]:
    out = dict(msg)
    content = out.get("content")
    if isinstance(content, str) and len(content) > keep_chars:
        out["content"] = content[-keep_chars:]
    return out


def _fit_window(window: list[dict[str, Any]], max_tokens: int) -> list[dict[str, Any]]:
    """Shrink *window* from the front so ``estimate_tokens`` stays in budget."""
    window = list(window)
    while len(window) > 2 and estimate_tokens(window) > max_tokens:
        window = window[1:]
    if estimate_tokens(window) <= max_tokens:
        return window
    for idx in (0, -1):
        if estimate_tokens(window) <= max_tokens:
            break
        content = window[idx].get("content")
        if not isinstance(content, str) or len(content) <= 32:
            continue
        lo, hi = 32, len(content)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            trial = list(window)
            trial[idx] = _crop_content(window[idx], mid)
            if estimate_tokens(trial) <= max_tokens:
                lo = mid
            else:
                hi = mid - 1
        window[idx] = _crop_content(window[idx], lo)
    return window


def window_messages(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Keep a trailing user→assistant suffix that fits *max_tokens*.

    mlx_lm truncates from the left and ``--mask-prompt`` only trains the
    last assistant turn.  A 200k-token session truncated to 2048 has no
    completion tokens left, so val/train loss becomes NaN.
    """
    messages = [m for m in messages if isinstance(m, dict)]
    if not messages or max_tokens <= 0:
        return []
    if estimate_tokens(messages) <= max_tokens:
        return list(messages)

    end = len(messages)
    while end > 0 and messages[end - 1].get("role") != "assistant":
        end -= 1
    if end == 0:
        return []

    start = end
    while start > 0:
        trial = messages[start - 1 : end]
        if estimate_tokens(trial) > max_tokens:
            break
        start -= 1
    window = list(messages[start:end])
    while len(window) > 2 and window[0].get("role") != "user":
        window = window[1:]
    if window and window[0].get("role") != "user":
        for i in range(start - 1, -1, -1):
            if messages[i].get("role") == "user":
                trial = [messages[i]] + window
                if estimate_tokens(trial) <= max_tokens:
                    window = trial
                else:
                    cropped = _fit_window(trial, max_tokens)
                    if any(m.get("role") == "user" for m in cropped) and any(
                        m.get("role") == "assistant" for m in cropped
                    ):
                        window = cropped
                break
    window = _fit_window(window, max_tokens)
    if not any(m.get("role") == "user" for m in window):
        return []
    if not any(m.get("role") == "assistant" for m in window):
        return []
    return window


def supervised_windows(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
) -> list[list[dict[str, Any]]]:
    """One window per assistant turn ``--mask-prompt`` should learn.

    The loss only covers the last assistant message.  A session that ends
    in a written summary therefore never teaches a tool call.  Emit a
    window ending on each tool-call turn, plus the final reply.
    """
    messages = [m for m in messages if isinstance(m, dict)]
    targets = [
        i
        for i, msg in enumerate(messages)
        if msg.get("role") == "assistant"
        and (msg.get("tool_calls") or i == len(messages) - 1)
    ]
    if not targets:
        window = window_messages(messages, max_tokens=max_tokens)
        return [window] if window else []
    out: list[list[dict[str, Any]]] = []
    seen: set[str] = set()
    for index in targets:
        window = window_messages(messages[: index + 1], max_tokens=max_tokens)
        if not window or window[-1].get("role") != "assistant":
            continue
        if messages[index].get("tool_calls") and not window[-1].get("tool_calls"):
            continue
        key = json.dumps(window, default=str, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(window)
    return out


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_lora_data(
    sft_jsonl: Path,
    dest_dir: Path,
    *,
    max_seq_length: int = DEFAULT_MAX_SEQ,
) -> Path:
    """Split SFT JSONL into ``train.jsonl`` / ``valid.jsonl`` for mlx_lm.lora.

    ``mlx_lm.lora --data`` expects a directory, not a file.  Each session
    becomes one window per tool-call turn plus the final reply, trimmed
    so ``--mask-prompt`` still has completion tokens.  The shortest
    remaining row is held out as valid.
    """
    if not sft_jsonl.is_file():
        raise FileNotFoundError(f"SFT dataset not found: {sft_jsonl}")
    raw_lines = [ln for ln in sft_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not raw_lines:
        raise ValueError(f"SFT dataset is empty: {sft_jsonl}")
    budget = max(256, int(max_seq_length * 0.6))
    rows: list[str] = []
    dropped = 0
    for line in raw_lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            dropped += 1
            continue
        if not isinstance(obj, dict):
            dropped += 1
            continue
        windows = supervised_windows(list(obj.get("messages") or []), max_tokens=budget)
        if not windows:
            dropped += 1
            continue
        for messages in windows:
            next_obj = dict(obj)
            next_obj["messages"] = messages
            next_obj.pop("tools", None)
            rows.append(json.dumps(next_obj, default=str))
    if not rows:
        raise ValueError(
            "Every SFT example is empty after length windowing. "
            "Use shorter sessions or a smaller max sequence length."
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    if len(rows) == 1:
        train, valid = rows, rows
    else:
        valid_line = min(rows, key=len)
        valid = [valid_line]
        train = [r for r in rows if r != valid_line]
        if not train:
            train = valid
    (dest_dir / "train.jsonl").write_text("\n".join(train) + "\n", encoding="utf-8")
    (dest_dir / "valid.jsonl").write_text("\n".join(valid) + "\n", encoding="utf-8")
    if dropped:
        logger.info("Windowed SFT: kept %d, dropped %d overlong/empty rows", len(rows), dropped)
    return dest_dir


def write_moe_lora_config(path: Path) -> Path:
    """Write a YAML config for hybrid and MoE students.

    Targets full attention and the feed-forward (plus a shared expert when
    the block has one).  Keys the model does not contain are ignored.
    """
    lines = [
        "fine_tune_type: lora",
        f"num_layers: {DEFAULT_NUM_LAYERS}",
        "lora_parameters:",
        "  rank: 16",
        "  dropout: 0.0",
        "  scale: 20.0",
        "  keys:",
    ]
    for key in _SAFE_LORA_KEYS:
        lines.append(f"    - {key}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_lora_argv(
    *,
    student_id: str,
    data_dir: Path,
    adapter_path: Path,
    iters: int = DEFAULT_ITERS,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    num_layers: int = DEFAULT_NUM_LAYERS,
    max_seq_length: int = DEFAULT_MAX_SEQ,
    grad_checkpoint: bool | None = None,
    config_path: Path | None = None,
) -> list[str]:
    """Return the LoRA trainer argument vector.  No subprocess."""
    if grad_checkpoint is None:
        grad_checkpoint = True
    if looks_moe(student_id):
        launcher = [sys.executable, str(_LORA_CLI)]
    else:
        launcher = [sys.executable, "-m", "mlx_lm", "lora"]
    argv = [
        *launcher,
        "--train",
        "--model", student_id,
        "--data", str(data_dir),
        "--adapter-path", str(adapter_path),
        "--fine-tune-type", "lora",
        "--mask-prompt",
        "--num-layers", str(num_layers),
        "--batch-size", str(batch_size),
        "--iters", str(iters),
        "--learning-rate", str(learning_rate),
        "--max-seq-length", str(max_seq_length),
    ]
    if config_path is not None:
        argv.extend(["--config", str(config_path)])
    if grad_checkpoint:
        argv.append("--grad-checkpoint")
    return argv


def train_lora(
    *,
    sft_jsonl: Path,
    student_id: str,
    teacher_id: str = "",
    output_name: str = "activity",
    adapter_path: Path | None = None,
    ram_gb: float = 0.0,
    allow_family_mismatch: bool = True,
    run: bool = True,
    runner: Any | None = None,
    iters: int = DEFAULT_ITERS,
    unique_name: bool = False,
    purpose: str = "",
) -> dict[str, Any]:
    """Prepare data, write adapter metadata, optionally invoke mlx_lm.lora.

    ``run=False`` skips the subprocess (tests).  ``runner`` is an optional
    callable ``(argv: list[str]) -> int`` replacing :func:`subprocess.call`.
    ``unique_name=True`` writes a timestamped folder and ``otto-distill/…``
    catalog id so retrains never collide.
    """
    from backend.distillation.catalog import make_identity

    pair = validate_pair(
        teacher_id or student_id,
        student_id,
        ram_gb=ram_gb,
        allow_family_mismatch=allow_family_mismatch,
        check_student_fit=ram_gb > 0,
        check_teacher_fit=False,
    )
    dataset_sha = _sha256_file(sft_jsonl)
    identity = make_identity(
        student_id=pair.student_id,
        kind=output_name,
        dataset_sha=dataset_sha,
        unique=unique_name and adapter_path is None,
    )
    out = adapter_path or (adapters_dir() / identity["slug"])
    out.mkdir(parents=True, exist_ok=True)
    data_dir = runs_dir() / identity["slug"]
    if data_dir.exists():
        shutil.rmtree(data_dir)
    profile = student_profile(pair.student_id)
    prepare_lora_data(sft_jsonl, data_dir, max_seq_length=profile["max_seq"])
    config_path = write_moe_lora_config(data_dir / "lora.yaml") if profile["use_safe_keys"] else None
    pair.warnings.extend(profile["warnings"])
    write_adapter_meta(
        out,
        base_repo_id=pair.student_id,
        teacher_model_id=pair.teacher_id,
        dataset_sha=dataset_sha,
        extra={
            "iters": iters,
            "warnings": pair.warnings,
            "catalog_id": identity["catalog_id"],
            "display_name": identity["display_name"],
            "kind": identity["kind"],
            "purpose": (purpose or "").strip(),
        },
    )
    argv = build_lora_argv(
        student_id=pair.student_id,
        data_dir=data_dir,
        adapter_path=out,
        iters=iters,
        max_seq_length=profile["max_seq"],
        config_path=config_path,
    )
    result: dict[str, Any] = {
        "adapter_path": str(out),
        "student_id": pair.student_id,
        "teacher_id": pair.teacher_id,
        "argv": argv,
        "warnings": pair.warnings,
        "dataset_sha": dataset_sha,
        "catalog_id": identity["catalog_id"],
        "display_name": identity["display_name"],
        "purpose": (purpose or "").strip(),
        "kind": identity["kind"],
        "exit_code": None,
    }
    if not run:
        return result
    call = runner or subprocess.call
    logger.info("Starting LoRA train: %s", " ".join(argv))
    result["exit_code"] = int(call(argv))
    if result["exit_code"] != 0:
        raise ModelPairError(
            f"mlx_lm.lora exited {result['exit_code']}. See the train log."
        )
    return result
