"""Fuse a trained LoRA into a standalone MLX model for oMLX/Turbo.

oMLX skips directories that contain ``adapter_config.json`` and loads
via ``mlx_lm.load(path)`` with no ``adapter_path``.  Fusing writes a
normal model folder (``config.json`` + weights) that the scanner accepts.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from backend.distillation.adapter_meta import patch_adapter_meta, read_adapter_meta
from backend.distillation.paths import fused_model_path

logger = logging.getLogger(__name__)


def fused_ready(path: str | Path) -> bool:
    """True when *path* looks like an MLX model oMLX will register."""
    p = Path(path)
    if not p.is_dir():
        return False
    if (p / "adapter_config.json").is_file() and not (p / "config.json").is_file():
        return False
    has_weights = any(p.glob("*.safetensors")) or any(p.glob("*.npz"))
    return (p / "config.json").is_file() and has_weights


def omlx_model_id_for(adapter_path: str | Path) -> str:
    """oMLX short id = fused folder name = adapter directory name."""
    return Path(adapter_path).expanduser().name


def build_fuse_argv(
    *,
    student_id: str,
    adapter_path: Path,
    save_path: Path,
) -> list[str]:
    """Return ``python -m mlx_lm.fuse`` argv.  No subprocess."""
    return [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model", student_id,
        "--adapter-path", str(adapter_path),
        "--save-path", str(save_path),
    ]


def fuse_lora(
    *,
    adapter_path: Path,
    student_id: str = "",
    save_path: Path | None = None,
    run: bool = True,
    runner: Any | None = None,
) -> dict[str, Any]:
    """Merge LoRA into student weights for oMLX.

    ``run=False`` skips the subprocess (tests).  Already-fused destinations
    are returned as-is.
    """
    import subprocess

    adapter = Path(adapter_path).expanduser()
    if not adapter.is_dir():
        raise FileNotFoundError(f"Adapter not found: {adapter}")
    meta = read_adapter_meta(adapter) or {}
    base = (student_id or str(meta.get("base_repo_id") or "")).strip()
    if not base:
        raise ValueError(f"No student/base repo on adapter {adapter}")
    dest = Path(save_path) if save_path else fused_model_path(adapter.name)
    argv = build_fuse_argv(student_id=base, adapter_path=adapter, save_path=dest)
    result: dict[str, Any] = {
        "adapter_path": str(adapter),
        "student_id": base,
        "fused_path": str(dest),
        "omlx_model_id": dest.name,
        "argv": argv,
        "skipped": False,
        "exit_code": None,
    }
    if fused_ready(dest):
        result["skipped"] = True
        result["exit_code"] = 0
        patch_adapter_meta(adapter, {"fused_path": str(dest)})
        return result
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not run:
        return result
    call = runner or subprocess.call
    logger.info("Fusing LoRA into %s", dest)
    result["exit_code"] = int(call(argv))
    if result["exit_code"] != 0:
        raise RuntimeError(
            f"mlx_lm.fuse exited {result['exit_code']}. See the fuse log."
        )
    if not fused_ready(dest):
        raise RuntimeError(
            f"mlx_lm.fuse finished but {dest} is not a loadable MLX model."
        )
    patch_adapter_meta(adapter, {"fused_path": str(dest)})
    return result
