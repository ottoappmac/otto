"""On-disk layout under ``<app_data>/distillation/``."""

from __future__ import annotations

from pathlib import Path

from backend.config import get_app_data_dir


def distillation_root(
    *,
    app_data: Path | None = None,
    data_dir: str = "",
) -> Path:
    """Return the distillation root, creating the standard subdirs.

    ``data_dir`` overrides the default ``<app_data>/distillation`` when
    ``DistillationConfig.data_dir`` is set.
    """
    if data_dir.strip():
        root = Path(data_dir).expanduser()
    else:
        root = (app_data or get_app_data_dir()) / "distillation"
    for name in ("activity", "adapters", "runs", "fused", "uploads"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def activity_dir(*, app_data: Path | None = None, data_dir: str = "") -> Path:
    return distillation_root(app_data=app_data, data_dir=data_dir) / "activity"


def adapters_dir(*, app_data: Path | None = None, data_dir: str = "") -> Path:
    return distillation_root(app_data=app_data, data_dir=data_dir) / "adapters"


def runs_dir(*, app_data: Path | None = None, data_dir: str = "") -> Path:
    return distillation_root(app_data=app_data, data_dir=data_dir) / "runs"


def fused_models_dir(*, app_data: Path | None = None, data_dir: str = "") -> Path:
    """Standalone MLX weights for Turbo/oMLX (fused LoRA, not adapter dirs)."""
    return distillation_root(app_data=app_data, data_dir=data_dir) / "fused"


def fused_model_path(
    slug: str,
    *,
    app_data: Path | None = None,
    data_dir: str = "",
) -> Path:
    """``fused/<slug>/`` — oMLX registers this folder name as the model id."""
    name = (slug or "").strip().strip("/")
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    return fused_models_dir(app_data=app_data, data_dir=data_dir) / (name or "adapter")


def trajectories_path(*, app_data: Path | None = None, data_dir: str = "") -> Path:
    return activity_dir(app_data=app_data, data_dir=data_dir) / "trajectories.jsonl"


def sft_path(*, app_data: Path | None = None, data_dir: str = "") -> Path:
    return activity_dir(app_data=app_data, data_dir=data_dir) / "sft.jsonl"


def uploads_dir(*, app_data: Path | None = None, data_dir: str = "") -> Path:
    return distillation_root(app_data=app_data, data_dir=data_dir) / "uploads"
