"""Model distillation: collect trajectories, format SFT data, train LoRA.

This package is the offline Path A/B pipeline.  Live adapter loading lives
on the MLX inference path (``ChatMLXText.adapter_path``); this package does
not import ``chat_models`` at module load so tests and CLI scripts stay
light.
"""

from __future__ import annotations

DEFAULT_TEACHER_MODEL_ID = "mlx-community/Qwen3-32B-4bit"
DEFAULT_STUDENT_MODEL_ID = "mlx-community/Qwen3-8B-4bit"

__all__ = [
    "DEFAULT_STUDENT_MODEL_ID",
    "DEFAULT_TEACHER_MODEL_ID",
]
