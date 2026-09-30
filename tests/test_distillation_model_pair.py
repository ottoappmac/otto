"""Teacher/student pair validation."""

from __future__ import annotations

import pytest

from backend.distillation.model_pair import ModelPairError, family_stem, validate_pair


def test_defaults_are_same_family_qwen():
    result = validate_pair(
        "mlx-community/Qwen3-32B-4bit",
        "mlx-community/Qwen3-8B-4bit",
        ram_gb=128,
        check_student_fit=True,
        check_teacher_fit=False,
    )
    assert result.same_family is True
    assert family_stem(result.teacher_row.family) == "qwen"
    assert family_stem(result.student_row.family) == "qwen"


def test_cloud_teacher_is_refused():
    with pytest.raises(ModelPairError, match="local MLX"):
        validate_pair("claude-sonnet-4-6", "mlx-community/Qwen3-8B-4bit")


def test_family_mismatch_warns_when_allowed():
    result = validate_pair(
        "mlx-community/Llama-3.3-70B-Instruct-4bit",
        "mlx-community/Qwen3-8B-4bit",
        ram_gb=128,
        allow_family_mismatch=True,
        check_student_fit=False,
        check_teacher_fit=False,
    )
    assert result.same_family is False
    assert result.warnings


def test_family_mismatch_refused_by_default():
    with pytest.raises(ModelPairError, match="family"):
        validate_pair(
            "mlx-community/Llama-3.3-70B-Instruct-4bit",
            "mlx-community/Qwen3-8B-4bit",
            ram_gb=128,
            allow_family_mismatch=False,
            check_student_fit=False,
            check_teacher_fit=False,
        )


def test_student_over_refused_on_small_machine():
    with pytest.raises(ModelPairError, match="does not fit"):
        validate_pair(
            "mlx-community/Qwen3-8B-4bit",
            "mlx-community/Qwen3-32B-4bit",
            ram_gb=8,
            wired_limit_gb=6,
            allow_family_mismatch=True,
            check_student_fit=True,
            check_teacher_fit=False,
        )


def test_teacher_over_refused_when_requested():
    with pytest.raises(ModelPairError, match="Teacher"):
        validate_pair(
            "mlx-community/Qwen3-32B-4bit",
            "mlx-community/Qwen3-8B-4bit",
            ram_gb=8,
            wired_limit_gb=6,
            check_student_fit=False,
            check_teacher_fit=True,
        )
