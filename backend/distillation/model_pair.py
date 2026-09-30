"""Validate a user-chosen teacher/student pair against the MLX catalog."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from backend.distillation import DEFAULT_STUDENT_MODEL_ID, DEFAULT_TEACHER_MODEL_ID
from backend.mlx_catalog import CURATED, CatalogRow, score_row

logger = logging.getLogger(__name__)

# Repo ids / names that are never valid distillation teachers.
_CLOUD_RE = re.compile(
    r"claude|anthropic|gpt-4|gpt-5|openai|gemini|command-a|chatgpt",
    re.IGNORECASE,
)


class ModelPairError(ValueError):
    """Hard refusal — do not train or fabricate with this pair."""


@dataclass
class ModelPairResult:
    teacher_id: str
    student_id: str
    teacher_row: CatalogRow | None
    student_row: CatalogRow | None
    teacher_fits: str = "unknown"
    student_fits: str = "unknown"
    same_family: bool = True
    warnings: list[str] = field(default_factory=list)

    def refuse_if_student_over(self) -> None:
        if self.student_fits == "over":
            raise ModelPairError(
                f"Student {self.student_id!r} does not fit this machine "
                f"(fits={self.student_fits}). Pick a smaller student."
            )

    def refuse_if_teacher_over(self) -> None:
        if self.teacher_fits == "over":
            raise ModelPairError(
                f"Teacher {self.teacher_id!r} does not fit this machine "
                f"(fits={self.teacher_fits}). Pick a smaller teacher."
            )


def catalog_row(repo_id: str) -> CatalogRow | None:
    rid = repo_id.strip()
    for row in CURATED:
        if row.repo_id == rid:
            return row
    return None


def family_stem(family: str) -> str:
    """``qwen3`` / ``qwen-vlm`` → ``qwen``; ``llama3`` → ``llama``."""
    fam = (family or "").strip().lower()
    for prefix in ("qwen", "llama", "mistral", "gemma", "phi", "deepseek"):
        if fam.startswith(prefix):
            return prefix
    return fam


def _looks_cloud(repo_id: str) -> bool:
    return bool(_CLOUD_RE.search(repo_id))


def _fit_for(row: CatalogRow | None, *, ram_gb: float, wired_limit_gb: float) -> str:
    if row is None or ram_gb <= 0:
        return "unknown"
    scored = score_row(
        row,
        ram_gb=ram_gb,
        wired_limit_gb=wired_limit_gb,
        free_disk_gb=ram_gb,
        ctx_len=8192,
        kv_bits=4,
    )
    return str(scored.get("fits") or "unknown")


def validate_pair(
    teacher_id: str,
    student_id: str,
    *,
    ram_gb: float = 0.0,
    wired_limit_gb: float = 0.0,
    privacy_lock: bool = False,
    allow_family_mismatch: bool = False,
    check_teacher_fit: bool = False,
    check_student_fit: bool = True,
) -> ModelPairResult:
    """Validate teacher/student ids.  Raises :class:`ModelPairError` on hard refuse."""
    teacher_id = (teacher_id or DEFAULT_TEACHER_MODEL_ID).strip()
    student_id = (student_id or DEFAULT_STUDENT_MODEL_ID).strip()
    if not teacher_id or not student_id:
        raise ModelPairError("Teacher and student model ids are required")

    if _looks_cloud(teacher_id) or _looks_cloud(student_id):
        raise ModelPairError(
            "Distillation only accepts local MLX catalog / Hub repo ids "
            f"(got teacher={teacher_id!r}, student={student_id!r})"
        )
    if privacy_lock and _looks_cloud(teacher_id):
        raise ModelPairError("Privacy lock forbids a cloud teacher")

    teacher_row = catalog_row(teacher_id)
    student_row = catalog_row(student_id)
    result = ModelPairResult(
        teacher_id=teacher_id,
        student_id=student_id,
        teacher_row=teacher_row,
        student_row=student_row,
        teacher_fits=_fit_for(teacher_row, ram_gb=ram_gb, wired_limit_gb=wired_limit_gb or ram_gb),
        student_fits=_fit_for(student_row, ram_gb=ram_gb, wired_limit_gb=wired_limit_gb or ram_gb),
    )

    t_stem = family_stem(teacher_row.family if teacher_row else "")
    s_stem = family_stem(student_row.family if student_row else "")
    if t_stem and s_stem and t_stem != s_stem:
        result.same_family = False
        msg = (
            f"Teacher family {t_stem!r} differs from student family {s_stem!r}. "
            "The formatter restamps tool calls into the student template, but "
            "same-family pairs train more reliably."
        )
        if not allow_family_mismatch:
            raise ModelPairError(msg + " Pass allow_family_mismatch to proceed.")
        result.warnings.append(msg)

    if student_row and "tools" not in (student_row.role or []):
        result.warnings.append(
            f"Student {student_id} is not tagged as a tool-calling model in the catalog."
        )

    t_params = teacher_row.params_b if teacher_row else 0.0
    s_params = student_row.params_b if student_row else 0.0
    if t_params and s_params and t_params < s_params:
        result.warnings.append(
            f"Teacher ({t_params}B) is smaller than student ({s_params}B)."
        )
    if teacher_id == student_id:
        result.warnings.append("Teacher and student are the same model.")

    if check_student_fit:
        result.refuse_if_student_over()
    if check_teacher_fit:
        result.refuse_if_teacher_over()
    return result
