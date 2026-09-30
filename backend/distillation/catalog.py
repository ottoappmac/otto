"""Unique catalog identity for trained LoRA adapters.

Hub ``CURATED`` stays Hub-only.  Each successful train gets an
``otto-distill/<slug>`` id that the MLX picker can list and that
inference resolves back to ``(student_repo_id, adapter_path)``.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.distillation.adapter_meta import read_adapter_meta
from backend.distillation.fuse import fused_ready, omlx_model_id_for
from backend.distillation.paths import adapters_dir, fused_model_path
from backend.mlx_catalog import CURATED, CatalogRow

logger = logging.getLogger(__name__)

CATALOG_PREFIX = "otto-distill"

_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def is_distill_catalog_id(repo_id: str) -> bool:
    return (repo_id or "").strip().startswith(f"{CATALOG_PREFIX}/")


def student_short(student_id: str) -> str:
    """``mlx-community/Qwen3-8B-4bit`` → ``Qwen3-8B-4bit``."""
    raw = (student_id or "").strip().rstrip("/")
    if not raw:
        return "student"
    return raw.rsplit("/", 1)[-1]


def _slug(text: str) -> str:
    s = _SLUG_RE.sub("-", (text or "").strip()).strip("-")
    return s or "adapter"


def make_identity(
    *,
    student_id: str,
    kind: str = "activity",
    dataset_sha: str = "",
    now: datetime | None = None,
    unique: bool = True,
) -> dict[str, str]:
    """Return ``slug``, ``catalog_id``, and ``display_name`` for one train.

    ``unique=True`` (the UI job) stamps time + dataset so retrains never
    collide.  ``unique=False`` keeps ``kind`` as the folder name so the
    CLI ``--name activity`` path still overwrites in place.
    """
    when = now or datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    short = student_short(student_id)
    kind_slug = _slug(kind) or "activity"
    if unique:
        stamp = when.strftime("%Y%m%d-%H%M%S")
        slug = f"{_slug(short)}-{kind_slug}-{stamp}"
        sha = (dataset_sha or "").strip()
        if sha:
            slug = f"{slug}-{sha[:8]}"
    else:
        slug = kind_slug
    catalog_id = f"{CATALOG_PREFIX}/{slug}"
    display_name = (
        f"{short} distilled ({kind_slug} · "
        f"{when.strftime('%d %b %H:%M').lstrip('0')})"
    )
    return {
        "slug": slug,
        "catalog_id": catalog_id,
        "display_name": display_name,
        "kind": kind_slug,
    }


def _adapter_ready(path: Path) -> bool:
    """True when mlx_lm has written weights (not just pre-train metadata)."""
    return (path / "adapters.safetensors").is_file() or (path / "adapter_config.json").is_file()


def _dir_size_mb(path: Path) -> float:
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    except OSError:
        return 0.0
    return round(total / (1024 * 1024), 1)


def iter_trained_adapters(
    *,
    app_data: Path | None = None,
    data_dir: str = "",
) -> list[dict[str, Any]]:
    """Scan ``adapters/`` for catalog-ready LoRAs."""
    root = adapters_dir(app_data=app_data, data_dir=data_dir)
    if not root.is_dir():
        return []
    found: list[dict[str, Any]] = []
    for path in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_dir():
            continue
        if not _adapter_ready(path):
            continue
        meta = read_adapter_meta(path) or {}
        base = str(meta.get("base_repo_id") or "").strip()
        if not base:
            continue
        identity_id = str(meta.get("catalog_id") or "").strip()
        if not identity_id:
            identity_id = f"{CATALOG_PREFIX}/{path.name}"
        display = str(meta.get("display_name") or "").strip()
        if not display:
            display = f"{student_short(base)} distilled ({path.name})"
        fused = Path(str(meta.get("fused_path") or "")).expanduser() if meta.get("fused_path") else fused_model_path(path.name)
        fused_ok = fused_ready(fused)
        found.append({
            "adapter_path": str(path),
            "base_repo_id": base,
            "teacher_model_id": str(meta.get("teacher_model_id") or ""),
            "catalog_id": identity_id,
            "display_name": display,
            "dataset_sha": str(meta.get("dataset_sha") or ""),
            "trained_at": str(meta.get("trained_at") or ""),
            "size_mb": _dir_size_mb(path),
            "kind": str(meta.get("kind") or ""),
            "purpose": str(meta.get("purpose") or ""),
            "fused_path": str(fused) if fused_ok else "",
            "omlx_model_id": omlx_model_id_for(path) if fused_ok else "",
            "meta": meta,
        })
    return found


def resolve_catalog_id(
    repo_id: str,
    *,
    app_data: Path | None = None,
    data_dir: str = "",
) -> dict[str, Any] | None:
    """Look up a trained adapter by ``otto-distill/…`` id."""
    rid = (repo_id or "").strip()
    if not is_distill_catalog_id(rid):
        return None
    for rec in iter_trained_adapters(app_data=app_data, data_dir=data_dir):
        if rec["catalog_id"] == rid:
            return rec
    return None


def catalog_id_for_omlx_model(
    model_id: str,
    *,
    app_data: Path | None = None,
    data_dir: str = "",
) -> str | None:
    """Map a Turbo/oMLX fused slug back to ``otto-distill/…``, or None."""
    raw = (model_id or "").strip()
    if not raw:
        return None
    short = raw.rsplit("/", 1)[-1]
    for rec in iter_trained_adapters(app_data=app_data, data_dir=data_dir):
        oid = str(rec.get("omlx_model_id") or "").strip()
        if oid and (oid == raw or oid == short or oid.rsplit("/", 1)[-1] == short):
            return str(rec["catalog_id"])
    return None


def set_adapter_purpose(catalog_id: str, purpose: str) -> dict[str, Any] | None:
    """Patch ``purpose`` on a trained adapter.  Returns the public row or None."""
    from backend.distillation.adapter_meta import patch_adapter_meta

    rec = resolve_catalog_id(catalog_id)
    if rec is None:
        return None
    updated = patch_adapter_meta(rec["adapter_path"], {"purpose": (purpose or "").strip()})
    if updated is None:
        return None
    rec["purpose"] = str(updated.get("purpose") or "")
    rec["meta"] = updated
    return rec


def _student_catalog_row(base_repo_id: str) -> CatalogRow | None:
    for row in CURATED:
        if row.repo_id == base_repo_id:
            return row
    return None


def catalog_rows_from_adapters(
    *,
    app_data: Path | None = None,
    data_dir: str = "",
) -> list[CatalogRow]:
    """Build MLX catalog rows for local distilled adapters (not Hub)."""
    rows: list[CatalogRow] = []
    for rec in iter_trained_adapters(app_data=app_data, data_dir=data_dir):
        student = _student_catalog_row(rec["base_repo_id"])
        teacher = rec.get("teacher_model_id") or "your sessions"
        purpose = str(rec.get("purpose") or "").strip()
        blurb = purpose or (
            f"LoRA on {student_short(rec['base_repo_id'])} from {student_short(str(teacher))}."
        )
        rows.append(CatalogRow(
            repo_id=rec["catalog_id"],
            family=student.family if student else "qwen3",
            display_name=rec["display_name"],
            blurb=blurb,
            weights_gb=student.weights_gb if student else 0.0,
            params_b=student.params_b if student else 0.0,
            quant=student.quant if student else "4bit",
            role=list(student.role) if student else ["text", "tools"],
            capability_tags=["chat", "tools", "distilled"],
            n_layers=student.n_layers if student else 0,
            n_kv_heads=student.n_kv_heads if student else 0,
            head_dim=student.head_dim if student else 0,
            max_position=student.max_position if student else 32768,
            requires_token=False,
            tier=student.tier if student else "balanced",
            featured=True,
            source="distill",
            adapter_path=rec["adapter_path"],
            base_repo_id=rec["base_repo_id"],
        ))
    return rows
