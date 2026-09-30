"""Adapter sidecar metadata — binds a LoRA to the student it was trained on."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

META_FILENAME = "adapter_meta.json"


def meta_path(adapter_path: str | Path) -> Path:
    p = Path(adapter_path)
    if p.is_file():
        return p.parent / META_FILENAME
    return p / META_FILENAME


def read_adapter_meta(adapter_path: str | Path) -> dict[str, Any] | None:
    """Return ``adapter_meta.json`` or ``None`` if it is missing/corrupt."""
    path = meta_path(adapter_path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Corrupt adapter metadata at %s", path)
        return None
    if not isinstance(data, dict):
        return None
    return data


def write_adapter_meta(
    adapter_path: str | Path,
    *,
    base_repo_id: str,
    teacher_model_id: str = "",
    dataset_sha: str = "",
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write metadata next to a trained adapter.  Returns the meta file path."""
    dest = meta_path(adapter_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "base_repo_id": base_repo_id,
        "teacher_model_id": teacher_model_id,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_sha": dataset_sha,
    }
    if extra:
        payload.update(extra)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return dest


def patch_adapter_meta(
    adapter_path: str | Path,
    updates: dict[str, Any],
) -> dict[str, Any] | None:
    """Merge *updates* into an existing sidecar.  Returns the new dict or None."""
    meta = read_adapter_meta(adapter_path)
    if not meta:
        return None
    meta.update(updates)
    dest = meta_path(adapter_path)
    dest.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def adapter_matches_base(adapter_path: str | Path, model_id: str) -> bool:
    """True when the adapter may be loaded onto *model_id*.

    Missing metadata is treated as compatible so third-party LoRAs still
    load; a *mismatched* ``base_repo_id`` is a hard refuse.
    """
    meta = read_adapter_meta(adapter_path)
    if not meta:
        return True
    base = str(meta.get("base_repo_id") or "").strip()
    if not base:
        return True
    return base == model_id.strip()
