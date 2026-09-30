"""Store user-supplied distillation JSONL under ``distillation/uploads/``."""

from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.distillation.paths import uploads_dir
from backend.distillation.validate import normalize_to_sft, validate_bytes

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


def _root(app_data: Path | None = None, data_dir: str = "") -> Path:
    return uploads_dir(app_data=app_data, data_dir=data_dir)


def _safe_stem(name: str) -> str:
    stem = Path(name or "").name
    if stem.lower().endswith(".jsonl"):
        stem = stem[:-6]
    elif stem.lower().endswith(".json"):
        stem = stem[:-5]
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip(".-")[:48]
    return cleaned or "dataset"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dir_for(upload_id: str, *, app_data: Path | None = None, data_dir: str = "") -> Path | None:
    if not _ID_RE.match(upload_id or ""):
        return None
    path = _root(app_data=app_data, data_dir=data_dir) / upload_id
    if not path.is_dir():
        return None
    return path


def save_upload(
    raw: bytes,
    *,
    filename: str,
    app_data: Path | None = None,
    data_dir: str = "",
) -> dict[str, Any]:
    """Validate and persist a file.  Invalid files are not stored."""
    report = validate_bytes(raw, filename=filename)
    if not report["ok"]:
        return {"saved": False, "upload": None, "report": report}
    upload_id = f"{_safe_stem(filename)}-{uuid.uuid4().hex[:8]}"
    dest = _root(app_data=app_data, data_dir=data_dir) / upload_id
    dest.mkdir(parents=True, exist_ok=True)
    data_path = dest / "data.jsonl"
    data_path.write_bytes(raw)
    rec = {
        "id": upload_id,
        "filename": Path(filename).name or "upload.jsonl",
        "bytes": len(raw),
        "uploaded_at": _now(),
        "n_valid": report["n_valid"],
        "n_invalid": report["n_invalid"],
        "n_rows": report["n_rows"],
        "format": report["format"],
        "container": report["container"],
        "n_messages": report["n_messages"],
        "n_tool_calls": report["n_tool_calls"],
    }
    (dest / "meta.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    return {"saved": True, "upload": rec, "report": report}


def list_uploads(*, app_data: Path | None = None, data_dir: str = "") -> list[dict[str, Any]]:
    root = _root(app_data=app_data, data_dir=data_dir)
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        rec = _read_meta(child)
        if rec:
            rows.append(rec)
    rows.sort(key=lambda r: str(r.get("uploaded_at") or ""), reverse=True)
    return rows


def get_upload(
    upload_id: str,
    *,
    app_data: Path | None = None,
    data_dir: str = "",
) -> dict[str, Any] | None:
    dest = _dir_for(upload_id, app_data=app_data, data_dir=data_dir)
    if dest is None:
        return None
    rec = _read_meta(dest)
    if rec is None:
        return None
    rec["path"] = str(dest / "data.jsonl")
    return rec


def delete_upload(
    upload_id: str,
    *,
    app_data: Path | None = None,
    data_dir: str = "",
) -> bool:
    dest = _dir_for(upload_id, app_data=app_data, data_dir=data_dir)
    if dest is None:
        return False
    shutil.rmtree(dest)
    return True


def revalidate(
    upload_id: str,
    *,
    app_data: Path | None = None,
    data_dir: str = "",
) -> dict[str, Any] | None:
    rec = get_upload(upload_id, app_data=app_data, data_dir=data_dir)
    if rec is None:
        return None
    path = Path(rec["path"])
    report = validate_bytes(path.read_bytes(), filename=rec["filename"])
    rec["n_valid"] = report["n_valid"]
    rec["n_invalid"] = report["n_invalid"]
    rec["n_rows"] = report["n_rows"]
    rec["format"] = report["format"]
    rec["container"] = report["container"]
    rec["n_messages"] = report["n_messages"]
    rec["n_tool_calls"] = report["n_tool_calls"]
    dest = _dir_for(upload_id, app_data=app_data, data_dir=data_dir)
    if dest is not None:
        slim = {k: rec[k] for k in rec if k != "path"}
        (dest / "meta.json").write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
    return {"upload": rec, "report": report}


def write_sft_from_upload(
    upload_id: str,
    dest: Path,
    *,
    app_data: Path | None = None,
    data_dir: str = "",
) -> dict[str, Any]:
    rec = get_upload(upload_id, app_data=app_data, data_dir=data_dir)
    if rec is None:
        raise FileNotFoundError(f"Unknown upload {upload_id!r}")
    path = Path(rec["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Upload data missing: {path}")
    report = normalize_to_sft(path, dest)
    if report.get("n_written", 0) <= 0:
        raise ValueError("Upload has no valid training rows.")
    return report


def _read_meta(dest: Path) -> dict[str, Any] | None:
    meta = dest / "meta.json"
    data = dest / "data.jsonl"
    if not dest.is_dir() or not data.is_file():
        return None
    rec: dict[str, Any] = {}
    if meta.is_file():
        try:
            loaded = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = {}
        if isinstance(loaded, dict):
            rec.update(loaded)
    rec.setdefault("id", dest.name)
    rec.setdefault("filename", dest.name)
    rec.setdefault("bytes", data.stat().st_size)
    rec.setdefault("uploaded_at", "")
    return rec
