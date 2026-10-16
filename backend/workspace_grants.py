"""Persistent allowed-folder grants.

The store lives in the Otto app-data directory, not inside a session, so
a folder the user allows can be reused. Writes are atomic (temp file plus
replace). Callers validate the path with :mod:`backend.workspace_policy`
before inserting a user grant.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from backend.schemas import FolderGrant

logger = logging.getLogger(__name__)


class GrantStoreError(RuntimeError):
    pass


def grants_path() -> Path:
    from backend.config import get_app_data_dir
    return get_app_data_dir() / "workspace_grants.json"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _load() -> list[FolderGrant]:
    path = grants_path()
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("workspace grants unreadable: %s", exc)
        return []
    items = raw.get("grants") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return []
    grants: list[FolderGrant] = []
    for item in items:
        try:
            grants.append(FolderGrant.model_validate(item))
        except Exception:
            logger.debug("skipping malformed folder grant", exc_info=True)
    return grants


def _save(grants: list[FolderGrant]) -> None:
    path = grants_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"grants": [g.model_dump(mode="json") for g in grants]},
        indent=2,
    )
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(payload + "\n", encoding="utf-8")
    os.replace(tmp, path)


def list_grants() -> list[FolderGrant]:
    return _load()


def get_grant(grant_id: str) -> FolderGrant | None:
    if not grant_id:
        return None
    for grant in _load():
        if grant.id == grant_id:
            return grant
    return None


def find_by_realpath(realpath: str) -> FolderGrant | None:
    target = str(Path(realpath).expanduser().resolve()) if realpath else ""
    if not target:
        return None
    for grant in _load():
        try:
            if str(Path(grant.realpath).resolve()) == target:
                return grant
        except OSError:
            if grant.realpath == target:
                return grant
    return None


def upsert_grant(
    realpath: Path,
    *,
    mode: str = "write",
    source: str = "user",
    label: str | None = None,
) -> FolderGrant:
    """Insert a grant or update the mode of the grant for this exact path."""
    if mode not in ("read", "write"):
        raise GrantStoreError(f"mode must be read or write, got {mode!r}")
    resolved = realpath.expanduser().resolve()
    grants = _load()
    now = _now()
    for index, grant in enumerate(grants):
        try:
            same = Path(grant.realpath).resolve() == resolved
        except OSError:
            same = grant.realpath == str(resolved)
        if not same:
            continue
        updated = grant.model_copy(update={
            "mode": mode,
            "label": label or grant.label or resolved.name or "project",
            "last_used": now,
            "realpath": str(resolved),
        })
        grants[index] = updated
        _save(grants)
        return updated
    grant = FolderGrant(
        id=uuid.uuid4().hex,
        realpath=str(resolved),
        label=label or resolved.name or "project",
        mode=mode,
        created_at=now,
        last_used=now,
        source=source,
    )
    grants.append(grant)
    _save(grants)
    return grant


def update_grant(
    grant_id: str,
    *,
    mode: str | None = None,
    label: str | None = None,
) -> FolderGrant | None:
    grants = _load()
    for index, grant in enumerate(grants):
        if grant.id != grant_id:
            continue
        changes: dict = {"last_used": _now()}
        if mode is not None:
            if mode not in ("read", "write"):
                raise GrantStoreError(f"mode must be read or write, got {mode!r}")
            changes["mode"] = mode
        if label is not None and label.strip():
            changes["label"] = label.strip()
        updated = grant.model_copy(update=changes)
        grants[index] = updated
        _save(grants)
        return updated
    return None


def touch_grant(grant_id: str) -> None:
    grants = _load()
    for index, grant in enumerate(grants):
        if grant.id != grant_id:
            continue
        grants[index] = grant.model_copy(update={"last_used": _now()})
        _save(grants)
        return


def revoke_grant(grant_id: str) -> FolderGrant | None:
    grants = _load()
    kept: list[FolderGrant] = []
    removed: FolderGrant | None = None
    for grant in grants:
        if grant.id == grant_id and removed is None:
            removed = grant
            continue
        kept.append(grant)
    if removed is None:
        return None
    _save(kept)
    return removed
