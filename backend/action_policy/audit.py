"""Append-only record of policy decisions.

One JSON object per line under the Otto app-data directory. A failed
write is logged and swallowed: the decision still stands. Args are
truncated and secret-shaped keys are dropped before anything hits disk.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_SECRET_KEY = re.compile(r"(secret|token|password|api_key|credential|authorization)", re.I)
_MAX_STRING = 400
_MAX_READ = 200


def _redact(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        return "…"
    if isinstance(value, str):
        if len(value) > _MAX_STRING:
            return value[:_MAX_STRING] + "…"
        return value
    if isinstance(value, dict):
        out = {}
        for key, item in list(value.items())[:20]:
            if _SECRET_KEY.search(str(key)):
                out[str(key)] = "[redacted]"
            else:
                out[str(key)] = _redact(item, depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [_redact(item, depth + 1) for item in list(value)[:10]]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:_MAX_STRING]


def audit_path() -> "os.PathLike[str]":
    from backend.config import get_app_data_dir
    return get_app_data_dir() / "action_audit.jsonl"


def record(
    *,
    session_id: str | None,
    agent: str | None,
    tool: str,
    server_id: str | None,
    category: str | None,
    effect: str,
    reason: str,
    attended: bool,
    args: dict | None,
    labels: tuple[str, ...] | list[str] = (),
) -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id or "",
        "agent": agent or "",
        "tool": tool,
        "server_id": server_id or "",
        "category": category or "",
        "effect": effect,
        "reason": reason,
        "attended": attended,
        "labels": list(labels),
        "args": _redact(args or {}),
    }
    try:
        path = audit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
        with _LOCK:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line)
    except Exception:
        logger.warning("action audit write failed", exc_info=True)
    try:
        from backend.session_transcript import append_event
        if session_id:
            append_event(
                session_id,
                "action_policy",
                f"{effect}: {tool} ({category or 'ungated'}) — {reason}",
                metadata={k: entry[k] for k in ("tool", "category", "effect", "reason", "labels")},
            )
    except Exception:
        logger.debug("action audit transcript write failed", exc_info=True)


def recent(limit: int = 50) -> list[dict]:
    """Newest last. Missing file is an empty log, not an error."""
    path = audit_path()
    if not os.path.isfile(path):
        return []
    limit = max(1, min(int(limit), _MAX_READ))
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()[-limit:]
    except OSError:
        logger.warning("action audit read failed", exc_info=True)
        return []
    out: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            out.append(item)
    return out
