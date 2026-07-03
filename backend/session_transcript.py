"""Append-only JSONL session transcripts with rotation.

Each session gets one file: ``<session-id>.jsonl``
Each line is a self-contained JSON object with a UTC timestamp,
message type, role, content, and optional tool metadata.

Separate from the UI message history (``.messages.json``) which truncates
tool results to 500 chars.  Transcripts store **full** content so dream
mode can grep for specific error messages, env var names, etc.

Older transcripts are rotated based on age and count limits.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from backend.config import get_app_data_dir

logger = logging.getLogger(__name__)

TranscriptEventType = Literal[
    "user",
    "assistant",
    "tool_call",
    "tool_result",
    "system",
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _transcripts_dir() -> Path:
    d = get_app_data_dir() / "transcripts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _transcript_path(session_id: str) -> Path:
    return _transcripts_dir() / f"{session_id}.jsonl"


# ---------------------------------------------------------------------------
# Append
# ---------------------------------------------------------------------------


def _build_record(
    event_type: TranscriptEventType,
    content: Any,
    *,
    role: str | None = None,
    tool_name: str | None = None,
    tool_call_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": event_type,
    }
    if role:
        record["role"] = role
    if tool_name:
        record["tool"] = tool_name
    if tool_call_id:
        record["tool_call_id"] = tool_call_id
    record["content"] = content
    if metadata:
        record["meta"] = metadata
    return record


def append_event(
    session_id: str,
    event_type: TranscriptEventType,
    content: Any,
    *,
    role: str | None = None,
    tool_name: str | None = None,
    tool_call_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append a single event to the session transcript (blocking)."""
    record = _build_record(
        event_type,
        content,
        role=role,
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        metadata=metadata,
    )
    with _transcript_path(session_id).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


async def append_event_async(
    session_id: str,
    event_type: TranscriptEventType,
    content: Any,
    **kwargs: Any,
) -> None:
    """Non-blocking wrapper — moves the sync file write to a thread."""
    await asyncio.to_thread(
        append_event, session_id, event_type, content, **kwargs
    )


# ---------------------------------------------------------------------------
# Manifest scan
# ---------------------------------------------------------------------------


def list_transcripts_since(since_ms: float) -> list[dict[str, Any]]:
    """Return transcript metadata for sessions modified after *since_ms*.

    Gives dream mode a lightweight gate check — which sessions to consider
    without reading file contents.
    """
    results: list[dict[str, Any]] = []
    for p in _transcripts_dir().glob("*.jsonl"):
        try:
            stat = p.stat()
        except OSError:
            continue
        mtime_ms = stat.st_mtime * 1000
        if mtime_ms > since_ms:
            results.append({
                "session_id": p.stem,
                "mtime_ms": mtime_ms,
                "size_bytes": stat.st_size,
            })
    return sorted(results, key=lambda r: r["mtime_ms"], reverse=True)


def delete_transcript(session_id: str) -> bool:
    """Remove a single transcript file.  Returns ``True`` if it existed."""
    p = _transcript_path(session_id)
    try:
        p.unlink()
        return True
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


class TranscriptRotator:
    """Rotate old transcript files based on age and count.

    Call :meth:`rotate` periodically — e.g. at the start of each dream
    consolidation run, or on app startup.

    *watermark_ms*, when provided, prevents deletion of any file modified
    after that timestamp.  This ensures transcripts the dream agent hasn't
    processed yet survive startup rotation.
    """

    def __init__(
        self,
        *,
        max_age_days: int = 30,
        max_files: int = 200,
        archive: bool = False,
    ) -> None:
        self._max_age_days = max_age_days
        self._max_files = max_files
        self._archive = archive

    def rotate(self, *, watermark_ms: float = 0.0) -> int:
        """Remove transcripts exceeding age or count limits.

        Files modified after *watermark_ms* are never removed — they haven't
        been consolidated yet.

        Returns the number of files removed / archived.
        """
        tdir = _transcripts_dir()
        files = sorted(tdir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        now = time.time()
        cutoff = now - (self._max_age_days * 86400)
        watermark_s = watermark_ms / 1000.0 if watermark_ms else 0.0
        removed = 0

        for p in files:
            mtime = p.stat().st_mtime
            if mtime < cutoff and (watermark_s == 0.0 or mtime < watermark_s):
                if self._archive:
                    archive_dir = tdir / "archive"
                    archive_dir.mkdir(exist_ok=True)
                    p.rename(archive_dir / p.name)
                else:
                    p.unlink()
                removed += 1

        remaining = sorted(tdir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        while len(remaining) > self._max_files:
            oldest = remaining.pop(0)
            if watermark_s and oldest.stat().st_mtime >= watermark_s:
                break
            oldest.unlink()
            removed += 1

        if removed:
            logger.info("Rotated %d transcript file(s)", removed)
        return removed

    async def rotate_async(self, *, watermark_ms: float = 0.0) -> int:
        return await asyncio.to_thread(self.rotate, watermark_ms=watermark_ms)
