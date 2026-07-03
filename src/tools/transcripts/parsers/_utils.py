"""Shared utilities for transcript parsers."""

from __future__ import annotations

import json
from pathlib import Path


def count_lines(path: Path) -> int:
    """Fast line count without loading the whole file into memory."""
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def tail_read(path: Path, max_bytes: int = 4096) -> str:
    """Read the last *max_bytes* of a file as UTF-8 text."""
    size = path.stat().st_size
    tail_bytes = min(size, max_bytes)
    with open(path, "rb") as f:
        f.seek(max(0, size - tail_bytes))
        return f.read().decode("utf-8", errors="replace")


def _parse_tail_metadata(path: Path) -> tuple[str, bool]:
    """Read the file tail and extract the last timestamp and active flag."""
    last_ts = ""
    is_active = True
    tail = tail_read(path)

    for raw in reversed(tail.strip().splitlines()):
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not last_ts:
            ts = rec.get("timestamp", "")
            if ts:
                last_ts = ts
        if rec.get("type") == "last-prompt":
            is_active = False
        if last_ts:
            break

    return last_ts, is_active


def scan_tail_for_activity(
    path: Path,
    known_line_count: int = 0,
) -> dict:
    """Lightweight activity check on an append-only JSONL file.

    Returns a dict with ``has_new_activity``, ``new_lines``,
    ``total_lines``, ``size_bytes``, ``last_timestamp``, and
    ``is_active``.  Works for both Claude Code and Cowork JSONL.
    """
    stat = path.stat()
    current_lines = count_lines(path)
    new_lines = max(0, current_lines - known_line_count)

    last_ts, is_active = _parse_tail_metadata(path)

    return {
        "has_new_activity": new_lines > 0,
        "new_lines": new_lines,
        "total_lines": current_lines,
        "size_bytes": stat.st_size,
        "last_timestamp": last_ts,
        "is_active": is_active,
    }


def scan_activity_by_size(
    path: Path,
    known_size: int = 0,
) -> dict:
    """Activity check using byte-size comparison — O(1) via a single stat call.

    More efficient than :func:`scan_tail_for_activity` for repeated polling
    because the size comparison avoids iterating every line in the file.
    Only reads the tail when the caller needs timestamp/active metadata.
    """
    stat = path.stat()
    current_size = stat.st_size
    has_new = current_size > known_size

    last_ts, is_active = _parse_tail_metadata(path)

    return {
        "has_new_activity": has_new,
        "new_bytes": max(0, current_size - known_size),
        "size_bytes": current_size,
        "last_timestamp": last_ts,
        "is_active": is_active,
    }
