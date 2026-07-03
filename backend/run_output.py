"""Shared helpers for inspecting schedule/trigger run output directories.

Both the schedule tools and the trigger tools expose "list a run's outcome"
and "read a file a run produced" capabilities over an identical on-disk
layout (``<owner>/<id>/runs/<run_id>/files/...``).  The file-listing and
file-reading logic is security-sensitive (it guards against path traversal
out of the run directory), so it lives here once rather than being copied
into each tool module.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

# Caps so a single tool call can never flood the context window with an
# unbounded number of runs or a huge output file.
MAX_RUNS_RETURNED = 50
MAX_OUTPUT_FILE_BYTES = 20_000
MAX_FILES_LISTED = 50


def fmt_duration(started: Optional[datetime], finished: Optional[datetime]) -> str:
    """Human-readable run duration, or "" when it can't be computed."""
    if not started or not finished:
        return ""
    try:
        secs = int((finished - started).total_seconds())
    except Exception:
        return ""
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m {secs % 60}s"
    return f"{secs // 3600}h {(secs % 3600) // 60}m"


def list_run_files(files_dir: Path, with_sizes: bool = False) -> list[str]:
    """Relative paths of run output files, capped at ``MAX_FILES_LISTED``."""
    if not files_dir.exists():
        return []
    entries: list[str] = []
    for fp in sorted(files_dir.rglob("*")):
        if not fp.is_file():
            continue
        rel = fp.relative_to(files_dir).as_posix()
        entries.append(f"{rel} ({fp.stat().st_size}b)" if with_sizes else rel)
        if len(entries) >= MAX_FILES_LISTED:
            break
    return entries


def read_run_output(
    files_dir: Path,
    run_id: str,
    file_path: str,
    owner_label: str,
) -> str:
    """List or read an output file under ``files_dir`` for a single run.

    With an empty ``file_path`` this returns a listing of available files;
    otherwise it returns the (capped, traversal-checked) contents of the
    requested file.  ``owner_label`` is a human string such as
    ``"schedule 'daily'"`` used only in user-facing messages.

    The caller is responsible for validating that the owner ID and ``run_id``
    are safe path segments before computing ``files_dir``.
    """
    available = list_run_files(files_dir)
    if not available:
        return f"Run '{run_id}' for {owner_label} has no output files."

    if not file_path:
        return (
            f"Output files for run '{run_id}':\n"
            + "\n".join(f"- {p}" for p in available)
            + "\n\nCall again with file_path set to read one."
        )

    resolved = (files_dir / file_path).resolve()
    if not resolved.is_relative_to(files_dir.resolve()):
        return f"Error: '{file_path}' resolves outside the run output directory."
    if not resolved.is_file():
        return (
            f"Error: '{file_path}' not found in run '{run_id}'. Available files:\n"
            + "\n".join(f"- {p}" for p in available)
        )

    total_bytes = resolved.stat().st_size
    with resolved.open("rb") as fh:
        raw = fh.read(MAX_OUTPUT_FILE_BYTES)
    try:
        # errors="replace" keeps a multi-byte char split at the read
        # boundary from masquerading as a binary file.
        text = (
            raw.decode("utf-8")
            if len(raw) == total_bytes
            else raw.decode("utf-8", errors="replace")
        )
    except UnicodeDecodeError:
        return (
            f"'{file_path}' ({total_bytes} bytes) appears to be binary and cannot "
            f"be shown as text."
        )

    if total_bytes > MAX_OUTPUT_FILE_BYTES:
        text += f"\n\n…[truncated, {total_bytes} bytes total]"
    return f"Contents of `{file_path}` from run '{run_id}':\n\n{text}"
