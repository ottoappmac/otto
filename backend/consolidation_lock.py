"""File-based consolidation lock for dream mode scheduling.

The lock file's **mtime** serves double duty as ``last_consolidated_at``.
The file body holds the PID of the current holder.

Mirrors the Claude Code lock-file pattern: a single ``stat()`` call is
enough to decide whether consolidation is overdue, and the write-then-
verify-read sequence handles casual races on a single-user desktop app.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from backend.config import get_app_data_dir

logger = logging.getLogger(__name__)

_STALE_MS = 3600 * 1000  # 1 hour — assume holder is dead after this


def _lock_path() -> Path:
    return get_app_data_dir() / "memory" / ".consolidate-lock"


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


def last_consolidated_at() -> float:
    """Return epoch-ms of the last successful consolidation, or ``0``."""
    try:
        return _lock_path().stat().st_mtime * 1000
    except FileNotFoundError:
        return 0.0


# ---------------------------------------------------------------------------
# Acquire / release
# ---------------------------------------------------------------------------


def _now_ms() -> float:
    return time.time() * 1000


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def try_acquire() -> float | None:
    """Attempt to acquire the consolidation lock.

    Returns the **previous** mtime (epoch-ms) on success — this is the
    ``since`` watermark the dream agent should use when scanning transcripts.

    Returns ``None`` when the lock is already held by a live process.
    """
    p = _lock_path()
    p.parent.mkdir(parents=True, exist_ok=True)

    prev_mtime_ms = 0.0
    try:
        stat = p.stat()
        prev_mtime_ms = stat.st_mtime * 1000
        holder_pid = int(p.read_text().strip())
        age_ms = _now_ms() - prev_mtime_ms
        if age_ms < _STALE_MS and _is_pid_alive(holder_pid):
            return None
    except (FileNotFoundError, ValueError):
        pass

    p.write_text(str(os.getpid()))

    # Verify we won the race (best-effort for single-desktop use)
    try:
        if int(p.read_text().strip()) != os.getpid():
            return None
    except (FileNotFoundError, ValueError):
        return None

    return prev_mtime_ms


def release(*, update_mtime: bool = True, rollback_to_ms: float | None = None) -> None:
    """Release the consolidation lock.

    By default the lock file's mtime is left at *now*, recording the
    consolidation timestamp.

    Pass *rollback_to_ms* to restore the previous mtime on failure so
    the next attempt retries promptly.
    """
    p = _lock_path()
    if not p.exists():
        return

    p.write_text("")

    if rollback_to_ms is not None:
        epoch_s = rollback_to_ms / 1000
        os.utime(p, (epoch_s, epoch_s))
    elif update_mtime:
        p.touch()
