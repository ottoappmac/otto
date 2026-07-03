"""LangChain tool: ``get_activity_recap``.

Returns a concise summary of everything that happened since the last server
boot or the last ambient-suggestion sweep — whichever is more recent.

Data sources covered
--------------------
* **Sessions** — agent sessions that started since the reference time.
* **Schedule runs** — cron jobs that fired since the reference time.
* **Trigger runs** — event-driven triggers that fired since the reference time.
* **Activity** — per-app time totals from the local on-device activity DB.
* **Ambient hints** — suggestions that were generated, accepted, or dismissed.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _since_label(since_ts: float) -> str:
    mins = int((time.time() - since_ts) / 60)
    if mins < 60:
        return f"{mins}m ago"
    hours = mins // 60
    if hours < 24:
        return f"{hours}h ago"
    return f"{hours // 24}d ago"


# ---------------------------------------------------------------------------
# Reference-time helpers
# ---------------------------------------------------------------------------


def _reference_time() -> tuple[float, str]:
    """Return (unix_timestamp, label) for the recap window start.

    Picks the *most recent* of:
    - Server boot time
    - Last ambient sweep time

    Falls back to "last 24 hours" when neither is available.
    """
    candidates: list[tuple[float, str]] = []

    try:
        from backend.server import get_boot_time
        bt = get_boot_time()
        if bt > 0:
            candidates.append((bt, "last boot"))
    except Exception:
        pass

    try:
        from backend.ambient_agent import get_last_sweep_time
        st = get_last_sweep_time()
        if st:
            candidates.append((st, "last ambient sweep"))
    except Exception:
        pass

    if candidates:
        ts, label = max(candidates, key=lambda x: x[0])
        return ts, label

    fallback = time.time() - 86400
    return fallback, "last 24 hours"


# ---------------------------------------------------------------------------
# Per-source gatherers (all sync — called from the tool which is sync)
# ---------------------------------------------------------------------------


def _sessions_since(since_ts: float) -> list[str]:
    lines: list[str] = []
    try:
        from backend.state import session_mgr  # type: ignore[import]

        # list_history is synchronous
        sessions = session_mgr.list_history()
        cutoff = datetime.fromtimestamp(since_ts, tz=timezone.utc)
        recent = [s for s in sessions if s.created_at >= cutoff]
        recent.sort(key=lambda s: s.created_at)
        for s in recent:
            duration_str = ""
            try:
                secs = int((s.updated_at - s.created_at).total_seconds())
                if secs >= 60:
                    duration_str = f", {secs // 60}m"
            except Exception:
                pass
            status = getattr(s, "status", "") or ""
            status_str = f" [{status}]" if status else ""
            lines.append(
                f"  • {s.created_at.strftime('%H:%M')} — {s.title or 'Untitled'}"
                f" (agent: {s.agent_name or 'default'}{duration_str}){status_str}"
            )
    except Exception:
        logger.debug("[recap] sessions gather failed", exc_info=True)
    return lines


def _schedule_runs_since(since_ts: float) -> list[str]:
    lines: list[str] = []
    try:
        from backend.scheduler import load_all_schedules, load_runs
        cutoff = datetime.fromtimestamp(since_ts, tz=timezone.utc)
        for sched in load_all_schedules():
            for run in load_runs(sched.id, limit=20):
                run_ts = run.started_at
                if run_ts.tzinfo is None:
                    run_ts = run_ts.replace(tzinfo=timezone.utc)
                if run_ts < cutoff:
                    continue
                elapsed = ""
                if run.finished_at:
                    fin = run.finished_at
                    if fin.tzinfo is None:
                        fin = fin.replace(tzinfo=timezone.utc)
                    secs = int((fin - run_ts).total_seconds())
                    if secs >= 60:
                        elapsed = f", {secs // 60}m"
                err = f" ⚠ {run.error[:60]}" if run.error else ""
                lines.append(
                    f"  • {run_ts.strftime('%H:%M')} [{run.status}] schedule={sched.id}"
                    f" (agent: {sched.agent_name or 'default'}{elapsed}){err}"
                )
    except Exception:
        logger.debug("[recap] schedule runs gather failed", exc_info=True)
    return lines


def _trigger_runs_since(since_ts: float) -> list[str]:
    lines: list[str] = []
    try:
        from backend.trigger_manager import load_all_triggers, load_runs
        cutoff = datetime.fromtimestamp(since_ts, tz=timezone.utc)
        for trig in load_all_triggers():
            for run in load_runs(trig.id, limit=20):
                run_ts = run.started_at
                if run_ts.tzinfo is None:
                    run_ts = run_ts.replace(tzinfo=timezone.utc)
                if run_ts < cutoff:
                    continue
                elapsed = ""
                if run.finished_at:
                    fin = run.finished_at
                    if fin.tzinfo is None:
                        fin = fin.replace(tzinfo=timezone.utc)
                    secs = int((fin - run_ts).total_seconds())
                    if secs >= 60:
                        elapsed = f", {secs // 60}m"
                err = f" ⚠ {run.error[:60]}" if run.error else ""
                lines.append(
                    f"  • {run_ts.strftime('%H:%M')} [{run.status}] trigger={trig.id}"
                    f" (type: {trig.type}{elapsed}){err}"
                )
    except Exception:
        logger.debug("[recap] trigger runs gather failed", exc_info=True)
    return lines


def _activity_since(since_ts: float) -> list[str]:
    lines: list[str] = []
    try:
        from backend.activity_tracker import daily_summary
        summary = daily_summary(int(since_ts), int(time.time()))
        if not summary["apps"]:
            return []
        total_min = summary["total_seconds"] // 60
        lines.append(f"  Total tracked: {total_min}m")
        for a in summary["apps"][:15]:
            mins = a["seconds"] // 60
            if mins < 1:
                continue
            lines.append(f"  • {a['app']:30s}  {mins:>4d}m")
    except Exception:
        logger.debug("[recap] activity gather failed", exc_info=True)
    return lines


def _ambient_hints_since(since_ts: float) -> list[str]:
    """Return hint lines for hints created or acted on since since_ts."""
    lines: list[str] = []
    try:
        import sqlite3
        from backend.config import get_app_data_dir

        db_path = get_app_data_dir() / "ambient.db"
        if not db_path.exists():
            return []

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """SELECT title, status, confidence, kind, created_at, acted_at
                   FROM hints
                   WHERE created_at >= ? OR acted_at >= ?
                   ORDER BY COALESCE(acted_at, created_at) DESC
                   LIMIT 30""",
                (since_ts, since_ts),
            ).fetchall()
        finally:
            conn.close()

        for r in rows:
            ts = r["acted_at"] or r["created_at"]
            when = datetime.fromtimestamp(ts).strftime("%H:%M")
            status_icon = {
                "accepted": "✓",
                "dismissed": "✗",
                "snoozed": "⏸",
                "pending": "●",
                "shown": "◉",
                "expired": "–",
            }.get(r["status"], "?")
            lines.append(
                f"  • {when} {status_icon} [{r['kind']}] {r['title'][:70]}"
                f" (conf: {r['confidence']:.0%})"
            )
    except Exception:
        logger.debug("[recap] ambient hints gather failed", exc_info=True)
    return lines


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


def build_recap_tools() -> list:
    """Return the recap tool for injection into the agent graph."""

    @tool
    def get_activity_recap(since: str = "") -> str:
        """Summarise everything that has happened since the last server boot
        or the last ambient-suggestion sweep (whichever is more recent).

        Covers: agent sessions run, scheduled jobs that fired, event triggers
        that fired, macOS app-usage time, and ambient suggestions generated /
        accepted / dismissed.

        Args:
            since: Optional override for the start of the recap window.
                   Accepts "boot" (server start), "sweep" (last ambient sweep),
                   "today", "yesterday", "Nh" (e.g. "4h"), "Nd" (e.g. "2d"),
                   an ISO date, or an epoch integer.  Leave empty to use the
                   default (most recent of boot or last sweep).
        """
        # Resolve the reference timestamp.
        if since:
            since_ts, label = _resolve_since_override(since)
        else:
            since_ts, label = _reference_time()

        window_label = (
            f"since {label} ({_fmt_ts(since_ts)}, {_since_label(since_ts)})"
        )

        sections: list[str] = [f"## Activity recap — {window_label}\n"]

        # --- Sessions ---
        session_lines = _sessions_since(since_ts)
        if session_lines:
            sections.append(f"### Sessions ({len(session_lines)})")
            sections.extend(session_lines)
        else:
            sections.append("### Sessions\n  (none)")

        # --- Schedule runs ---
        sched_lines = _schedule_runs_since(since_ts)
        if sched_lines:
            sections.append(f"\n### Schedule runs ({len(sched_lines)})")
            sections.extend(sched_lines)
        else:
            sections.append("\n### Schedule runs\n  (none)")

        # --- Trigger runs ---
        trig_lines = _trigger_runs_since(since_ts)
        if trig_lines:
            sections.append(f"\n### Trigger runs ({len(trig_lines)})")
            sections.extend(trig_lines)
        else:
            sections.append("\n### Trigger runs\n  (none)")

        # --- Activity ---
        act_lines = _activity_since(since_ts)
        if act_lines:
            sections.append("\n### macOS app activity")
            sections.extend(act_lines)
        else:
            sections.append("\n### macOS app activity\n  (none recorded)")

        # --- Ambient hints ---
        hint_lines = _ambient_hints_since(since_ts)
        if hint_lines:
            sections.append(f"\n### Ambient suggestions ({len(hint_lines)})")
            sections.append(
                "  Legend: ✓ accepted  ✗ dismissed  ⏸ snoozed  ● pending  ◉ shown"
            )
            sections.extend(hint_lines)
        else:
            sections.append("\n### Ambient suggestions\n  (none)")

        return "\n".join(sections)

    return [get_activity_recap]


def _resolve_since_override(since: str) -> tuple[float, str]:
    """Parse the optional *since* override argument."""
    s = since.strip().lower()
    now = time.time()

    if s in ("boot", "startup"):
        try:
            from backend.server import get_boot_time
            bt = get_boot_time()
            if bt > 0:
                return bt, "boot"
        except Exception:
            pass
        return now - 3600, "boot (unknown, using 1h)"

    if s in ("sweep", "last_sweep", "ambient"):
        try:
            from backend.ambient_agent import get_last_sweep_time
            st = get_last_sweep_time()
            if st:
                return st, "last ambient sweep"
        except Exception:
            pass
        return now - 3600, "sweep (unknown, using 1h)"

    if s == "today":
        d = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return d.timestamp(), "today"

    if s == "yesterday":
        from datetime import timedelta
        d = (datetime.now() - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return d.timestamp(), "yesterday"

    if s.endswith("h") and s[:-1].isdigit():
        h = int(s[:-1])
        return now - h * 3600, f"last {h}h"

    if s.endswith("d") and s[:-1].isdigit():
        d_val = int(s[:-1])
        return now - d_val * 86400, f"last {d_val}d"

    try:
        return float(s), since
    except ValueError:
        pass

    try:
        dt = datetime.fromisoformat(since)
        return dt.timestamp(), since
    except ValueError:
        pass

    # Fallback — default window
    ref, label = _reference_time()
    return ref, label
