"""LangChain tools that expose ambient-assistant suggestions to the chat agent.

These let the agent (including Otto, the voice front-end, which feeds its
transcripts into the same orchestrator) read the suggestions the ambient
engine has surfaced and act on them — list them, accept one (and act on its
proposed prompt inline), dismiss, or snooze.

The ambient store (:mod:`backend.ambient_store`) is async (aiosqlite), but
LangChain tools in this codebase are synchronous and run inside the agent's
event loop — so ``asyncio.run`` is unavailable.  We therefore talk to the
``ambient.db`` SQLite file directly with the stdlib ``sqlite3`` driver, the
same approach used by :func:`backend.recap_tools._ambient_hints_since`.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Statuses a suggestion can be in while it is still actionable.
_ACTIONABLE = ("pending", "shown", "snoozed")


def _db_path():
    from backend.config import get_app_data_dir
    return get_app_data_dir() / "ambient.db"


def _connect():
    import sqlite3

    path = _db_path()
    if not path.exists():
        return None
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _fmt_age(created_at: float) -> str:
    mins = int((time.time() - created_at) / 60)
    if mins < 60:
        return f"{mins}m ago"
    hours = mins // 60
    if hours < 24:
        return f"{hours}h ago"
    return f"{hours // 24}d ago"


def build_ambient_tools() -> list:
    """Build ambient-suggestion tools for injection into the agent graph."""

    @tool
    def list_ambient_suggestions() -> str:
        """List the ambient assistant's current actionable suggestions.

        The ambient assistant watches the user's recent sessions, activity, and
        memory and periodically surfaces proactive suggestions — one-off tasks,
        recurring schedules, or event-driven triggers. Use this to tell the user
        what suggestions are waiting, or before acting on one. Each suggestion is
        listed with its ID (pass it to accept/dismiss/snooze), kind, confidence,
        rationale, and the ready-to-run proposed prompt.
        """
        conn = _connect()
        if conn is None:
            return "No ambient suggestions (the ambient assistant has not run yet)."
        try:
            now = time.time()
            rows = conn.execute(
                """SELECT id, title, rationale, proposed_prompt, suggested_agent,
                          kind, confidence, schedule_cron, status, created_at,
                          snoozed_until
                   FROM hints
                   WHERE status IN ('pending', 'shown')
                      OR (status = 'snoozed' AND snoozed_until IS NOT NULL
                          AND snoozed_until <= ?)
                   ORDER BY confidence DESC, created_at ASC""",
                (now,),
            ).fetchall()

            # Mark freshly-surfaced "pending" suggestions as "shown" so the
            # state stays in sync with the dashboard UI.
            pending_ids = [r["id"] for r in rows if r["status"] == "pending"]
            if pending_ids:
                conn.executemany(
                    "UPDATE hints SET status='shown', shown_at=? WHERE id=? AND status='pending'",
                    [(now, hid) for hid in pending_ids],
                )
                conn.commit()
        finally:
            conn.close()

        if not rows:
            return "No ambient suggestions are pending right now."

        lines: list[str] = [f"{len(rows)} ambient suggestion(s) pending:\n"]
        for r in rows:
            agent = f", agent: {r['suggested_agent']}" if r["suggested_agent"] else ""
            cron = f", cron: `{r['schedule_cron']}`" if r["schedule_cron"] else ""
            lines.append(
                f"- **{r['title']}**  (id: `{r['id']}`)\n"
                f"  kind: {r['kind']}, confidence: {r['confidence']:.0%}, "
                f"{_fmt_age(r['created_at'])}{agent}{cron}\n"
                f"  why: {r['rationale']}\n"
                f"  proposed prompt: {r['proposed_prompt']}"
            )
        return "\n".join(lines)

    @tool
    def accept_ambient_suggestion(suggestion_id: str) -> str:
        """Accept an ambient suggestion by its ID and retrieve its proposed prompt.

        Marks the suggestion as accepted and returns the proposed prompt text.
        After calling this, carry out that proposed prompt yourself in the current
        conversation (or, for a schedule/trigger suggestion, use the schedule /
        trigger creation tools to set up the automation it describes).

        Args:
            suggestion_id: The suggestion's ID (from list_ambient_suggestions).
        """
        conn = _connect()
        if conn is None:
            return "No ambient suggestions database found."
        try:
            row = conn.execute(
                """SELECT title, proposed_prompt, suggested_agent, kind,
                          schedule_cron, status
                   FROM hints WHERE id=?""",
                (suggestion_id,),
            ).fetchone()
            if row is None:
                return f"No ambient suggestion found with id {suggestion_id!r}."
            if row["status"] not in _ACTIONABLE:
                return (
                    f"Suggestion {suggestion_id!r} is already '{row['status']}' "
                    f"and can't be accepted."
                )
            conn.execute(
                "UPDATE hints SET status='accepted', acted_at=? WHERE id=?",
                (time.time(), suggestion_id),
            )
            conn.commit()
        finally:
            conn.close()

        agent = (
            f"\nSuggested agent: {row['suggested_agent']}"
            if row["suggested_agent"]
            else ""
        )
        cron = (
            f"\nSchedule (cron): {row['schedule_cron']}"
            if row["schedule_cron"]
            else ""
        )
        return (
            f"Accepted '{row['title']}' (kind: {row['kind']}).{agent}{cron}\n\n"
            f"Now carry out this proposed prompt:\n{row['proposed_prompt']}"
        )

    @tool
    def dismiss_ambient_suggestion(suggestion_id: str) -> str:
        """Dismiss an ambient suggestion so it is no longer surfaced.

        Args:
            suggestion_id: The suggestion's ID (from list_ambient_suggestions).
        """
        conn = _connect()
        if conn is None:
            return "No ambient suggestions database found."
        try:
            cur = conn.execute(
                """UPDATE hints SET status='dismissed', acted_at=?
                   WHERE id=? AND status IN ('pending','shown','snoozed')""",
                (time.time(), suggestion_id),
            )
            conn.commit()
            ok = cur.rowcount > 0
        finally:
            conn.close()
        if not ok:
            return f"No actionable suggestion found with id {suggestion_id!r}."
        return f"Dismissed suggestion {suggestion_id!r}."

    @tool
    def snooze_ambient_suggestion(suggestion_id: str, hours: int = 4) -> str:
        """Snooze an ambient suggestion for a number of hours.

        Args:
            suggestion_id: The suggestion's ID (from list_ambient_suggestions).
            hours: How long to snooze, between 1 and 168 (default 4).
        """
        if not 1 <= hours <= 168:
            return "hours must be between 1 and 168."
        until = time.time() + hours * 3600
        conn = _connect()
        if conn is None:
            return "No ambient suggestions database found."
        try:
            cur = conn.execute(
                """UPDATE hints SET status='snoozed', snoozed_until=?
                   WHERE id=? AND status IN ('pending','shown')""",
                (until, suggestion_id),
            )
            conn.commit()
            ok = cur.rowcount > 0
        finally:
            conn.close()
        if not ok:
            return f"No actionable suggestion found with id {suggestion_id!r}."
        when = datetime.fromtimestamp(until).strftime("%Y-%m-%d %H:%M")
        return f"Snoozed suggestion {suggestion_id!r} until {when}."

    return [
        list_ambient_suggestions,
        accept_ambient_suggestion,
        dismiss_ambient_suggestion,
        snooze_ambient_suggestion,
    ]
