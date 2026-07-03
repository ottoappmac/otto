"""SQLite-backed persistence layer for ambient assistant hints.

Each hint moves through a simple lifecycle::

    pending → shown → accepted | dismissed | snoozed | expired

Deduplication is keyed on a topic hash so the same suggestion cannot
resurface within ``AmbientConfig.cooldown_hours`` hours.

The database lives at ``<app_data>/ambient.db`` and is created on first
use via :func:`get_store` / :func:`init_db`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from backend.config import get_app_data_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS hints (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    rationale   TEXT NOT NULL,
    proposed_prompt TEXT NOT NULL,
    suggested_agent TEXT,
    kind        TEXT NOT NULL DEFAULT 'task',
    confidence  REAL NOT NULL DEFAULT 0.8,
    sources     TEXT NOT NULL DEFAULT '[]',
    topic_hash  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    session_id  TEXT,
    created_at  REAL NOT NULL,
    shown_at    REAL,
    acted_at    REAL,
    snoozed_until REAL,
    schedule_cron TEXT,
    origin      TEXT NOT NULL DEFAULT 'ambient',
    target_kind TEXT,
    target_id   TEXT
);

CREATE INDEX IF NOT EXISTS idx_hints_status   ON hints (status);
CREATE INDEX IF NOT EXISTS idx_hints_topic    ON hints (topic_hash);
CREATE INDEX IF NOT EXISTS idx_hints_created  ON hints (created_at);
"""

# Each statement is applied independently; failures (column already exists) are
# ignored so existing databases pick up new columns on next startup.
_MIGRATIONS = (
    "ALTER TABLE hints ADD COLUMN schedule_cron TEXT",
    "ALTER TABLE hints ADD COLUMN origin TEXT NOT NULL DEFAULT 'ambient'",
    "ALTER TABLE hints ADD COLUMN target_kind TEXT",
    "ALTER TABLE hints ADD COLUMN target_id TEXT",
)

# ---------------------------------------------------------------------------
# Hint model
# ---------------------------------------------------------------------------

HintStatus = str  # pending | shown | accepted | dismissed | snoozed | expired

_PENDING_STATUSES = frozenset(["pending", "shown", "snoozed"])


class AmbientHint:
    """Immutable value object for a single ambient hint."""

    __slots__ = (
        "id", "title", "rationale", "proposed_prompt",
        "suggested_agent", "kind", "confidence", "sources",
        "topic_hash", "status", "session_id",
        "created_at", "shown_at", "acted_at", "snoozed_until",
        "schedule_cron", "origin", "target_kind", "target_id",
    )

    def __init__(self, row: aiosqlite.Row) -> None:
        (
            self.id,
            self.title,
            self.rationale,
            self.proposed_prompt,
            self.suggested_agent,
            self.kind,
            self.confidence,
            sources_json,
            self.topic_hash,
            self.status,
            self.session_id,
            self.created_at,
            self.shown_at,
            self.acted_at,
            self.snoozed_until,
            self.schedule_cron,
            self.origin,
            self.target_kind,
            self.target_id,
        ) = row
        self.sources: list[str] = json.loads(sources_json or "[]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "rationale": self.rationale,
            "proposed_prompt": self.proposed_prompt,
            "suggested_agent": self.suggested_agent,
            "kind": self.kind,
            "confidence": self.confidence,
            "sources": self.sources,
            "topic_hash": self.topic_hash,
            "status": self.status,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "shown_at": self.shown_at,
            "acted_at": self.acted_at,
            "snoozed_until": self.snoozed_until,
            "schedule_cron": self.schedule_cron,
            "origin": self.origin,
            "target_kind": self.target_kind,
            "target_id": self.target_id,
        }


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def _db_path() -> Path:
    return get_app_data_dir() / "ambient.db"


def _topic_hash(title: str) -> str:
    """Stable dedup key derived from a normalised hint title."""
    normalised = re.sub(r"[^\w\s]", "", title.lower()).strip()
    normalised = re.sub(r"\s+", " ", normalised)
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]


class AmbientStore:
    """Async interface to the ambient hints database."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def init(self) -> None:
        """Create the schema if it doesn't exist yet, and run any pending migrations."""
        async with aiosqlite.connect(_db_path()) as db:
            await db.executescript(_CREATE_SQL)
            # Migrations: add columns to pre-existing databases. Each is applied
            # independently; "duplicate column" errors are expected and ignored.
            for stmt in _MIGRATIONS:
                try:
                    await db.execute(stmt)
                    await db.commit()
                except Exception:
                    pass

    async def add_hints(
        self,
        hints: list[dict[str, Any]],
        cooldown_hours: int = 4,
        max_per_day: int = 10,
    ) -> list[str]:
        """Persist a batch of new hints, skipping duplicates and rate-limit overflows.

        Returns the ids of hints that were actually inserted.
        """
        now = time.time()
        cooldown_secs = cooldown_hours * 3600
        day_start = now - 86400

        async with self._lock:
            async with aiosqlite.connect(_db_path()) as db:
                db.row_factory = aiosqlite.Row

                # Count active (non-dismissed, non-expired) hints created today
                # for the rate cap. Dismissed hints don't count against the
                # daily limit so users can freely dismiss and still receive
                # new suggestions.
                cur = await db.execute(
                    "SELECT COUNT(*) FROM hints WHERE created_at >= ? AND status NOT IN ('dismissed', 'expired')",
                    (day_start,),
                )
                row = await cur.fetchone()
                day_count = row[0] if row else 0

                inserted: list[str] = []
                for h in hints:
                    if day_count >= max_per_day:
                        logger.debug("[ambient] daily hint cap reached (%d)", max_per_day)
                        break

                    topic_hash = _topic_hash(h.get("title", ""))

                    # Skip if an equivalent hint was shown or accepted recently.
                    cur2 = await db.execute(
                        """SELECT id FROM hints
                           WHERE topic_hash = ?
                             AND created_at >= ?
                             AND status NOT IN ('dismissed', 'expired')""",
                        (topic_hash, now - cooldown_secs),
                    )
                    recent = await cur2.fetchone()
                    if recent:
                        logger.debug(
                            "[ambient] skipping duplicate hint (cooldown): %s",
                            h.get("title"),
                        )
                        continue

                    hint_id = _new_id()
                    await db.execute(
                        """INSERT INTO hints
                           (id, title, rationale, proposed_prompt, suggested_agent,
                            kind, confidence, sources, topic_hash, status, created_at,
                            schedule_cron)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            hint_id,
                            h.get("title", ""),
                            h.get("rationale", ""),
                            h.get("proposed_prompt", ""),
                            h.get("suggested_agent"),
                            h.get("kind", "task"),
                            float(h.get("confidence", 0.8)),
                            json.dumps(h.get("sources", [])),
                            topic_hash,
                            "pending",
                            now,
                            h.get("schedule_cron"),
                        ),
                    )
                    inserted.append(hint_id)
                    day_count += 1

                await db.commit()
                return inserted

    async def add_eval_suggestion(self, hint: dict[str, Any]) -> Optional[str]:
        """Insert a single evaluation-triggered suggestion.

        Unlike :meth:`add_hints`, this bypasses the daily cap (a genuinely
        failing run should always surface).  Before inserting it retires any
        previous pending eval suggestions for the same target so that only the
        latest suggestion is ever visible:

        * schedule / trigger — keyed on ``(target_kind, target_id)``
        * manual             — keyed on ``session_id`` (one suggestion per run)

        Returns the new hint id, or ``None`` when an identical suggestion for
        the same session already exists (idempotent re-evaluation guard).
        """
        now = time.time()
        topic_hash = _topic_hash(hint.get("title", ""))
        session_id = hint.get("session_id")
        target_kind = hint.get("target_kind")
        target_id = hint.get("target_id")

        async with self._lock:
            async with aiosqlite.connect(_db_path()) as db:
                db.row_factory = aiosqlite.Row

                # Idempotency: skip if this exact session already has a pending
                # suggestion with the same title (re-evaluation guard).
                cur = await db.execute(
                    """SELECT id FROM hints
                       WHERE topic_hash = ?
                         AND origin = 'evaluation'
                         AND ((session_id IS ?) OR (session_id = ?))
                         AND status NOT IN ('dismissed', 'expired')""",
                    (topic_hash, session_id, session_id),
                )
                if await cur.fetchone():
                    logger.debug(
                        "[ambient] skipping duplicate eval suggestion: %s",
                        hint.get("title"),
                    )
                    return None

                # Retire stale suggestions for the same target so only the
                # newest one appears in the Suggestions inbox.
                if target_kind in ("schedule", "trigger") and target_id:
                    await db.execute(
                        """UPDATE hints SET status = 'expired'
                           WHERE origin = 'evaluation'
                             AND target_kind = ?
                             AND target_id = ?
                             AND status NOT IN ('dismissed', 'expired', 'accepted')""",
                        (target_kind, target_id),
                    )
                elif session_id:
                    # For manual runs, retire any older eval suggestions for
                    # the same session (shouldn't happen normally, but guards
                    # against concurrent evaluations of the same session).
                    await db.execute(
                        """UPDATE hints SET status = 'expired'
                           WHERE origin = 'evaluation'
                             AND target_kind = 'manual'
                             AND ((session_id IS ?) OR (session_id = ?))
                             AND status NOT IN ('dismissed', 'expired', 'accepted')""",
                        (session_id, session_id),
                    )

                hint_id = _new_id()
                await db.execute(
                    """INSERT INTO hints
                       (id, title, rationale, proposed_prompt, suggested_agent,
                        kind, confidence, sources, topic_hash, status, session_id,
                        created_at, schedule_cron, origin, target_kind, target_id)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        hint_id,
                        hint.get("title", ""),
                        hint.get("rationale", ""),
                        hint.get("proposed_prompt", ""),
                        hint.get("suggested_agent"),
                        hint.get("kind", "task"),
                        float(hint.get("confidence", 0.9)),
                        json.dumps(hint.get("sources", ["evaluation"])),
                        topic_hash,
                        "pending",
                        session_id,
                        now,
                        hint.get("schedule_cron"),
                        "evaluation",
                        hint.get("target_kind"),
                        hint.get("target_id"),
                    ),
                )
                await db.commit()
                return hint_id

    async def list_pending(self, quiet: bool = False) -> list[AmbientHint]:
        """Return all actionable hints (pending or snoozed-but-elapsed).

        When *quiet* is True (inside quiet hours) snoozed hints whose
        ``snoozed_until`` has elapsed are not yet returned — they remain
        snoozed until the quiet window ends.
        """
        now = time.time()
        async with aiosqlite.connect(_db_path()) as db:
            db.row_factory = aiosqlite.Row
            if quiet:
                cur = await db.execute(
                    "SELECT * FROM hints WHERE status IN ('pending', 'shown') ORDER BY confidence DESC, created_at ASC",
                )
            else:
                cur = await db.execute(
                    """SELECT * FROM hints
                       WHERE status IN ('pending', 'shown')
                          OR (status = 'snoozed' AND snoozed_until IS NOT NULL AND snoozed_until <= ?)
                       ORDER BY confidence DESC, created_at ASC""",
                    (now,),
                )
            rows = await cur.fetchall()
            return [AmbientHint(r) for r in rows]

    async def pending_count(self) -> int:
        async with aiosqlite.connect(_db_path()) as db:
            cur = await db.execute(
                "SELECT COUNT(*) FROM hints WHERE status IN ('pending', 'shown')",
            )
            row = await cur.fetchone()
            return row[0] if row else 0

    async def mark_shown(self, hint_id: str) -> None:
        async with aiosqlite.connect(_db_path()) as db:
            await db.execute(
                "UPDATE hints SET status='shown', shown_at=? WHERE id=? AND status='pending'",
                (time.time(), hint_id),
            )
            await db.commit()

    async def accept(self, hint_id: str, session_id: Optional[str] = None) -> bool:
        async with aiosqlite.connect(_db_path()) as db:
            cur = await db.execute(
                """UPDATE hints SET status='accepted', acted_at=?, session_id=?
                   WHERE id=? AND status IN ('pending','shown','snoozed')""",
                (time.time(), session_id, hint_id),
            )
            await db.commit()
            return cur.rowcount > 0

    async def dismiss(self, hint_id: str) -> bool:
        async with aiosqlite.connect(_db_path()) as db:
            cur = await db.execute(
                """UPDATE hints SET status='dismissed', acted_at=?
                   WHERE id=? AND status IN ('pending','shown','snoozed')""",
                (time.time(), hint_id),
            )
            await db.commit()
            return cur.rowcount > 0

    async def snooze(self, hint_id: str, hours: int = 4) -> bool:
        until = time.time() + hours * 3600
        async with aiosqlite.connect(_db_path()) as db:
            cur = await db.execute(
                """UPDATE hints SET status='snoozed', snoozed_until=?
                   WHERE id=? AND status IN ('pending','shown')""",
                (until, hint_id),
            )
            await db.commit()
            return cur.rowcount > 0

    async def get(self, hint_id: str) -> Optional[AmbientHint]:
        async with aiosqlite.connect(_db_path()) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM hints WHERE id=?", (hint_id,)
            )
            row = await cur.fetchone()
            return AmbientHint(row) if row else None

    async def expire_old(self, max_age_days: int = 7) -> int:
        """Mark old pending/shown hints as expired. Returns count expired."""
        cutoff = time.time() - max_age_days * 86400
        async with aiosqlite.connect(_db_path()) as db:
            cur = await db.execute(
                """UPDATE hints SET status='expired'
                   WHERE status IN ('pending','shown','snoozed') AND created_at < ?""",
                (cutoff,),
            )
            await db.commit()
            return cur.rowcount


def _new_id() -> str:
    import uuid
    return str(uuid.uuid4())


# Module-level singleton initialised by the server lifespan.
_store: Optional[AmbientStore] = None


async def get_store() -> AmbientStore:
    global _store  # noqa: PLW0603
    if _store is None:
        _store = AmbientStore()
        await _store.init()
    return _store
