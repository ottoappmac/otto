"""In-memory buffer for Claude Code HTTP hook events.

Stores incoming hook events keyed by Claude Code ``session_id`` and
provides an :class:`asyncio.Event`-based signalling mechanism so that
:func:`wait_for_activity` in the MCP server can return instantly when
a hook fires instead of polling the filesystem.

Thread-safe: all mutations go through a :class:`threading.Lock` so the
buffer can be written to from FastAPI request handlers and read from
the MCP server's asyncio loop concurrently.

Usage::

    from tools.transcripts.hook_event_buffer import hook_buffer

    # Push from a route handler
    hook_buffer.push("session-abc", event_payload)

    # Await from the MCP server
    arrived = await hook_buffer.wait_for_event("session-abc", timeout=60)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_MAX_EVENTS_PER_SESSION = 5_000
_MAX_SESSIONS = 200
_STALE_SESSION_SECONDS = 7_200  # 2 hours


@dataclass
class _SessionBucket:
    """Events and metadata for a single Claude Code session."""

    events: list[dict[str, Any]] = field(default_factory=list)
    transcript_path: str = ""
    last_event_time: float = 0.0
    last_event_mono: float = 0.0
    session_ended: bool = False

    # Per-session asyncio.Event — set() when a new hook arrives,
    # clear()ed after a consumer wakes up.
    _notify: asyncio.Event | None = None

    def get_notify_event(self) -> asyncio.Event:
        """Lazily create the asyncio.Event for wakeup signalling."""
        if self._notify is None:
            self._notify = asyncio.Event()
        return self._notify


class HookEventBuffer:
    """Singleton buffer for Claude Code HTTP hook events.

    All public methods are safe to call from any thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: dict[str, _SessionBucket] = {}

    # ------------------------------------------------------------------
    # Write path (called from FastAPI route handlers)
    # ------------------------------------------------------------------

    def push(self, session_id: str, event: dict[str, Any]) -> None:
        """Store an incoming hook event and wake any waiters."""
        now_mono = time.monotonic()
        now_wall = time.time()

        with self._lock:
            self._maybe_evict_stale(now_mono)
            bucket = self._buckets.get(session_id)
            if bucket is None:
                bucket = _SessionBucket()
                self._buckets[session_id] = bucket

            bucket.last_event_time = now_wall
            bucket.last_event_mono = now_mono

            transcript = event.get("transcript_path", "")
            if transcript:
                bucket.transcript_path = transcript

            if event.get("hook_event_name") == "SessionEnd":
                bucket.session_ended = True

            if len(bucket.events) < _MAX_EVENTS_PER_SESSION:
                bucket.events.append(event)

            notify = bucket._notify

        if notify is not None:
            try:
                notify.set()
            except RuntimeError:
                pass

    # ------------------------------------------------------------------
    # Read path (called from MCP server / eval pipeline)
    # ------------------------------------------------------------------

    def has_new_events(self, session_id: str, since_mono: float = 0.0) -> bool:
        """True if events arrived for *session_id* after *since_mono*."""
        with self._lock:
            bucket = self._buckets.get(session_id)
            if bucket is None:
                return False
            return bucket.last_event_mono > since_mono

    async def wait_for_event(
        self,
        session_id: str,
        timeout: float = 60.0,
        since_mono: float = 0.0,
    ) -> bool:
        """Block until a hook event arrives or *timeout* elapses.

        If events already arrived after *since_mono*, returns ``True``
        immediately without blocking.

        Returns ``True`` if an event arrived, ``False`` on timeout.
        """
        with self._lock:
            bucket = self._buckets.get(session_id)
            if bucket is None:
                bucket = _SessionBucket()
                self._buckets[session_id] = bucket

            if bucket.last_event_mono > since_mono and bucket.events:
                return True

            notify = bucket.get_notify_event()
            notify.clear()

        try:
            await asyncio.wait_for(notify.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def get_events(self, session_id: str) -> list[dict[str, Any]]:
        """Return all buffered events for *session_id* (non-destructive)."""
        with self._lock:
            bucket = self._buckets.get(session_id)
            if bucket is None:
                return []
            return list(bucket.events)

    def drain_events(self, session_id: str) -> list[dict[str, Any]]:
        """Return and remove all buffered events for *session_id*."""
        with self._lock:
            bucket = self._buckets.get(session_id)
            if bucket is None:
                return []
            events = bucket.events
            bucket.events = []
            return events

    def get_tool_trajectory(self, session_id: str) -> list[dict[str, Any]]:
        """Return PostToolUse / PostToolUseFailure events for quality gating."""
        with self._lock:
            bucket = self._buckets.get(session_id)
            if bucket is None:
                return []
            return [
                e for e in bucket.events
                if e.get("hook_event_name") in (
                    "PostToolUse", "PostToolUseFailure",
                )
            ]

    def get_transcript_path(self, session_id: str) -> str | None:
        """Return the transcript path associated with *session_id*."""
        with self._lock:
            bucket = self._buckets.get(session_id)
            if bucket is None:
                return None
            return bucket.transcript_path or None

    def session_ended(self, session_id: str) -> bool:
        """True if a SessionEnd event was received for *session_id*."""
        with self._lock:
            bucket = self._buckets.get(session_id)
            return bucket is not None and bucket.session_ended

    def find_session_for_transcript(self, transcript_path: str) -> str | None:
        """Reverse lookup: find the session_id whose transcript_path matches."""
        with self._lock:
            for sid, bucket in self._buckets.items():
                if bucket.transcript_path == transcript_path:
                    return sid
            return None

    def active_session_count(self) -> int:
        """Number of sessions currently tracked."""
        with self._lock:
            return len(self._buckets)

    def session_stats(self, session_id: str) -> dict[str, Any]:
        """Return summary stats for a session (for the get_hook_status tool)."""
        with self._lock:
            bucket = self._buckets.get(session_id)
            if bucket is None:
                return {"hooks_active": False, "events_buffered": 0}
            return {
                "hooks_active": True,
                "events_buffered": len(bucket.events),
                "transcript_path": bucket.transcript_path,
                "session_ended": bucket.session_ended,
                "last_event_time": bucket.last_event_time,
            }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self, max_age_seconds: float = _STALE_SESSION_SECONDS) -> int:
        """Evict sessions with no events in *max_age_seconds*. Returns count."""
        now = time.monotonic()
        removed = 0
        with self._lock:
            stale = [
                sid for sid, b in self._buckets.items()
                if (now - b.last_event_mono) > max_age_seconds
            ]
            for sid in stale:
                del self._buckets[sid]
                removed += 1
        if removed:
            logger.info("Evicted %d stale hook session(s)", removed)
        return removed

    def _maybe_evict_stale(self, now_mono: float) -> None:
        """Inline eviction when we're over the session cap."""
        if len(self._buckets) < _MAX_SESSIONS:
            return
        oldest_sid = min(
            self._buckets, key=lambda s: self._buckets[s].last_event_mono,
        )
        del self._buckets[oldest_sid]
        logger.debug("Evicted oldest session %s (cap reached)", oldest_sid)


# Module-level singleton
hook_buffer = HookEventBuffer()
