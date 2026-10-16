"""One event stream per session, shared by every connected view.

Producers keep calling ``put`` on the object stored in
``message_queues``. Each ``put`` stamps a monotonic ``seq``, keeps a
ring of recent frames, and copies the frame to every subscriber. Two
views of the same run therefore each receive the full stream, instead
of splitting a single queue between them.

``replay(since)`` returns frames the caller missed. If the ring no
longer contains the next sequence, the caller must reload persisted
messages.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

_RING = 2000
_QUEUE = 2000


class SessionEventHub:
    def __init__(self, maxlen: int = _RING) -> None:
        self.seq = 0
        self.ring: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self.subscribers: list[asyncio.Queue] = []

    def stamp(self, item: dict[str, Any]) -> dict[str, Any]:
        if "seq" in item:
            return item
        self.seq += 1
        item["seq"] = self.seq
        self.ring.append(dict(item))
        return item

    async def put(self, item: dict[str, Any]) -> None:
        frame = self.stamp(item)
        stale: list[asyncio.Queue] = []
        for queue in list(self.subscribers):
            payload = dict(frame)
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                stale.append(queue)
        for queue in stale:
            try:
                queue.get_nowait()
                queue.put_nowait(dict(frame))
            except Exception:
                self.unsubscribe(queue)

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE)
        self.subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        try:
            self.subscribers.remove(queue)
        except ValueError:
            pass

    def replay(self, since: int) -> tuple[str, list[dict[str, Any]]]:
        """``('ok', frames)`` or ``('gap', [])`` when the ring dropped the next frame."""
        if since < 0:
            since = 0
        if not self.ring:
            if since == 0 or since >= self.seq:
                return "ok", []
            return "gap", []
        oldest = int(self.ring[0].get("seq") or 0)
        if since == 0 and oldest > 1:
            return "gap", []
        if since and since + 1 < oldest:
            return "gap", []
        return "ok", [dict(frame) for frame in self.ring if int(frame.get("seq") or 0) > since]


def get_hub(session_id: str):
    """Return the hub for *session_id*, creating it if needed.

    Imported lazily so route modules and this module can both see
    ``message_queues``.
    """
    from backend.state import message_queues

    current = message_queues.get(session_id)
    if isinstance(current, SessionEventHub):
        return current
    hub = SessionEventHub()
    message_queues[session_id] = hub
    return hub
