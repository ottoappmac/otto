"""Session event hub fans one frame out to every subscriber."""

from __future__ import annotations

import asyncio

from backend.session_events import SessionEventHub


def test_two_subscribers_each_receive_the_frame():
    hub = SessionEventHub()
    first = hub.subscribe()
    second = hub.subscribe()

    async def _run():
        await hub.put({"type": "agent", "content": "hi"})
        return await first.get(), await second.get()

    a, b = asyncio.run(_run())
    assert a["content"] == "hi"
    assert b["content"] == "hi"
    assert a["seq"] == b["seq"] == 1


def test_replay_since_and_gap():
    hub = SessionEventHub(maxlen=2)

    async def _fill():
        for i in range(3):
            await hub.put({"type": "agent", "content": str(i)})

    asyncio.run(_fill())
    status, frames = hub.replay(0)
    assert status == "gap"
    status, frames = hub.replay(1)
    assert status == "ok"
    assert [frame["content"] for frame in frames] == ["1", "2"]


def test_live_subscribe_does_not_require_replay():
    hub = SessionEventHub()

    async def _run():
        await hub.put({"type": "agent", "content": "before"})
        sub = hub.subscribe()
        assert sub.empty()
        await hub.put({"type": "agent", "content": "after"})
        frame = await sub.get()
        return frame["content"]

    assert asyncio.run(_run()) == "after"


def test_a_full_subscriber_gets_resync_instead_of_a_hole():
    hub = SessionEventHub()

    async def _run():
        sub = hub.subscribe(maxsize=1)
        await hub.put({"type": "agent", "content": "one"})
        await hub.put({"type": "agent", "content": "two"})
        return await sub.get()

    frame = asyncio.run(_run())
    assert frame["type"] == "resync"
    assert "dropped" in frame["content"]
