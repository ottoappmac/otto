"""Unit tests for the sticky host-global desktop lease.

The lease (``backend.builtin_mcps.macos_osascript._desktop_lock``) is what
keeps two concurrent macOS desktop agents from stealing the foreground from
each other mid-sequence.  The key invariant under test: one owner holds the
screen across its whole burst (including read-only observation calls) and a
second owner only gets it once the first goes idle — never mid-call.

These tests drive the pure-Python lease directly (no real ``flock`` contention
across processes is needed — in-process the owner table and a per-fd
``flock`` agree).  Each test resets the module's global lease state and runs
its own event loop so a stale ``asyncio.Lock`` can't leak across loops.
"""

from __future__ import annotations

import asyncio

import pytest

import backend.builtin_mcps.macos_osascript._desktop_lock as dl


@pytest.fixture(autouse=True)
def _reset_lease(tmp_path, monkeypatch):
    """Point the lock file at a temp path and reset all global lease state."""
    monkeypatch.setenv("OTTO_DESKTOP_LOCK_PATH", str(tmp_path / "desktop.lock"))
    original_idle = dl.LEASE_IDLE_SECS
    _hard_reset()
    yield
    _hard_reset()
    dl.LEASE_IDLE_SECS = original_idle


def _hard_reset() -> None:
    """Drop any held flock and clear the in-memory lease singletons."""
    if dl._held_fd is not None:
        try:
            dl._flock_unlock(dl._held_fd)
        except OSError:
            pass
        try:
            import os
            os.close(dl._held_fd)
        except OSError:
            pass
    dl._held_fd = None
    dl._held_owner = None
    dl._in_flight = 0
    dl._idle_deadline = 0.0
    if dl._reaper is not None:
        dl._reaper.cancel()
    dl._reaper = None
    # Force a fresh asyncio.Lock for the next event loop.
    dl._state_lock = None


def test_same_owner_is_reentrant():
    async def go():
        await dl.acquire_desktop("A", wait_timeout=2)
        await dl.acquire_desktop("A", wait_timeout=2)
        assert dl._held_owner == "A"
        assert dl._in_flight == 2
        await dl.end_desktop_call("A")
        assert dl._in_flight == 1
        await dl.end_desktop_call("A")
        assert dl._in_flight == 0

    asyncio.run(go())


def test_other_owner_blocked_while_call_in_flight():
    """A busy owner (a call still in flight) never yields the screen."""

    async def go():
        # A acquires and does NOT end — a call is in flight.
        await dl.acquire_desktop("A", wait_timeout=2)
        with pytest.raises(dl.DesktopBusy):
            await dl.acquire_desktop("B", wait_timeout=0.3)
        assert dl._held_owner == "A"

    asyncio.run(go())


def test_handoff_after_idle():
    """Once the owner goes idle past the deadline, a new owner takes over."""
    monkeypatch_idle(0.2)

    async def go():
        await dl.acquire_desktop("A", wait_timeout=2)
        await dl.end_desktop_call("A")  # starts the idle countdown
        assert dl._held_owner == "A"
        await asyncio.sleep(0.35)
        # B's acquire reclaims the idle lease without waiting for the reaper.
        await dl.acquire_desktop("B", wait_timeout=2)
        assert dl._held_owner == "B"

    asyncio.run(go())


def test_idle_owner_does_not_block_before_deadline():
    """A just-ended owner still owns the screen until the idle window lapses."""
    monkeypatch_idle(5.0)

    async def go():
        await dl.acquire_desktop("A", wait_timeout=2)
        await dl.end_desktop_call("A")
        # Within the idle window B must still wait (and time out quickly here).
        with pytest.raises(dl.DesktopBusy):
            await dl.acquire_desktop("B", wait_timeout=0.3)
        # ...but the original owner can resume instantly.
        await dl.acquire_desktop("A", wait_timeout=2)
        assert dl._held_owner == "A"

    asyncio.run(go())


def test_release_frees_immediately():
    async def go():
        await dl.acquire_desktop("A", wait_timeout=2)  # in flight, not ended
        await dl.release_desktop("A")
        assert dl._held_owner is None
        # B can take it right away even though A never ended its call.
        await dl.acquire_desktop("B", wait_timeout=1)
        assert dl._held_owner == "B"

    asyncio.run(go())


def test_context_manager_is_sticky():
    """The context manager holds the screen past block exit (idle hand-off)."""
    monkeypatch_idle(5.0)

    async def go():
        async with dl.desktop_lock("A") as waited_ms:
            assert isinstance(waited_ms, int)
            assert dl._in_flight == 1
        # Block exited but the lease is sticky: still owned by A, now idle.
        assert dl._held_owner == "A"
        assert dl._in_flight == 0

    asyncio.run(go())


def test_reaper_releases_when_left_idle():
    """With no competing owner, the background reaper frees the lease on idle."""
    monkeypatch_idle(0.2)

    async def go():
        await dl.acquire_desktop("A", wait_timeout=2)
        await dl.end_desktop_call("A")
        # Give the reaper time to fire on its own.
        await asyncio.sleep(0.5)
        assert dl._held_owner is None

    asyncio.run(go())


def monkeypatch_idle(seconds: float) -> None:
    """Shrink the idle window so hand-off tests run fast."""
    dl.LEASE_IDLE_SECS = seconds
