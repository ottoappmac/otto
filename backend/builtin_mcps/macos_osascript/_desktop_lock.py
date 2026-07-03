"""Host-global *sticky* desktop lease shared by every macOS GUI-driving surface.

macOS renders exactly one foreground window per login session, and any
synthesized keystroke / click lands on whatever is frontmost at that
instant.  So at most ONE agent may drive the screen at a time — and that
constraint spans *every* agent run on the host, not just calls within a
single session.

Why a *sticky* lease and not a per-call lock
--------------------------------------------
The first cut serialized *individual* tool calls behind a cross-process
``fcntl.flock``: every focus-stealing call acquired the lock, ran, and
released it.  That is necessary but **not sufficient**.  A real desktop
action is a multi-call sequence — ``activate_app`` → ``get_screen_controls``
→ ``click`` (at coordinates computed from the scan).  With a per-call lock
the screen is *released between those calls*, so a second agent can grab
it, ``activate`` its own app, and become frontmost.  Now the first agent's
``get_screen_controls`` reads a backgrounded window (which often collapses
to just the menu bar) and its coordinate ``click`` lands on the wrong app.
Two simultaneous desktop agents spin forever re-scanning and mis-clicking.

The fix is to hold the screen across the *whole burst*, not one call.  An
agent ("owner") acquires the lease on its first desktop tool call and keeps
it across every subsequent call — including read-only observation tools —
until it has been **idle** for ``LEASE_IDLE_SECS`` (no desktop tool ran in
that window) or the process dies.  A second agent's desktop tools wait for
the lease and, if it never frees within ``LEASE_WAIT_SECS``, get a clean
"desktop busy" signal to back off and retry.  This keeps each agent's
activate→observe→act sequence atomic while still handing the screen off
once the active agent goes quiet.

Mechanics
---------
* **Cross-process exclusivity** is still a ``fcntl.flock`` on a fixed host
  path.  Whichever process owns the lease keeps that ``flock`` held between
  calls; other processes' non-blocking ``flock`` attempts fail with
  ``EWOULDBLOCK`` and poll.
* **In-process coordination** (two sessions in the same backend process,
  each with its own ``macos-native`` toolkit) is an owner table guarded by
  an ``asyncio.Lock``.  Owner ids distinguish the sessions; flock on two
  separate open file descriptions of the same file still conflicts even
  within one process, so the two paths agree.
* **In-flight guard**: the lease can only expire while *no* call from the
  owner is running (``in_flight == 0``).  The idle clock starts when the
  owner's last call finishes, so a long single call can never have the
  screen yanked out from under it mid-action.
* **Idle reaper**: a small background task frees the ``flock`` once the
  owner has been idle past its deadline, so the screen is released even if
  the owner never makes another call.  If the owning process dies the OS
  drops the ``flock`` automatically, so a crashed agent can't wedge the
  screen for everyone else.

This module deliberately depends only on the standard library so it works
identically whether imported as
``backend.builtin_mcps.macos_osascript._desktop_lock`` in the main process
or as the bundled sibling ``_desktop_lock`` inside the MCP subprocess venv
(which has neither ``backend`` nor ``src`` on its path).
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import errno
import logging
import os
import platform
import time
from pathlib import Path
from typing import AsyncIterator

logger = logging.getLogger("otto.desktop_lock")


# ---------------------------------------------------------------------------
# Per-run owner override.
#
# The in-process ``macos-native`` MCP connection — and therefore its
# lease-wrapped tools — is cached once per server id and SHARED across every
# session in the backend process.  An owner baked into the tool wrapper at
# connection-load time would then be identical for two concurrent desktop
# agents, making ``acquire_desktop`` re-entrant (``_held_owner == owner``) and
# the lease a no-op between them — which is exactly how two scheduled desktop
# tasks ended up fighting over the foreground.
#
# Each agent invocation instead sets this contextvar to a unique-but-stable id
# for the duration of its run, and the wrapper resolves the effective owner per
# call (see :func:`resolve_owner`).  Distinct owners ⇒ the lease serializes the
# concurrent runs as intended.
# ---------------------------------------------------------------------------
active_desktop_owner: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "active_desktop_owner", default=None,
)


def resolve_owner(fallback: str) -> str:
    """Return the current async context's desktop owner, or *fallback*.

    ``fallback`` is the per-connection owner baked in at tool-wrap time; it is
    used only when no per-run owner was set (e.g. direct, non-subagent calls).
    """
    return active_desktop_owner.get() or fallback


# How long a queued call waits for the screen before giving up with a clean
# "desktop busy" signal instead of blocking the agent indefinitely.
LEASE_WAIT_SECS: int = max(
    1, int(os.environ.get("OTTO_DESKTOP_LEASE_WAIT_SECS", "300") or "300")
)

# How long the owner may stay idle (no desktop tool call) before the lease
# is handed off to another agent.  Small enough that a quiet agent frees the
# screen promptly, large enough to absorb the LLM round-trips *between* an
# agent's desktop calls so its activate→observe→act burst stays atomic.
LEASE_IDLE_SECS: float = max(
    1.0, float(os.environ.get("OTTO_DESKTOP_LEASE_IDLE_SECS", "20") or "20")
)

# Poll cadence while waiting for the lease.  Small enough that hand-off
# between agents feels instant, large enough not to spin the CPU.
_POLL_INTERVAL_SECS = 0.1


class DesktopBusy(Exception):
    """Raised when the desktop lease can't be acquired within the wait cap."""

    def __init__(self, waited_ms: int) -> None:
        super().__init__("desktop is busy with another agent")
        self.waited_ms = waited_ms


# ---------------------------------------------------------------------------
# Lease state (per process; coordinates in-process owners + the flock fd).
# All mutations happen under ``_state_lock`` which is created lazily on the
# running loop the first time it's needed.
# ---------------------------------------------------------------------------

_held_fd: int | None = None
_held_owner: str | None = None
_in_flight: int = 0
_idle_deadline: float = 0.0  # monotonic; only meaningful while _in_flight == 0
_reaper: "asyncio.Task | None" = None
_state_lock: "asyncio.Lock | None" = None


def _get_state_lock() -> "asyncio.Lock":
    global _state_lock
    if _state_lock is None:
        _state_lock = asyncio.Lock()
    return _state_lock


def _app_data_dir() -> Path:
    """Resolve the Otto app-data directory without importing ``backend``.

    Mirrors ``backend.config.get_app_data_dir`` (minus the one-time
    "George" -> "Otto" migration, which the main process has already run by
    the time any agent is driving the desktop).  Kept dependency-free so the
    isolated MCP subprocess venv can import this module.
    """
    system = platform.system()
    if system == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    elif system == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "Otto"


def lock_path() -> Path:
    """Absolute path of the host-global desktop lock file.

    Overridable via ``OTTO_DESKTOP_LOCK_PATH`` (mainly for tests).  The
    default is a fixed location under the app-data dir, shared by every
    process, so the lease is genuinely machine-wide.
    """
    override = os.environ.get("OTTO_DESKTOP_LOCK_PATH")
    if override:
        return Path(override)
    return _app_data_dir() / "desktop.lock"


def desktop_busy_message(waited_ms: int | None = None) -> str:
    """Standard user-facing text for a call that never got the screen."""
    return (
        "Desktop is busy: another agent is driving the screen and it did not "
        f"free up within {LEASE_WAIT_SECS}s. Only one agent can control the "
        "foreground (mouse/keyboard/window focus) at a time across the whole "
        "machine. Retry shortly, or prefer a background approach (app "
        "dictionary verbs, read-only Accessibility scans, or query_mail_store) "
        "that doesn't need the screen."
    )


def default_owner() -> str:
    """A stable owner id for callers that don't supply one.

    Each ``macos-osascript`` subprocess serves exactly one agent session, so
    a per-process id is a correct owner identity there.  In-process callers
    (``macos-native``) pass an explicit per-session owner instead.
    """
    return f"pid-{os.getpid()}"


# ---------------------------------------------------------------------------
# Low-level flock helpers (POSIX advisory lock on the shared file).
# ---------------------------------------------------------------------------

def _flock_exclusive_nonblocking(fd: int) -> None:
    """Take an exclusive, non-blocking flock; raise ``BlockingIOError`` if held.

    ``fcntl`` is POSIX-only.  On the off chance this module is imported on a
    platform without it (it shouldn't be — both consumers are macOS-gated),
    degrade to a no-op so callers don't crash; the lease simply provides no
    mutual exclusion there.
    """
    try:
        import fcntl
    except ImportError:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
            raise BlockingIOError(exc.errno, "desktop lock held by another agent")
        raise


def _flock_unlock(fd: int) -> None:
    try:
        import fcntl
    except ImportError:
        return
    fcntl.flock(fd, fcntl.LOCK_UN)


def _free_locked() -> None:
    """Release the flock and clear the in-memory holder. Call under the lock."""
    global _held_fd, _held_owner, _in_flight, _idle_deadline
    if _held_fd is not None:
        try:
            _flock_unlock(_held_fd)
        except OSError:
            pass
        try:
            os.close(_held_fd)
        except OSError:
            pass
    _held_fd = None
    _held_owner = None
    _in_flight = 0
    _idle_deadline = 0.0


def _maybe_reclaim_locked() -> None:
    """Free an idle holder so a different owner can take over. Under the lock.

    Only reclaims when the holder has no call in flight and its idle deadline
    has passed.  This is the in-process fast path that lets a waiting owner
    take over without waiting for the reaper task to fire.
    """
    if _held_owner is None:
        return
    if _in_flight == 0 and time.monotonic() >= _idle_deadline:
        _free_locked()


def _ensure_reaper() -> None:
    """Start the idle-reaper task if one isn't already running. Under the lock."""
    global _reaper
    if _reaper is not None and not _reaper.done():
        return
    try:
        _reaper = asyncio.ensure_future(_reaper_loop())
    except RuntimeError:
        # No running loop (shouldn't happen from an async caller); the next
        # waiting owner's _maybe_reclaim_locked will free an idle lease.
        _reaper = None


async def _reaper_loop() -> None:
    """Free the lease once the owner has been idle past its deadline.

    Re-reads the deadline every iteration so a refreshing owner keeps the
    screen, and bails out the moment a call goes back in flight (a busy owner
    can't expire — the next idle transition restarts the reaper).
    """
    global _reaper
    lock = _get_state_lock()
    while True:
        async with lock:
            if _held_owner is None:
                _reaper = None
                return
            if _in_flight > 0:
                # Active again — stop reaping; end_desktop_call() restarts us.
                _reaper = None
                return
            remaining = _idle_deadline - time.monotonic()
            if remaining <= 0:
                logger.info(
                    "desktop lease idle %.0fs — releasing for hand-off (owner=%s)",
                    LEASE_IDLE_SECS, _held_owner,
                )
                _free_locked()
                _reaper = None
                return
        await asyncio.sleep(max(_POLL_INTERVAL_SECS, min(remaining, LEASE_IDLE_SECS)))


# ---------------------------------------------------------------------------
# Public sticky-lease API.
# ---------------------------------------------------------------------------

async def acquire_desktop(
    owner: str, *, wait_timeout: float = LEASE_WAIT_SECS,
) -> int:
    """Begin a guarded desktop section for ``owner``; returns ms spent waiting.

    Ensures ``owner`` holds the host-global screen lease and marks one call
    as in flight (so the lease can't expire mid-call).  Re-entrant: nested
    calls from the same owner (``launch_app`` → ``activate_app``) just bump
    the in-flight count.  Raises :class:`DesktopBusy` if the screen can't be
    secured within ``wait_timeout`` seconds.

    Every successful ``acquire_desktop`` MUST be paired with exactly one
    :func:`end_desktop_call` (use the :func:`desktop_lock` context manager,
    which does this for you).
    """
    global _held_fd, _held_owner, _in_flight, _idle_deadline

    lock = _get_state_lock()
    started = time.monotonic()
    waited_logged = False

    while True:
        async with lock:
            _maybe_reclaim_locked()

            if _held_owner == owner:
                _in_flight += 1
                return int((time.monotonic() - started) * 1000)

            if _held_owner is None:
                path = lock_path()
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                except OSError:
                    pass
                fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
                try:
                    _flock_exclusive_nonblocking(fd)
                except BlockingIOError:
                    # Another *process* holds the screen — close and retry.
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                else:
                    _held_fd = fd
                    _held_owner = owner
                    _in_flight = 1
                    _idle_deadline = 0.0
                    waited_ms = int((time.monotonic() - started) * 1000)
                    if waited_ms:
                        logger.info(
                            "desktop lease acquired after %d ms wait (owner=%s)",
                            waited_ms, owner,
                        )
                    return waited_ms
            # else: held by a different in-process owner that's still active.

        if not waited_logged:
            waited_logged = True
            logger.info(
                "desktop lease busy; owner=%s waiting up to %.0fs for the screen",
                owner, wait_timeout,
            )
        if time.monotonic() - started >= wait_timeout:
            raise DesktopBusy(int((time.monotonic() - started) * 1000))
        await asyncio.sleep(_POLL_INTERVAL_SECS)


async def end_desktop_call(owner: str) -> None:
    """End a guarded section started by :func:`acquire_desktop`.

    Does NOT free the screen — that's the whole point of the sticky lease.
    When the owner's last in-flight call finishes, the idle countdown starts
    and the reaper is (re)armed so the lease frees after ``LEASE_IDLE_SECS``
    of quiet.
    """
    global _in_flight, _idle_deadline
    lock = _get_state_lock()
    async with lock:
        if _held_owner != owner:
            return
        if _in_flight > 0:
            _in_flight -= 1
        if _in_flight == 0:
            _idle_deadline = time.monotonic() + LEASE_IDLE_SECS
            _ensure_reaper()


async def release_desktop(owner: str) -> None:
    """Force-free the lease for ``owner`` immediately (e.g. session teardown).

    Safe to call even if ``owner`` doesn't hold the lease.
    """
    lock = _get_state_lock()
    async with lock:
        if _held_owner == owner:
            _free_locked()


@contextlib.asynccontextmanager
async def desktop_lock(
    owner: str | None = None, *, wait_timeout: float = LEASE_WAIT_SECS,
) -> AsyncIterator[int]:
    """Hold the sticky desktop lease for the duration of the ``with`` block.

    Yields the milliseconds spent waiting for the screen (``0`` when it was
    free or already held by this owner).  Raises :class:`DesktopBusy` if the
    lease can't be acquired within ``wait_timeout``.  Unlike a plain mutex,
    the screen is **not** handed back when the block exits — the owner keeps
    it until it goes idle (see module docstring) so a multi-call burst stays
    atomic.  ``owner`` defaults to a stable per-process id, correct for the
    single-session ``macos-osascript`` subprocess.
    """
    o = owner or default_owner()
    waited_ms = await acquire_desktop(o, wait_timeout=wait_timeout)
    try:
        yield waited_ms
    finally:
        await end_desktop_call(o)
