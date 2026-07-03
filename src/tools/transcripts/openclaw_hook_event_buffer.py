"""In-memory event buffer for OpenClaw session watcher events.

Mirrors :mod:`~tools.transcripts.hook_event_buffer` but is keyed by
OpenClaw ``sessionId`` values.  The watcher pushes synthetic events
when new session files appear, and the MCP server's
:func:`wait_for_activity` can consume them for instant wake-up
instead of polling the filesystem.

Thread-safe: all mutations go through a :class:`threading.Lock` so
the buffer can be written to from the watcher and read from the
MCP server's asyncio loop concurrently.

Usage::

    from tools.transcripts.openclaw_hook_event_buffer import oc_hook_buffer

    # Push from the watcher
    oc_hook_buffer.push("session-abc", event_payload)

    # Await from the MCP server
    arrived = await oc_hook_buffer.wait_for_event("session-abc", timeout=60)
"""

from tools.transcripts.hook_event_buffer import HookEventBuffer

oc_hook_buffer = HookEventBuffer()
