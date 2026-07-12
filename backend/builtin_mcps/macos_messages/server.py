#!/usr/bin/env python3
"""Built-in MCP server: macOS Messages.

Typed tools over Apple Messages. Messages is unusual among the scriptable
macOS apps in that its AppleScript dictionary can *send* and enumerate
services / buddies / chats, but exposes **no readable message element** —
so message history can only be read from the ``chat.db`` SQLite store.
This server therefore spans two surfaces:

* **Apple Events** (Automation permission) — ``send_message``,
  ``list_chats``, ``list_buddies``.
* **SQLite** (Full Disk Access) — ``read_messages`` reads recent messages
  straight out of ``~/Library/Messages/chat.db``, mirroring the approach
  ``macos-osascript``'s ``query_mail_store`` takes for Apple Mail.

There is intentionally no update/delete tool: Messages' dictionary can't
edit or delete sent messages, so those operations aren't possible here.

Trust boundaries:

* The send/enumerate scripts are background Apple Events — none of them
  ``activate`` Messages or synthesize input, so this server never needs
  the cross-process desktop lease; calls can run concurrently.
* ``read_messages`` opens chat.db strictly read-only (``mode=ro``) and
  requires Full Disk Access; a clear TCC error is surfaced when it's not
  granted.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from mcp.server.fastmcp import FastMCP

try:
    from ._helpers import (  # type: ignore[import-not-found]
        APPLESCRIPT_HANDLERS,
        FIELD_SEP,
        RECORD_SEP,
        clamp_limit,
        cocoa_to_iso,
        decode_attributed_body,
        escape_applescript_string,
        find_chat_db,
        normalize_service,
        parse_records,
    )
except ImportError:
    from _helpers import (  # type: ignore[no-redef]
        APPLESCRIPT_HANDLERS,
        FIELD_SEP,
        RECORD_SEP,
        clamp_limit,
        cocoa_to_iso,
        decode_attributed_body,
        escape_applescript_string,
        find_chat_db,
        normalize_service,
        parse_records,
    )

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("otto.mcp.macos_messages")

_MAX_TIMEOUT_SECS = 120
_DEFAULT_TIMEOUT_SECS = 30
_MAX_OUTPUT_BYTES = 64 * 1024

mcp = FastMCP("macOS Messages")


def _truncate(blob: bytes) -> tuple[str, bool]:
    truncated = len(blob) > _MAX_OUTPUT_BYTES
    if truncated:
        blob = blob[:_MAX_OUTPUT_BYTES]
    try:
        return blob.decode("utf-8", errors="replace"), truncated
    except Exception:
        return blob.decode("latin-1", errors="replace"), truncated


def _clamp_timeout(timeout: int, default: int = _DEFAULT_TIMEOUT_SECS) -> int:
    try:
        timeout = int(timeout)
    except (TypeError, ValueError):
        timeout = default
    if timeout <= 0:
        timeout = default
    return min(timeout, _MAX_TIMEOUT_SECS)


async def _run_osascript(script: str, timeout: int = _DEFAULT_TIMEOUT_SECS) -> dict[str, Any]:
    """Spawn ``osascript -e <script>`` and collect its result.

    ``create_subprocess_exec`` (no shell) delivers the full script as a
    single argv entry. The body is wrapped in AppleScript's own
    ``with timeout of`` block a few seconds shorter than the outer
    subprocess timeout so a hung Apple Event raises a fast, clean script
    error instead of eating the whole wall-clock budget.
    """
    timeout = _clamp_timeout(timeout)
    inner_timeout = max(5, timeout - 3)
    full_script = (
        f"{APPLESCRIPT_HANDLERS}\nwith timeout of {inner_timeout} seconds\n"
        f"{script}\nend timeout\n"
    )
    started = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            "osascript", "-e", full_script,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return {
            "ok": False, "stdout": "", "timed_out": False, "duration_ms": 0,
            "stderr": "osascript binary not found — is this running on macOS?",
        }

    timed_out = False
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
    except asyncio.TimeoutError:
        timed_out = True
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            stdout_bytes, stderr_bytes = await proc.communicate()
        except Exception:
            stdout_bytes, stderr_bytes = b"", b""

    duration_ms = int((time.monotonic() - started) * 1000)
    stdout_text, _ = _truncate(stdout_bytes or b"")
    stderr_text, _ = _truncate(stderr_bytes or b"")
    exit_code = proc.returncode if proc.returncode is not None else -1

    return {
        "ok": (not timed_out) and exit_code == 0,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "timed_out": timed_out,
        "duration_ms": duration_ms,
    }


# ---------------------------------------------------------------------------
# Create (send) — Apple Events
# ---------------------------------------------------------------------------


@mcp.tool()
async def send_message(
    to: str,
    body: str,
    service: str = "iMessage",
) -> dict[str, Any]:
    """Send a message to a phone number or email handle.

    Args:
        to: The recipient handle — a phone number (ideally E.164, e.g.
            ``"+15551234567"``) or an iMessage email address.
        body: The text to send (required).
        service: ``"iMessage"`` (default) or ``"SMS"``. ``SMS`` only works
            when the Mac has Text Message Forwarding set up with an iPhone;
            otherwise use ``iMessage``.

    Returns:
        dict with ``ok`` and ``stderr`` on failure. A failure here often
        means the handle isn't reachable on the chosen service (e.g.
        sending ``iMessage`` to a number with no iMessage account) — retry
        with the other service or verify the handle.
    """
    if not to or not to.strip():
        return {"ok": False, "stderr": "to must be a non-empty recipient handle"}
    if not body or not body.strip():
        return {"ok": False, "stderr": "body must be a non-empty string"}

    svc = normalize_service(service)
    if svc is None:
        return {"ok": False, "stderr": f"service must be 'iMessage' or 'SMS'; got {service!r}"}

    esc_to = escape_applescript_string(to)
    esc_body = escape_applescript_string(body)
    # `service type` is an enumerated constant (iMessage / SMS), not a
    # string, so it's interpolated as a bare token, not a quoted literal.
    script = f"""
tell application "Messages"
\tset targetService to 1st service whose service type = {svc}
\tset targetBuddy to buddy "{esc_to}" of targetService
\tsend "{esc_body}" to targetBuddy
end tell
"""
    res = await _run_osascript(script)
    if not res["ok"]:
        return {
            "ok": False,
            "stderr": res["stderr"] or (
                f"failed to send to {to!r} via {svc} — the handle may not be "
                f"reachable on {svc}; try the other service or check the number/email"
            ),
        }
    return {"ok": True, "stderr": ""}


# ---------------------------------------------------------------------------
# Read (enumerate) — Apple Events
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_chats(limit: int = 50) -> dict[str, Any]:
    """List existing chats (conversations) known to Messages.

    Args:
        limit: Max chats to return (1-500).

    Returns:
        dict with ``ok``, ``chats`` (list of dicts with ``id``, ``name``,
        ``participants`` (list of handle strings)), and ``stderr``. Use a
        participant handle with ``send_message`` or ``read_messages``.
    """
    limit = clamp_limit(limit)
    fields = ["id", "name", "participants"]
    script = f"""
set out to ""
tell application "Messages"
\tset theChats to every chat
\tset n to (count of theChats)
\tif n > {limit} then set n to {limit}
\trepeat with i from 1 to n
\t\tset ch to item i of theChats
\t\tset partHandles to ""
\t\ttry
\t\t\trepeat with p in participants of ch
\t\t\t\tif partHandles is not "" then set partHandles to partHandles & ","
\t\t\t\tset partHandles to partHandles & my safeText(handle of p)
\t\t\tend repeat
\t\tend try
\t\tset out to out & my safeText(id of ch) & "{FIELD_SEP}" & my safeText(name of ch) & "{FIELD_SEP}" & partHandles & "{RECORD_SEP}"
\tend repeat
end tell
return out
"""
    res = await _run_osascript(script)
    if not res["ok"]:
        return {"ok": False, "chats": [], "stderr": res["stderr"] or "failed to list chats"}
    records = parse_records(res["stdout"], fields)
    chats = [
        {
            "id": r["id"],
            "name": r["name"],
            "participants": [h for h in r["participants"].split(",") if h],
        }
        for r in records
    ]
    return {"ok": True, "chats": chats, "stderr": ""}


@mcp.tool()
async def list_buddies(service: str = "iMessage", limit: int = 200) -> dict[str, Any]:
    """List known buddies (contacts) on a service.

    Args:
        service: ``"iMessage"`` (default) or ``"SMS"``.
        limit: Max buddies to return (1-500).

    Returns:
        dict with ``ok``, ``buddies`` (list of dicts with ``handle``,
        ``name``), and ``stderr``. Use a ``handle`` with ``send_message``
        or ``read_messages``.
    """
    limit = clamp_limit(limit)
    svc = normalize_service(service)
    if svc is None:
        return {"ok": False, "buddies": [], "stderr": f"service must be 'iMessage' or 'SMS'; got {service!r}"}

    fields = ["handle", "name"]
    script = f"""
set out to ""
tell application "Messages"
\tset targetService to 1st service whose service type = {svc}
\tset theBuddies to buddies of targetService
\tset n to (count of theBuddies)
\tif n > {limit} then set n to {limit}
\trepeat with i from 1 to n
\t\tset b to item i of theBuddies
\t\tset out to out & my safeText(handle of b) & "{FIELD_SEP}" & my safeText(name of b) & "{RECORD_SEP}"
\tend repeat
end tell
return out
"""
    res = await _run_osascript(script)
    if not res["ok"]:
        return {"ok": False, "buddies": [], "stderr": res["stderr"] or "failed to list buddies"}
    records = parse_records(res["stdout"], fields)
    buddies = [{"handle": r["handle"], "name": r["name"]} for r in records]
    return {"ok": True, "buddies": buddies, "stderr": ""}


# ---------------------------------------------------------------------------
# Read (history) — SQLite (Full Disk Access)
# ---------------------------------------------------------------------------


@mcp.tool()
async def read_messages(
    handle: str = "",
    limit: int = 50,
    days_back: int = 0,
) -> dict[str, Any]:
    """Read recent messages from the Messages SQLite store (chat.db).

    Messages' AppleScript dictionary can't read message history, so this
    reads ``~/Library/Messages/chat.db`` directly — which **requires Full
    Disk Access** for the process running Otto's backend. If the call
    returns ``ok=false`` with a permissions error, grant it in System
    Settings → Privacy & Security → Full Disk Access.

    Args:
        handle: Filter to one conversation partner — a phone number or
            email substring (case-insensitive). Empty string (default)
            returns the most recent messages across all conversations.
        limit: Max messages to return (1-500), most recent first.
        days_back: Only messages within this many days. ``0`` (default)
            means no lower bound.

    Returns:
        dict with:
        * ``ok``       — bool
        * ``messages`` — list of dicts with ``text``, ``handle`` (the
          other party's number/email), ``from_me`` (bool), ``date``
          (ISO-8601 UTC), ``service``, most recent first
        * ``count``    — number of messages returned
        * ``db_path``  — the database file read
        * ``stderr``   — error message on failure
    """
    import sqlite3

    limit = clamp_limit(limit)
    db_path = find_chat_db()
    if not db_path:
        return {
            "ok": False, "messages": [], "count": 0, "db_path": "",
            "stderr": (
                "Messages database not found at ~/Library/Messages/chat.db. "
                "Either Messages has never been used or Full Disk Access is "
                "not granted — grant it in System Settings → Privacy & "
                "Security → Full Disk Access."
            ),
        }

    conditions = ["m.text IS NOT NULL OR m.attributedBody IS NOT NULL"]
    params: list[Any] = []
    if handle.strip():
        conditions.append("lower(h.id) LIKE ?")
        params.append(f"%{handle.strip().lower()}%")
    where = " AND ".join(f"({c})" for c in conditions)

    # Over-fetch a little so post-query days_back filtering (needed because
    # chat.db's date units differ across macOS versions — see
    # _helpers.cocoa_to_iso) still yields up to `limit` rows.
    fetch_n = limit if days_back <= 0 else min(limit * 4, 2000)
    sql = f"""
        SELECT m.text, m.attributedBody, m.is_from_me, m.date, m.service,
               h.id AS handle
        FROM message m
        LEFT JOIN handle h ON m.handle_id = h.ROWID
        WHERE {where}
        ORDER BY m.date DESC
        LIMIT ?
    """
    params.append(fetch_n)

    def _run_query() -> list[tuple]:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    try:
        rows = await asyncio.to_thread(_run_query)
    except sqlite3.OperationalError as exc:
        err = str(exc)
        if "unable to open" in err.lower() or "permission" in err.lower() or "authorization" in err.lower():
            err = (
                f"Cannot open {db_path}: {exc}. Grant Full Disk Access in "
                "System Settings → Privacy & Security → Full Disk Access."
            )
        return {"ok": False, "messages": [], "count": 0, "db_path": db_path, "stderr": err}
    except Exception as exc:
        return {"ok": False, "messages": [], "count": 0, "db_path": db_path, "stderr": str(exc)}

    cutoff_iso = ""
    if days_back > 0:
        cutoff_iso = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()

    messages: list[dict[str, Any]] = []
    for text, attributed, is_from_me, date_val, service, hdl in rows:
        body = text or ""
        if not body and attributed:
            body = decode_attributed_body(attributed)
        iso = cocoa_to_iso(date_val)
        if cutoff_iso and iso and iso < cutoff_iso:
            continue
        messages.append({
            "text": body,
            "handle": hdl or "",
            "from_me": bool(is_from_me),
            "date": iso,
            "service": service or "",
        })
        if len(messages) >= limit:
            break

    return {
        "ok": True,
        "messages": messages,
        "count": len(messages),
        "db_path": db_path,
        "stderr": "",
    }


if __name__ == "__main__":
    mcp.run()
