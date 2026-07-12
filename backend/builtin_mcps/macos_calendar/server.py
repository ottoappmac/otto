#!/usr/bin/env python3
"""Built-in MCP server: macOS Calendar.

Structured Create/Read/Update/Delete tools over Apple Calendar,
implemented by generating AppleScript against Calendar's own dictionary
and running it via ``osascript``. Unlike ``macos-osascript``'s generic
``run_osascript``, every tool takes typed arguments and builds the
correct AppleScript internally — the agent never authors raw AppleScript
or hand-builds a locale-safe ``date`` object.

Tools:

* Read:   ``list_calendars``, ``list_events``, ``get_event``.
* Create: ``create_event``.
* Update: ``update_event``.
* Delete: ``delete_event``.

Trust boundaries:

* Every script here is a background Apple Event — none of them
  ``activate`` Calendar or synthesize input, so this server never needs
  the cross-process desktop lease; calls can run concurrently.
* macOS gates Apple Events behind a per-app **Automation** TCC prompt
  the user must approve (System Settings → Privacy & Security →
  Automation → grant the calling app access to Calendar). No Full Disk
  Access is required.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

try:
    from ._helpers import (  # type: ignore[import-not-found]
        APPLESCRIPT_HANDLERS,
        FIELD_SEP,
        RECORD_SEP,
        applescript_date_block,
        calendar_reference,
        clamp_limit,
        escape_applescript_string,
        event_reference,
        parse_iso_datetime,
        parse_records,
        parse_single_record,
    )
except ImportError:
    from _helpers import (  # type: ignore[no-redef]
        APPLESCRIPT_HANDLERS,
        FIELD_SEP,
        RECORD_SEP,
        applescript_date_block,
        calendar_reference,
        clamp_limit,
        escape_applescript_string,
        event_reference,
        parse_iso_datetime,
        parse_records,
        parse_single_record,
    )

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("otto.mcp.macos_calendar")

_MAX_TIMEOUT_SECS = 120
_DEFAULT_TIMEOUT_SECS = 30
_LIST_DEFAULT_TIMEOUT_SECS = 60
_MAX_OUTPUT_BYTES = 64 * 1024

mcp = FastMCP("macOS Calendar")


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
    subprocess timeout so a hung Apple Event (Calendar's ``whose``
    filters can be slow on large calendars) raises a fast, clean script
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
# Read
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_calendars() -> dict[str, Any]:
    """List every calendar.

    Use this to discover the exact ``calendar_name`` strings the other
    tools expect (``list_events``, ``create_event``, …).

    Returns:
        dict with ``ok``, ``calendars`` (list of dicts with ``name``,
        ``writable`` (bool), ``description``), and ``stderr`` on failure.
    """
    fields = ["name", "writable", "description"]
    script = f"""
set out to ""
tell application "Calendar"
\trepeat with c in every calendar
\t\tset out to out & my safeText(name of c) & "{FIELD_SEP}" & (writable of c as string) & "{FIELD_SEP}" & my safeText(description of c) & "{RECORD_SEP}"
\tend repeat
end tell
return out
"""
    res = await _run_osascript(script)
    if not res["ok"]:
        return {"ok": False, "calendars": [], "stderr": res["stderr"] or "failed to list calendars"}
    records = parse_records(res["stdout"], fields)
    calendars = [
        {
            "name": r["name"],
            "writable": r["writable"] == "true",
            "description": r["description"],
        }
        for r in records
    ]
    return {"ok": True, "calendars": calendars, "stderr": ""}


@mcp.tool()
async def list_events(
    calendar_name: str = "",
    start_date: str = "",
    end_date: str = "",
    limit: int = 100,
    timeout_seconds: int = _LIST_DEFAULT_TIMEOUT_SECS,
) -> dict[str, Any]:
    """List events overlapping a date range, in one calendar or all.

    Args:
        calendar_name: Restrict to one calendar (see ``list_calendars``).
            Empty string (default) sweeps every calendar.
        start_date: Range start as an ISO-ish string (``"2026-07-10"`` or
            ``"2026-07-10 09:00"``). Empty string defaults to today at
            midnight.
        end_date: Range end (same formats). Empty string defaults to 30
            days after the start.
        limit: Max events to return (1-200), across the whole sweep.
        timeout_seconds: Wall-clock cap. Defaults to 60s (Calendar's
            ``whose`` range filter can be slow on large calendars);
            capped at 120s.

    Returns:
        dict with ``ok``, ``events`` (list of dicts with ``uid``,
        ``summary``, ``start_date``, ``end_date``, ``location``,
        ``all_day``, ``calendar_name``), and ``stderr``. Use ``uid`` with
        ``get_event``/``update_event``/``delete_event`` (pass the same
        ``calendar_name`` back).

    Note: recurring events are matched by their master only — the
    AppleScript dictionary does not expand individual occurrences, so a
    weekly meeting shows once, not once per week in the range.
    """
    limit = clamp_limit(limit)
    timeout = _clamp_timeout(timeout_seconds, default=_LIST_DEFAULT_TIMEOUT_SECS)

    start_comps = parse_iso_datetime(start_date) if start_date.strip() else None
    if start_date.strip() and start_comps is None:
        return {"ok": False, "events": [], "stderr": f"could not parse start_date={start_date!r}"}
    end_comps = parse_iso_datetime(end_date) if end_date.strip() else None
    if end_date.strip() and end_comps is None:
        return {"ok": False, "events": [], "stderr": f"could not parse end_date={end_date!r}"}

    now = datetime.now()
    if start_comps is None:
        base = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start_comps = (base.year, base.month, base.day, 0, 0, 0)
    if end_comps is None:
        s = datetime(*start_comps)
        e = s + timedelta(days=30)
        end_comps = (e.year, e.month, e.day, e.hour, e.minute, e.second)

    date_lines = applescript_date_block("rangeStart", start_comps) + applescript_date_block("rangeEnd", end_comps)
    # An event overlaps [rangeStart, rangeEnd] iff it starts before the
    # range ends AND ends after the range starts.
    whose = "(start date <= rangeEnd) and (end date >= rangeStart)"

    def _scan(cal_expr: str, cal_label_expr: str) -> str:
        return f"""
\t\tset evs to {{}}
\t\ttry
\t\t\tset evs to (every event of {cal_expr} whose {whose})
\t\ton error
\t\t\tset evs to {{}}
\t\tend try
\t\tset ecnt to (count of evs)
\t\trepeat with ei from 1 to ecnt
\t\t\tif foundCount >= {limit} then exit repeat
\t\t\tset ev to item ei of evs
\t\t\tset out to out & my safeText(uid of ev) & "{FIELD_SEP}" & my safeText(summary of ev) & "{FIELD_SEP}" & my safeText(start date of ev) & "{FIELD_SEP}" & my safeText(end date of ev) & "{FIELD_SEP}" & my safeText(location of ev) & "{FIELD_SEP}" & (allday event of ev as string) & "{FIELD_SEP}" & {cal_label_expr} & "{RECORD_SEP}"
\t\t\tset foundCount to foundCount + 1
\t\tend repeat
"""

    if calendar_name.strip():
        cal_ref = calendar_reference(calendar_name)
        esc_label = escape_applescript_string(calendar_name)
        body = _scan(cal_ref, f'"{esc_label}"')
        sweep = f'tell application "Calendar"{body}end tell'
    else:
        body = _scan("cal", "calName")
        sweep = f"""tell application "Calendar"
\trepeat with cal in every calendar
\t\tset calName to my safeText(name of cal)
{body}\t\tif foundCount >= {limit} then exit repeat
\tend repeat
end tell"""

    fields = ["uid", "summary", "start_date", "end_date", "location", "all_day", "calendar_name"]
    script = f"""
set out to ""
set foundCount to 0
{date_lines}{sweep}
return out
"""
    res = await _run_osascript(script, timeout=timeout)
    if not res["ok"]:
        if res["timed_out"] or "1712" in res["stderr"] or "timed out" in res["stderr"].lower():
            stderr = (
                f"list_events timed out after {timeout}s — narrow the range "
                "with start_date/end_date, scope to one calendar_name, or "
                "raise timeout_seconds (up to 120)."
            )
        else:
            stderr = res["stderr"] or "failed to list events"
        return {"ok": False, "events": [], "stderr": stderr}
    records = parse_records(res["stdout"], fields)
    events = [
        {
            "uid": r["uid"],
            "summary": r["summary"],
            "start_date": r["start_date"],
            "end_date": r["end_date"],
            "location": r["location"],
            "all_day": r["all_day"] == "true",
            "calendar_name": r["calendar_name"],
        }
        for r in records
        if r["uid"]
    ]
    return {"ok": True, "events": events, "stderr": ""}


@mcp.tool()
async def get_event(uid: str, calendar_name: str = "") -> dict[str, Any]:
    """Fetch one event's full detail by uid.

    Args:
        uid: The event's ``uid`` (from ``list_events``).
        calendar_name: The calendar it lives in (see ``list_calendars``).
            Empty string searches every calendar — slower; pass it when
            you can.

    Returns:
        dict with ``ok``, ``uid``, ``summary``, ``description``,
        ``location``, ``start_date``, ``end_date``, ``all_day``, ``url``,
        ``calendar_name``, and ``stderr``.
    """
    if not uid or not uid.strip():
        return {"ok": False, "stderr": "uid must be a non-empty string"}
    cal_ref = calendar_reference(calendar_name) if calendar_name.strip() else None
    ev_ref = event_reference(uid, cal_ref)
    fields = [
        "uid", "summary", "description", "location", "start_date",
        "end_date", "all_day", "url",
    ]
    script = f"""
tell application "Calendar"
\tset ev to {ev_ref}
\tset out to my safeText(uid of ev) & "{FIELD_SEP}" & my safeText(summary of ev) & "{FIELD_SEP}" & my safeText(description of ev) & "{FIELD_SEP}" & my safeText(location of ev) & "{FIELD_SEP}" & my safeText(start date of ev) & "{FIELD_SEP}" & my safeText(end date of ev) & "{FIELD_SEP}" & (allday event of ev as string) & "{FIELD_SEP}" & my safeText(url of ev)
end tell
return out
"""
    res = await _run_osascript(script)
    if not res["ok"]:
        return {
            "ok": False,
            "stderr": res["stderr"] or f"event uid={uid!r} not found (calendar_name={calendar_name!r})",
        }
    record = parse_single_record(res["stdout"], fields)
    return {
        "ok": True,
        "uid": record["uid"],
        "summary": record["summary"],
        "description": record["description"],
        "location": record["location"],
        "start_date": record["start_date"],
        "end_date": record["end_date"],
        "all_day": record["all_day"] == "true",
        "url": record["url"],
        "calendar_name": calendar_name.strip(),
        "stderr": "",
    }


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


@mcp.tool()
async def create_event(
    calendar_name: str,
    summary: str,
    start_date: str,
    end_date: str = "",
    location: str = "",
    description: str = "",
    all_day: bool = False,
) -> dict[str, Any]:
    """Create a new calendar event.

    Args:
        calendar_name: Which (writable) calendar to add to (see
            ``list_calendars``). Required — Calendar has no "default"
            calendar concept in its dictionary.
        summary: Event title (required).
        start_date: Start as an ISO-ish string (``"2026-07-10 09:00"`` or
            ``"2026-07-10"``). Parsed in Python (locale-safe).
        end_date: End (same formats). Empty string defaults to one hour
            after the start (or the same day, for ``all_day``).
        location: Optional location text.
        description: Optional notes.
        all_day: If ``True``, create an all-day event (the time portion of
            the dates is ignored).

    Returns:
        dict with ``ok``, ``uid`` (the new event's uid), and ``stderr``.
    """
    if not calendar_name or not calendar_name.strip():
        return {"ok": False, "stderr": "calendar_name is required (see list_calendars)"}
    if not summary or not summary.strip():
        return {"ok": False, "stderr": "summary must be a non-empty string"}

    start_comps = parse_iso_datetime(start_date)
    if start_comps is None:
        return {"ok": False, "stderr": f"could not parse start_date={start_date!r} (use e.g. 2026-07-10 09:00)"}

    if end_date.strip():
        end_comps = parse_iso_datetime(end_date)
        if end_comps is None:
            return {"ok": False, "stderr": f"could not parse end_date={end_date!r}"}
    else:
        s = datetime(*start_comps)
        e = s + timedelta(hours=1)
        end_comps = (e.year, e.month, e.day, e.hour, e.minute, e.second)

    date_lines = applescript_date_block("startDate", start_comps) + applescript_date_block("endDate", end_comps)

    props = [
        f'summary:"{escape_applescript_string(summary)}"',
        "start date:startDate",
        "end date:endDate",
    ]
    if all_day:
        props.append("allday event:true")
    if location:
        props.append(f'location:"{escape_applescript_string(location)}"')
    if description:
        props.append(f'description:"{escape_applescript_string(description)}"')

    cal_ref = calendar_reference(calendar_name)
    props_str = "{" + ", ".join(props) + "}"
    script = f"""
{date_lines}tell application "Calendar"
\tset newEv to make new event at end of events of {cal_ref} with properties {props_str}
\treturn uid of newEv
end tell
"""
    res = await _run_osascript(script)
    if not res["ok"]:
        return {"ok": False, "stderr": res["stderr"] or "failed to create event"}
    return {"ok": True, "uid": res["stdout"].strip(), "stderr": ""}


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


@mcp.tool()
async def update_event(
    uid: str,
    calendar_name: str = "",
    summary: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    location: Optional[str] = None,
    description: Optional[str] = None,
    all_day: Optional[bool] = None,
) -> dict[str, Any]:
    """Update an existing event. Pass only the fields you want to change.

    All change arguments are independently optional; at least one must be
    given.

    Args:
        uid: The event's ``uid`` (from ``list_events``).
        calendar_name: The calendar it lives in (see ``list_calendars``).
            Empty string searches every calendar — pass it when you can.
        summary: New title. ``None`` leaves it unchanged.
        start_date: New start (ISO-ish string). ``None`` leaves unchanged.
        end_date: New end. ``None`` leaves unchanged.
        location: New location. ``None`` leaves unchanged.
        description: New notes. ``None`` leaves unchanged.
        all_day: Toggle all-day. ``None`` leaves unchanged.

    Returns:
        dict with ``ok`` and ``stderr`` on failure.
    """
    if not uid or not uid.strip():
        return {"ok": False, "stderr": "uid must be a non-empty string"}
    if all(v is None for v in (summary, start_date, end_date, location, description, all_day)):
        return {"ok": False, "stderr": "nothing to update — pass at least one field to change"}

    cal_ref = calendar_reference(calendar_name) if calendar_name.strip() else None
    ev_ref = event_reference(uid, cal_ref)

    date_lines = ""
    set_lines: list[str] = []
    if summary is not None:
        set_lines.append(f'\tset summary of ev to "{escape_applescript_string(summary)}"')
    if location is not None:
        set_lines.append(f'\tset location of ev to "{escape_applescript_string(location)}"')
    if description is not None:
        set_lines.append(f'\tset description of ev to "{escape_applescript_string(description)}"')
    if all_day is not None:
        set_lines.append(f"\tset allday event of ev to {'true' if all_day else 'false'}")
    if start_date is not None:
        comps = parse_iso_datetime(start_date)
        if comps is None:
            return {"ok": False, "stderr": f"could not parse start_date={start_date!r}"}
        date_lines += applescript_date_block("startDate", comps)
        set_lines.append("\tset start date of ev to startDate")
    if end_date is not None:
        comps = parse_iso_datetime(end_date)
        if comps is None:
            return {"ok": False, "stderr": f"could not parse end_date={end_date!r}"}
        date_lines += applescript_date_block("endDate", comps)
        set_lines.append("\tset end date of ev to endDate")

    script = (
        f"{date_lines}tell application \"Calendar\"\n"
        f"\tset ev to {ev_ref}\n"
        + "\n".join(set_lines)
        + "\nend tell"
    )
    res = await _run_osascript(script)
    if not res["ok"]:
        return {"ok": False, "stderr": res["stderr"] or f"failed to update event uid={uid!r}"}
    return {"ok": True, "stderr": ""}


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


@mcp.tool()
async def delete_event(uid: str, calendar_name: str = "") -> dict[str, Any]:
    """Delete an event permanently.

    Args:
        uid: The event's ``uid`` (from ``list_events``).
        calendar_name: The calendar it lives in (see ``list_calendars``).
            Empty string searches every calendar.

    Returns:
        dict with ``ok`` and ``stderr`` on failure.
    """
    if not uid or not uid.strip():
        return {"ok": False, "stderr": "uid must be a non-empty string"}
    cal_ref = calendar_reference(calendar_name) if calendar_name.strip() else None
    ev_ref = event_reference(uid, cal_ref)
    script = f'tell application "Calendar" to delete {ev_ref}'
    res = await _run_osascript(script)
    if not res["ok"]:
        return {"ok": False, "stderr": res["stderr"] or f"failed to delete event uid={uid!r}"}
    return {"ok": True, "stderr": ""}


if __name__ == "__main__":
    mcp.run()
