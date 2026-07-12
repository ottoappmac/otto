#!/usr/bin/env python3
"""Built-in MCP server: macOS Reminders.

Structured Create/Read/Update/Delete tools over Apple Reminders,
implemented by generating AppleScript against Reminders' own dictionary
and running it via ``osascript``. Unlike ``macos-osascript``'s generic
``run_osascript``, every tool here takes typed arguments and builds the
correct AppleScript internally — the agent never has to author raw
AppleScript, guess the priority integer scale, or hand-build a
locale-safe ``date`` object.

Tools:

* Read:   ``list_lists``, ``list_reminders``, ``get_reminder``.
* Create: ``create_reminder``.
* Update: ``update_reminder`` (rename, re-note, re-date, re-prioritise,
  complete/uncomplete).
* Delete: ``delete_reminder``.

Trust boundaries:

* Every script here is a background Apple Event — none of them
  ``activate`` Reminders or synthesize keyboard/mouse input, so unlike
  ``macos-osascript`` this server never needs the cross-process desktop
  lease; any number of calls can run concurrently.
* macOS gates Apple Events behind a per-app **Automation** TCC prompt
  the user must approve (System Settings → Privacy & Security →
  Automation → grant the calling app access to Reminders). No Full Disk
  Access is required.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

try:
    from ._helpers import (  # type: ignore[import-not-found]
        APPLESCRIPT_HANDLERS,
        FIELD_SEP,
        RECORD_SEP,
        applescript_date_block,
        clamp_limit,
        escape_applescript_string,
        int_to_priority,
        list_reference,
        parse_iso_datetime,
        parse_records,
        parse_single_record,
        priority_to_int,
        reminder_reference,
    )
except ImportError:
    from _helpers import (  # type: ignore[no-redef]
        APPLESCRIPT_HANDLERS,
        FIELD_SEP,
        RECORD_SEP,
        applescript_date_block,
        clamp_limit,
        escape_applescript_string,
        int_to_priority,
        list_reference,
        parse_iso_datetime,
        parse_records,
        parse_single_record,
        priority_to_int,
        reminder_reference,
    )

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("otto.mcp.macos_reminders")

_MAX_TIMEOUT_SECS = 120
_DEFAULT_TIMEOUT_SECS = 30
_MAX_OUTPUT_BYTES = 64 * 1024

mcp = FastMCP("macOS Reminders")


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
    single argv entry — no shell metachar escaping and no risk of
    chaining unrelated commands. The body is wrapped in AppleScript's own
    ``with timeout of`` block, set a few seconds shorter than the outer
    subprocess timeout, so a hung Apple Event raises a fast, clean script
    error instead of silently eating the whole wall-clock budget.
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
async def list_lists() -> dict[str, Any]:
    """List every reminder list (and the count of reminders in each).

    Use this to discover the exact ``list_name`` strings the other tools
    expect (``list_reminders``, ``create_reminder``, …).

    Returns:
        dict with ``ok``, ``lists`` (list of dicts with ``name``, ``id``,
        ``reminder_count``), and ``stderr`` on failure.
    """
    fields = ["name", "id", "reminder_count"]
    script = f"""
set out to ""
tell application "Reminders"
\trepeat with lst in every list
\t\tset out to out & my safeText(name of lst) & "{FIELD_SEP}" & my safeText(id of lst) & "{FIELD_SEP}" & my safeText(count of reminders of lst) & "{RECORD_SEP}"
\tend repeat
end tell
return out
"""
    res = await _run_osascript(script)
    if not res["ok"]:
        return {"ok": False, "lists": [], "stderr": res["stderr"] or "failed to list reminder lists"}
    records = parse_records(res["stdout"], fields)
    lists = [
        {
            "name": r["name"],
            "id": r["id"],
            "reminder_count": int(r["reminder_count"] or 0),
        }
        for r in records
    ]
    return {"ok": True, "lists": lists, "stderr": ""}


@mcp.tool()
async def list_reminders(
    list_name: str = "",
    include_completed: bool = False,
    limit: int = 100,
) -> dict[str, Any]:
    """List reminders in one list (incomplete only, by default).

    Args:
        list_name: The list to read (see ``list_lists``). Empty string
            (default) uses the account's ``default list``.
        include_completed: If ``True``, also return completed reminders.
            Defaults to ``False`` (the common "what's still open?" case).
        limit: Max reminders to return (1-200).

    Returns:
        dict with ``ok``, ``reminders`` (list of dicts with ``id``,
        ``name``, ``body``, ``completed``, ``due_date``, ``priority``,
        ``list_name``), and ``stderr``. Use ``id`` with ``get_reminder``/
        ``update_reminder``/``delete_reminder`` (pass the same
        ``list_name`` back).
    """
    limit = clamp_limit(limit)
    list_ref = list_reference(list_name)
    if include_completed:
        col_expr = f"reminders of {list_ref}"
    else:
        col_expr = f"(reminders of {list_ref} whose completed is false)"

    fields = ["id", "name", "body", "completed", "due_date", "priority"]
    script = f"""
set out to ""
tell application "Reminders"
\tset theList to {list_ref}
\tset listName to my safeText(name of theList)
\tset rems to {col_expr}
\tset n to (count of rems)
\tif n > {limit} then set n to {limit}
\trepeat with i from 1 to n
\t\tset r to item i of rems
\t\tset out to out & my safeText(id of r) & "{FIELD_SEP}" & my safeText(name of r) & "{FIELD_SEP}" & my safeText(body of r) & "{FIELD_SEP}" & (completed of r as string) & "{FIELD_SEP}" & my safeText(due date of r) & "{FIELD_SEP}" & my safeText(priority of r) & "{RECORD_SEP}"
\tend repeat
end tell
return out
"""
    res = await _run_osascript(script)
    if not res["ok"]:
        return {
            "ok": False, "reminders": [],
            "stderr": res["stderr"] or f"failed to list reminders in list_name={list_name!r}",
        }
    records = parse_records(res["stdout"], fields)
    resolved_list = list_name.strip()
    reminders = [
        {
            "id": r["id"],
            "name": r["name"],
            "body": r["body"],
            "completed": r["completed"] == "true",
            "due_date": r["due_date"],
            "priority": int_to_priority(r["priority"]),
            "list_name": resolved_list,
        }
        for r in records
        if r["id"]
    ]
    return {"ok": True, "reminders": reminders, "stderr": ""}


@mcp.tool()
async def get_reminder(reminder_id: str, list_name: str = "") -> dict[str, Any]:
    """Fetch one reminder's full detail by id.

    Args:
        reminder_id: The reminder's ``id`` (from ``list_reminders``).
        list_name: The list the reminder lives in (see ``list_lists``).
            Empty string searches every reminder in the app — slower, but
            works when you don't know the list. Pass it whenever you can.

    Returns:
        dict with ``ok``, ``id``, ``name``, ``body``, ``completed``,
        ``due_date``, ``remind_me_date``, ``priority``, ``creation_date``,
        ``modification_date``, ``list_name``, and ``stderr``.
    """
    if not reminder_id or not reminder_id.strip():
        return {"ok": False, "stderr": "reminder_id must be a non-empty string"}

    list_ref = list_reference(list_name) if list_name.strip() else None
    rem_ref = reminder_reference(reminder_id, list_ref)
    fields = [
        "id", "name", "body", "completed", "due_date", "remind_me_date",
        "priority", "creation_date", "modification_date", "container",
    ]
    script = f"""
tell application "Reminders"
\tset r to {rem_ref}
\tset out to my safeText(id of r) & "{FIELD_SEP}" & my safeText(name of r) & "{FIELD_SEP}" & my safeText(body of r) & "{FIELD_SEP}" & (completed of r as string) & "{FIELD_SEP}" & my safeText(due date of r) & "{FIELD_SEP}" & my safeText(remind me date of r) & "{FIELD_SEP}" & my safeText(priority of r) & "{FIELD_SEP}" & my safeText(creation date of r) & "{FIELD_SEP}" & my safeText(modification date of r) & "{FIELD_SEP}" & my safeText(name of container of r)
end tell
return out
"""
    res = await _run_osascript(script)
    if not res["ok"]:
        return {
            "ok": False,
            "stderr": res["stderr"] or f"reminder {reminder_id!r} not found (list_name={list_name!r})",
        }
    record = parse_single_record(res["stdout"], fields)
    return {
        "ok": True,
        "id": record["id"],
        "name": record["name"],
        "body": record["body"],
        "completed": record["completed"] == "true",
        "due_date": record["due_date"],
        "remind_me_date": record["remind_me_date"],
        "priority": int_to_priority(record["priority"]),
        "creation_date": record["creation_date"],
        "modification_date": record["modification_date"],
        "list_name": record["container"],
        "stderr": "",
    }


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


@mcp.tool()
async def create_reminder(
    name: str,
    list_name: str = "",
    body: str = "",
    due_date: str = "",
    remind_me_date: str = "",
    priority: str = "none",
) -> dict[str, Any]:
    """Create a new reminder.

    Args:
        name: The reminder title (required).
        list_name: Which list to add it to (see ``list_lists``). Empty
            string (default) uses the account's ``default list``.
        body: Optional notes attached to the reminder.
        due_date: Optional due date/time as an ISO-ish string —
            ``"2026-07-10"``, ``"2026-07-10 14:30"``, or
            ``"2026-07-10T14:30:00"``. Parsed in Python (locale-safe).
            When given and ``remind_me_date`` is empty, the same time is
            also used as the alert time so Reminders actually notifies.
        remind_me_date: Optional explicit alert time (same formats). Set
            this to schedule an alert at a different time than the due
            date.
        priority: ``"none"`` (default), ``"low"``, ``"medium"``, or
            ``"high"``.

    Returns:
        dict with ``ok``, ``id`` (the new reminder's id), and ``stderr``.
    """
    if not name or not name.strip():
        return {"ok": False, "stderr": "name must be a non-empty string"}

    prio = priority_to_int(priority)
    if prio is None:
        return {"ok": False, "stderr": f"priority must be none/low/medium/high; got {priority!r}"}

    props = [f'name:"{escape_applescript_string(name)}"']
    if body:
        props.append(f'body:"{escape_applescript_string(body)}"')
    if prio:
        props.append(f"priority:{prio}")

    date_lines = ""
    due_prop = ""
    if due_date.strip():
        comps = parse_iso_datetime(due_date)
        if comps is None:
            return {"ok": False, "stderr": f"could not parse due_date={due_date!r} (use e.g. 2026-07-10 14:30)"}
        date_lines += applescript_date_block("dueDate", comps)
        props.append("due date:dueDate")
        due_prop = "dueDate"

    remind_prop = ""
    if remind_me_date.strip():
        comps = parse_iso_datetime(remind_me_date)
        if comps is None:
            return {"ok": False, "stderr": f"could not parse remind_me_date={remind_me_date!r}"}
        date_lines += applescript_date_block("remindDate", comps)
        remind_prop = "remindDate"
    elif due_prop:
        # No explicit alert time but a due date was given — reuse it so
        # Reminders raises a notification (a due date alone doesn't alert).
        remind_prop = due_prop

    if remind_prop:
        props.append(f"remind me date:{remind_prop}")

    list_ref = list_reference(list_name)
    props_str = "{" + ", ".join(props) + "}"
    script = f"""
{date_lines}tell application "Reminders"
\tset newRem to make new reminder at end of reminders of {list_ref} with properties {props_str}
\treturn id of newRem
end tell
"""
    res = await _run_osascript(script)
    if not res["ok"]:
        return {"ok": False, "stderr": res["stderr"] or "failed to create reminder"}
    return {"ok": True, "id": res["stdout"].strip(), "stderr": ""}


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


@mcp.tool()
async def update_reminder(
    reminder_id: str,
    list_name: str = "",
    name: Optional[str] = None,
    body: Optional[str] = None,
    completed: Optional[bool] = None,
    due_date: Optional[str] = None,
    remind_me_date: Optional[str] = None,
    priority: Optional[str] = None,
) -> dict[str, Any]:
    """Update an existing reminder. Pass only the fields you want to change.

    All change arguments are independently optional; at least one must be
    given.

    Args:
        reminder_id: The reminder's ``id`` (from ``list_reminders``).
        list_name: The list it lives in (see ``list_lists``). Empty
            string searches every reminder in the app — pass it when you
            can to keep the lookup fast.
        name: New title. ``None`` leaves it unchanged.
        body: New notes. ``None`` leaves it unchanged.
        completed: ``True`` to complete, ``False`` to reopen. ``None``
            leaves it unchanged.
        due_date: New due date/time (ISO-ish string, same formats as
            ``create_reminder``). Empty string clears it; ``None`` leaves
            it unchanged.
        remind_me_date: New alert time. Empty string clears it; ``None``
            leaves it unchanged.
        priority: New priority (``none``/``low``/``medium``/``high``).
            ``None`` leaves it unchanged.

    Returns:
        dict with ``ok`` and ``stderr`` on failure.
    """
    if not reminder_id or not reminder_id.strip():
        return {"ok": False, "stderr": "reminder_id must be a non-empty string"}
    if all(v is None for v in (name, body, completed, due_date, remind_me_date, priority)):
        return {"ok": False, "stderr": "nothing to update — pass at least one field to change"}

    list_ref = list_reference(list_name) if list_name.strip() else None
    rem_ref = reminder_reference(reminder_id, list_ref)

    date_lines = ""
    set_lines: list[str] = []
    if name is not None:
        set_lines.append(f'\tset name of r to "{escape_applescript_string(name)}"')
    if body is not None:
        set_lines.append(f'\tset body of r to "{escape_applescript_string(body)}"')
    if completed is not None:
        set_lines.append(f"\tset completed of r to {'true' if completed else 'false'}")
    if priority is not None:
        prio = priority_to_int(priority)
        if prio is None:
            return {"ok": False, "stderr": f"priority must be none/low/medium/high; got {priority!r}"}
        set_lines.append(f"\tset priority of r to {prio}")
    if due_date is not None:
        if due_date.strip():
            comps = parse_iso_datetime(due_date)
            if comps is None:
                return {"ok": False, "stderr": f"could not parse due_date={due_date!r}"}
            date_lines += applescript_date_block("dueDate", comps)
            set_lines.append("\tset due date of r to dueDate")
        else:
            set_lines.append("\tset due date of r to missing value")
    if remind_me_date is not None:
        if remind_me_date.strip():
            comps = parse_iso_datetime(remind_me_date)
            if comps is None:
                return {"ok": False, "stderr": f"could not parse remind_me_date={remind_me_date!r}"}
            date_lines += applescript_date_block("remindDate", comps)
            set_lines.append("\tset remind me date of r to remindDate")
        else:
            set_lines.append("\tset remind me date of r to missing value")

    script = (
        f"{date_lines}tell application \"Reminders\"\n"
        f"\tset r to {rem_ref}\n"
        + "\n".join(set_lines)
        + "\nend tell"
    )
    res = await _run_osascript(script)
    if not res["ok"]:
        return {"ok": False, "stderr": res["stderr"] or f"failed to update reminder {reminder_id!r}"}
    return {"ok": True, "stderr": ""}


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


@mcp.tool()
async def delete_reminder(reminder_id: str, list_name: str = "") -> dict[str, Any]:
    """Delete a reminder permanently.

    Reminders has no Trash — deletion is immediate and irreversible, so
    prefer ``update_reminder(..., completed=True)`` when the intent is
    just to mark something done.

    Args:
        reminder_id: The reminder's ``id`` (from ``list_reminders``).
        list_name: The list it lives in (see ``list_lists``). Empty
            string searches every reminder in the app.

    Returns:
        dict with ``ok`` and ``stderr`` on failure.
    """
    if not reminder_id or not reminder_id.strip():
        return {"ok": False, "stderr": "reminder_id must be a non-empty string"}
    list_ref = list_reference(list_name) if list_name.strip() else None
    rem_ref = reminder_reference(reminder_id, list_ref)
    script = f'tell application "Reminders" to delete {rem_ref}'
    res = await _run_osascript(script)
    if not res["ok"]:
        return {"ok": False, "stderr": res["stderr"] or f"failed to delete reminder {reminder_id!r}"}
    return {"ok": True, "stderr": ""}


if __name__ == "__main__":
    mcp.run()
