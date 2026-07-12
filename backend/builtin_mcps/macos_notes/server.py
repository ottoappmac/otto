#!/usr/bin/env python3
"""Built-in MCP server: macOS Notes.

Structured Create/Read/Update/Delete tools over Apple Notes, implemented
by generating AppleScript against Notes' own dictionary and running it via
``osascript``. Unlike ``macos-osascript``'s generic ``run_osascript``,
every tool takes typed arguments and builds the correct AppleScript
internally — including converting a plain title + body into the HTML
``body`` Notes actually stores, so the agent never hand-writes markup.

Tools:

* Read:   ``list_folders``, ``list_notes``, ``get_note``, ``search_notes``.
* Create: ``create_note``.
* Update: ``update_note`` (retitle, replace body, or append text).
* Delete: ``delete_note``.

Trust boundaries:

* Every script here is a background Apple Event — none of them
  ``activate`` Notes or synthesize input, so this server never needs the
  cross-process desktop lease; calls can run concurrently.
* macOS gates Apple Events behind a per-app **Automation** TCC prompt the
  user must approve (System Settings → Privacy & Security → Automation →
  grant the calling app access to Notes). No Full Disk Access is required.
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
        build_note_html,
        clamp_limit,
        escape_applescript_string,
        keyword_whose_clause,
        lookup_container,
        note_container,
        note_reference,
        parse_records,
        parse_single_record,
    )
except ImportError:
    from _helpers import (  # type: ignore[no-redef]
        APPLESCRIPT_HANDLERS,
        FIELD_SEP,
        RECORD_SEP,
        build_note_html,
        clamp_limit,
        escape_applescript_string,
        keyword_whose_clause,
        lookup_container,
        note_container,
        note_reference,
        parse_records,
        parse_single_record,
    )

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("otto.mcp.macos_notes")

_MAX_TIMEOUT_SECS = 120
_DEFAULT_TIMEOUT_SECS = 30
_SEARCH_DEFAULT_TIMEOUT_SECS = 60
_MAX_OUTPUT_BYTES = 64 * 1024

mcp = FastMCP("macOS Notes")


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
# Read
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_folders(account_name: str = "") -> dict[str, Any]:
    """List note folders, optionally scoped to one account.

    Args:
        account_name: Restrict to one account's folders. Empty string
            (default) lists every folder across every account.

    Returns:
        dict with ``ok``, ``folders`` (list of dicts with ``name``,
        ``note_count``, ``account_name``), and ``stderr`` on failure.
    """
    fields = ["name", "note_count", "account_name"]
    if account_name.strip():
        esc = escape_applescript_string(account_name)
        loop = f"""
\tset acct to account "{esc}"
\tset acctName to my safeText(name of acct)
\trepeat with f in every folder of acct
\t\tset out to out & my safeText(name of f) & "{FIELD_SEP}" & my safeText(count of notes of f) & "{FIELD_SEP}" & acctName & "{RECORD_SEP}"
\tend repeat
"""
    else:
        loop = f"""
\trepeat with acct in every account
\t\tset acctName to my safeText(name of acct)
\t\trepeat with f in every folder of acct
\t\t\tset out to out & my safeText(name of f) & "{FIELD_SEP}" & my safeText(count of notes of f) & "{FIELD_SEP}" & acctName & "{RECORD_SEP}"
\t\tend repeat
\tend repeat
"""
    script = f"""
set out to ""
tell application "Notes"{loop}end tell
return out
"""
    res = await _run_osascript(script)
    if not res["ok"]:
        return {"ok": False, "folders": [], "stderr": res["stderr"] or "failed to list folders"}
    records = parse_records(res["stdout"], fields)
    folders = [
        {
            "name": r["name"],
            "note_count": int(r["note_count"] or 0),
            "account_name": r["account_name"],
        }
        for r in records
    ]
    return {"ok": True, "folders": folders, "stderr": ""}


@mcp.tool()
async def list_notes(
    folder_name: str = "",
    account_name: str = "",
    limit: int = 50,
) -> dict[str, Any]:
    """List notes (metadata only) in one folder or across all folders.

    Returns lightweight metadata — no note body, which would force Notes
    to materialise every note. Call ``get_note`` for a single note's
    content.

    Args:
        folder_name: Restrict to one folder (see ``list_folders``). Empty
            string (default) lists notes across every folder.
        account_name: Scopes ``folder_name`` to one account when the same
            folder name exists in multiple accounts.
        limit: Max notes to return (1-200).

    Returns:
        dict with ``ok``, ``notes`` (list of dicts with ``id``, ``name``,
        ``modification_date``, ``folder_name``), and ``stderr``. Use
        ``id`` with ``get_note``/``update_note``/``delete_note``.
    """
    limit = clamp_limit(limit)
    container = lookup_container(folder_name, account_name)
    col_expr = f"notes of {container}" if container else "every note"

    fields = ["id", "name", "modification_date", "folder_name"]
    script = f"""
set out to ""
tell application "Notes"
\tset theNotes to {col_expr}
\tset n to (count of theNotes)
\tif n > {limit} then set n to {limit}
\trepeat with i from 1 to n
\t\tset nt to item i of theNotes
\t\tset fName to ""
\t\ttry
\t\t\tset fName to name of container of nt
\t\tend try
\t\tset out to out & my safeText(id of nt) & "{FIELD_SEP}" & my safeText(name of nt) & "{FIELD_SEP}" & my safeText(modification date of nt) & "{FIELD_SEP}" & fName & "{RECORD_SEP}"
\tend repeat
end tell
return out
"""
    res = await _run_osascript(script)
    if not res["ok"]:
        return {
            "ok": False, "notes": [],
            "stderr": res["stderr"] or f"failed to list notes (folder_name={folder_name!r})",
        }
    records = parse_records(res["stdout"], fields)
    notes = [
        {
            "id": r["id"],
            "name": r["name"],
            "modification_date": r["modification_date"],
            "folder_name": r["folder_name"],
        }
        for r in records
        if r["id"]
    ]
    return {"ok": True, "notes": notes, "stderr": ""}


@mcp.tool()
async def get_note(
    note_id: str,
    folder_name: str = "",
    account_name: str = "",
) -> dict[str, Any]:
    """Fetch one note's full content by id.

    Args:
        note_id: The note's ``id`` (from ``list_notes``/``search_notes``).
        folder_name: The folder it lives in (see ``list_folders``). Empty
            string searches every note in the app — slower; pass it when
            you can.
        account_name: Scopes ``folder_name`` to one account.

    Returns:
        dict with ``ok``, ``id``, ``name``, ``plaintext`` (body as plain
        text), ``body`` (the raw HTML Notes stores), ``creation_date``,
        ``modification_date``, ``folder_name``, and ``stderr``.
    """
    if not note_id or not note_id.strip():
        return {"ok": False, "stderr": "note_id must be a non-empty string"}
    container = lookup_container(folder_name, account_name)
    ref = note_reference(note_id, container)
    fields = ["id", "name", "plaintext", "body", "creation_date", "modification_date", "folder_name"]
    script = f"""
tell application "Notes"
\tset nt to {ref}
\tset fName to ""
\ttry
\t\tset fName to name of container of nt
\tend try
\tset out to my safeText(id of nt) & "{FIELD_SEP}" & my safeText(name of nt) & "{FIELD_SEP}" & my safeText(plaintext of nt) & "{FIELD_SEP}" & my safeText(body of nt) & "{FIELD_SEP}" & my safeText(creation date of nt) & "{FIELD_SEP}" & my safeText(modification date of nt) & "{FIELD_SEP}" & fName
end tell
return out
"""
    res = await _run_osascript(script)
    if not res["ok"]:
        return {
            "ok": False,
            "stderr": res["stderr"] or f"note {note_id!r} not found (folder_name={folder_name!r})",
        }
    record = parse_single_record(res["stdout"], fields)
    return {
        "ok": True,
        "id": record["id"],
        "name": record["name"],
        "plaintext": record["plaintext"],
        "body": record["body"],
        "creation_date": record["creation_date"],
        "modification_date": record["modification_date"],
        "folder_name": record["folder_name"],
        "stderr": "",
    }


@mcp.tool()
async def search_notes(
    query: str,
    folder_name: str = "",
    account_name: str = "",
    search_body: bool = False,
    limit: int = 50,
    timeout_seconds: int = _SEARCH_DEFAULT_TIMEOUT_SECS,
) -> dict[str, Any]:
    """Search notes by keyword, matching the title by default.

    Call with the default ``search_body=False`` first (fast — matches the
    note name/title only) and re-run with ``search_body=True`` only if
    that comes back empty and the keyword might live in the body. Body
    search forces Notes to materialise every candidate note's plain text
    and is much slower.

    Args:
        query: Keyword to match (case-insensitive substring).
        folder_name: Restrict to one folder (fastest). Empty string
            searches across every folder.
        account_name: Scopes ``folder_name`` to one account.
        search_body: If ``True``, also match body text (slower).
        limit: Max matches to return (1-200).
        timeout_seconds: Wall-clock cap. Defaults to 60s; capped at 120s.

    Returns:
        dict with ``ok``, ``notes`` (list of dicts with ``id``, ``name``,
        ``modification_date``, ``folder_name``), and ``stderr``.
    """
    if not query or not query.strip():
        return {"ok": False, "notes": [], "stderr": "query must be a non-empty string"}
    limit = clamp_limit(limit)
    timeout = _clamp_timeout(timeout_seconds, default=_SEARCH_DEFAULT_TIMEOUT_SECS)
    container = lookup_container(folder_name, account_name)
    whose = keyword_whose_clause(query, include_body=search_body)
    base = f"notes of {container}" if container else "notes"
    col_expr = f"({base} whose {whose})"

    fields = ["id", "name", "modification_date", "folder_name"]
    script = f"""
set out to ""
tell application "Notes"
\tset matches to {{}}
\ttry
\t\tset matches to {col_expr}
\ton error
\t\tset matches to {{}}
\tend try
\tset n to (count of matches)
\tif n > {limit} then set n to {limit}
\trepeat with i from 1 to n
\t\tset nt to item i of matches
\t\tset fName to ""
\t\ttry
\t\t\tset fName to name of container of nt
\t\tend try
\t\tset out to out & my safeText(id of nt) & "{FIELD_SEP}" & my safeText(name of nt) & "{FIELD_SEP}" & my safeText(modification date of nt) & "{FIELD_SEP}" & fName & "{RECORD_SEP}"
\tend repeat
end tell
return out
"""
    res = await _run_osascript(script, timeout=timeout)
    if not res["ok"]:
        if res["timed_out"] or "1712" in res["stderr"] or "timed out" in res["stderr"].lower():
            hint = "drop search_body=True or " if search_body else ""
            stderr = (
                f"search timed out after {timeout}s — {hint}scope to a "
                "folder_name or raise timeout_seconds (up to 120)."
            )
        else:
            stderr = res["stderr"] or "search failed"
        return {"ok": False, "notes": [], "stderr": stderr}
    records = parse_records(res["stdout"], fields)
    notes = [
        {
            "id": r["id"],
            "name": r["name"],
            "modification_date": r["modification_date"],
            "folder_name": r["folder_name"],
        }
        for r in records
        if r["id"]
    ]
    return {"ok": True, "notes": notes, "stderr": ""}


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


@mcp.tool()
async def create_note(
    title: str,
    body: str = "",
    folder_name: str = "",
    account_name: str = "",
) -> dict[str, Any]:
    """Create a new note.

    Args:
        title: The note title — becomes the first line, which Notes styles
            as the title and uses as the note's ``name`` (required).
        body: Optional plain-text body. Newlines are preserved. You pass
            plain text; this tool builds the HTML Notes stores internally.
        folder_name: Which folder to create it in (see ``list_folders``).
            Empty string (default) uses the default account's default
            folder — where the Notes UI puts a new note.
        account_name: Scopes ``folder_name`` to one account.

    Returns:
        dict with ``ok``, ``id`` (the new note's id), and ``stderr``.
    """
    if not title or not title.strip():
        return {"ok": False, "stderr": "title must be a non-empty string"}
    html = build_note_html(title, body)
    esc_html = escape_applescript_string(html)
    container = note_container(folder_name, account_name)
    script = f"""
tell application "Notes"
\tset newNote to make new note at end of notes of {container} with properties {{body:"{esc_html}"}}
\treturn id of newNote
end tell
"""
    res = await _run_osascript(script)
    if not res["ok"]:
        return {"ok": False, "stderr": res["stderr"] or "failed to create note"}
    return {"ok": True, "id": res["stdout"].strip(), "stderr": ""}


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


@mcp.tool()
async def update_note(
    note_id: str,
    folder_name: str = "",
    account_name: str = "",
    title: Optional[str] = None,
    body: Optional[str] = None,
    append_text: Optional[str] = None,
) -> dict[str, Any]:
    """Update a note: replace its content, and/or append text to the end.

    Exactly one of the two content operations applies:

    * Pass ``title`` and/or ``body`` to REPLACE the note's whole content
      with a freshly built title + body. If you pass only one of them the
      other becomes empty, so pass both when replacing.
    * Pass ``append_text`` to add lines to the END of the existing note
      without touching what's already there.

    At least one of ``title``/``body``/``append_text`` must be given.
    ``append_text`` takes precedence if combined with the others.

    Args:
        note_id: The note's ``id`` (from ``list_notes``/``search_notes``).
        folder_name: The folder it lives in. Empty string searches every
            note — pass it when you can.
        account_name: Scopes ``folder_name`` to one account.
        title: New title (used when replacing).
        body: New plain-text body (used when replacing).
        append_text: Plain text to append to the end of the note.

    Returns:
        dict with ``ok`` and ``stderr`` on failure.
    """
    if not note_id or not note_id.strip():
        return {"ok": False, "stderr": "note_id must be a non-empty string"}
    if title is None and body is None and append_text is None:
        return {"ok": False, "stderr": "nothing to update — pass title/body (replace) or append_text"}

    container = lookup_container(folder_name, account_name)
    ref = note_reference(note_id, container)

    if append_text is not None:
        add_html = build_note_html("", append_text)
        esc_add = escape_applescript_string(add_html)
        set_line = f'\tset body of nt to (body of nt) & "{esc_add}"'
    else:
        html = build_note_html(title or "", body or "")
        esc_html = escape_applescript_string(html)
        set_line = f'\tset body of nt to "{esc_html}"'

    script = f"""
tell application "Notes"
\tset nt to {ref}
{set_line}
end tell
"""
    res = await _run_osascript(script)
    if not res["ok"]:
        return {"ok": False, "stderr": res["stderr"] or f"failed to update note {note_id!r}"}
    return {"ok": True, "stderr": ""}


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


@mcp.tool()
async def delete_note(
    note_id: str,
    folder_name: str = "",
    account_name: str = "",
) -> dict[str, Any]:
    """Delete a note (moves it to the account's "Recently Deleted" folder).

    Notes keeps deleted notes in "Recently Deleted" for ~30 days, so this
    is recoverable from the Notes UI within that window.

    Args:
        note_id: The note's ``id`` (from ``list_notes``/``search_notes``).
        folder_name: The folder it lives in. Empty string searches every
            note.
        account_name: Scopes ``folder_name`` to one account.

    Returns:
        dict with ``ok`` and ``stderr`` on failure.
    """
    if not note_id or not note_id.strip():
        return {"ok": False, "stderr": "note_id must be a non-empty string"}
    container = lookup_container(folder_name, account_name)
    ref = note_reference(note_id, container)
    script = f'tell application "Notes" to delete {ref}'
    res = await _run_osascript(script)
    if not res["ok"]:
        return {"ok": False, "stderr": res["stderr"] or f"failed to delete note {note_id!r}"}
    return {"ok": True, "stderr": ""}


if __name__ == "__main__":
    mcp.run()
