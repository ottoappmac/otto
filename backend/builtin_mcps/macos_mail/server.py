#!/usr/bin/env python3
"""Built-in MCP server: macOS Mail.

Structured Create/Read/Update/Delete tools over Apple Mail, implemented
by generating AppleScript against Mail's own dictionary (confirmed
against a real ``sdef /System/Applications/Mail.app`` dump) and running
it via ``osascript``. Unlike ``macos-osascript``'s generic ``run_osascript``,
every tool here takes typed arguments and builds the correct AppleScript
internally — the agent never has to author raw AppleScript or guess
Mail's quirky property names (``message id`` vs ``id``, etc.).

Nine tools:

* Read: ``list_accounts``, ``list_mailboxes``, ``list_messages``,
  ``get_message``, ``search_messages``.
* Create: ``send_message``, ``create_draft``.
* Update: ``update_message``.
* Delete: ``delete_message``.

Trust boundaries:

* Every script here is a background Apple Event — none of them
  ``activate`` Mail or synthesize keyboard/mouse input, so unlike
  ``macos-osascript`` this server never needs the cross-process desktop
  lease; any number of calls (from this agent or others) can run
  concurrently.
* macOS gates Apple Events behind a per-app **Automation** TCC prompt
  the user must explicitly approve (System Settings → Privacy &
  Security → Automation → grant the calling app access to Mail). No
  Full Disk Access is required — that's what differentiates this MCP
  from ``macos-osascript``'s SQLite-backed ``query_mail_store``.
* Attachment paths are resolved to real host paths in Python *before*
  any AppleScript runs (see ``_helpers.resolve_attachment_path``), so a
  missing file is reported as a clear error up front rather than a
  half-composed message.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

try:
    from ._helpers import (  # type: ignore[import-not-found]
        APPLESCRIPT_HANDLERS,
        ATTACH_FIELD_SEP,
        ATTACH_SEP,
        FIELD_SEP,
        RECORD_SEP,
        applescript_date_literal,
        clamp_limit,
        escape_applescript_string,
        keyword_whose_clause,
        mailbox_reference,
        message_reference,
        parse_attachments,
        parse_records,
        parse_single_record,
        resolve_attachment_path,
    )
except ImportError:
    from _helpers import (  # type: ignore[no-redef]
        APPLESCRIPT_HANDLERS,
        ATTACH_FIELD_SEP,
        ATTACH_SEP,
        FIELD_SEP,
        RECORD_SEP,
        applescript_date_literal,
        clamp_limit,
        escape_applescript_string,
        keyword_whose_clause,
        mailbox_reference,
        message_reference,
        parse_attachments,
        parse_records,
        parse_single_record,
        resolve_attachment_path,
    )

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("otto.mcp.macos_mail")

_MAX_TIMEOUT_SECS = 120
_DEFAULT_TIMEOUT_SECS = 30
_SEARCH_DEFAULT_TIMEOUT_SECS = 60
_MAX_OUTPUT_BYTES = 64 * 1024

mcp = FastMCP("macOS Mail")


def _truncate(blob: bytes) -> tuple[str, bool]:
    truncated = len(blob) > _MAX_OUTPUT_BYTES
    if truncated:
        blob = blob[:_MAX_OUTPUT_BYTES]
    try:
        return blob.decode("utf-8", errors="replace"), truncated
    except Exception:
        return blob.decode("latin-1", errors="replace"), truncated


def _bulk_fetch(var_name: str, prop_expr: str, collection_expr: str, item_var: str) -> str:
    """AppleScript snippet: fetch ``prop_expr`` for every item of
    ``collection_expr`` into ``var_name``, one Apple Event for the whole
    collection (`property of aList` evaluates to a list of that property
    for every item — the standard AppleScript performance idiom, and the
    difference between a handful of round trips and one per property PER
    item in a loop).

    Falls back to a slower per-item loop (each wrapped in its own
    ``try``, defaulting to ``""``) if the bulk fetch itself errors — which
    it does the moment ANY single item's property access fails, e.g. a
    stale/orphaned mailbox reference in an otherwise-healthy list
    (confirmed against a real multi-account Mail setup: one busted
    mailbox reference is enough to fail `name of <the whole list>`).
    This keeps the fast path fast while staying correct on real mailboxes
    that have a few rotten entries.
    """
    return f"""
\tset {var_name} to {{}}
\ttry
\t\tset {var_name} to {prop_expr} of {collection_expr}
\ton error
\t\trepeat with {item_var} in {collection_expr}
\t\t\tset oneVal to ""
\t\t\ttry
\t\t\t\tset oneVal to {prop_expr} of {item_var}
\t\t\tend try
\t\t\tset end of {var_name} to oneVal
\t\tend repeat
\tend try
"""


def _clamp_timeout(timeout: int, default: int = _DEFAULT_TIMEOUT_SECS) -> int:
    try:
        timeout = int(timeout)
    except (TypeError, ValueError):
        timeout = default
    if timeout <= 0:
        timeout = default
    return min(timeout, _MAX_TIMEOUT_SECS)


async def _run_osascript(script: str, timeout: int) -> dict[str, Any]:
    """Spawn ``osascript -e <script>`` and collect its result.

    ``create_subprocess_exec`` (no shell) delivers the full script as a
    single argv entry — no shell metachar escaping and no risk of
    chaining unrelated commands.

    The script body is wrapped in AppleScript's own ``with timeout of``
    block, set a few seconds shorter than the outer subprocess timeout.
    Mail can flat-out hang (rather than erroring) on certain malformed
    element lookups — e.g. ``mailbox "inbox" of account "X"`` when that
    account's inbox is actually named ``"INBOX"`` — confirmed against a
    real IMAP account, where the mismatch doesn't fail fast with -1728
    like a genuinely nonexistent account does, it just never returns.
    Without this, every such lookup silently eats the *entire* outer
    timeout and loses its error detail (the process gets killed, so
    there's no AppleScript-level stderr to report). The native
    ``with timeout of`` block raises a normal, fast (-1712) script
    error instead, which flows through to the caller like any other
    AppleScript failure.
    """
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


def _session_files() -> str:
    return os.environ.get("SESSION_FILES", "")


def _mailbox_error_hint(mailbox_name: str, account_name: str) -> str:
    return (
        f"mailbox={mailbox_name!r} account={account_name!r} — pass the exact "
        "name from list_mailboxes/list_accounts, or one of the special "
        "aliases (inbox/sent/drafts/trash/junk/outbox)."
    )


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_accounts() -> dict[str, Any]:
    """List every configured Mail account.

    Use this to discover the exact ``account_name`` strings other tools
    expect (``list_mailboxes``, ``list_messages``, ``send_message``, …).

    Returns:
        dict with ``ok``, ``accounts`` (list of dicts with ``name``,
        ``id``, ``account_type``, ``full_name``, ``email_addresses``
        (list of str)), and ``stderr`` on failure.
    """
    fields = ["name", "id", "account_type", "full_name", "emails"]
    script = f"""
set out to ""
tell application "Mail"
\trepeat with a in every account
\t\tset out to out & my safeText(name of a) & "{FIELD_SEP}" & my safeText(id of a) & "{FIELD_SEP}" & my safeText(account type of a) & "{FIELD_SEP}" & my safeText(full name of a) & "{FIELD_SEP}" & (my joinList(email addresses of a, ",")) & "{RECORD_SEP}"
\tend repeat
end tell
return out
"""
    res = await _run_osascript(script, timeout=_DEFAULT_TIMEOUT_SECS)
    if not res["ok"]:
        return {"ok": False, "accounts": [], "stderr": res["stderr"] or "failed to list accounts"}
    records = parse_records(res["stdout"], fields)
    accounts = [
        {
            "name": r["name"],
            "id": r["id"],
            "account_type": r["account_type"],
            "full_name": r["full_name"],
            "email_addresses": [e for e in r["emails"].split(",") if e],
        }
        for r in records
    ]
    return {"ok": True, "accounts": accounts, "stderr": ""}


@mcp.tool()
async def list_mailboxes(account_name: str = "") -> dict[str, Any]:
    """List mailboxes (including nested folders), optionally scoped to one account.

    Args:
        account_name: Restrict to one account's mailboxes. Empty string
            (default) lists every mailbox across every account. Accepts
            either the account's display name (see ``list_accounts``)
            or one of its email addresses.

    Returns:
        dict with ``ok``, ``mailboxes`` (list of dicts with ``name``,
        ``unread_count``, ``account_name``, ``parent_mailbox`` — the
        last is ``""`` for a top-level mailbox), and ``stderr``.
    """
    fields = ["name", "unread_count", "account_name", "parent"]
    # NOTE: Mail's application-level `mailbox` element (`every mailbox`)
    # does NOT span every account despite what the sdef's class hierarchy
    # suggests — confirmed against a real multi-account setup, it only
    # returns the local "On My Mac" mailboxes. Each account's own mailboxes
    # only show up via `every mailbox of account "…"`. So an app-wide
    # listing has to sweep BOTH explicitly: the local list, then each
    # account's own list. This also sidesteps `account of <local mailbox>`
    # entirely, which reliably errors ("-1728 can't make … into … text")
    # for local mailboxes — since we already know the account per branch,
    # there's never a need to ask a mailbox which account owns it.
    esc_account = escape_applescript_string(account_name) if account_name.strip() else ""
    if esc_account:
        local_block = ""
        accounts_block = f"""
\tset acctBoxes to every mailbox of account (my resolveAccountName("{esc_account}"))
\tset n1 to (count of acctBoxes)
\tif n1 > 0 then
{_bulk_fetch("nameList1", "name", "acctBoxes", "b1")}{_bulk_fetch("countList1", "unread count", "acctBoxes", "b1")}\t\trepeat with j from 1 to n1
\t\t\tset parentName to ""
\t\t\ttry
\t\t\t\tset parentName to name of (container of (item j of acctBoxes))
\t\t\tend try
\t\t\tset out to out & my safeText(item j of nameList1) & "{FIELD_SEP}" & my safeText(item j of countList1) & "{FIELD_SEP}" & "{esc_account}" & "{FIELD_SEP}" & parentName & "{RECORD_SEP}"
\t\tend repeat
\tend if
"""
    else:
        local_block = f"""
\tset localBoxes to every mailbox
\tset n0 to (count of localBoxes)
\tif n0 > 0 then
{_bulk_fetch("nameList0", "name", "localBoxes", "b0")}{_bulk_fetch("countList0", "unread count", "localBoxes", "b0")}\t\trepeat with i from 1 to n0
\t\t\tset parentName to ""
\t\t\ttry
\t\t\t\tset parentName to name of (container of (item i of localBoxes))
\t\t\tend try
\t\t\tset out to out & my safeText(item i of nameList0) & "{FIELD_SEP}" & my safeText(item i of countList0) & "{FIELD_SEP}" & "" & "{FIELD_SEP}" & parentName & "{RECORD_SEP}"
\t\tend repeat
\tend if
"""
        accounts_block = f"""
\trepeat with acct in every account
\t\tset acctName to name of acct
\t\tset acctBoxes to every mailbox of acct
\t\tset n1 to (count of acctBoxes)
\t\tif n1 > 0 then
{_bulk_fetch("nameList1", "name", "acctBoxes", "b1")}{_bulk_fetch("countList1", "unread count", "acctBoxes", "b1")}\t\t\trepeat with j from 1 to n1
\t\t\t\tset parentName to ""
\t\t\t\ttry
\t\t\t\t\tset parentName to name of (container of (item j of acctBoxes))
\t\t\t\tend try
\t\t\t\tset out to out & my safeText(item j of nameList1) & "{FIELD_SEP}" & my safeText(item j of countList1) & "{FIELD_SEP}" & acctName & "{FIELD_SEP}" & parentName & "{RECORD_SEP}"
\t\t\tend repeat
\t\tend if
\tend repeat
"""

    script = f"""
set out to ""
tell application "Mail"{local_block}{accounts_block}end tell
return out
"""
    res = await _run_osascript(script, timeout=_DEFAULT_TIMEOUT_SECS)
    if not res["ok"]:
        return {"ok": False, "mailboxes": [], "stderr": res["stderr"] or "failed to list mailboxes"}
    records = parse_records(res["stdout"], fields)
    mailboxes = [
        {
            "name": r["name"],
            "unread_count": int(r["unread_count"] or 0),
            "account_name": r["account_name"],
            "parent_mailbox": r["parent"],
        }
        for r in records
    ]
    return {"ok": True, "mailboxes": mailboxes, "stderr": ""}


@mcp.tool()
async def list_messages(
    mailbox_name: str = "inbox",
    account_name: str = "",
    limit: int = 50,
    unread_only: bool = False,
    flagged_only: bool = False,
) -> dict[str, Any]:
    """List the most recent messages in one mailbox.

    Args:
        mailbox_name: Mailbox to list. Accepts a special alias
            (``inbox``/``sent``/``drafts``/``trash``/``junk``/``outbox``),
            a folder name (searched across every account when
            ``account_name`` is empty), or an exact name scoped by
            ``account_name``.
        account_name: Scopes ``mailbox_name`` to one account. Empty
            string uses the alias/app-wide-search resolution instead.
            Accepts either the account's display name (see
            ``list_accounts``) or one of its email addresses.
        limit: Max messages to return (1-200). Returns the most
            recently added messages first is NOT guaranteed — Mail
            orders a mailbox's ``messages`` by internal insertion order,
            which is usually but not strictly chronological.
        unread_only: Only messages with ``read status`` false.
        flagged_only: Only flagged messages.

    Returns:
        dict with ``ok``, ``messages`` (list of dicts with ``id``,
        ``subject``, ``sender``, ``date_received``, ``read``,
        ``flagged``), and ``stderr``. Use ``id`` with ``get_message``/
        ``update_message``/``delete_message`` (pass the same
        ``mailbox_name``/``account_name`` back).
    """
    limit = clamp_limit(limit)
    mbox_ref = mailbox_reference(mailbox_name, account_name)

    where_parts = []
    if unread_only:
        where_parts.append("read status is false")
    if flagged_only:
        where_parts.append("flagged status is true")
    if where_parts:
        col_expr = f'(messages of {mbox_ref} whose {" and ".join(where_parts)})'
    else:
        col_expr = f"messages of {mbox_ref}"

    fields = ["id", "subject", "sender", "date_received", "read", "flagged"]
    # Bulk-fetch every property across the whole slice in one Apple Event
    # each (`property of aList` evaluates to a list of that property for
    # every item) instead of one Apple Event per property PER message in a
    # `repeat` loop — the difference between ~6 round trips total and
    # ~6 x limit round trips, which is the difference between sub-second and
    # multi-minute (or outright AppleEvent-timeout, -1712) on a real mailbox.
    script = f"""
set out to ""
tell application "Mail"
\tset allMsgs to {col_expr}
\tset n to (count of allMsgs)
\tset startIdx to 1
\tif n > {limit} then set startIdx to n - {limit} + 1
\tif n > 0 then
\t\tset msgs to items startIdx thru n of allMsgs
{_bulk_fetch("idList", "id", "msgs", "m1")}{_bulk_fetch("subjList", "subject", "msgs", "m1")}{_bulk_fetch("sendList", "sender", "msgs", "m1")}{_bulk_fetch("dateList", "date received", "msgs", "m1")}{_bulk_fetch("readList", "read status", "msgs", "m1")}{_bulk_fetch("flagList", "flagged status", "msgs", "m1")}\t\tset cnt to count of idList
\t\trepeat with i from 1 to cnt
\t\t\tset out to out & my safeText(item i of idList) & "{FIELD_SEP}" & my safeText(item i of subjList) & "{FIELD_SEP}" & my safeText(item i of sendList) & "{FIELD_SEP}" & my safeText(item i of dateList) & "{FIELD_SEP}" & my safeText(item i of readList) & "{FIELD_SEP}" & my safeText(item i of flagList) & "{RECORD_SEP}"
\t\tend repeat
\tend if
end tell
return out
"""
    res = await _run_osascript(script, timeout=_DEFAULT_TIMEOUT_SECS)
    if not res["ok"]:
        return {
            "ok": False, "messages": [],
            "stderr": res["stderr"] or f"failed to list messages ({_mailbox_error_hint(mailbox_name, account_name)})",
        }
    records = parse_records(res["stdout"], fields)
    messages = [
        {
            "id": int(r["id"]),
            "subject": r["subject"],
            "sender": r["sender"],
            "date_received": r["date_received"],
            "read": r["read"] == "true",
            "flagged": r["flagged"] == "true",
        }
        for r in records
        if r["id"]  # a per-item fetch failure (see _bulk_fetch) defaults id to "" — drop it rather than crash
    ]
    return {"ok": True, "messages": messages, "stderr": ""}


@mcp.tool()
async def get_message(
    mailbox_name: str,
    message_id: int,
    account_name: str = "",
) -> dict[str, Any]:
    """Fetch one message's full content, headers, and attachment list.

    Args:
        mailbox_name: The mailbox the message currently lives in (as
            returned by ``list_messages``/``search_messages``).
        message_id: The message's ``id`` (from ``list_messages`` /
            ``search_messages``).
        account_name: Account the mailbox belongs to, if ``mailbox_name``
            isn't one of the special aliases. Accepts either the
            account's display name (see ``list_accounts``) or one of
            its email addresses.

    Returns:
        dict with ``ok``, ``subject``, ``sender``, ``content``,
        ``date_received``, ``date_sent``, ``read``, ``flagged``,
        ``message_id`` (the RFC ``Message-ID`` header, distinct from
        the numeric ``id``), ``attachments`` (list of dicts with
        ``name``, ``file_size``, ``downloaded``), and ``stderr``.
    """
    mbox_ref = mailbox_reference(mailbox_name, account_name)
    msg_ref = message_reference(message_id, mbox_ref)

    fields = [
        "subject", "sender", "content", "date_received", "date_sent",
        "read", "flagged", "rfc_message_id", "attachments",
    ]
    script = f"""
tell application "Mail"
\tset msgRef to {msg_ref}
\tset attOut to ""
\tset isFirst to true
\trepeat with att in mail attachments of msgRef
\t\tif not isFirst then set attOut to attOut & "{ATTACH_SEP}"
\t\tset isFirst to false
\t\tset attOut to attOut & my safeText(name of att) & "{ATTACH_FIELD_SEP}" & my safeText(file size of att) & "{ATTACH_FIELD_SEP}" & (downloaded of att as string)
\tend repeat
\tset out to my safeText(subject of msgRef) & "{FIELD_SEP}" & my safeText(sender of msgRef) & "{FIELD_SEP}" & my safeText(content of msgRef) & "{FIELD_SEP}" & my safeText(date received of msgRef) & "{FIELD_SEP}" & my safeText(date sent of msgRef) & "{FIELD_SEP}" & (read status of msgRef as string) & "{FIELD_SEP}" & (flagged status of msgRef as string) & "{FIELD_SEP}" & my safeText(message id of msgRef) & "{FIELD_SEP}" & attOut
end tell
return out
"""
    res = await _run_osascript(script, timeout=_DEFAULT_TIMEOUT_SECS)
    if not res["ok"]:
        return {
            "ok": False,
            "stderr": res["stderr"] or (
                f"message {message_id} not found in {_mailbox_error_hint(mailbox_name, account_name)}"
            ),
        }
    record = parse_single_record(res["stdout"], fields)
    return {
        "ok": True,
        "subject": record["subject"],
        "sender": record["sender"],
        "content": record["content"],
        "date_received": record["date_received"],
        "date_sent": record["date_sent"],
        "read": record["read"] == "true",
        "flagged": record["flagged"] == "true",
        "message_id": record["rfc_message_id"],
        "attachments": parse_attachments(record["attachments"]),
        "stderr": "",
    }


@mcp.tool()
async def search_messages(
    query: str,
    account_name: str = "",
    mailbox_name: str = "",
    days_back: int = 0,
    limit: int = 50,
    search_body: bool = False,
    timeout_seconds: int = _SEARCH_DEFAULT_TIMEOUT_SECS,
) -> dict[str, Any]:
    """Search for messages by keyword, matching subject/sender by default.

    Recommended strategy for "does mail about X exist" questions: call
    this with the default ``search_body=False`` first (fast — subject/
    sender are cached metadata Mail can filter without touching any
    message body) and only re-run with ``search_body=True`` if that
    comes back empty and body text genuinely might be the only place
    the keyword appears. Body search is inherently much slower — Mail
    must fetch/decode every candidate message's full body to evaluate
    it — and is the single biggest cause of this tool timing out on
    large/IMAP mailboxes. Whichever mode you use, narrowing with
    ``mailbox_name``/``account_name``/``days_back`` (instead of a full
    unscoped sweep) helps a lot, and the ``macos-osascript`` MCP's
    ``query_mail_store`` tool (reads Mail's local SQLite index
    directly, no Apple Events at all) is far faster still for a
    subject/sender-only search — at the cost of requiring Full Disk
    Access and never matching body text.

    Args:
        query: Keyword to match (case-insensitive substring) against
            subject and sender, plus body content if ``search_body``.
        account_name: Restrict the sweep to one account's mailboxes.
            Empty string (default, with ``mailbox_name`` also empty)
            sweeps every mailbox across every account. Accepts either
            the account's display name (see ``list_accounts``) or one
            of its email addresses.
        mailbox_name: Restrict to exactly one mailbox (fastest). When
            given, ``account_name`` scopes which account's copy of that
            mailbox to use, same as ``list_messages``.
        days_back: Only messages received within this many days.
            ``0`` (default) means no lower bound.
        limit: Max matches to return (1-200), across the whole sweep.
        search_body: If ``True``, also match body content — thorough
            but much slower (forces Mail to decode every candidate
            message). Defaults to ``False`` (subject/sender only).
        timeout_seconds: Wall-clock cap. Defaults to 60s (higher than
            other tools since a full sweep can take a while); capped
            at 120s.

    Returns:
        dict with ``ok``, ``messages`` (list of dicts with ``id``,
        ``subject``, ``sender``, ``date_received``, ``read``,
        ``flagged``, ``mailbox_name``, ``account_name`` — the last two
        so each hit is directly addressable by ``get_message``/
        ``update_message``/``delete_message``), and ``stderr``.
    """
    if not query or not query.strip():
        return {"ok": False, "messages": [], "stderr": "query must be a non-empty string"}

    limit = clamp_limit(limit)
    timeout = _clamp_timeout(timeout_seconds, default=_SEARCH_DEFAULT_TIMEOUT_SECS)
    whose_clause = keyword_whose_clause(query, include_body=search_body)
    date_literal = applescript_date_literal(days_back)
    if date_literal:
        whose_clause = f"({whose_clause}) and (date received >= {date_literal})"

    def _scan_block(mailbox_expr: str, mbox_label_expr: str, acct_label_expr: str) -> str:
        # Shared "filter this one mailbox's messages, bulk-fetch every
        # property across the whole match list in one Apple Event each,
        # append to `out`" body — reused by all three scope shapes below.
        # Bulk-fetching (rather than one Apple Event per property PER
        # message in a loop) is the difference between a handful of round
        # trips and thousands on a real inbox.
        return f"""
\t\tset filtered to {{}}
\t\ttry
\t\t\tset filtered to (messages of {mailbox_expr} whose {whose_clause})
\t\ton error
\t\t\tset filtered to {{}}
\t\tend try
\t\tset mcnt to (count of filtered)
\t\tif mcnt > 0 then
{_bulk_fetch("idList", "id", "filtered", "m1")}{_bulk_fetch("subjList", "subject", "filtered", "m1")}{_bulk_fetch("sendList", "sender", "filtered", "m1")}{_bulk_fetch("dateList", "date received", "filtered", "m1")}{_bulk_fetch("readList", "read status", "filtered", "m1")}{_bulk_fetch("flagList", "flagged status", "filtered", "m1")}\t\t\trepeat with i from 1 to mcnt
\t\t\t\tset out to out & my safeText(item i of idList) & "{FIELD_SEP}" & my safeText(item i of subjList) & "{FIELD_SEP}" & my safeText(item i of sendList) & "{FIELD_SEP}" & my safeText(item i of dateList) & "{FIELD_SEP}" & my safeText(item i of readList) & "{FIELD_SEP}" & my safeText(item i of flagList) & "{FIELD_SEP}" & {mbox_label_expr} & "{FIELD_SEP}" & {acct_label_expr} & "{RECORD_SEP}"
\t\t\t\tset foundCount to foundCount + 1
\t\t\t\tif foundCount >= {limit} then exit repeat
\t\t\tend repeat
\t\tend if
"""

    esc_mailbox_label = escape_applescript_string(mailbox_name)
    esc_account_label = escape_applescript_string(account_name)

    # NOTE: same app-wide-sweep caveat as list_mailboxes — Mail's
    # application-level `every mailbox` only covers local "On My Mac"
    # mailboxes, not every account's own mailboxes, and `account of
    # <local mailbox>` reliably errors. So each scope shape below is
    # generated explicitly rather than through one shared collection
    # expression, and never asks a mailbox which account owns it — the
    # account is always already known from how we got to that mailbox.
    if mailbox_name.strip():
        # Exactly one mailbox — fastest path, no sweep at all.
        mbox_ref = mailbox_reference(mailbox_name, account_name)
        body = _scan_block(mbox_ref, f'"{esc_mailbox_label}"', f'"{esc_account_label}"')
        sweep = f"tell application \"Mail\"{body}end tell"
    elif account_name.strip():
        # Every mailbox belonging to one account.
        body = _scan_block("mb", "mbName", f'"{esc_account_label}"')
        sweep = f"""tell application "Mail"
\tset acctBoxes to every mailbox of account (my resolveAccountName("{esc_account_label}"))
\tset boxCount to (count of acctBoxes)
{_bulk_fetch("boxNames", "name", "acctBoxes", "b1")}\trepeat with bi from 1 to boxCount
\t\tset mb to item bi of acctBoxes
\t\tset mbName to item bi of boxNames
{body}\t\tif foundCount >= {limit} then exit repeat
\tend repeat
end tell"""
    else:
        # Full app-wide sweep: local ("On My Mac") mailboxes, then every
        # account's own mailboxes.
        local_body = _scan_block("mb", "mbName", '""')
        acct_body = _scan_block("mb", "mbName", "acctName")
        sweep = f"""tell application "Mail"
\tset localBoxes to every mailbox
\tset localCount to (count of localBoxes)
{_bulk_fetch("localNames", "name", "localBoxes", "b0")}\trepeat with bi from 1 to localCount
\t\tset mb to item bi of localBoxes
\t\tset mbName to item bi of localNames
{local_body}\t\tif foundCount >= {limit} then exit repeat
\tend repeat
\tif foundCount < {limit} then
\t\trepeat with acct in every account
\t\t\tset acctName to name of acct
\t\t\tset acctBoxes to every mailbox of acct
\t\t\tset acctBoxCount to (count of acctBoxes)
{_bulk_fetch("acctBoxNames", "name", "acctBoxes", "b1")}\t\t\trepeat with bj from 1 to acctBoxCount
\t\t\t\tset mb to item bj of acctBoxes
\t\t\t\tset mbName to item bj of acctBoxNames
{acct_body}\t\t\t\tif foundCount >= {limit} then exit repeat
\t\t\tend repeat
\t\t\tif foundCount >= {limit} then exit repeat
\t\tend repeat
\tend if
end tell"""

    fields = ["id", "subject", "sender", "date_received", "read", "flagged", "mailbox_name", "account_name"]
    script = f"""
set out to ""
set foundCount to 0
{sweep}
return out
"""
    res = await _run_osascript(script, timeout=timeout)
    if not res["ok"]:
        # Two different layers can report a timeout here: the Python-side
        # subprocess watchdog (`timed_out`, script got killed, no stderr
        # detail) and AppleScript's own `with timeout of` block firing
        # first and returning a clean "-1712" error as normal stderr. Both
        # mean the same thing to the caller, so both get the same
        # actionable guidance instead of a raw AppleEvent error code.
        if res["timed_out"] or "1712" in res["stderr"] or "timed out" in res["stderr"].lower():
            if search_body:
                scope = "body content, subject, and sender"
                hint = "drop search_body=True (body decoding is the expensive part) or "
            else:
                scope = "subject and sender"
                hint = ""
            stderr = (
                f"search timed out after {timeout}s scanning {scope} — "
                f"{hint}narrow with mailbox_name/account_name/days_back, "
                "raise timeout_seconds (up to 120), or use macos-osascript's "
                "query_mail_store for a subject/sender-only search of a large mailbox"
            )
        else:
            stderr = res["stderr"] or "search failed"
        return {"ok": False, "messages": [], "stderr": stderr}
    records = parse_records(res["stdout"], fields)
    messages = [
        {
            "id": int(r["id"]),
            "subject": r["subject"],
            "sender": r["sender"],
            "date_received": r["date_received"],
            "read": r["read"] == "true",
            "flagged": r["flagged"] == "true",
            "mailbox_name": r["mailbox_name"],
            "account_name": r["account_name"],
        }
        for r in records
        if r["id"]  # a per-item fetch failure (see _bulk_fetch) defaults id to "" — drop it rather than crash
    ]
    return {"ok": True, "messages": messages, "stderr": ""}


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


def _resolve_attachments(attachments: Optional[list[str]]) -> tuple[list[Path], str]:
    """Resolve every attachment path up front; returns (paths, error)."""
    if not attachments:
        return [], ""
    session_files = _session_files()
    resolved: list[Path] = []
    for raw in attachments:
        path = resolve_attachment_path(raw, session_files=session_files)
        if path is None:
            return [], f"attachment not found: {raw!r} (checked host path and $SESSION_FILES)"
        resolved.append(path)
    return resolved, ""


def _compose_script(
    *,
    to: list[str],
    subject: str,
    body: str,
    cc: Optional[list[str]],
    bcc: Optional[list[str]],
    attachment_paths: list[Path],
    account_name: str,
    final_command: str,
) -> str:
    esc_subject = escape_applescript_string(subject)
    esc_body = escape_applescript_string(body)
    lines = [
        "tell application \"Mail\"",
        f'\tset newMsg to make new outgoing message with properties '
        f'{{subject:"{esc_subject}", content:"{esc_body}", visible:false}}',
        "\ttell newMsg",
    ]
    for addr in to:
        lines.append(
            f'\t\tmake new to recipient at end of to recipients '
            f'with properties {{address:"{escape_applescript_string(addr)}"}}'
        )
    for addr in (cc or []):
        lines.append(
            f'\t\tmake new cc recipient at end of cc recipients '
            f'with properties {{address:"{escape_applescript_string(addr)}"}}'
        )
    for addr in (bcc or []):
        lines.append(
            f'\t\tmake new bcc recipient at end of bcc recipients '
            f'with properties {{address:"{escape_applescript_string(addr)}"}}'
        )
    for path in attachment_paths:
        posix = str(path).replace("\\", "\\\\").replace('"', '\\"')
        lines.append(
            f'\t\tmake new attachment with properties '
            f'{{file name:(POSIX file "{posix}")}} at after the last paragraph'
        )
    lines.append("\tend tell")
    if account_name.strip():
        # NOTE: `item 1 of email addresses` errors ("-1700 can't make …
        # into type specifier") when evaluated *inside* a `tell account
        # "…"` block — confirmed against a real Mail install; the same
        # property read works fine addressed directly, so that's what's
        # used here instead of nesting a `tell account` block.
        esc_account = escape_applescript_string(account_name)
        lines.extend([
            f'\tset acctRefName to (my resolveAccountName("{esc_account}"))',
            "\tset acctEmailList to email addresses of account acctRefName",
            "\tset acctEmail to \"\"",
            "\tif (count of acctEmailList) > 0 then set acctEmail to item 1 of acctEmailList",
            "\tset acctFull to full name of account acctRefName",
            '\tset sender of newMsg to (acctFull & " <" & acctEmail & ">")',
        ])
    lines.append(f"\t{final_command} newMsg")
    lines.append("end tell")
    return "\n".join(lines)


async def _send_or_draft(
    *,
    to: list[str],
    subject: str,
    body: str,
    cc: Optional[list[str]],
    bcc: Optional[list[str]],
    attachments: Optional[list[str]],
    account_name: str,
    final_command: str,
) -> dict[str, Any]:
    if not to or not any((addr or "").strip() for addr in to):
        return {"ok": False, "stderr": "to must contain at least one recipient address"}
    if not subject and not body:
        return {"ok": False, "stderr": "subject and body are both empty"}

    attachment_paths, err = _resolve_attachments(attachments)
    if err:
        return {"ok": False, "stderr": err}

    script = _compose_script(
        to=to, subject=subject, body=body, cc=cc, bcc=bcc,
        attachment_paths=attachment_paths, account_name=account_name,
        final_command=final_command,
    )
    res = await _run_osascript(script, timeout=_DEFAULT_TIMEOUT_SECS)
    if not res["ok"]:
        return {"ok": False, "stderr": res["stderr"] or f"failed to {final_command} message"}
    return {"ok": True, "stderr": ""}


@mcp.tool()
async def send_message(
    to: list[str],
    subject: str,
    body: str,
    cc: Optional[list[str]] = None,
    bcc: Optional[list[str]] = None,
    attachments: Optional[list[str]] = None,
    account_name: str = "",
) -> dict[str, Any]:
    """Compose and immediately send a new email.

    Args:
        to: Recipient email addresses (at least one required).
        subject: Message subject.
        body: Plain-text message body.
        cc: Optional CC addresses.
        bcc: Optional BCC addresses.
        attachments: Optional file paths to attach. Accepts real host
            paths or virtual session paths (e.g. ``/output/report.pdf``,
            auto-remapped to ``$SESSION_FILES/output/report.pdf``).
            Every path is checked to exist BEFORE anything is sent —
            if any is missing, no message is composed at all.
        account_name: Which account to send from — either the account's
            display name (``list_accounts``' ``name`` field) or one of
            its ``email_addresses``. Empty string uses Mail's own
            default/currently-selected account.

    Returns:
        dict with ``ok`` and ``stderr`` on failure.
    """
    return await _send_or_draft(
        to=to, subject=subject, body=body, cc=cc, bcc=bcc,
        attachments=attachments, account_name=account_name,
        final_command="send",
    )


@mcp.tool()
async def create_draft(
    to: list[str],
    subject: str,
    body: str,
    cc: Optional[list[str]] = None,
    bcc: Optional[list[str]] = None,
    attachments: Optional[list[str]] = None,
    account_name: str = "",
) -> dict[str, Any]:
    """Compose a new email and save it as a draft WITHOUT sending it.

    Same arguments as ``send_message``. The draft is saved into the
    composing account's Drafts mailbox (list it back with
    ``list_messages(mailbox_name="drafts", account_name=...)``).
    """
    return await _send_or_draft(
        to=to, subject=subject, body=body, cc=cc, bcc=bcc,
        attachments=attachments, account_name=account_name,
        final_command="save",
    )


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


@mcp.tool()
async def update_message(
    mailbox_name: str,
    message_id: int,
    account_name: str = "",
    read: Optional[bool] = None,
    flagged: Optional[bool] = None,
    move_to_mailbox: str = "",
    move_to_account: str = "",
) -> dict[str, Any]:
    """Update an existing message's read/flagged status and/or move it.

    All of ``read``, ``flagged``, and ``move_to_mailbox`` are
    independently optional — pass only what you want to change. At
    least one must be given.

    Args:
        mailbox_name: The mailbox the message currently lives in.
        message_id: The message's ``id`` (from ``list_messages`` /
            ``search_messages``).
        account_name: Account the mailbox belongs to, if ``mailbox_name``
            isn't one of the special aliases. Accepts either the
            account's display name (see ``list_accounts``) or one of
            its email addresses.
        read: Set to ``True``/``False`` to mark read/unread. ``None``
            (default) leaves it unchanged.
        flagged: Set to ``True``/``False`` to flag/unflag. ``None``
            (default) leaves it unchanged.
        move_to_mailbox: Destination mailbox name (e.g. ``"trash"`` to
            move to Trash, or any folder name). Empty string (default)
            leaves the message in place.
        move_to_account: Account for ``move_to_mailbox``, same
            resolution rules as ``account_name``.

    Returns:
        dict with ``ok`` and ``stderr`` on failure.
    """
    if read is None and flagged is None and not move_to_mailbox.strip():
        return {"ok": False, "stderr": "nothing to update — pass read, flagged, and/or move_to_mailbox"}

    mbox_ref = mailbox_reference(mailbox_name, account_name)
    msg_ref = message_reference(message_id, mbox_ref)

    lines = ["tell application \"Mail\"", f"\tset msgRef to {msg_ref}"]
    if read is not None:
        lines.append(f"\tset read status of msgRef to {'true' if read else 'false'}")
    if flagged is not None:
        lines.append(f"\tset flagged status of msgRef to {'true' if flagged else 'false'}")
    if move_to_mailbox.strip():
        dest_ref = mailbox_reference(move_to_mailbox, move_to_account)
        lines.append(f"\tmove msgRef to {dest_ref}")
    lines.append("end tell")
    script = "\n".join(lines)

    res = await _run_osascript(script, timeout=_DEFAULT_TIMEOUT_SECS)
    if not res["ok"]:
        return {
            "ok": False,
            "stderr": res["stderr"] or f"failed to update message {message_id}",
        }
    return {"ok": True, "stderr": ""}


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


@mcp.tool()
async def delete_message(
    mailbox_name: str,
    message_id: int,
    account_name: str = "",
) -> dict[str, Any]:
    """Delete a message (Mail's standard soft-delete — moves it to Trash).

    Mirrors the Mail UI's Delete key: the message moves to the owning
    account's Trash mailbox (honoring that account's own
    "move deleted messages to trash" setting). Permanently emptying
    Trash is out of scope for this tool — call this again with
    ``mailbox_name="trash"`` to permanently remove an already-trashed
    message, if the account is configured to delete-on-repeat.

    Args:
        mailbox_name: The mailbox the message currently lives in.
        message_id: The message's ``id`` (from ``list_messages`` /
            ``search_messages``).
        account_name: Account the mailbox belongs to, if ``mailbox_name``
            isn't one of the special aliases. Accepts either the
            account's display name (see ``list_accounts``) or one of
            its email addresses.

    Returns:
        dict with ``ok`` and ``stderr`` on failure.
    """
    mbox_ref = mailbox_reference(mailbox_name, account_name)
    msg_ref = message_reference(message_id, mbox_ref)
    script = f'tell application "Mail" to delete {msg_ref}'

    res = await _run_osascript(script, timeout=_DEFAULT_TIMEOUT_SECS)
    if not res["ok"]:
        return {
            "ok": False,
            "stderr": res["stderr"] or f"failed to delete message {message_id}",
        }
    return {"ok": True, "stderr": ""}


if __name__ == "__main__":
    mcp.run()
