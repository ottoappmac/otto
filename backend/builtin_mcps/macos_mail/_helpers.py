"""Pure-Python helpers for the macos-mail MCP.

Lives in a separate module — free of the ``@mcp.tool()`` decorator — so
unit tests can exercise the logic without needing FastMCP's parameter
introspection (which the parent process's test env doesn't always agree
with; see :mod:`backend.builtin_mcps.macos_osascript._helpers` for the
same rationale). At runtime this MCP runs in its own uv-provisioned venv
where the decorator works fine.

Three responsibilities:

* Building the AppleScript *object references* Mail's dictionary expects
  (:func:`mailbox_reference`, :func:`message_reference`) so :mod:`server`
  never hand-rolls a reference string inline.
* Building the free-text search predicate (:func:`keyword_whose_clause`,
  :func:`applescript_date_literal`).
* Parsing the delimited text that the generated AppleScript emits back
  (:func:`parse_records`, :func:`parse_single_record`,
  :func:`parse_attachments`) and resolving attachment paths through the
  same virtual-session-path convention as every other file-touching tool
  in this codebase (:func:`resolve_attachment_path`).

Delimiter scheme
-----------------
AppleScript has no JSON encoder, so list/record results are serialized
as plain text using four ASCII separator control characters — chosen
specifically because they can never appear in real mail content and
require no escaping on either side:

* ``RECORD_SEP`` (0x1E) — between whole records in a list result.
* ``FIELD_SEP``  (0x1F) — between fields within one record.
* ``ATTACH_SEP`` (0x1D) — between attachment entries within the
  attachments sub-field of ``get_message``.
* ``ATTACH_FIELD_SEP`` (0x1C) — between the name/size/downloaded fields
  of one attachment entry.

Because parsing is strictly hierarchical (split the outer text first,
then split each isolated substring independently), reusing a character
at two different *levels* of nesting never collides.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

RECORD_SEP = "\x1e"
FIELD_SEP = "\x1f"
ATTACH_SEP = "\x1d"
ATTACH_FIELD_SEP = "\x1c"


# AppleScript handlers embedded verbatim at the top of every generated
# script. ``safeText`` guards against ``missing value`` — which raises a
# hard AppleScript error ("can't make missing value into type reference")
# the moment it's concatenated with ``&`` — so callers can freely build
# delimited output without per-property try/on error noise. ``joinList``
# is the same idea for a handful of list-valued properties (e.g. an
# account's ``email addresses``). ``resolveAccountName`` lets every
# ``account_name`` argument accept either an account's display name OR
# one of its email addresses — a very natural mistake to guess wrong
# the other way (some accounts happen to have their email address AS
# their display name, which makes it easy to assume that's always true).
APPLESCRIPT_HANDLERS = """\
on safeText(v)
\ttry
\t\tif v is missing value then return ""
\t\treturn v as string
\ton error
\t\treturn ""
\tend try
end safeText

on joinList(lst, sep)
\tset out to ""
\tset isFirst to true
\trepeat with itm in lst
\t\tif isFirst then
\t\t\tset isFirst to false
\t\telse
\t\t\tset out to out & sep
\t\tend if
\t\tset out to out & my safeText(itm)
\tend repeat
\treturn out
end joinList

on resolveAccountName(nameOrEmail)
\ttell application "Mail"
\t\trepeat with acct in every account
\t\t\tif (name of acct) is nameOrEmail then return name of acct
\t\t\ttry
\t\t\t\tif nameOrEmail is in (email addresses of acct) then return name of acct
\t\t\tend try
\t\tend repeat
\tend tell
\treturn nameOrEmail
end resolveAccountName
"""


# Case-insensitive aliases for Mail's top-level "special" mailboxes.
# These only resolve to an application-level property (``inbox``,
# ``sent mailbox``, …), which spans every account's copy of that special
# mailbox — they are NOT valid once an ``account_name`` is given, since
# Mail also exposes a same-named mailbox *inside* each account (e.g.
# ``mailbox "INBOX" of account "X"`` for an IMAP account) which callers
# reach through the ``account_name`` branch instead.
_MAILBOX_ALIASES: dict[str, str] = {
    "inbox": "inbox",
    "sent": "sent mailbox",
    "sent mailbox": "sent mailbox",
    "drafts": "drafts mailbox",
    "draft": "drafts mailbox",
    "drafts mailbox": "drafts mailbox",
    "trash": "trash mailbox",
    "deleted": "trash mailbox",
    "deleted messages": "trash mailbox",
    "trash mailbox": "trash mailbox",
    "junk": "junk mailbox",
    "spam": "junk mailbox",
    "junk mailbox": "junk mailbox",
    "outbox": "outbox",
}


def escape_applescript_string(value: str) -> str:
    """Escape a Python string for embedding in an AppleScript string literal."""
    return (value or "").replace("\\", "\\\\").replace('"', '\\"')


def mailbox_reference(mailbox_name: str, account_name: str = "") -> str:
    """Build the AppleScript object-reference expression for a mailbox.

    Resolution order:

    1. ``account_name`` given → ``mailbox "<name>" of account (my
       resolveAccountName("<account>"))`` (works for both the account's
       special mailboxes, e.g. an IMAP account's own ``"INBOX"``, and
       any user-created folder). Routing through the ``resolveAccountName``
       handler (see ``APPLESCRIPT_HANDLERS``) means ``account_name`` can
       be either the account's display name OR one of its email
       addresses — Mail's ``account "…"`` reference only matches the
       display name, but it's an easy mistake to pass an email address
       instead (especially since some accounts' display name IS their
       email address, making that assumption seem safe until it isn't).
    2. Otherwise, a case-insensitive alias table maps common special
       names (``inbox``, ``sent``, ``drafts``, ``trash``/``deleted``,
       ``junk``/``spam``, ``outbox``) to the application-level property
       that spans every account's copy of that mailbox.
    3. Otherwise, falls back to the application-level ``mailbox "<name>"``
       element, which searches every account's mailboxes by name — works
       when the name happens to be unique across accounts.
    """
    name = (mailbox_name or "").strip()
    account = (account_name or "").strip()
    if account:
        return (
            f'mailbox "{escape_applescript_string(name)}" '
            f'of account (my resolveAccountName("{escape_applescript_string(account)}"))'
        )
    alias = _MAILBOX_ALIASES.get(name.lower())
    if alias:
        return alias
    return f'mailbox "{escape_applescript_string(name)}"'


def message_reference(message_id: int, mailbox_ref: str) -> str:
    """Build a message reference scoped to one mailbox by numeric id.

    Uses AppleScript's ``whose`` form (``first message of <mailboxRef>
    whose id is <N>``) rather than the id-form reference (``message id
    <N> of <mailboxRef>``) the ``message`` class's sdef code
    (``"ID  "``) would suggest — confirmed against a real Mail install,
    the id-form only parses when its container is a bare literal
    property (e.g. ``message id 5 of inbox``); it's a hard syntax error
    (``-2741``) the moment the container is any compound expression,
    which is exactly what :func:`mailbox_reference` builds for every
    case except the handful of app-level aliases. ``whose id is`` works
    uniformly regardless of how the mailbox was resolved.
    """
    return f"(first message of {mailbox_ref} whose id is {int(message_id)})"


def keyword_whose_clause(keyword: str, include_body: bool = False) -> str:
    """Build the ``whose`` predicate matching subject/sender (and, optionally, body content).

    A single shared builder so escaping only needs to be right once.
    ``subject``/``sender`` are cached message metadata — Mail answers a
    ``whose`` filter on them without touching the message body at all,
    so this half of the clause is always cheap regardless of mailbox
    size. Matching against ``content`` is a different story: it forces
    Mail to fetch/decode *every candidate message's full body* to
    evaluate the clause, one at a time, which is the dominant cost on
    a large/IMAP mailbox and the direct cause of ``search_messages``
    timeouts. ``include_body`` lets callers opt out of that expensive
    half entirely — the sane default, since most keyword searches
    (a name, a company, a topic) show up in the subject or sender
    anyway.
    """
    esc = escape_applescript_string(keyword)
    clause = f'(subject contains "{esc}") or (sender contains "{esc}")'
    if include_body:
        clause += f' or (content contains "{esc}")'
    return clause


def applescript_date_literal(days_back: int) -> str:
    """AppleScript expression for "``days_back`` days ago", or ``""`` when unbounded.

    ``0`` (or any non-positive value) means "no lower bound" — callers
    should skip adding a date clause entirely rather than embed the
    empty string.
    """
    try:
        days_back = int(days_back)
    except (TypeError, ValueError):
        days_back = 0
    if days_back <= 0:
        return ""
    return f"((current date) - ({days_back} * days))"


def clamp_limit(n: int, cap: int = 200) -> int:
    """Bound a caller-supplied ``limit`` to ``[1, cap]``."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 1
    return max(1, min(n, cap))


def resolve_attachment_path(path: str, session_files: Optional[str] = None) -> Optional[Path]:
    """Resolve an attachment path to a real host filesystem path.

    Agents write files through a virtual filesystem where ``/`` maps to
    the session sandbox (``$SESSION_FILES``). ``osascript`` runs on the
    host, so a virtual path like ``/output/report.pdf`` must be remapped
    to ``$SESSION_FILES/output/report.pdf`` before it's embedded in a
    ``POSIX file "…"`` literal — mirrors
    ``macos_osascript/server.py:_resolve_script_path``.

    Returns ``None`` when no candidate exists so the caller can surface
    a clear "attachment not found" error *before* running any AppleScript
    (rather than partway through composing a message).
    """
    if not path:
        return None
    expanded = Path(os.path.expanduser(path)).resolve()
    if expanded.is_file():
        return expanded
    root = session_files if session_files is not None else os.environ.get("SESSION_FILES", "")
    if root:
        candidate = Path(root) / path.lstrip("/")
        if candidate.is_file():
            return candidate
        candidate2 = Path(root + path)
        if candidate2.is_file():
            return candidate2
    return None


def parse_records(text: str, fields: list[str]) -> list[dict[str, str]]:
    """Parse ``RECORD_SEP``/``FIELD_SEP``-delimited AppleScript output.

    ``osascript`` appends a trailing newline after a returned ``text``
    value, which would otherwise show up as a bogus trailing record —
    stripped up front. Records are emitted with a trailing ``RECORD_SEP``
    by every generated script, so a final empty segment is expected and
    dropped; short records (fewer values than ``fields``) pad with ``""``
    rather than raising, since a well-formed generator never emits one
    but a defensive parse is cheap.
    """
    if not text:
        return []
    text = text.rstrip("\n")
    records: list[dict[str, str]] = []
    for raw_record in text.split(RECORD_SEP):
        if not raw_record:
            continue
        values = raw_record.split(FIELD_SEP)
        records.append({
            field: (values[i] if i < len(values) else "")
            for i, field in enumerate(fields)
        })
    return records


def parse_single_record(text: str, fields: list[str]) -> dict[str, str]:
    """Parse one ``FIELD_SEP``-delimited AppleScript record (no ``RECORD_SEP``)."""
    if not text:
        return {field: "" for field in fields}
    values = text.rstrip("\n").split(FIELD_SEP)
    return {
        field: (values[i] if i < len(values) else "")
        for i, field in enumerate(fields)
    }


def parse_attachments(text: str) -> list[dict[str, str]]:
    """Parse the ``ATTACH_SEP``/``ATTACH_FIELD_SEP``-delimited attachments blob.

    Each entry is ``name``/``file_size``/``downloaded``. Used to decode
    the attachments sub-field embedded in one ``get_message`` record.
    """
    if not text:
        return []
    entries: list[dict[str, str]] = []
    for raw_entry in text.split(ATTACH_SEP):
        if not raw_entry:
            continue
        parts = raw_entry.split(ATTACH_FIELD_SEP)
        entries.append({
            "name": parts[0] if len(parts) > 0 else "",
            "file_size": parts[1] if len(parts) > 1 else "",
            "downloaded": parts[2] if len(parts) > 2 else "",
        })
    return entries
