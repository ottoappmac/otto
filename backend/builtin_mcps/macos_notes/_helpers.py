"""Pure-Python helpers for the macos-notes MCP.

Lives in a separate module — free of the ``@mcp.tool()`` decorator — so
unit tests can exercise the logic without needing FastMCP's parameter
introspection (mirrors :mod:`backend.builtin_mcps.macos_mail._helpers`).

Responsibilities:

* Building the AppleScript object references the Notes dictionary expects
  (:func:`note_container`, :func:`lookup_container`, :func:`note_reference`).
* Converting caller plaintext into the HTML ``body`` Notes stores
  (:func:`build_note_html`), so callers pass a title + plain text and
  never hand-write markup.
* Building the free-text search predicate (:func:`keyword_whose_clause`).
* Parsing the delimited text the generated AppleScript emits back
  (:func:`parse_records`, :func:`parse_single_record`).

Delimiter scheme
----------------
Two ASCII separator control characters that can never appear in a note's
title/body and need no escaping:

* ``RECORD_SEP`` (0x1E) — between whole records in a list result.
* ``FIELD_SEP``  (0x1F) — between fields within one record.
"""

from __future__ import annotations

from typing import Optional

RECORD_SEP = "\x1e"
FIELD_SEP = "\x1f"


# ``safeText`` guards against ``missing value`` (which raises a hard
# AppleScript error the moment it's concatenated with ``&``) so callers
# can build delimited output without per-property try/on error noise.
APPLESCRIPT_HANDLERS = """\
on safeText(v)
\ttry
\t\tif v is missing value then return ""
\t\treturn v as string
\ton error
\t\treturn ""
\tend try
end safeText
"""


def escape_applescript_string(value: str) -> str:
    """Escape a Python string for embedding in an AppleScript string literal."""
    return (value or "").replace("\\", "\\\\").replace('"', '\\"')


def clamp_limit(n: int, cap: int = 200) -> int:
    """Bound a caller-supplied ``limit`` to ``[1, cap]``."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 1
    return max(1, min(n, cap))


def html_escape(value: str) -> str:
    """Escape text for embedding inside a Notes HTML ``body``."""
    return (
        (value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def build_note_html(title: str, body: str) -> str:
    """Turn a plaintext ``title`` + ``body`` into the HTML Notes stores.

    Notes derives a note's ``name`` from the first line of its ``body``
    and auto-styles that first line as the title, so the title becomes the
    leading ``<div>`` and each subsequent body line becomes its own
    ``<div>`` (blank lines rendered as ``<div><br></div>`` to preserve
    spacing). All text is HTML-escaped first so ``<`` / ``&`` in the
    user's content can't break the markup.
    """
    parts: list[str] = []
    if title:
        parts.append(f"<div>{html_escape(title)}</div>")
    for line in (body or "").split("\n"):
        if line == "":
            parts.append("<div><br></div>")
        else:
            parts.append(f"<div>{html_escape(line)}</div>")
    return "".join(parts) or "<div><br></div>"


def note_container(folder_name: str, account_name: str) -> str:
    """AppleScript container to *create* a note in (with a default fallback).

    Resolution:

    * folder + account → ``folder "F" of account "A"``
    * folder only      → ``folder "F"`` (app-level; matches by name)
    * account only     → ``default folder of account "A"``
    * neither          → ``default folder of default account`` (where the
      Notes UI puts a new note when nothing is selected)
    """
    folder = (folder_name or "").strip()
    account = (account_name or "").strip()
    if folder and account:
        return (
            f'folder "{escape_applescript_string(folder)}" '
            f'of account "{escape_applescript_string(account)}"'
        )
    if folder:
        return f'folder "{escape_applescript_string(folder)}"'
    if account:
        return f'default folder of account "{escape_applescript_string(account)}"'
    return "default folder of default account"


def lookup_container(folder_name: str, account_name: str) -> Optional[str]:
    """AppleScript container to *look a note up* in, or ``None`` for app-wide.

    Unlike :func:`note_container` this has no default fallback: when the
    caller gives no folder, the lookup runs app-wide (``first note whose
    id is …``) rather than guessing a folder. An account-only hint is
    ignored for scoping (the ``note`` element isn't reliably addressable
    directly under an ``account`` across Notes versions) — id uniqueness
    still finds it app-wide.
    """
    folder = (folder_name or "").strip()
    account = (account_name or "").strip()
    if folder and account:
        return (
            f'folder "{escape_applescript_string(folder)}" '
            f'of account "{escape_applescript_string(account)}"'
        )
    if folder:
        return f'folder "{escape_applescript_string(folder)}"'
    return None


def note_reference(note_id: str, container: Optional[str]) -> str:
    """Build a note reference by id, optionally scoped to one container.

    Uses AppleScript's ``whose id is`` form for the same robustness
    reasons the macos-mail MCP settled on. ``container`` (from
    :func:`lookup_container`) bounds the search to one folder; ``None``
    searches every note in the app.
    """
    esc = escape_applescript_string(note_id)
    scope = f"of {container} " if container else ""
    return f'(first note {scope}whose id is "{esc}")'


def keyword_whose_clause(keyword: str, include_body: bool = False) -> str:
    """Build the ``whose`` predicate matching a note's name (and, optionally, body).

    Matching ``name`` is cheap (cached metadata). Matching body forces
    Notes to materialise every candidate note's ``plaintext`` to test the
    clause — much slower — so ``include_body`` is opt-in.
    """
    esc = escape_applescript_string(keyword)
    clause = f'(name contains "{esc}")'
    if include_body:
        clause += f' or (plaintext contains "{esc}")'
    return clause


def parse_records(text: str, fields: list[str]) -> list[dict[str, str]]:
    """Parse ``RECORD_SEP``/``FIELD_SEP``-delimited AppleScript output.

    Strips the trailing newline ``osascript`` appends and the trailing
    ``RECORD_SEP`` every generated script emits; short records pad with
    ``""`` rather than raising.
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
