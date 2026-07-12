"""Pure-Python helpers for the macos-calendar MCP.

Lives in a separate module — free of the ``@mcp.tool()`` decorator — so
unit tests can exercise the logic without needing FastMCP's parameter
introspection (mirrors :mod:`backend.builtin_mcps.macos_mail._helpers`).

Responsibilities:

* Building the AppleScript object references the Calendar dictionary
  expects (:func:`calendar_reference`, :func:`event_reference`).
* Turning caller-supplied ISO date strings into a locale-independent
  AppleScript ``date`` construction snippet
  (:func:`parse_iso_datetime`, :func:`applescript_date_block`).
* Parsing the delimited text the generated AppleScript emits back
  (:func:`parse_records`, :func:`parse_single_record`).

Delimiter scheme
----------------
Two ASCII separator control characters that can never appear in an
event's summary/notes and need no escaping:

* ``RECORD_SEP`` (0x1E) — between whole records in a list result.
* ``FIELD_SEP``  (0x1F) — between fields within one record.
"""

from __future__ import annotations

from datetime import datetime
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


def calendar_reference(calendar_name: str) -> str:
    """Build the AppleScript object reference for a calendar by name."""
    return f'calendar "{escape_applescript_string(calendar_name)}"'


def event_reference(uid: str, calendar_ref: Optional[str] = None) -> str:
    """Build an event reference by ``uid``, optionally scoped to one calendar.

    Uses AppleScript's ``whose uid is`` form (uniform whether the
    container is a compound ``calendar "…"`` expression or the app-level
    collection), matching the defensive reference style the macos-mail
    MCP settled on. Scoping to a ``calendar_ref`` bounds the search to one
    calendar; ``None`` searches every event in every calendar.
    """
    esc = escape_applescript_string(uid)
    container = f"of {calendar_ref} " if calendar_ref else ""
    return f'(first event {container}whose uid is "{esc}")'


def parse_iso_datetime(value: str) -> Optional[tuple[int, int, int, int, int, int]]:
    """Parse a caller date string into ``(year, month, day, h, m, s)``.

    Parsing happens in Python (locale-independent) so the generated
    AppleScript never calls ``date "…"``, whose string parsing depends on
    the host's regional format. Accepts common ISO-8601 shapes plus a
    bare date (midnight). Returns ``None`` for anything unparseable.
    """
    if not value or not value.strip():
        return None
    raw = value.strip().replace("T", " ")
    fmts = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H",
        "%Y-%m-%d",
    )
    for fmt in fmts:
        try:
            dt = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        return (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    try:
        dt = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)


def applescript_date_block(var_name: str, comps: tuple[int, int, int, int, int, int]) -> str:
    """Emit AppleScript that builds ``var_name`` as a ``date`` from components.

    Assigns fields off a fresh ``current date`` (locale-independent).
    ``day`` is reset to 1 before year/month are set so that setting the
    month can never overflow (today is the 31st, target month is
    February), then the real day is set last.
    """
    year, month, day, hours, minutes, seconds = comps
    return (
        f"set {var_name} to current date\n"
        f"set day of {var_name} to 1\n"
        f"set year of {var_name} to {year}\n"
        f"set month of {var_name} to {month}\n"
        f"set day of {var_name} to {day}\n"
        f"set hours of {var_name} to {hours}\n"
        f"set minutes of {var_name} to {minutes}\n"
        f"set seconds of {var_name} to {seconds}\n"
    )


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
