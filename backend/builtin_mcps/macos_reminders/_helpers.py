"""Pure-Python helpers for the macos-reminders MCP.

Lives in a separate module — free of the ``@mcp.tool()`` decorator — so
unit tests can exercise the logic without needing FastMCP's parameter
introspection (mirrors :mod:`backend.builtin_mcps.macos_mail._helpers`).
At runtime this MCP runs in its own uv-provisioned venv where the
decorator works fine.

Responsibilities:

* Building the AppleScript object references the Reminders dictionary
  expects (:func:`list_reference`, :func:`reminder_reference`).
* Turning caller-supplied ISO date strings into a locale-independent
  AppleScript ``date`` construction snippet
  (:func:`parse_iso_datetime`, :func:`applescript_date_block`).
* Mapping a friendly priority word to Reminders' integer scale
  (:func:`priority_to_int`).
* Parsing the delimited text the generated AppleScript emits back
  (:func:`parse_records`, :func:`parse_single_record`).

Delimiter scheme
----------------
AppleScript has no JSON encoder, so list/record results are serialized
as plain text using two ASCII separator control characters — chosen
because they can never appear in a reminder's title/notes and require
no escaping on either side:

* ``RECORD_SEP`` (0x1E) — between whole records in a list result.
* ``FIELD_SEP``  (0x1F) — between fields within one record.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

RECORD_SEP = "\x1e"
FIELD_SEP = "\x1f"


# ``safeText`` guards against ``missing value`` — which raises a hard
# AppleScript error ("can't make missing value into type reference") the
# moment it's concatenated with ``&`` — so callers can build delimited
# output without per-property try/on error noise.
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


# Reminders stores priority as an EventKit integer, not a friendly word.
# 0 = none, 1 = high, 5 = medium, 9 = low (the same scale the Reminders
# UI's High/Medium/Low buttons write).
_PRIORITY_WORDS: dict[str, int] = {
    "": 0,
    "none": 0,
    "high": 1,
    "medium": 5,
    "med": 5,
    "low": 9,
}
_PRIORITY_INTS = {0, 1, 5, 9}


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


def priority_to_int(priority: object) -> Optional[int]:
    """Map a friendly priority word (or raw int) to Reminders' integer scale.

    Accepts ``"none"``/``"low"``/``"medium"``/``"high"`` (case-insensitive)
    or a raw ``0``/``1``/``5``/``9``. Returns ``None`` for an unrecognised
    value so the caller can report a clear error rather than silently
    writing a bogus priority.
    """
    if isinstance(priority, bool):  # bool is an int subclass — reject explicitly
        return None
    if isinstance(priority, int):
        return priority if priority in _PRIORITY_INTS else None
    if isinstance(priority, str):
        return _PRIORITY_WORDS.get(priority.strip().lower())
    return None


def int_to_priority(value: object) -> str:
    """Inverse of :func:`priority_to_int` for surfacing a read priority."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return "none"
    return {0: "none", 1: "high", 5: "medium", 9: "low"}.get(n, "none")


def list_reference(list_name: str) -> str:
    """Build the AppleScript object reference for a reminder list.

    Empty ``list_name`` resolves to Reminders' ``default list`` (the
    account's primary list, e.g. "Reminders"), matching what the app
    itself does when the user adds a reminder without picking a list.
    """
    name = (list_name or "").strip()
    if not name:
        return "default list"
    return f'list "{escape_applescript_string(name)}"'


def reminder_reference(reminder_id: str, list_ref: Optional[str] = None) -> str:
    """Build a reminder reference by id, optionally scoped to one list.

    Uses AppleScript's ``whose id is`` form rather than a bare
    ``reminder id "…"`` reference: the ``whose`` form works uniformly
    whether the container is a compound ``list "…"`` expression or the
    app-level collection, matching the defensive choice made in the
    macos-mail MCP for the same class of ``-2741`` reference errors.

    Scoping to a ``list_ref`` bounds the search to one list (fast);
    ``None`` searches every reminder in the app.
    """
    esc = escape_applescript_string(reminder_id)
    container = f"of {list_ref} " if list_ref else ""
    return f'(first reminder {container}whose id is "{esc}")'


def parse_iso_datetime(value: str) -> Optional[tuple[int, int, int, int, int, int]]:
    """Parse a caller date string into ``(year, month, day, h, m, s)``.

    Parsing happens in Python (robust, locale-independent) so the
    generated AppleScript never has to call ``date "…"``, whose string
    parsing depends on the host's regional format. Accepts common
    ISO-8601 shapes plus a bare date:

    * ``"2026-07-10"``            → midnight
    * ``"2026-07-10 14:30"``      → to the minute
    * ``"2026-07-10T14:30:00"``   → to the second

    Returns ``None`` for anything unparseable so the caller can report a
    clear error instead of writing a wrong date.
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
    # Last resort: fromisoformat handles fractional seconds / offsets.
    try:
        dt = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)


def applescript_date_block(var_name: str, comps: tuple[int, int, int, int, int, int]) -> str:
    """Emit AppleScript that builds ``var_name`` as a ``date`` from components.

    Assigns the fields one at a time off a fresh ``current date`` rather
    than parsing a date literal — locale-independent. ``day`` is reset to
    1 *before* setting year/month so that setting the month can never
    overflow (e.g. today is the 31st and the target month is February),
    the classic AppleScript date-construction pitfall; the real day is
    then set last.
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

    ``osascript`` appends a trailing newline after a returned ``text``
    value (stripped up front). Every generated script emits a trailing
    ``RECORD_SEP`` per record, so a final empty segment is expected and
    dropped; short records pad with ``""`` rather than raising.
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
