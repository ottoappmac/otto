"""Pure-Python helpers for the macos-messages MCP.

Lives in a separate module — free of the ``@mcp.tool()`` decorator — so
unit tests can exercise the logic without needing FastMCP's parameter
introspection (mirrors :mod:`backend.builtin_mcps.macos_mail._helpers`).

Messages is a two-surface app, and this module carries helpers for both:

* **Apple Events** drive *sending* and enumerating services / buddies /
  chats — the Messages dictionary exposes ``send`` and the ``service`` /
  ``buddy`` / ``chat`` elements, but **no readable message element**.
  :func:`normalize_service`, :func:`escape_applescript_string`.
* **SQLite** is therefore the only way to *read* message history: Messages
  persists every message in ``~/Library/Messages/chat.db``. Reading it
  directly (like ``macos-osascript``'s ``query_mail_store`` does for Mail)
  needs Full Disk Access. :func:`find_chat_db`, :func:`cocoa_to_iso`,
  :func:`decode_attributed_body`.

Delimiter scheme (Apple Events path only)
-----------------------------------------
Two ASCII separator control characters that can never appear in a chat
name/handle and need no escaping:

* ``RECORD_SEP`` (0x1E) — between whole records in a list result.
* ``FIELD_SEP``  (0x1F) — between fields within one record.
"""

from __future__ import annotations

import glob
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

RECORD_SEP = "\x1e"
FIELD_SEP = "\x1f"

# Seconds between the Unix epoch (1970-01-01) and the Cocoa/Mac absolute
# time epoch (2001-01-01) that chat.db timestamps are measured from.
_COCOA_EPOCH_OFFSET = 978307200

_VALID_SERVICES = {"imessage": "iMessage", "sms": "SMS"}


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


def clamp_limit(n: int, cap: int = 500) -> int:
    """Bound a caller-supplied ``limit`` to ``[1, cap]``."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 1
    return max(1, min(n, cap))


def normalize_service(service: str) -> Optional[str]:
    """Normalise a service name to Messages' constant (``iMessage``/``SMS``).

    Returns ``None`` for an unrecognised value so the caller can report a
    clear error rather than composing a script against a bogus service.
    """
    if not service:
        return "iMessage"
    return _VALID_SERVICES.get(service.strip().lower())


def find_chat_db() -> Optional[str]:
    """Locate the user's Messages SQLite store, or ``None`` if absent.

    The canonical path is ``~/Library/Messages/chat.db``. Falls back to a
    glob in case a future macOS relocates it.
    """
    canonical = Path(os.path.expanduser("~/Library/Messages/chat.db"))
    if canonical.is_file():
        return str(canonical)
    matches = sorted(glob.glob(os.path.expanduser("~/Library/Messages/chat*.db")))
    return matches[0] if matches else None


def cocoa_to_iso(raw: object) -> str:
    """Convert a chat.db ``date`` value to an ISO-8601 UTC string.

    chat.db stores message dates as Mac absolute time. Modern macOS uses
    *nanoseconds* since 2001-01-01; older releases used *seconds*. Detect
    which by magnitude (nanosecond values are ~1e18, second values ~6e8)
    and normalise to seconds before applying the epoch offset. Returns
    ``""`` for a missing/unparseable value.
    """
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return ""
    if value == 0:
        return ""
    # Nanosecond timestamps are enormous; anything past ~1e11 is ns.
    if abs(value) > 10**11:
        value = value / 1_000_000_000
    unix_ts = value + _COCOA_EPOCH_OFFSET
    try:
        return datetime.fromtimestamp(unix_ts, tz=timezone.utc).isoformat()
    except (OSError, OverflowError, ValueError):
        return ""


def decode_attributed_body(data: Optional[bytes]) -> str:
    """Best-effort extraction of message text from an ``attributedBody`` blob.

    On Ventura and later Messages sometimes leaves the ``message.text``
    column NULL and stores the body only in ``attributedBody`` — an
    ``NSAttributedString`` archived as an old-style ``typedstream``. There
    is no stable public parser, so this is a heuristic: skip to the
    archived ``NSString`` payload and read its length-prefixed UTF-8
    bytes. Returns ``""`` when the shape isn't recognised (the caller then
    surfaces whatever the ``text`` column had, possibly empty).
    """
    if not data:
        return ""
    try:
        marker = b"NSString"
        idx = data.find(marker)
        if idx == -1:
            return ""
        # Skip the class name and the fixed archiver bytes that follow it
        # (\x01\x94\x84\x01) before the length-prefixed string payload.
        pos = idx + len(marker) + 5
        if pos >= len(data):
            return ""
        length_byte = data[pos]
        pos += 1
        if length_byte == 0x81:
            # 0x81 signals a 2-byte little-endian length follows.
            if pos + 2 > len(data):
                return ""
            length = int.from_bytes(data[pos:pos + 2], "little")
            pos += 2
        else:
            length = length_byte
        text = data[pos:pos + length]
        return text.decode("utf-8", errors="replace")
    except Exception:
        return ""


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
