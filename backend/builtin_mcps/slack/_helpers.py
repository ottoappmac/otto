"""Pure-Python helpers for the Slack MCP.

Lives in a separate module (no ``@mcp.tool()`` decoration) so unit tests can
exercise request/response handling without importing FastMCP or making a
network call — mirrors the split used by
:mod:`backend.builtin_mcps.macos_osascript._helpers`.

Slack's Web API is unusual among the platforms wired up here: it always
responds with HTTP 200, even for a logical failure, and encodes success in
an ``"ok"`` boolean field plus an ``"error"`` slug.  Every response must be
run through :func:`parse_slack_payload` before use — relying on
``httpx``'s ``raise_for_status()`` alone would silently treat a failed call
as a success.
"""

from __future__ import annotations

from typing import Any, Optional

BASE_URL = "https://slack.com/api"

DEFAULT_LIMIT = 20
MAX_LIMIT = 200


class SlackAPIError(RuntimeError):
    """Raised when Slack responds with ``{"ok": false, ...}``."""

    def __init__(self, error: str, *, response_metadata: Optional[dict] = None):
        self.error = error
        self.response_metadata = response_metadata or {}
        super().__init__(f"Slack API error: {error}")


def build_headers(token: str) -> dict[str, str]:
    """Return the standard Slack Web API request headers."""
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }


def parse_slack_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a Slack Web API JSON body and return it on success.

    Raises :class:`SlackAPIError` when ``payload["ok"]`` is falsy — the
    only reliable failure signal Slack provides, since the HTTP status
    code is 200 either way.
    """
    if not isinstance(payload, dict):
        raise SlackAPIError("malformed_response")
    if not payload.get("ok"):
        raise SlackAPIError(
            str(payload.get("error") or "unknown_error"),
            response_metadata=payload.get("response_metadata"),
        )
    return payload


def clamp_limit(limit: int, *, default: int = DEFAULT_LIMIT, max_limit: int = MAX_LIMIT) -> int:
    """Clamp a user-supplied page size into ``[1, max_limit]``.

    Falls back to *default* for non-positive or non-numeric input rather
    than raising — pagination knobs should never be the reason a tool call
    fails.
    """
    try:
        value = int(limit)
    except (TypeError, ValueError):
        return default
    if value <= 0:
        return default
    return min(value, max_limit)


def next_cursor(payload: dict[str, Any]) -> str:
    """Extract the pagination cursor Slack returns for ``*.list`` endpoints."""
    meta = payload.get("response_metadata") or {}
    return str(meta.get("next_cursor") or "")


def format_channel(raw: dict[str, Any]) -> dict[str, Any]:
    """Shrink a Slack conversation object down to the fields agents need."""
    topic = (raw.get("topic") or {}).get("value", "")
    purpose = (raw.get("purpose") or {}).get("value", "")
    return {
        "id": raw.get("id", ""),
        "name": raw.get("name", ""),
        "is_channel": bool(raw.get("is_channel")),
        "is_private": bool(raw.get("is_private")),
        "is_archived": bool(raw.get("is_archived")),
        "is_member": bool(raw.get("is_member")),
        "num_members": raw.get("num_members"),
        "topic": topic,
        "purpose": purpose,
    }


def format_message(raw: dict[str, Any]) -> dict[str, Any]:
    """Shrink a Slack message object down to the fields agents need."""
    return {
        "ts": raw.get("ts", ""),
        "user": raw.get("user", raw.get("bot_id", "")),
        "text": raw.get("text", ""),
        "thread_ts": raw.get("thread_ts", ""),
        "reply_count": raw.get("reply_count", 0),
        "reactions": [
            {"name": r.get("name", ""), "count": r.get("count", 0)}
            for r in raw.get("reactions", []) or []
        ],
    }


def format_user(raw: dict[str, Any]) -> dict[str, Any]:
    """Shrink a Slack user object down to the fields agents need."""
    profile = raw.get("profile") or {}
    return {
        "id": raw.get("id", ""),
        "name": raw.get("name", ""),
        "real_name": raw.get("real_name", profile.get("real_name", "")),
        "is_bot": bool(raw.get("is_bot")),
        "is_admin": bool(raw.get("is_admin")),
        "deleted": bool(raw.get("deleted")),
        "tz": raw.get("tz", ""),
        "email": profile.get("email", ""),
    }


def normalize_emoji(emoji: str) -> str:
    """Strip surrounding colons from a reaction name (``:thumbsup:`` -> ``thumbsup``)."""
    return (emoji or "").strip().strip(":")
