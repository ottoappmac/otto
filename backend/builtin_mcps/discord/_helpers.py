"""Pure-Python helpers for the Discord MCP.

Lives in a separate module (no ``@mcp.tool()`` decoration) so unit tests
can exercise request/response handling without importing FastMCP or
making a network call — mirrors the split used by
:mod:`backend.builtin_mcps.macos_osascript._helpers`.

Unlike Slack, Discord's REST API uses normal HTTP status codes for
failures (4xx/5xx), but rate limiting (``429``) carries the retry delay
in the *response body* (``retry_after``, seconds) rather than purely in a
header, so it needs its own classifier rather than a bare
``raise_for_status()``.
"""

from __future__ import annotations

from typing import Any, Optional

API_BASE = "https://discord.com/api/v10"

# Discord's API guidelines require a descriptive User-Agent identifying
# the client and a contact URL. See https://discord.com/developers/docs/reference#user-agent
USER_AGENT = "Otto (https://github.com/ottoappmac/otto, builtin-discord-mcp)"

DEFAULT_LIMIT = 20
MAX_LIMIT = 100


class DiscordAPIError(RuntimeError):
    """Raised for a non-2xx Discord REST response."""

    def __init__(self, status_code: int, message: str, *, retry_after: Optional[float] = None):
        self.status_code = status_code
        self.message = message
        self.retry_after = retry_after
        super().__init__(f"Discord API error ({status_code}): {message}")


def build_headers(token: str) -> dict[str, str]:
    """Return the standard Discord bot request headers."""
    return {
        "Authorization": f"Bot {token}",
        "User-Agent": USER_AGENT,
        "Content-Type": "application/json",
    }


def classify_error(status_code: int, body: Any) -> DiscordAPIError:
    """Build a :class:`DiscordAPIError` from a non-2xx status + JSON body.

    Discord's rate-limit responses (429) put the retry delay in the body
    (``{"retry_after": 1.5, ...}``); other failures use ``{"message": ...,
    "code": ...}``. Both shapes are handled defensively since a transient
    proxy error could return a non-JSON body.
    """
    if isinstance(body, dict):
        message = str(body.get("message") or body.get("error") or "request failed")
        retry_after = body.get("retry_after")
        try:
            retry_after = float(retry_after) if retry_after is not None else None
        except (TypeError, ValueError):
            retry_after = None
    else:
        message = str(body) if body else "request failed"
        retry_after = None
    return DiscordAPIError(status_code, message, retry_after=retry_after)


def clamp_limit(limit: int, *, default: int = DEFAULT_LIMIT, max_limit: int = MAX_LIMIT) -> int:
    """Clamp a user-supplied page size into ``[1, max_limit]``."""
    try:
        value = int(limit)
    except (TypeError, ValueError):
        return default
    if value <= 0:
        return default
    return min(value, max_limit)


def format_guild(raw: dict[str, Any]) -> dict[str, Any]:
    """Shrink a Discord guild object down to the fields agents need."""
    return {
        "id": raw.get("id", ""),
        "name": raw.get("name", ""),
        "owner": bool(raw.get("owner")),
        "approximate_member_count": raw.get("approximate_member_count"),
        "description": raw.get("description") or "",
    }


# Discord channel type codes that represent a text-postable channel.
# https://discord.com/developers/docs/resources/channel#channel-object-channel-types
_TEXT_CHANNEL_TYPES = frozenset({0, 5})  # GUILD_TEXT, GUILD_ANNOUNCEMENT


def format_channel(raw: dict[str, Any]) -> dict[str, Any]:
    """Shrink a Discord channel object down to the fields agents need."""
    return {
        "id": raw.get("id", ""),
        "name": raw.get("name", ""),
        "type": raw.get("type"),
        "is_text_channel": raw.get("type") in _TEXT_CHANNEL_TYPES,
        "topic": raw.get("topic") or "",
        "position": raw.get("position"),
        "parent_id": raw.get("parent_id"),
    }


def format_message(raw: dict[str, Any]) -> dict[str, Any]:
    """Shrink a Discord message object down to the fields agents need."""
    author = raw.get("author") or {}
    return {
        "id": raw.get("id", ""),
        "channel_id": raw.get("channel_id", ""),
        "author": author.get("username", ""),
        "author_id": author.get("id", ""),
        "content": raw.get("content", ""),
        "timestamp": raw.get("timestamp", ""),
        "edited_timestamp": raw.get("edited_timestamp"),
        "reactions": [
            {"emoji": (r.get("emoji") or {}).get("name", ""), "count": r.get("count", 0)}
            for r in raw.get("reactions", []) or []
        ],
    }


def format_member(raw: dict[str, Any]) -> dict[str, Any]:
    """Shrink a Discord guild-member object down to the fields agents need."""
    user = raw.get("user") or {}
    return {
        "user_id": user.get("id", ""),
        "username": user.get("username", ""),
        "nick": raw.get("nick") or "",
        "roles": raw.get("roles", []),
        "joined_at": raw.get("joined_at"),
    }


def find_matching_channels(
    channels: list[dict[str, Any]], name: str,
) -> list[dict[str, Any]]:
    """Find formatted channels whose name matches ``name``.

    Discord channel names are case-insensitive and never include the
    leading ``#`` that clients display, so both are normalized away
    before comparing. Falls back to a substring match only when there's
    no exact match, so a query like ``"general"`` doesn't ambiguously
    match ``"general-chat"`` and ``"general"`` at once when an exact
    channel of that name exists.
    """
    needle = (name or "").strip().lstrip("#").casefold()
    if not needle:
        return []
    exact = [c for c in channels if c.get("name", "").casefold() == needle]
    if exact:
        return exact
    return [c for c in channels if needle in c.get("name", "").casefold()]


def encode_emoji(emoji: str) -> str:
    """URL-encode an emoji for use in a reaction endpoint path segment.

    Unicode emoji (e.g. ``"\U0001F44D"``) are sent as-is in the URL (the
    HTTP client percent-encodes them); custom guild emoji must be passed
    as ``name:id`` per Discord's reaction API.
    """
    return (emoji or "").strip()
