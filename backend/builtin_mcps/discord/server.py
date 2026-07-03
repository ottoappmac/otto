#!/usr/bin/env python3
"""Built-in MCP server: Discord.

Read + write tools over the Discord REST API (`discord.com/api/v10`):

* ``list_guilds`` / ``get_guild_info`` / ``list_guild_members``
* ``list_channels``
* ``get_channel_messages`` / ``send_message`` / ``add_reaction``

Auth is a single Bot Token (``DISCORD_BOT_TOKEN``).  The user creates an
application + bot at https://discord.com/developers/applications, copies
the bot token, enables any privileged Gateway intents the tools need
(notably **Server Members Intent** for ``list_guild_members``), and
generates an OAuth2 invite URL (scope ``bot``) to add the bot to a server
with the permissions it needs (Read Messages, Send Messages, Add
Reactions, ...).

This file is the canonical source for the ``discord`` builtin MCP.  The
backend copies it into ``mcp_server/discord/`` on every startup and runs
it inside a uv-provisioned venv (``mcp[cli]`` + ``httpx``) — see
:mod:`backend.builtin_mcps.registry`.

Trust boundaries:
* Only Discord's documented public REST API is reached.
* ``DISCORD_BOT_TOKEN`` is read from the environment (hydrated from the
  OS keychain by the parent backend at spawn time) and never logged or
  echoed back in a tool result.
* Rate-limit (``429``) and other non-2xx responses are turned into a
  structured ``{"error": ...}`` result rather than a raw exception so the
  agent can see what happened and back off / retry deliberately.
"""

from __future__ import annotations

import logging
import os
from typing import Any
from urllib.parse import quote

import httpx
from mcp.server.fastmcp import FastMCP

# Pure-Python helpers live in a sibling module so unit tests can import
# them without going through the @mcp.tool() decorator. Two import paths
# are supported because this file runs in two contexts: spawned directly
# as ``python <path-to-server.py>`` in production (script dir on
# sys.path, ``_helpers`` resolves as a top-level module) vs. imported as
# ``backend.builtin_mcps.discord.server`` from tests (relative import).
try:
    from ._helpers import (  # type: ignore[import-not-found]
        API_BASE,
        DiscordAPIError,
        build_headers,
        clamp_limit,
        classify_error,
        encode_emoji,
        find_matching_channels,
        format_channel,
        format_guild,
        format_member,
        format_message,
    )
except ImportError:
    from _helpers import (  # type: ignore[no-redef]
        API_BASE,
        DiscordAPIError,
        build_headers,
        clamp_limit,
        classify_error,
        encode_emoji,
        find_matching_channels,
        format_channel,
        format_guild,
        format_member,
        format_message,
    )

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("otto.mcp.discord")

DEFAULT_TIMEOUT = 30.0

mcp = FastMCP("Discord")


def _token() -> str:
    token = (os.environ.get("DISCORD_BOT_TOKEN") or "").strip()
    if not token:
        raise ValueError(
            "DISCORD_BOT_TOKEN is not set. Set it via Tools page → Discord → "
            "Credentials, using the bot token from your application's "
            "Bot page on the Discord Developer Portal."
        )
    return token


def _client() -> httpx.Client:
    return httpx.Client(timeout=DEFAULT_TIMEOUT)


def _request(method: str, path: str, **kwargs: Any) -> Any:
    with _client() as client:
        resp = client.request(
            method, f"{API_BASE}{path}", headers=build_headers(_token()), **kwargs,
        )
        if resp.status_code >= 300:
            try:
                body = resp.json()
            except ValueError:
                body = resp.text
            raise classify_error(resp.status_code, body)
        if resp.status_code == 204 or not resp.content:
            return {}
        return resp.json()


def _error_result(exc: DiscordAPIError) -> dict[str, Any]:
    result: dict[str, Any] = {"error": exc.message, "status_code": exc.status_code}
    if exc.retry_after is not None:
        result["retry_after_seconds"] = exc.retry_after
    return result


def _resolve_channel_id(
    channel_id: str, channel_name: str, guild_id: str,
) -> tuple[str, dict[str, Any] | None]:
    """Resolve a channel_id, either passed directly or looked up by name.

    Discord's API only ever addresses channels by snowflake id, so a
    ``channel_name`` lookup requires listing the channels of one guild (if
    ``guild_id`` is given) or every guild the bot is in (if not) and
    matching by name. Returns ``(id, None)`` on success or ``("", error)``
    on failure — callers should return the error dict as-is.
    """
    if channel_id:
        return channel_id, None
    if not channel_name:
        return "", {"error": "Either channel_id or channel_name is required."}

    try:
        if guild_id:
            guild_ids = [guild_id]
        else:
            guilds_payload = _request(
                "GET", "/users/@me/guilds", params={"limit": 100},
            )
            guild_ids = [g["id"] for g in guilds_payload]
    except DiscordAPIError as exc:
        return "", _error_result(exc)

    matches: list[dict[str, Any]] = []
    for gid in guild_ids:
        try:
            channels = _request("GET", f"/guilds/{gid}/channels")
        except DiscordAPIError:
            continue
        for c in find_matching_channels(channels, channel_name):
            matches.append({"guild_id": gid, **format_channel(c)})

    if not matches:
        scope = f"guild {guild_id}" if guild_id else "any accessible server"
        return "", {"error": f"No channel named {channel_name!r} found in {scope}."}
    if len(matches) > 1:
        return "", {
            "error": (
                f"Multiple channels named {channel_name!r} found. Pass "
                "channel_id or guild_id to disambiguate."
            ),
            "candidates": matches,
        }
    return matches[0]["id"], None


@mcp.tool()
def list_guilds(limit: int = 100) -> dict[str, Any]:
    """List servers (guilds) the bot is a member of.

    Args:
        limit: Max guilds to return (1-100).
    """
    try:
        payload = _request(
            "GET", "/users/@me/guilds", params={"limit": clamp_limit(limit, default=100)},
        )
    except DiscordAPIError as exc:
        return _error_result(exc)
    return {"guilds": [format_guild(g) for g in payload]}


@mcp.tool()
def get_guild_info(guild_id: str) -> dict[str, Any]:
    """Get metadata for a single server.

    Args:
        guild_id: Discord guild (server) id.
    """
    try:
        payload = _request(
            "GET", f"/guilds/{guild_id}", params={"with_counts": "true"},
        )
    except DiscordAPIError as exc:
        return _error_result(exc)
    return format_guild(payload)


@mcp.tool()
def list_guild_members(guild_id: str, limit: int = 100) -> dict[str, Any]:
    """List members of a server.

    Args:
        guild_id: Discord guild (server) id.
        limit: Max members to return (1-100).

    Requires the "Server Members Intent" to be enabled for the bot in the
    Discord Developer Portal.
    """
    try:
        payload = _request(
            "GET", f"/guilds/{guild_id}/members",
            params={"limit": clamp_limit(limit, default=100)},
        )
    except DiscordAPIError as exc:
        return _error_result(exc)
    return {"members": [format_member(m) for m in payload]}


@mcp.tool()
def list_channels(guild_id: str) -> dict[str, Any]:
    """List channels in a server.

    Args:
        guild_id: Discord guild (server) id.
    """
    try:
        payload = _request("GET", f"/guilds/{guild_id}/channels")
    except DiscordAPIError as exc:
        return _error_result(exc)
    return {"channels": [format_channel(c) for c in payload]}


@mcp.tool()
def get_channel_messages(
    channel_id: str = "",
    channel_name: str = "",
    guild_id: str = "",
    limit: int = 20,
    before: str = "",
    after: str = "",
) -> dict[str, Any]:
    """Get recent messages from a channel, newest first.

    Args:
        channel_id: Discord channel id. Provide this or channel_name.
        channel_name: Channel name to resolve to an id (case-insensitive,
            "#" prefix optional). Ignored if channel_id is set. If
            multiple channels share this name across servers, pass
            guild_id too, or use channel_id directly.
        guild_id: Optional server (guild) id to scope the channel_name
            lookup to. Recommended once you already know it — otherwise
            every server the bot is in gets searched.
        limit: Max messages to return (1-100).
        before: Optional message id — only messages before this are returned.
        after: Optional message id — only messages after this are returned.
    """
    resolved_id, err = _resolve_channel_id(channel_id, channel_name, guild_id)
    if err:
        return err
    try:
        params: dict[str, Any] = {"limit": clamp_limit(limit)}
        if before:
            params["before"] = before
        if after:
            params["after"] = after
        payload = _request(
            "GET", f"/channels/{resolved_id}/messages", params=params,
        )
    except DiscordAPIError as exc:
        return _error_result(exc)
    return {"messages": [format_message(m) for m in payload]}


@mcp.tool()
def send_message(
    content: str, channel_id: str = "", channel_name: str = "", guild_id: str = "",
) -> dict[str, Any]:
    """Post a message to a channel.

    Args:
        content: Message body (up to 2000 characters; Discord markdown
            formatting is supported).
        channel_id: Discord channel id. Provide this or channel_name.
        channel_name: Channel name to resolve to an id (case-insensitive,
            "#" prefix optional). Ignored if channel_id is set. If
            multiple channels share this name across servers, pass
            guild_id too, or use channel_id directly.
        guild_id: Optional server (guild) id to scope the channel_name
            lookup to. Recommended once you already know it — otherwise
            every server the bot is in gets searched.
    """
    if not content or not content.strip():
        return {"error": "content is required"}
    resolved_id, err = _resolve_channel_id(channel_id, channel_name, guild_id)
    if err:
        return err
    try:
        payload = _request(
            "POST", f"/channels/{resolved_id}/messages", json={"content": content},
        )
    except DiscordAPIError as exc:
        return _error_result(exc)
    return format_message(payload)


@mcp.tool()
def add_reaction(
    message_id: str,
    emoji: str,
    channel_id: str = "",
    channel_name: str = "",
    guild_id: str = "",
) -> dict[str, Any]:
    """Add an emoji reaction to a message.

    Args:
        message_id: Id of the message to react to.
        emoji: Unicode emoji (e.g. ``"\U0001F44D"``) or custom emoji as
            ``name:id``.
        channel_id: Discord channel id. Provide this or channel_name.
        channel_name: Channel name to resolve to an id (case-insensitive,
            "#" prefix optional). Ignored if channel_id is set. If
            multiple channels share this name across servers, pass
            guild_id too, or use channel_id directly.
        guild_id: Optional server (guild) id to scope the channel_name
            lookup to. Recommended once you already know it — otherwise
            every server the bot is in gets searched.
    """
    resolved_id, err = _resolve_channel_id(channel_id, channel_name, guild_id)
    if err:
        return err
    try:
        encoded = quote(encode_emoji(emoji), safe="")
        _request(
            "PUT",
            f"/channels/{resolved_id}/messages/{message_id}/reactions/{encoded}/@me",
        )
    except DiscordAPIError as exc:
        return _error_result(exc)
    return {"ok": True}


if __name__ == "__main__":
    mcp.run()
