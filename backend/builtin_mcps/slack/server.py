#!/usr/bin/env python3
"""Built-in MCP server: Slack.

Read + write tools over the Slack Web API (``slack.com/api``):

* ``list_channels`` / ``get_channel_info`` / ``join_channel``
* ``get_channel_history`` / ``get_thread_replies``
* ``send_message`` / ``add_reaction``
* ``list_users`` / ``get_user_info``

Auth is a single Bot User OAuth Token (``SLACK_BOT_TOKEN``, starts with
``xoxb-``).  The user creates a Slack App at https://api.slack.com/apps,
grants it Bot Token Scopes (at minimum: ``channels:read``,
``channels:history``, ``chat:write``, ``users:read``; add
``groups:read``/``groups:history`` for private channels and
``reactions:write`` for ``add_reaction``), installs it to their workspace,
and pastes the resulting Bot User OAuth Token into Otto's credentials
dialog for this MCP.

This file is the canonical source for the ``slack`` builtin MCP.  The
backend copies it into ``mcp_server/slack/`` on every startup and runs it
inside a uv-provisioned venv (``mcp[cli]`` + ``httpx``) — see
:mod:`backend.builtin_mcps.registry`.

Trust boundaries:
* Only Slack's documented public Web API is reached.
* ``SLACK_BOT_TOKEN`` is read from the environment (hydrated from the OS
  keychain by the parent backend at spawn time) and never logged or
  echoed back in a tool result.
* Slack always answers HTTP 200, even for a failed call — every response
  is run through ``_helpers.parse_slack_payload`` so a logical failure
  (``{"ok": false, "error": "..."}``) surfaces as a clear tool error
  instead of being treated as success.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

# Pure-Python helpers live in a sibling module so unit tests can import
# them without going through the @mcp.tool() decorator. Two import paths
# are supported because this file runs in two contexts: spawned directly
# as ``python <path-to-server.py>`` in production (script dir on
# sys.path, ``_helpers`` resolves as a top-level module) vs. imported as
# ``backend.builtin_mcps.slack.server`` from tests (relative import).
try:
    from ._helpers import (  # type: ignore[import-not-found]
        BASE_URL,
        SlackAPIError,
        build_headers,
        clamp_limit,
        format_channel,
        format_message,
        format_user,
        next_cursor,
        normalize_emoji,
        parse_slack_payload,
    )
except ImportError:
    from _helpers import (  # type: ignore[no-redef]
        BASE_URL,
        SlackAPIError,
        build_headers,
        clamp_limit,
        format_channel,
        format_message,
        format_user,
        next_cursor,
        normalize_emoji,
        parse_slack_payload,
    )

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("otto.mcp.slack")

DEFAULT_TIMEOUT = 30.0

mcp = FastMCP("Slack")


def _token() -> str:
    token = (os.environ.get("SLACK_BOT_TOKEN") or "").strip()
    if not token:
        raise ValueError(
            "SLACK_BOT_TOKEN is not set. Set it via Tools page → Slack → "
            "Credentials, using the Bot User OAuth Token from your Slack "
            "App's OAuth & Permissions page."
        )
    return token


def _client() -> httpx.Client:
    return httpx.Client(timeout=DEFAULT_TIMEOUT)


def _get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    with _client() as client:
        resp = client.get(
            f"{BASE_URL}/{path}", params=params, headers=build_headers(_token()),
        )
        resp.raise_for_status()
        return parse_slack_payload(resp.json())


def _post(path: str, body: dict[str, Any]) -> dict[str, Any]:
    with _client() as client:
        resp = client.post(
            f"{BASE_URL}/{path}", json=body, headers=build_headers(_token()),
        )
        resp.raise_for_status()
        return parse_slack_payload(resp.json())


@mcp.tool()
def list_channels(
    types: str = "public_channel,private_channel",
    limit: int = 100,
    cursor: str = "",
) -> dict[str, Any]:
    """List channels visible to the bot.

    Args:
        types: Comma-separated conversation types
            (``public_channel``, ``private_channel``, ``mpim``, ``im``).
        limit: Max channels per page (1-200).
        cursor: Pagination cursor from a previous call's ``next_cursor``.

    Only returns private channels the bot has been invited to.
    """
    try:
        params: dict[str, Any] = {
            "types": types,
            "limit": clamp_limit(limit, default=100, max_limit=200),
            "exclude_archived": "true",
        }
        if cursor:
            params["cursor"] = cursor
        payload = _get("conversations.list", params)
    except SlackAPIError as exc:
        return {"error": exc.error}
    channels = [format_channel(c) for c in payload.get("channels", [])]
    return {"channels": channels, "next_cursor": next_cursor(payload)}


@mcp.tool()
def get_channel_info(channel_id: str) -> dict[str, Any]:
    """Get metadata for a single channel.

    Args:
        channel_id: Slack conversation id (e.g. ``"C0123ABCD"``).
    """
    try:
        payload = _get("conversations.info", {"channel": channel_id})
    except SlackAPIError as exc:
        return {"error": exc.error}
    return format_channel(payload.get("channel", {}))


@mcp.tool()
def join_channel(channel_id: str) -> dict[str, Any]:
    """Join a public channel so the bot can read/post in it.

    Args:
        channel_id: Slack conversation id of a *public* channel. Private
            channels require a human to invite the bot instead.
    """
    try:
        payload = _post("conversations.join", {"channel": channel_id})
    except SlackAPIError as exc:
        return {"error": exc.error}
    return {"joined": True, "channel": format_channel(payload.get("channel", {}))}


@mcp.tool()
def get_channel_history(
    channel_id: str,
    limit: int = 20,
    oldest: str = "",
    latest: str = "",
    cursor: str = "",
) -> dict[str, Any]:
    """Get recent messages from a channel, newest first.

    Args:
        channel_id: Slack conversation id.
        limit: Max messages per page (1-200).
        oldest: Optional Slack timestamp (``"1234567890.123456"``) — only
            messages after this are returned.
        latest: Optional Slack timestamp — only messages before this are
            returned.
        cursor: Pagination cursor from a previous call's ``next_cursor``.
    """
    try:
        params: dict[str, Any] = {
            "channel": channel_id,
            "limit": clamp_limit(limit),
        }
        if oldest:
            params["oldest"] = oldest
        if latest:
            params["latest"] = latest
        if cursor:
            params["cursor"] = cursor
        payload = _get("conversations.history", params)
    except SlackAPIError as exc:
        return {"error": exc.error}
    messages = [format_message(m) for m in payload.get("messages", [])]
    return {"messages": messages, "next_cursor": next_cursor(payload)}


@mcp.tool()
def get_thread_replies(
    channel_id: str, thread_ts: str, limit: int = 50, cursor: str = "",
) -> dict[str, Any]:
    """Get all replies in a message thread.

    Args:
        channel_id: Slack conversation id.
        thread_ts: Timestamp of the thread's parent message.
        limit: Max replies per page (1-200).
        cursor: Pagination cursor from a previous call's ``next_cursor``.
    """
    try:
        params: dict[str, Any] = {
            "channel": channel_id,
            "ts": thread_ts,
            "limit": clamp_limit(limit, default=50),
        }
        if cursor:
            params["cursor"] = cursor
        payload = _get("conversations.replies", params)
    except SlackAPIError as exc:
        return {"error": exc.error}
    messages = [format_message(m) for m in payload.get("messages", [])]
    return {"messages": messages, "next_cursor": next_cursor(payload)}


@mcp.tool()
def send_message(channel_id: str, text: str, thread_ts: str = "") -> dict[str, Any]:
    """Post a message to a channel, or reply in a thread.

    Args:
        channel_id: Slack conversation id.
        text: Message body (Slack ``mrkdwn`` formatting is supported).
        thread_ts: Optional parent message timestamp — when set, this
            posts as a threaded reply instead of a new top-level message.
    """
    if not text or not text.strip():
        return {"error": "text is required"}
    try:
        body: dict[str, Any] = {"channel": channel_id, "text": text}
        if thread_ts:
            body["thread_ts"] = thread_ts
        payload = _post("chat.postMessage", body)
    except SlackAPIError as exc:
        return {"error": exc.error}
    return {
        "ok": True,
        "channel": payload.get("channel", channel_id),
        "ts": payload.get("ts", ""),
    }


@mcp.tool()
def add_reaction(channel_id: str, timestamp: str, emoji: str) -> dict[str, Any]:
    """Add an emoji reaction to a message.

    Args:
        channel_id: Slack conversation id.
        timestamp: Timestamp of the message to react to.
        emoji: Emoji name, with or without colons (e.g. ``"thumbsup"`` or
            ``":thumbsup:"``).
    """
    try:
        body = {
            "channel": channel_id,
            "timestamp": timestamp,
            "name": normalize_emoji(emoji),
        }
        _post("reactions.add", body)
    except SlackAPIError as exc:
        return {"error": exc.error}
    return {"ok": True}


@mcp.tool()
def list_users(limit: int = 200, cursor: str = "") -> dict[str, Any]:
    """List members of the workspace.

    Args:
        limit: Max users per page (1-200).
        cursor: Pagination cursor from a previous call's ``next_cursor``.
    """
    try:
        params: dict[str, Any] = {"limit": clamp_limit(limit, default=200, max_limit=200)}
        if cursor:
            params["cursor"] = cursor
        payload = _get("users.list", params)
    except SlackAPIError as exc:
        return {"error": exc.error}
    users = [format_user(u) for u in payload.get("members", [])]
    return {"users": users, "next_cursor": next_cursor(payload)}


@mcp.tool()
def get_user_info(user_id: str) -> dict[str, Any]:
    """Get profile details for one user.

    Args:
        user_id: Slack user id (e.g. ``"U0123ABCD"``).
    """
    try:
        payload = _get("users.info", {"user": user_id})
    except SlackAPIError as exc:
        return {"error": exc.error}
    return format_user(payload.get("user", {}))


if __name__ == "__main__":
    mcp.run()
