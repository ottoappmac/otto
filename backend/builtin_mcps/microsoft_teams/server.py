#!/usr/bin/env python3
"""Built-in MCP server: Microsoft Teams.

Read-only tools over the Microsoft Graph API (`graph.microsoft.com/v1.0`)
for Teams:

* ``list_teams`` / ``list_channels`` / ``get_channel_info``
* ``list_channel_members`` / ``get_channel_messages``
* ``list_users`` / ``get_user_info``

Auth is app-only OAuth2 client-credentials against Microsoft Entra ID
(``TEAMS_TENANT_ID`` / ``TEAMS_CLIENT_ID`` / ``TEAMS_CLIENT_SECRET``).
The user registers an app in the Entra admin center, grants it
**application** permissions (``Team.ReadBasic.All``,
``Channel.ReadBasic.All``, ``ChannelMessage.Read.All``,
``TeamMember.Read.All``, ``User.Read.All``), has a tenant admin grant
consent, and pastes the tenant id / client id / client secret into Otto's
credentials dialog for this MCP.

**This MCP is intentionally read-only.** Microsoft Graph does not allow
app-only (application-permission) tokens to post Teams channel or chat
messages at all — sending messages requires a delegated (signed-in user)
token, which is a fundamentally different auth model than every other
built-in MCP here. There is no ``send_message`` tool and there will not
be one without adding a full delegated OAuth flow.

Separately, reading channel messages (``get_channel_messages``) needs
``ChannelMessage.Read.All``, which Microsoft classifies as a *Protected
API*: admin consent in the Entra portal is necessary but **not
sufficient** — the tenant must additionally submit Microsoft's Protected
APIs request form and wait for approval. See
:data:`_helpers.PROTECTED_API_HINT`, which is appended to the error
whenever Graph returns 403 on that endpoint.

This file is the canonical source for the ``microsoft-teams`` builtin
MCP.  The backend copies it into ``mcp_server/microsoft_teams/`` on every
startup and runs it inside a uv-provisioned venv (``mcp[cli]`` +
``httpx``) — see :mod:`backend.builtin_mcps.registry`.

Trust boundaries:
* Only Microsoft Graph's documented public REST API is reached.
* ``TEAMS_CLIENT_SECRET`` is read from the environment (hydrated from
  the OS keychain by the parent backend at spawn time) and never logged
  or echoed back in a tool result.
* The app-only access token is cached in-process (module-level, never
  written to disk) and refreshed a few minutes ahead of its real expiry
  — see :func:`_helpers.token_is_valid`.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import httpx
from mcp.server.fastmcp import FastMCP

# Pure-Python helpers live in a sibling module so unit tests can import
# them without going through the @mcp.tool() decorator. Two import paths
# are supported because this file runs in two contexts: spawned directly
# as ``python <path-to-server.py>`` in production (script dir on
# sys.path, ``_helpers`` resolves as a top-level module) vs. imported as
# ``backend.builtin_mcps.microsoft_teams.server`` from tests (relative
# import).
try:
    from ._helpers import (  # type: ignore[import-not-found]
        GRAPH_BASE,
        PROTECTED_API_HINT,
        TeamsAPIError,
        build_token_request,
        clamp_top,
        classify_error,
        format_channel,
        format_channel_message,
        format_member,
        format_team,
        format_user,
        parse_token_response,
        teams_list_params,
        token_is_valid,
    )
except ImportError:
    from _helpers import (  # type: ignore[no-redef]
        GRAPH_BASE,
        PROTECTED_API_HINT,
        TeamsAPIError,
        build_token_request,
        clamp_top,
        classify_error,
        format_channel,
        format_channel_message,
        format_member,
        format_team,
        format_user,
        parse_token_response,
        teams_list_params,
        token_is_valid,
    )

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("otto.mcp.microsoft_teams")

DEFAULT_TIMEOUT = 30.0

mcp = FastMCP("Microsoft Teams")

# Module-level token cache: {"access_token": str, "expires_at": float}.
# Deliberately in-memory only (never persisted) — each MCP subprocess
# re-authenticates on first use after a restart.
_token_cache: Optional[dict[str, Any]] = None


def _credentials() -> tuple[str, str, str]:
    tenant_id = (os.environ.get("TEAMS_TENANT_ID") or "").strip()
    client_id = (os.environ.get("TEAMS_CLIENT_ID") or "").strip()
    client_secret = (os.environ.get("TEAMS_CLIENT_SECRET") or "").strip()
    if not (tenant_id and client_id and client_secret):
        raise ValueError(
            "TEAMS_TENANT_ID / TEAMS_CLIENT_ID / TEAMS_CLIENT_SECRET are not "
            "all set. Configure them via Tools page → Microsoft Teams → "
            "Credentials, using the values from your Entra ID app registration."
        )
    return tenant_id, client_id, client_secret


def _fetch_token() -> dict[str, Any]:
    tenant_id, client_id, client_secret = _credentials()
    url, data = build_token_request(tenant_id, client_id, client_secret)
    with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
        resp = client.post(url, data=data)
    if resp.status_code >= 300:
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        raise classify_error(resp.status_code, body)
    return parse_token_response(resp.json(), now=time.time())


def _access_token() -> str:
    global _token_cache
    if not token_is_valid(_token_cache, now=time.time()):
        _token_cache = _fetch_token()
    return _token_cache["access_token"]


def _request(method: str, path: str = "", *, params: Optional[dict[str, Any]] = None, full_url: str = "") -> Any:
    url = full_url or f"{GRAPH_BASE}{path}"
    with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
        resp = client.request(
            method, url,
            params=None if full_url else params,
            headers={"Authorization": f"Bearer {_access_token()}"},
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


def _error_result(exc: TeamsAPIError, *, protected_api_endpoint: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"error": exc.message, "status_code": exc.status_code}
    if protected_api_endpoint and exc.status_code == 403:
        result["hint"] = PROTECTED_API_HINT
    return result


@mcp.tool()
def list_teams(top: int = 50, next_link: str = "") -> dict[str, Any]:
    """List Teams-enabled groups in the tenant the app can see.

    Args:
        top: Max teams per page (1-50).
        next_link: Full pagination URL from a previous call's
            ``next_link`` — when set, ``top`` is ignored.
    """
    try:
        if next_link:
            payload = _request("GET", full_url=next_link)
        else:
            params = teams_list_params()
            params["$top"] = str(clamp_top(top))
            payload = _request("GET", "/groups", params=params)
    except TeamsAPIError as exc:
        return _error_result(exc)
    teams = [format_team(t) for t in payload.get("value", [])]
    return {"teams": teams, "next_link": payload.get("@odata.nextLink", "")}


@mcp.tool()
def list_channels(team_id: str) -> dict[str, Any]:
    """List channels in a team.

    Args:
        team_id: Id of the team (Graph group id with a Team facet).
    """
    try:
        payload = _request("GET", f"/teams/{team_id}/channels")
    except TeamsAPIError as exc:
        return _error_result(exc)
    return {"channels": [format_channel(c) for c in payload.get("value", [])]}


@mcp.tool()
def get_channel_info(team_id: str, channel_id: str) -> dict[str, Any]:
    """Get metadata for a single channel.

    Args:
        team_id: Id of the team.
        channel_id: Id of the channel.
    """
    try:
        payload = _request("GET", f"/teams/{team_id}/channels/{channel_id}")
    except TeamsAPIError as exc:
        return _error_result(exc)
    return format_channel(payload)


@mcp.tool()
def list_channel_members(team_id: str, channel_id: str) -> dict[str, Any]:
    """List members of a channel.

    Args:
        team_id: Id of the team.
        channel_id: Id of the channel.
    """
    try:
        payload = _request("GET", f"/teams/{team_id}/channels/{channel_id}/members")
    except TeamsAPIError as exc:
        return _error_result(exc)
    return {"members": [format_member(m) for m in payload.get("value", [])]}


@mcp.tool()
def get_channel_messages(team_id: str, channel_id: str, top: int = 20, next_link: str = "") -> dict[str, Any]:
    """Get recent messages in a channel (best-effort — see Known limitations).

    Args:
        team_id: Id of the team.
        channel_id: Id of the channel.
        top: Max messages per page (1-50).
        next_link: Full pagination URL from a previous call's
            ``next_link`` — when set, ``top`` is ignored.

    Requires the ``ChannelMessage.Read.All`` application permission AND
    Microsoft's separate Protected API approval for this tenant — admin
    consent alone returns 403. See this MCP's README.
    """
    try:
        if next_link:
            payload = _request("GET", full_url=next_link)
        else:
            params = {"$top": str(clamp_top(top))}
            payload = _request("GET", f"/teams/{team_id}/channels/{channel_id}/messages", params=params)
    except TeamsAPIError as exc:
        return _error_result(exc, protected_api_endpoint=True)
    messages = [format_channel_message(m) for m in payload.get("value", [])]
    return {"messages": messages, "next_link": payload.get("@odata.nextLink", "")}


@mcp.tool()
def list_users(top: int = 50, next_link: str = "") -> dict[str, Any]:
    """List users in the tenant.

    Args:
        top: Max users per page (1-50).
        next_link: Full pagination URL from a previous call's
            ``next_link`` — when set, ``top`` is ignored.
    """
    try:
        if next_link:
            payload = _request("GET", full_url=next_link)
        else:
            params = {"$top": str(clamp_top(top)), "$select": "id,displayName,mail,userPrincipalName,jobTitle"}
            payload = _request("GET", "/users", params=params)
    except TeamsAPIError as exc:
        return _error_result(exc)
    return {"users": [format_user(u) for u in payload.get("value", [])], "next_link": payload.get("@odata.nextLink", "")}


@mcp.tool()
def get_user_info(user_id: str) -> dict[str, Any]:
    """Get profile details for one user.

    Args:
        user_id: Graph user id or userPrincipalName (email).
    """
    try:
        payload = _request("GET", f"/users/{user_id}")
    except TeamsAPIError as exc:
        return _error_result(exc)
    return format_user(payload)


if __name__ == "__main__":
    mcp.run()
