"""Pure-Python helpers for the Microsoft Teams MCP.

Lives in a separate module (no ``@mcp.tool()`` decoration) so unit tests
can exercise token-cache and response-parsing logic without importing
FastMCP or making a network call — mirrors the split used by
:mod:`backend.builtin_mcps.macos_osascript._helpers`.

This MCP authenticates app-only (OAuth2 client-credentials grant) against
Microsoft Entra ID / Graph — there is deliberately no delegated/browser
login flow here, so it stays on Otto's existing "static vault secret"
auth model (``TEAMS_CLIENT_ID`` / ``TEAMS_CLIENT_SECRET`` /
``TEAMS_TENANT_ID``) instead of routing through :mod:`backend.auth`.

Important Graph limitation baked into this module and ``server.py``:
**application permissions cannot send Teams channel or chat messages at
all** (Microsoft restricts that to delegated/user tokens — app-only POST
to the messages endpoint is reserved for migration/import scenarios and
returns 403). So this MCP is read-only by design; see
:func:`classify_error` / ``is_protected_api`` for the other app-only gate
(``ChannelMessage.Read.All`` requires Microsoft's separate Protected APIs
approval on top of admin consent).
"""

from __future__ import annotations

from typing import Any, Optional

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
TOKEN_URL_TEMPLATE = "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
GRAPH_DEFAULT_SCOPE = "https://graph.microsoft.com/.default"

# Refresh this many seconds before actual expiry so a tool call never
# races a token that's about to die mid-request.
TOKEN_REFRESH_LEEWAY_SECS = 300

DEFAULT_TOP = 20
MAX_TOP = 50

# Surfaced when a 403 is plausibly the Microsoft Graph "Protected API"
# gate rather than a plain missing-permission error — admin consent
# alone is not enough for ChannelMessage.Read.All; the tenant must also
# submit Microsoft's Protected APIs request form and wait for approval.
PROTECTED_API_HINT = (
    "Microsoft Graph denied this request (403). If you have already "
    "granted admin consent for ChannelMessage.Read.All, note that Teams "
    "channel-message reads are a Microsoft 'Protected API' — admin "
    "consent alone is not enough. The tenant must also submit "
    "Microsoft's Protected APIs request form (search Microsoft Learn "
    "for 'Teams protected APIs') and wait for per-tenant approval "
    "(reviewed weekly). Listing teams/channels/members does not require "
    "this approval."
)


class TeamsAPIError(RuntimeError):
    """Raised for a non-2xx Microsoft Graph / token-endpoint response."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Microsoft Graph error ({status_code}): {message}")


def build_token_request(
    tenant_id: str, client_id: str, client_secret: str,
) -> tuple[str, dict[str, str]]:
    """Build the (url, form-data) pair for an OAuth2 client-credentials grant."""
    url = TOKEN_URL_TEMPLATE.format(tenant_id=tenant_id)
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": GRAPH_DEFAULT_SCOPE,
    }
    return url, data


def parse_token_response(body: dict[str, Any], *, now: float) -> dict[str, Any]:
    """Turn a token-endpoint JSON body into a cache entry.

    Returns ``{"access_token": ..., "expires_at": <epoch seconds>}``.
    Raises :class:`TeamsAPIError` (status 400) if the response has no
    usable access token — happens if the tenant rejects the grant for a
    reason that still returns HTTP 200 with an error-shaped body (rare,
    but defensive).
    """
    token = body.get("access_token")
    if not token:
        raise TeamsAPIError(400, str(body.get("error_description") or "no access_token in response"))
    try:
        expires_in = float(body.get("expires_in", 3600))
    except (TypeError, ValueError):
        expires_in = 3600.0
    return {"access_token": str(token), "expires_at": now + expires_in}


def token_is_valid(cache: Optional[dict[str, Any]], *, now: float, leeway: float = TOKEN_REFRESH_LEEWAY_SECS) -> bool:
    """Whether a cached token entry is still usable.

    Conservative: a missing/malformed cache entry is treated as invalid
    so callers always fall through to a fresh token request rather than
    risk spawning a request with a stale/garbage token.
    """
    if not cache or "access_token" not in cache or "expires_at" not in cache:
        return False
    try:
        expires_at = float(cache["expires_at"])
    except (TypeError, ValueError):
        return False
    return (now + leeway) < expires_at


def classify_error(status_code: int, body: Any) -> TeamsAPIError:
    """Build a :class:`TeamsAPIError` from a non-2xx status + JSON body."""
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            message = str(err.get("message") or "request failed")
        else:
            message = str(err or body.get("error_description") or "request failed")
    else:
        message = str(body) if body else "request failed"
    return TeamsAPIError(status_code, message)


def clamp_top(value: int, *, default: int = DEFAULT_TOP, max_top: int = MAX_TOP) -> int:
    """Clamp a user-supplied ``$top`` page size into ``[1, max_top]``."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    if v <= 0:
        return default
    return min(v, max_top)


def teams_list_params() -> dict[str, str]:
    """Query params for listing every Team-enabled group in the tenant."""
    return {
        "$filter": "resourceProvisioningOptions/Any(x:x eq 'Team')",
        "$select": "id,displayName,description",
    }


def format_team(raw: dict[str, Any]) -> dict[str, Any]:
    """Shrink a Graph group/team object down to the fields agents need."""
    return {
        "id": raw.get("id", ""),
        "display_name": raw.get("displayName", ""),
        "description": raw.get("description") or "",
    }


def format_channel(raw: dict[str, Any]) -> dict[str, Any]:
    """Shrink a Graph channel object down to the fields agents need."""
    return {
        "id": raw.get("id", ""),
        "display_name": raw.get("displayName", ""),
        "description": raw.get("description") or "",
        "membership_type": raw.get("membershipType", ""),
    }


def format_member(raw: dict[str, Any]) -> dict[str, Any]:
    """Shrink a Graph conversationMember object down to the fields agents need."""
    return {
        "id": raw.get("id", ""),
        "display_name": raw.get("displayName", ""),
        "roles": raw.get("roles", []),
        "email": raw.get("email", ""),
    }


def format_user(raw: dict[str, Any]) -> dict[str, Any]:
    """Shrink a Graph user object down to the fields agents need."""
    return {
        "id": raw.get("id", ""),
        "display_name": raw.get("displayName", ""),
        "mail": raw.get("mail") or raw.get("userPrincipalName", ""),
        "job_title": raw.get("jobTitle") or "",
    }


def format_channel_message(raw: dict[str, Any]) -> dict[str, Any]:
    """Shrink a Graph channelMessage object down to the fields agents need."""
    sender = (raw.get("from") or {}).get("user") or {}
    body = raw.get("body") or {}
    return {
        "id": raw.get("id", ""),
        "from": sender.get("displayName", ""),
        "created_at": raw.get("createdDateTime", ""),
        "subject": raw.get("subject") or "",
        "content": body.get("content", ""),
        "content_type": body.get("contentType", "text"),
    }
