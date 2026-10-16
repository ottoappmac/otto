"""Per-launch token for the local Otto API.

The backend binds to localhost, but any process running as the same
user could call it. This token is the bar for that. It is not a
defence against a process that can read the token file: the file is
mode 0600 in the app-data directory, and a program with the user's
privileges can open it. What it does stop is a sandboxed shell or MCP
subprocess, which cannot read that directory and does not inherit the
token in its environment.

``OTTO_API_TOKEN_REQUIRED`` is read on each request. It defaults to on.
Tests set it to ``0`` before importing the app.
"""

from __future__ import annotations

import hmac
import logging
import os
import secrets
from urllib.parse import parse_qs

logger = logging.getLogger(__name__)

_TOKEN: str | None = None


def token_required() -> bool:
    raw = os.environ.get("OTTO_API_TOKEN_REQUIRED", "1").strip().lower()
    return raw not in ("0", "false", "no", "off", "")


def token_path():
    from backend.config import get_app_data_dir
    return get_app_data_dir() / "api_token"


def ensure_token() -> str:
    """Return the process token, creating the file if this process has none.

    An ``OTTO_API_TOKEN`` from the environment is taken once and then
    removed, so child processes (MCP servers, ``execute``) do not
    inherit it. A later reload still sees it if the parent shell
    exported it into the new process.
    """
    global _TOKEN
    if _TOKEN:
        return _TOKEN
    supplied = os.environ.pop("OTTO_API_TOKEN", "").strip()
    token = supplied or secrets.token_urlsafe(32)
    path = token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        logger.warning("api token: could not chmod 0600 %s", path)
    _TOKEN = token
    return token


def _presented(scope: dict) -> str:
    headers = {
        key.decode("latin1").lower(): value.decode("latin1")
        for key, value in scope.get("headers") or []
    }
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    if headers.get("x-otto-token"):
        return headers["x-otto-token"].strip()
    query = scope.get("query_string") or b""
    if query:
        parsed = parse_qs(query.decode("latin1"), keep_blank_values=False)
        values = parsed.get("token") or []
        if values:
            return values[0]
    return ""


class ApiTokenMiddleware:
    """Reject HTTP and WebSocket requests that do not carry the token."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") not in ("http", "websocket") or not token_required():
            await self.app(scope, receive, send)
            return
        if scope.get("type") == "http" and scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)
            return
        expected = ensure_token()
        got = _presented(scope)
        if got and hmac.compare_digest(got, expected):
            await self.app(scope, receive, send)
            return
        body = b'{"error":"missing or invalid API token"}'
        if scope.get("type") == "websocket":
            await send({
                "type": "websocket.http.response.start",
                "status": 401,
                "headers": [(b"content-type", b"application/json")],
            })
            await send({"type": "websocket.http.response.body", "body": body})
            return
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        })
        await send({"type": "http.response.body", "body": body})
