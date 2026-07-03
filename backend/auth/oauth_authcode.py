"""OAuth 2.0 authorization code grant with loopback ``redirect_uri``.

Use this when:

* The vendor only supports the auth-code flow (no device endpoint).
* The vendor accepts a registered loopback ``redirect_uri`` such as
  ``http://127.0.0.1`` (the standard for native / installed
  applications — see RFC 8252).

Flow
----
1. Bind a TCP socket on ``127.0.0.1:0`` so the OS picks a free port.
2. Build the authorize URL with ``response_type=code``, ``client_id``,
   ``redirect_uri=http://127.0.0.1:<port>/callback``, ``state``,
   ``code_challenge`` (PKCE S256), and the requested ``scope``.
3. Open the URL in the user's default browser.
4. Accept exactly one HTTP request on the bound port; verify ``state``;
   extract ``code``.
5. Exchange ``code`` + ``code_verifier`` at ``token_url`` for
   ``access_token`` (+ optional ``refresh_token``).

PKCE is on by default — there's no reason for a public native client
to skip it, and the major issuers all support it.

Refresh uses the standard ``grant_type=refresh_token`` exchange.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import secrets
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from backend.auth.base import AuthBundle, NeedsLoginError, register_provider
from backend.auth.utils import (
    expiry_from_seconds,
    is_bundle_expired,
    isoformat,
    now_utc,
    open_in_default_browser,
    project_env,
)

if TYPE_CHECKING:
    from backend.config import MCPAuthConfig


logger = logging.getLogger(__name__)


# Default flow timeout — the user has this long to complete the consent
# screen before the loopback listener gives up.  Generous enough to
# survive 2FA prompts on a slow phone but not so long that a forgotten
# tab pins a port forever.
_LOGIN_TIMEOUT_SECS = 300


_SUCCESS_HTML = (
    "<!doctype html><html><head><title>Login complete</title>"
    "<style>body{font-family:system-ui,sans-serif;max-width:480px;"
    "margin:80px auto;text-align:center;color:#1f2937}</style></head>"
    "<body><h2>Login complete</h2>"
    "<p>You can close this tab and return to Otto.</p></body></html>"
)

_ERROR_HTML = (
    "<!doctype html><html><head><title>Login failed</title>"
    "<style>body{font-family:system-ui,sans-serif;max-width:480px;"
    "margin:80px auto;text-align:center;color:#991b1b}</style></head>"
    "<body><h2>Login failed</h2>"
    "<p>The provider returned an error. You can close this tab and "
    "try again from Settings.</p></body></html>"
)


@register_provider
class OAuthAuthCodeProvider:
    """3LO auth-code provider for desktop / native installed clients."""

    kind: ClassVar[str] = "oauth_authcode"

    async def acquire(
        self, auth: "MCPAuthConfig", server_id: str,
    ) -> AuthBundle:
        if not auth.auth_url or not auth.token_url or not auth.client_id:
            raise NeedsLoginError(
                server_id, kind=self.kind,
                reason="auth_url, token_url, and client_id must be configured",
            )

        port = _pick_loopback_port()
        redirect_uri = f"http://127.0.0.1:{port}/callback"
        state = secrets.token_urlsafe(24)
        verifier = secrets.token_urlsafe(64)
        challenge = _pkce_challenge(verifier)

        authorize_url = self._build_authorize_url(
            auth, redirect_uri, state, challenge,
        )

        logger.info(
            "OAuth auth-code [%s]: listening on %s, opening %s",
            server_id, redirect_uri, auth.auth_url,
        )

        # Run the blocking single-shot HTTP server in a worker thread so
        # the FastAPI event loop stays free.  The ``open_in_default_browser``
        # call is non-blocking but kept on the same thread for ordering.
        async def _wait_for_code() -> str:
            return await asyncio.to_thread(
                _wait_for_callback, port, state, _LOGIN_TIMEOUT_SECS,
            )

        opened = open_in_default_browser(authorize_url)
        if not opened:
            logger.warning(
                "OAuth auth-code [%s]: could not open browser; user must visit %s",
                server_id, authorize_url,
            )

        try:
            code = await _wait_for_code()
        except TimeoutError:
            raise NeedsLoginError(
                server_id, kind=self.kind, reason="login timed out",
            )
        except RuntimeError as exc:
            raise NeedsLoginError(
                server_id, kind=self.kind, reason=str(exc),
            )

        return await self._exchange_code(auth, code, redirect_uri, verifier)

    async def refresh(
        self, auth: "MCPAuthConfig", server_id: str, bundle: AuthBundle,
    ) -> AuthBundle | None:
        if not bundle:
            return None
        if not is_bundle_expired(bundle):
            return bundle

        refresh_token = bundle.get("refresh_token")
        if not refresh_token or not auth.token_url or not auth.client_id:
            return None

        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": auth.client_id,
        }
        if auth.client_secret:
            data["client_secret"] = auth.client_secret

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(auth.token_url, data=data)
        except httpx.HTTPError as exc:
            logger.warning(
                "OAuth auth-code refresh [%s] network error: %s", server_id, exc,
            )
            return None

        if resp.status_code != 200:
            logger.info(
                "OAuth auth-code refresh [%s] rejected (HTTP %d): %s",
                server_id, resp.status_code, resp.text[:300],
            )
            return None

        return _bundle_from_token_response(resp.json(), refresh_token)

    def is_expired(self, bundle: AuthBundle) -> bool:
        return is_bundle_expired(bundle)

    def env_for(
        self, auth: "MCPAuthConfig", bundle: AuthBundle,
    ) -> dict[str, str]:
        return project_env(auth, bundle)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_authorize_url(
        auth: "MCPAuthConfig",
        redirect_uri: str,
        state: str,
        code_challenge: str,
    ) -> str:
        params: dict[str, str] = {
            "response_type": "code",
            "client_id": auth.client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        if auth.scopes:
            params["scope"] = " ".join(auth.scopes)
        # Merge provider-specific extras last so they can override defaults
        # (e.g. Google needs ``access_type=offline`` to return a refresh
        # token; Okta may need ``nonce``; Auth0 may need ``audience``).
        if auth.extra_auth_params:
            params.update(auth.extra_auth_params)
        sep = "&" if "?" in auth.auth_url else "?"
        return f"{auth.auth_url}{sep}{urlencode(params)}"

    async def _exchange_code(
        self,
        auth: "MCPAuthConfig",
        code: str,
        redirect_uri: str,
        code_verifier: str,
    ) -> AuthBundle:
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": auth.client_id,
            "code_verifier": code_verifier,
        }
        if auth.client_secret:
            data["client_secret"] = auth.client_secret

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(auth.token_url, data=data)
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Failed to reach token endpoint: {exc}"
            ) from exc

        if resp.status_code != 200:
            raise RuntimeError(
                f"Token exchange rejected (HTTP {resp.status_code}): "
                f"{resp.text[:300]}"
            )
        return _bundle_from_token_response(resp.json())


# ---------------------------------------------------------------------------
# Loopback HTTP listener — single-shot, single-thread, single-port
# ---------------------------------------------------------------------------


def _pick_loopback_port() -> int:
    """Reserve a free TCP port on 127.0.0.1 by binding-then-closing.

    There's a tiny race between releasing the port here and re-binding
    inside ``HTTPServer`` — acceptable on a desktop host where nothing
    else is racing for the same ephemeral port.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def _pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _wait_for_callback(port: int, expected_state: str, timeout_secs: int) -> str:
    """Block on a one-shot HTTP server and return the parsed ``code``.

    Validates the ``state`` parameter to prevent CSRF.  Surfaces
    provider-side errors (``?error=...``) as :class:`RuntimeError`.
    Raises :class:`TimeoutError` if no request lands within
    ``timeout_secs``.
    """

    captured: dict[str, Any] = {}

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 — http.server protocol
            parsed = urlparse(self.path)
            params = {k: v[0] for k, v in parse_qs(parsed.query).items()}

            if "error" in params:
                captured["error"] = (
                    f"{params['error']}: {params.get('error_description', '')}"
                )
                self._respond(400, _ERROR_HTML)
                return

            if params.get("state") != expected_state:
                captured["error"] = "state mismatch"
                self._respond(400, _ERROR_HTML)
                return

            code = params.get("code", "")
            if not code:
                captured["error"] = "missing authorization code"
                self._respond(400, _ERROR_HTML)
                return

            captured["code"] = code
            self._respond(200, _SUCCESS_HTML)

        def _respond(self, status: int, body: str) -> None:
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            # Silence the default access-log line so a successful login
            # doesn't dump the auth code into stderr.
            return

    server = HTTPServer(("127.0.0.1", port), _Handler)
    server.timeout = max(timeout_secs, 1)
    try:
        server.handle_request()
    finally:
        server.server_close()

    if "error" in captured:
        raise RuntimeError(captured["error"])
    if "code" not in captured:
        raise TimeoutError("no callback received within timeout")
    return str(captured["code"])


def _bundle_from_token_response(
    body: dict[str, Any], fallback_refresh: str | None = None,
) -> AuthBundle:
    """Same shape as the device-flow conversion, factored to module scope."""
    bundle: AuthBundle = {
        "access_token": str(body.get("access_token", "")),
        "token_type": str(body.get("token_type", "Bearer")),
        "obtained_iso": isoformat(now_utc()),
    }
    if "expires_in" in body:
        bundle["expiry_iso"] = expiry_from_seconds(body["expires_in"])

    refresh = body.get("refresh_token") or fallback_refresh
    if refresh:
        bundle["refresh_token"] = str(refresh)

    known = {"access_token", "refresh_token", "token_type", "expires_in"}
    extra = {k: str(v) for k, v in body.items() if k not in known}
    if extra:
        bundle["extra"] = extra
    return bundle
