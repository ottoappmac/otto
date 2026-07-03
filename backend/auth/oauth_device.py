"""RFC 8628 device authorization grant.

The device flow is the right choice when:

* The MCP runs on a host that can launch the user's browser but isn't
  a registered web client (no public ``redirect_uri``).
* The vendor exposes the standard three endpoints
  (``device_authorization``, ``token``, optional ``refresh_token``).

Most modern OAuth providers support this — Google, Microsoft / Entra
ID, GitHub, Auth0, Okta, AWS IAM Identity Center.  The wire protocol
is identical across them; only the endpoint URLs and the ``client_id``
change, so a single provider serves every issuer.

Flow
----
1. POST ``device_url`` with ``client_id`` (+ optional ``client_secret``,
   ``scope``).  Receive ``device_code``, ``user_code``,
   ``verification_uri[_complete]``, ``expires_in``, ``interval``.
2. Open ``verification_uri_complete`` in the user's browser.  The user
   approves on whatever device is convenient.
3. Poll ``token_url`` every ``interval`` seconds with
   ``grant_type=urn:ietf:params:oauth:grant-type:device_code`` and the
   ``device_code``.  RFC 8628 §3.5 dictates ``authorization_pending`` /
   ``slow_down`` / ``access_denied`` / ``expired_token`` semantics.
4. On success, persist ``access_token`` + ``refresh_token`` (when
   present) + computed expiry.

Refresh uses the standard ``grant_type=refresh_token`` exchange.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, ClassVar

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


_DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
_REFRESH_GRANT = "refresh_token"
_DEFAULT_INTERVAL = 5  # seconds — applied if the issuer omits "interval"
_SLOW_DOWN_BUMP = 5    # RFC 8628 §3.5: increase poll interval by ≥5s on slow_down


@register_provider
class OAuthDeviceProvider:
    """Standards-compliant device authorization grant.

    Stateless — every method takes the auth spec + bundle as args, so
    a single instance is fine across MCPs.  All HTTP work goes through
    ``httpx.AsyncClient`` with a short connect timeout so a slow OAuth
    server can't tie up the FastAPI worker indefinitely.
    """

    kind: ClassVar[str] = "oauth_device"

    async def acquire(
        self, auth: "MCPAuthConfig", server_id: str,
    ) -> AuthBundle:
        if not auth.device_url or not auth.token_url or not auth.client_id:
            raise NeedsLoginError(
                server_id, kind=self.kind,
                reason="device_url, token_url, and client_id must be configured",
            )

        async with httpx.AsyncClient(timeout=20.0) as client:
            device = await self._start_device_authorization(client, auth)
            verification_url = (
                device.get("verification_uri_complete")
                or device.get("verification_uri")
                or ""
            )
            if not verification_url:
                raise RuntimeError(
                    "Device authorization response missing verification URI"
                )

            logger.info(
                "OAuth device flow [%s]: opening verification URL %s",
                server_id, verification_url,
            )
            open_in_default_browser(verification_url)

            return await self._poll_for_token(client, auth, device, server_id)

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
            "grant_type": _REFRESH_GRANT,
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
                "OAuth device refresh [%s] network error: %s", server_id, exc,
            )
            return None

        if resp.status_code != 200:
            logger.info(
                "OAuth device refresh [%s] rejected (HTTP %d): %s",
                server_id, resp.status_code, resp.text[:300],
            )
            return None

        return self._bundle_from_token_response(resp.json(), refresh_token)

    def is_expired(self, bundle: AuthBundle) -> bool:
        return is_bundle_expired(bundle)

    def env_for(
        self, auth: "MCPAuthConfig", bundle: AuthBundle,
    ) -> dict[str, str]:
        return project_env(auth, bundle)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _start_device_authorization(
        self, client: httpx.AsyncClient, auth: "MCPAuthConfig",
    ) -> dict[str, Any]:
        data: dict[str, str] = {"client_id": auth.client_id}
        if auth.client_secret:
            data["client_secret"] = auth.client_secret
        if auth.scopes:
            data["scope"] = " ".join(auth.scopes)

        try:
            resp = await client.post(auth.device_url, data=data)
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Failed to reach device authorization endpoint: {exc}"
            ) from exc

        if resp.status_code != 200:
            raise RuntimeError(
                f"Device authorization rejected (HTTP {resp.status_code}): "
                f"{resp.text[:300]}"
            )
        body = resp.json()
        if not body.get("device_code"):
            raise RuntimeError(
                "Device authorization response missing device_code"
            )
        return body

    async def _poll_for_token(
        self,
        client: httpx.AsyncClient,
        auth: "MCPAuthConfig",
        device: dict[str, Any],
        server_id: str,
    ) -> AuthBundle:
        device_code = device["device_code"]
        interval = max(int(device.get("interval", _DEFAULT_INTERVAL)), 1)
        expires_in = int(device.get("expires_in", 600))
        deadline = now_utc().timestamp() + expires_in

        data = {
            "grant_type": _DEVICE_GRANT,
            "device_code": device_code,
            "client_id": auth.client_id,
        }
        if auth.client_secret:
            data["client_secret"] = auth.client_secret

        while True:
            await asyncio.sleep(interval)

            if now_utc().timestamp() >= deadline:
                raise NeedsLoginError(
                    server_id, kind=self.kind, reason="device code expired",
                )

            try:
                resp = await client.post(auth.token_url, data=data)
            except httpx.HTTPError as exc:
                # Transient: keep polling — RFC 8628 mandates the client
                # keep retrying until ``expires_in`` runs out.
                logger.debug(
                    "Device flow poll [%s] transient error: %s", server_id, exc,
                )
                continue

            if resp.status_code == 200:
                return self._bundle_from_token_response(resp.json())

            err = self._parse_error(resp)
            if err == "authorization_pending":
                continue
            if err == "slow_down":
                interval += _SLOW_DOWN_BUMP
                continue
            if err == "access_denied":
                raise NeedsLoginError(
                    server_id, kind=self.kind, reason="access_denied",
                )
            if err == "expired_token":
                raise NeedsLoginError(
                    server_id, kind=self.kind, reason="device code expired",
                )
            raise RuntimeError(
                f"Token endpoint returned {resp.status_code}: {resp.text[:300]}"
            )

    @staticmethod
    def _parse_error(resp: httpx.Response) -> str:
        try:
            return str(resp.json().get("error", "")).lower()
        except Exception:
            return ""

    @staticmethod
    def _bundle_from_token_response(
        body: dict[str, Any], fallback_refresh: str | None = None,
    ) -> AuthBundle:
        """Convert an OAuth token endpoint response into our bundle shape."""
        bundle: AuthBundle = {
            "access_token": str(body.get("access_token", "")),
            "token_type": str(body.get("token_type", "Bearer")),
            "obtained_iso": isoformat(now_utc()),
        }
        if "expires_in" in body:
            bundle["expiry_iso"] = expiry_from_seconds(body["expires_in"])

        # Some issuers (Google, Auth0) only send the refresh token on
        # the very first exchange — preserve the previous one across
        # silent refreshes so we don't lose the ability to renew later.
        refresh = body.get("refresh_token") or fallback_refresh
        if refresh:
            bundle["refresh_token"] = str(refresh)

        # Stash any non-standard fields under ``extra`` so vendor-specific
        # data (e.g. ``id_token``, ``scope``, ``user_id``) survives a
        # round-trip without polluting the typed bundle slots.
        known = {
            "access_token", "refresh_token", "token_type", "expires_in",
        }
        extra = {k: str(v) for k, v in body.items() if k not in known}
        if extra:
            bundle["extra"] = extra
        return bundle
