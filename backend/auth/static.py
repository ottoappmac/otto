"""Static credential provider — the existing "paste a string" behaviour.

This is the default for every MCP that doesn't opt into a richer flow.
It keeps the historical contract verbatim:

* ``MCPServerConfig.required_secrets`` lists the env-var names the
  subprocess needs.
* The user fills them in Settings → Credentials, which writes one entry
  per name into the OS keychain via the credential vault.
* The MCP manager hydrates those names back into ``os.environ`` at
  spawn time.

Modelling this as a ``StaticPasteProvider`` lets the manager call
``get_provider(auth.kind)`` uniformly across every flow without a
special case for the static path.

There is no acquisition flow, no refresh, no expiry — the user pastes
once and the value stays valid until they overwrite it.  ``acquire``
raises so it's a hard error if a route ever tries to "log in" a static
MCP, which would always be a frontend bug.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from backend.auth.base import AuthBundle, NeedsLoginError, register_provider

if TYPE_CHECKING:
    from backend.config import MCPAuthConfig


@register_provider
class StaticPasteProvider:
    """Provider that maps directly to the existing per-name vault rows."""

    kind: ClassVar[str] = "static"

    async def acquire(
        self, auth: "MCPAuthConfig", server_id: str,
    ) -> AuthBundle:
        # The frontend should never POST /auth/login for a static MCP.
        # Returning a stub bundle here would mask that bug; a clear
        # error keeps the contract honest.
        raise NeedsLoginError(
            server_id, kind=self.kind,
            reason="static credentials are entered via the credentials dialog",
        )

    async def refresh(
        self, auth: "MCPAuthConfig", server_id: str, bundle: AuthBundle,
    ) -> AuthBundle | None:
        # Static creds don't expire.  The vault layer is already the
        # source of truth — there's nothing to renew.
        return bundle or None

    def is_expired(self, bundle: AuthBundle) -> bool:
        return False

    def env_for(
        self, auth: "MCPAuthConfig", bundle: AuthBundle,
    ) -> dict[str, str]:
        # Static credentials are projected by ``mcp_manager._hydrate_secrets``
        # directly out of the vault using ``required_secrets``; this
        # provider doesn't carry a bundle of its own.
        return {}
