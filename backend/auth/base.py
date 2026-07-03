"""Auth provider Protocol, shared types, and registry.

The Protocol is intentionally narrow — four async/sync methods cover
acquisition, refresh, expiry inspection, and the projection from a
token bundle to env vars.  Anything provider-specific (OAuth URLs,
browser landing pages, header names) lives on
:class:`backend.config.MCPAuthConfig` so a single ``AuthProvider``
implementation handles every flavour of its kind.

Trust boundary
--------------
Providers run in the **parent** backend process.  They have read/write
access to :mod:`backend.credential_vault` (where bundles live) but the
generated MCP subprocess and the LLM never see ``AuthBundle`` directly.
The MCP manager calls :meth:`AuthProvider.env_for` and forwards only
that flat ``dict[str, str]`` into the subprocess environment.
"""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Protocol,
    TypedDict,
    runtime_checkable,
)

if TYPE_CHECKING:
    from backend.config import MCPAuthConfig


logger = logging.getLogger(__name__)


class AuthBundle(TypedDict, total=False):
    """The shape persisted in the credential vault for non-static auth.

    All fields are optional because different flows populate different
    subsets — an OAuth device flow returns ``access_token`` +
    ``refresh_token`` + ``expiry_iso``; a browser-capture flow may return
    only ``access_token`` + ``expiry_iso``; an opaque vendor flow may
    stash everything under ``extra``.

    ``extra`` carries provider-specific fields (e.g. an AWS SSO
    ``accountId`` / ``roleName``) so the bundle stays loosely typed at
    rest while still being introspectable.
    """

    access_token: str
    refresh_token: str
    token_type: str
    expiry_iso: str
    obtained_iso: str
    extra: dict[str, str]


class NeedsLoginError(RuntimeError):
    """Raised when the manager can't hydrate env because no bundle is usable.

    The HTTP layer turns this into a structured 400 response so the
    frontend can swap its "Set credentials" button for a "Login" button
    targeting :func:`backend.routes.mcp.login_mcp_auth`.

    Attributes mirror the fields the frontend needs to render the
    affordance — never include token contents here.
    """

    def __init__(self, server_id: str, kind: str, reason: str = "no_bundle"):
        self.server_id = server_id
        self.kind = kind
        self.reason = reason
        super().__init__(
            f"MCP {server_id!r} requires {kind!r} login (reason: {reason})"
        )


@runtime_checkable
class AuthProvider(Protocol):
    """Contract every auth-flow implementation must satisfy."""

    kind: ClassVar[str]
    """Registry key — must match :class:`MCPAuthConfig.kind`."""

    async def acquire(
        self, auth: "MCPAuthConfig", server_id: str,
    ) -> AuthBundle:
        """Run the interactive flow once and return a fresh bundle.

        Called from a route handler when the user clicks "Login" in
        Settings.  Implementations may block on a browser interaction —
        callers should ``await`` with a generous timeout (the FastAPI
        route is fine; user gesture means HTTP keep-alive isn't a
        concern here).

        Should raise :exc:`NeedsLoginError` only if the flow itself
        depends on a precondition the caller should fix before retrying
        (e.g. ``client_id`` not configured).  Other failures should
        raise :exc:`RuntimeError` with an actionable message — the route
        layer turns those into 500-style responses.
        """
        ...

    async def refresh(
        self, auth: "MCPAuthConfig", server_id: str, bundle: AuthBundle,
    ) -> AuthBundle | None:
        """Try to mint a fresh access token without user interaction.

        Returns the new bundle on success, ``None`` if refresh isn't
        possible (no refresh token, refresh endpoint rejected the
        request, flow doesn't support silent renewal).  ``None`` is the
        signal for the manager to raise :exc:`NeedsLoginError` and let
        the user re-authenticate.

        MUST be a no-op when ``not self.is_expired(bundle)`` — callers
        rely on this so they can call ``refresh`` before every spawn
        without paying network cost on the happy path.
        """
        ...

    def is_expired(self, bundle: AuthBundle) -> bool:
        """Return True iff the access token in ``bundle`` is past its expiry.

        Should be conservative (treat unknown / unparseable expiries as
        expired) so we never spawn a subprocess with a stale token.
        """
        ...

    def env_for(
        self, auth: "MCPAuthConfig", bundle: AuthBundle,
    ) -> dict[str, str]:
        """Project a bundle into the env dict the subprocess receives.

        Honors ``auth.env_mapping`` so an MCP author can declare e.g.
        ``{"GITHUB_TOKEN": "access_token"}`` and have ``access_token``
        from the bundle land under ``GITHUB_TOKEN`` in the subprocess
        environment.  Bundle keys not referenced in the mapping are NOT
        forwarded — defence in depth, the subprocess never sees more of
        the bundle than the spec asked for.
        """
        ...


_REGISTRY: dict[str, type[AuthProvider]] = {}


def register_provider(cls: type[AuthProvider]) -> type[AuthProvider]:
    """Class decorator: add ``cls`` to the lookup table keyed by ``cls.kind``.

    Used by each provider module so importing :mod:`backend.auth`
    populates the registry as a side-effect of import.  Re-registering
    the same kind raises — provider replacement is a deliberate edit
    rather than something to do silently at import time.
    """
    kind = getattr(cls, "kind", None)
    if not isinstance(kind, str) or not kind:
        raise TypeError(f"{cls.__name__} is missing a non-empty 'kind' ClassVar")
    if kind in _REGISTRY and _REGISTRY[kind] is not cls:
        raise RuntimeError(
            f"Auth provider kind {kind!r} already registered to "
            f"{_REGISTRY[kind].__name__}; cannot replace with {cls.__name__}"
        )
    _REGISTRY[kind] = cls
    logger.debug("Registered auth provider %s → %s", kind, cls.__name__)
    return cls


def get_provider(kind: str) -> AuthProvider:
    """Instantiate the registered provider for *kind*.

    Providers are stateless — they read everything they need from
    ``MCPAuthConfig`` and the persisted bundle — so a fresh instance
    per call is fine and avoids any cross-MCP state leakage.
    """
    cls = _REGISTRY.get(kind)
    if cls is None:
        raise KeyError(
            f"Unknown auth kind {kind!r}.  Registered: {sorted(_REGISTRY)}"
        )
    return cls()


def available_kinds() -> list[str]:
    """Return every registered ``kind`` (for diagnostics + frontend hints)."""
    return sorted(_REGISTRY)


__all__ = [
    "AuthBundle",
    "AuthProvider",
    "NeedsLoginError",
    "available_kinds",
    "get_provider",
    "register_provider",
]
