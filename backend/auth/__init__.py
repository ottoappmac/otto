"""Pluggable authentication providers for MCP servers.

The :mod:`backend.mcp_builder` and :mod:`backend.mcp_manager` modules
treat every credential the same way: a flat string projected into
``os.environ`` at subprocess spawn time.  That works for API keys, PATs,
bot tokens, and other "paste me a string" credentials.

Some integrations need an interactive login flow instead — OAuth 2.0
device or auth-code grants, or browser-based bearer-token capture for
SaaS that authenticates through SSO.  Each flow runs in the **parent**
backend process; the resulting tokens are stored as a structured bundle
in the OS keychain and projected into the MCP subprocess environment as
plain env vars at spawn time, so the subprocess and the LLM remain on
the same flat-env contract they have today.

Public API
----------
* :class:`AuthBundle` — the persisted token shape.
* :class:`AuthProvider` — Protocol every provider implements.
* :class:`NeedsLoginError` — raised when the manager can't hydrate env
  because no usable bundle exists; callers turn this into a UI prompt.
* :func:`get_provider` — registry lookup keyed by ``MCPAuthConfig.kind``.
* :func:`available_kinds` — every registered provider id.

Built-in providers (each in its own module):

==================== ============================================
``static``           Paste-a-string (current behaviour, no flow).
``oauth_device``     RFC 8628 device authorization grant.
``oauth_authcode``   3LO auth code with loopback ``redirect_uri``.
``browser_capture``  CDP-based bearer-token sniff for SSO logins.
==================== ============================================
"""

from __future__ import annotations

from backend.auth.base import (
    AuthBundle,
    AuthProvider,
    NeedsLoginError,
    available_kinds,
    get_provider,
    register_provider,
)

# Importing each module registers it via the @register_provider decorator.
# Order is alphabetical so the registry is deterministic — only matters
# for diagnostics like ``available_kinds()`` output.
from backend.auth import browser_capture  # noqa: F401
from backend.auth import oauth_authcode  # noqa: F401
from backend.auth import oauth_device  # noqa: F401
from backend.auth import static  # noqa: F401


__all__ = [
    "AuthBundle",
    "AuthProvider",
    "NeedsLoginError",
    "available_kinds",
    "get_provider",
    "register_provider",
]
