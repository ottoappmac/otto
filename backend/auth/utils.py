"""Shared helpers for auth providers.

Three concerns live here:

* JWT / ISO expiry inspection — every flow needs a "is this still good?"
  check before spawn, and the logic is identical regardless of how the
  token was acquired.
* System browser discovery — both the OAuth auth-code provider (opens
  the consent screen) and the browser-capture provider (launches a CDP
  Chrome) need to find a usable browser binary.
* Env projection — turning an :class:`AuthBundle` into a flat env dict
  using the user-declared ``env_mapping``.

Keeping these out of the per-provider modules avoids importing PyJWT
or platform-specific code three times and gives tests a single
attachment point.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from backend.auth.base import AuthBundle
    from backend.config import MCPAuthConfig


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Expiry helpers
#
# Tokens carry expiries in three places, depending on the issuer:
#  * ``bundle["expiry_iso"]`` — what we computed at acquisition time.
#  * The JWT ``exp`` claim — when the token IS a JWT.
#  * Vendor-specific ``extra`` fields — last-resort.
#
# We treat any unparseable / missing expiry as "expired" so the manager
# never spawns a subprocess holding a stale token.
# ---------------------------------------------------------------------------

# Refresh slightly before the wire expiry to absorb clock skew + the
# few seconds it takes to spawn the MCP subprocess.
_EXPIRY_SKEW = timedelta(seconds=60)


def parse_iso(ts: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp into an aware UTC datetime, or None."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def expiry_from_seconds(expires_in: int | float) -> str:
    """Build an absolute ISO expiry from a relative ``expires_in`` integer.

    OAuth token endpoints return ``expires_in`` (seconds remaining); we
    persist absolute timestamps so reload after a process restart still
    works without recomputing the offset.
    """
    return isoformat(now_utc() + timedelta(seconds=int(expires_in)))


def jwt_exp_iso(token: str | None) -> str | None:
    """Return the ISO expiry encoded in *token*'s ``exp`` claim, or None.

    Pure decoder — does NOT verify the signature (we're inspecting our
    own freshly-minted bearer tokens to learn their expiry, not
    validating arbitrary input).  Falls back through PyJWT, then a
    manual base64 decode, then giving up.
    """
    if not token or token.count(".") != 2:
        return None

    exp: int | float | None = None
    try:
        import jwt  # type: ignore

        claims = jwt.decode(token, options={"verify_signature": False})
        exp = claims.get("exp")
    except Exception:
        exp = _manual_jwt_exp(token)

    if exp is None:
        return None
    try:
        return isoformat(datetime.fromtimestamp(float(exp), tz=timezone.utc))
    except (OverflowError, OSError, ValueError):
        return None


def _manual_jwt_exp(token: str) -> int | float | None:
    """Fallback JWT ``exp`` decoder used when PyJWT isn't available."""
    try:
        payload_b64 = token.split(".")[1]
        # JWT uses URL-safe base64 without padding — restore the padding.
        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
        return payload.get("exp")
    except Exception:
        return None


def is_bundle_expired(bundle: "AuthBundle") -> bool:
    """Whether the access token in *bundle* should be considered stale.

    Conservative: an empty / unparseable / missing expiry is treated as
    expired so the manager refreshes (or surfaces a NeedsLogin) rather
    than launching the subprocess and crashing on first call.
    """
    if not bundle:
        return True

    expiry = parse_iso(bundle.get("expiry_iso"))
    if expiry is None:
        # No tracked expiry — try the JWT itself as a fallback.
        expiry_iso = jwt_exp_iso(bundle.get("access_token"))
        expiry = parse_iso(expiry_iso)

    if expiry is None:
        return True
    return now_utc() + _EXPIRY_SKEW >= expiry


# ---------------------------------------------------------------------------
# Env projection
#
# Lifted into a helper so both built-in providers and any future
# third-party provider produce the same env dict from the same bundle.
# ---------------------------------------------------------------------------


def project_env(
    auth: "MCPAuthConfig", bundle: "AuthBundle",
) -> dict[str, str]:
    """Map ``auth.env_mapping`` over *bundle*, dropping unset values.

    ``env_mapping`` is ``{ENV_VAR_NAME: bundle_field}`` where
    ``bundle_field`` is one of the typed keys (``access_token``,
    ``refresh_token``, ``token_type``, ``expiry_iso``,
    ``obtained_iso``) OR ``"extra.<name>"`` to pull from the loose
    ``extra`` sub-dict.

    Missing fields are silently dropped — the subprocess gets only the
    env vars whose values exist.  This keeps the contract simple: an
    MCP that lists ``GITHUB_TOKEN`` in ``required_secrets`` knows the
    var will be set when the env is hydrated, OR the manager will have
    raised :exc:`NeedsLoginError` upstream.
    """
    if not bundle:
        return {}
    extra = bundle.get("extra") or {}
    out: dict[str, str] = {}
    for env_name, source in (auth.env_mapping or {}).items():
        if source.startswith("extra."):
            value = extra.get(source[len("extra."):], "")
        else:
            value = bundle.get(source, "")  # type: ignore[arg-type]
        if value:
            out[env_name] = str(value)
    return out


# ---------------------------------------------------------------------------
# System browser discovery
#
# ``browser_capture`` needs a Chromium-family binary it can launch with
# ``--remote-debugging-port``; ``oauth_authcode`` only needs to open a
# URL and is happy with whatever the user's default browser is.
# ---------------------------------------------------------------------------


def find_chromium_browser() -> Optional[str]:
    """Locate a Chrome / Edge / Chromium executable for CDP capture.

    Returns ``None`` when nothing usable is installed.  The caller
    surfaces that as a clear "install Chrome" error rather than
    falling back to the default browser, since CDP is Chromium-only.
    """
    if sys.platform == "darwin":
        candidates = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        ]
    elif sys.platform == "win32":
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        local = os.environ.get("LOCALAPPDATA", "")
        candidates = [
            rf"{pf}\Google\Chrome\Application\chrome.exe",
            rf"{pf86}\Google\Chrome\Application\chrome.exe",
            rf"{local}\Google\Chrome\Application\chrome.exe",
            rf"{pf}\Microsoft\Edge\Application\msedge.exe",
            rf"{pf86}\Microsoft\Edge\Application\msedge.exe",
        ]
    else:
        candidates = [
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/usr/bin/chromium-browser",
            "/usr/bin/chromium",
            "/usr/bin/microsoft-edge",
        ]
    for path in candidates:
        if Path(path).exists():
            return path
    return None


def open_in_default_browser(url: str) -> bool:
    """Open *url* using the OS default browser; return True on success.

    Used by ``oauth_device`` and ``oauth_authcode`` — both flows just
    need the user to land on the verification page; we don't care
    which browser does it.
    """
    try:
        import webbrowser

        return webbrowser.open(url, new=1, autoraise=True)
    except Exception as exc:
        logger.warning("Failed to open browser for %s: %s", url, exc)
        return False


# ---------------------------------------------------------------------------
# Host allowlist (defence against malicious manifests)
#
# An LLM-authored MCP manifest could try to send the user through a
# phishing intermediary during a ``browser_capture`` flow.  We require
# every redirect destination to share a registrable suffix with at
# least one entry in ``MCPAuthConfig.allowed_hosts``.
# ---------------------------------------------------------------------------


def host_is_allowed(url: str, allowed_hosts: list[str]) -> bool:
    """Whether the host of *url* matches any entry in *allowed_hosts*.

    Match is "exact OR ends with ``.<entry>``" so an entry of
    ``example.com`` covers ``api.example.com`` but NOT
    ``evil-example.com``.  Empty allowlist means "no restriction" —
    used by built-in providers that have no manifest exposure (e.g.
    ``oauth_device`` which only ever talks to URLs the spec author
    declared statically).
    """
    if not allowed_hosts:
        return True
    try:
        from urllib.parse import urlparse

        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    if not host:
        return False
    for entry in allowed_hosts:
        e = entry.strip().lower().lstrip(".")
        if not e:
            continue
        if host == e or host.endswith("." + e):
            return True
    return False


__all__ = [
    "expiry_from_seconds",
    "find_chromium_browser",
    "host_is_allowed",
    "is_bundle_expired",
    "isoformat",
    "jwt_exp_iso",
    "now_utc",
    "open_in_default_browser",
    "parse_iso",
    "project_env",
]
