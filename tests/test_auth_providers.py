"""Unit tests for the :mod:`backend.auth` provider package.

Coverage:

* ``utils.is_bundle_expired`` / ``parse_iso`` / ``jwt_exp_iso``
* ``utils.project_env`` honours ``env_mapping`` and the ``extra.*`` shorthand
* ``utils.host_is_allowed`` enforces the suffix match the
  ``browser_capture`` provider relies on
* The provider registry is populated for every shipped kind, returns
  Protocol-compatible instances, and rejects unknown kinds
* ``StaticPasteProvider`` is a no-op for env projection (so the manager
  uses its existing ``_hydrate_static_secrets`` path) and refuses to
  trigger an interactive flow
* OAuth-device + OAuth-authcode silent ``refresh`` paths short-circuit
  on a non-expired bundle, hit the configured ``token_url`` for an
  expired bundle, and return ``None`` when refresh is impossible
* ``BrowserCaptureProvider.refresh`` is correctly "no path" — it returns
  ``None`` on expired bundles so the manager raises ``NeedsLoginError``

The tests intentionally avoid touching real browsers or OAuth servers;
all HTTP calls are stubbed via ``httpx.MockTransport`` so the suite
runs deterministically in CI.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest

from backend.auth import (
    NeedsLoginError,
    available_kinds,
    get_provider,
)
from backend.auth.base import AuthProvider
from backend.auth.utils import (
    expiry_from_seconds,
    host_is_allowed,
    is_bundle_expired,
    isoformat,
    jwt_exp_iso,
    now_utc,
    parse_iso,
    project_env,
)
from backend.auth.oauth_authcode import OAuthAuthCodeProvider
from backend.auth.oauth_device import OAuthDeviceProvider
from backend.auth.browser_capture import BrowserCaptureProvider
from backend.auth.static import StaticPasteProvider
from backend.config import MCPAuthConfig


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------


class TestExpiryHelpers:
    def test_parse_iso_aware_returns_utc(self):
        dt = parse_iso("2030-01-01T00:00:00+09:00")
        assert dt is not None
        assert dt.tzinfo == timezone.utc
        assert dt.hour == 15  # 00:00 JST -> 15:00 UTC the prior day
        assert dt.day == 31

    def test_parse_iso_naive_assumed_utc(self):
        dt = parse_iso("2030-01-01T00:00:00")
        assert dt is not None
        assert dt.tzinfo == timezone.utc

    def test_parse_iso_invalid_returns_none(self):
        assert parse_iso("garbage") is None
        assert parse_iso("") is None
        assert parse_iso(None) is None

    def test_expiry_from_seconds_is_in_future(self):
        iso = expiry_from_seconds(60)
        dt = parse_iso(iso)
        assert dt is not None
        assert dt > now_utc()

    def test_is_bundle_expired_uses_explicit_expiry(self):
        future = isoformat(now_utc() + timedelta(minutes=10))
        past = isoformat(now_utc() - timedelta(minutes=10))
        assert not is_bundle_expired({"access_token": "x", "expiry_iso": future})
        assert is_bundle_expired({"access_token": "x", "expiry_iso": past})

    def test_is_bundle_expired_treats_missing_as_expired(self):
        assert is_bundle_expired({})
        assert is_bundle_expired({"access_token": "x"})

    def test_jwt_exp_iso_decodes_payload(self):
        # Hand-crafted JWT with exp set far in the future; signature ignored.
        import base64
        import json

        future_ts = int((now_utc() + timedelta(hours=1)).timestamp())
        header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
        payload = base64.urlsafe_b64encode(
            json.dumps({"exp": future_ts}).encode("utf-8"),
        ).rstrip(b"=").decode()
        token = f"{header}.{payload}.signature"

        iso = jwt_exp_iso(token)
        assert iso is not None
        dt = parse_iso(iso)
        assert dt is not None
        assert dt > now_utc()

    def test_jwt_exp_iso_returns_none_for_non_jwt(self):
        assert jwt_exp_iso("not-a-jwt") is None
        assert jwt_exp_iso(None) is None


class TestProjectEnv:
    def test_maps_typed_keys(self):
        auth = MCPAuthConfig(
            kind="oauth_device",
            env_mapping={"GITHUB_TOKEN": "access_token", "GITHUB_TYPE": "token_type"},
        )
        bundle = {"access_token": "T", "token_type": "Bearer", "obtained_iso": "2030"}
        env = project_env(auth, bundle)
        assert env == {"GITHUB_TOKEN": "T", "GITHUB_TYPE": "Bearer"}

    def test_drops_missing_fields(self):
        auth = MCPAuthConfig(
            kind="oauth_device",
            env_mapping={"GITHUB_TOKEN": "access_token", "EXTRA": "refresh_token"},
        )
        env = project_env(auth, {"access_token": "T"})
        assert env == {"GITHUB_TOKEN": "T"}

    def test_extra_shorthand(self):
        auth = MCPAuthConfig(
            kind="oauth_device",
            env_mapping={"VENDOR_ID": "extra.vendor_user_id"},
        )
        env = project_env(auth, {"extra": {"vendor_user_id": "u-99"}})
        assert env == {"VENDOR_ID": "u-99"}

    def test_empty_bundle_returns_empty_dict(self):
        auth = MCPAuthConfig(kind="oauth_device", env_mapping={"X": "access_token"})
        assert project_env(auth, {}) == {}


class TestHostIsAllowed:
    def test_empty_allowlist_means_anywhere(self):
        assert host_is_allowed("https://anything.example", [])

    def test_exact_match(self):
        assert host_is_allowed("https://api.example.com", ["api.example.com"])

    def test_subdomain_match(self):
        assert host_is_allowed("https://eu.api.example.com", ["example.com"])

    def test_lookalike_rejected(self):
        # "evil-example.com" must NOT pass an "example.com" allowlist.
        assert not host_is_allowed("https://evil-example.com", ["example.com"])

    def test_invalid_url_rejected(self):
        assert not host_is_allowed("not-a-url", ["example.com"])


# ---------------------------------------------------------------------------
# Registry + Protocol shape
# ---------------------------------------------------------------------------


class TestProviderRegistry:
    def test_all_expected_kinds_registered(self):
        kinds = set(available_kinds())
        assert kinds == {"static", "oauth_device", "oauth_authcode", "browser_capture"}

    @pytest.mark.parametrize("kind", [
        "static", "oauth_device", "oauth_authcode", "browser_capture",
    ])
    def test_get_provider_returns_protocol_satisfying_instance(self, kind):
        provider = get_provider(kind)
        assert isinstance(provider, AuthProvider)
        assert provider.kind == kind

    def test_get_provider_rejects_unknown_kind(self):
        with pytest.raises(KeyError):
            get_provider("definitely-not-a-real-kind")


# ---------------------------------------------------------------------------
# Static provider
# ---------------------------------------------------------------------------


class TestStaticProvider:
    @pytest.mark.asyncio
    async def test_acquire_raises_needs_login(self):
        provider = StaticPasteProvider()
        with pytest.raises(NeedsLoginError):
            await provider.acquire(MCPAuthConfig(), "stripe")

    def test_env_for_returns_empty(self):
        provider = StaticPasteProvider()
        assert provider.env_for(MCPAuthConfig(), {"access_token": "x"}) == {}

    def test_is_expired_always_false(self):
        provider = StaticPasteProvider()
        assert not provider.is_expired({})


# ---------------------------------------------------------------------------
# OAuth device — refresh path with httpx.MockTransport
# ---------------------------------------------------------------------------


@pytest.fixture
def device_auth() -> MCPAuthConfig:
    return MCPAuthConfig(
        kind="oauth_device",
        client_id="abc123",
        device_url="https://example.test/device",
        token_url="https://example.test/token",
        scopes=["repo"],
        env_mapping={"GITHUB_TOKEN": "access_token"},
    )


@pytest.fixture
def authcode_auth() -> MCPAuthConfig:
    return MCPAuthConfig(
        kind="oauth_authcode",
        client_id="abc123",
        auth_url="https://example.test/authorize",
        token_url="https://example.test/token",
        scopes=["repo"],
        env_mapping={"GITHUB_TOKEN": "access_token"},
    )


def _mock_transport(handler):
    return httpx.MockTransport(handler)


def _patch_httpx(monkeypatch, module_path: str, handler):
    """Replace ``<module>.httpx.AsyncClient`` with a transport-injecting
    factory.

    We capture the real :class:`httpx.AsyncClient` BEFORE monkey-patching
    so the inner factory can still call it — otherwise the lambda
    recurses into itself once the import-time alias is overwritten.
    """
    real_client = httpx.AsyncClient

    def _factory(*args, **kwargs):
        kwargs["transport"] = _mock_transport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(f"{module_path}.httpx.AsyncClient", _factory)


class TestOAuthDeviceRefresh:
    @pytest.mark.asyncio
    async def test_refresh_returns_bundle_when_not_expired(self, device_auth):
        provider = OAuthDeviceProvider()
        future = isoformat(now_utc() + timedelta(hours=1))
        bundle = {"access_token": "T", "refresh_token": "R", "expiry_iso": future}
        result = await provider.refresh(device_auth, "github", bundle)
        assert result == bundle

    @pytest.mark.asyncio
    async def test_refresh_returns_none_without_refresh_token(self, device_auth):
        provider = OAuthDeviceProvider()
        past = isoformat(now_utc() - timedelta(hours=1))
        bundle = {"access_token": "T", "expiry_iso": past}
        assert await provider.refresh(device_auth, "github", bundle) is None

    @pytest.mark.asyncio
    async def test_refresh_calls_token_endpoint(self, monkeypatch, device_auth):
        provider = OAuthDeviceProvider()
        past = isoformat(now_utc() - timedelta(hours=1))
        bundle = {"access_token": "old", "refresh_token": "R", "expiry_iso": past}

        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = bytes(request.content).decode("utf-8")
            return httpx.Response(200, json={
                "access_token": "new",
                "expires_in": 3600,
                "token_type": "Bearer",
            })

        _patch_httpx(monkeypatch, "backend.auth.oauth_device", handler)

        result = await provider.refresh(device_auth, "github", bundle)
        assert result is not None
        assert result["access_token"] == "new"
        # Refresh-token rotation: when the issuer omits a new
        # ``refresh_token`` we MUST preserve the old one so the user
        # doesn't get pushed back to a full login on the next renewal.
        assert result["refresh_token"] == "R"
        assert captured["url"] == "https://example.test/token"
        assert "grant_type=refresh_token" in captured["body"]
        assert "refresh_token=R" in captured["body"]

    @pytest.mark.asyncio
    async def test_refresh_returns_none_on_endpoint_rejection(
        self, monkeypatch, device_auth,
    ):
        provider = OAuthDeviceProvider()
        past = isoformat(now_utc() - timedelta(hours=1))
        bundle = {"access_token": "old", "refresh_token": "R", "expiry_iso": past}

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": "invalid_grant"})

        _patch_httpx(monkeypatch, "backend.auth.oauth_device", handler)
        assert await provider.refresh(device_auth, "github", bundle) is None


class TestOAuthAuthCodeRefresh:
    @pytest.mark.asyncio
    async def test_refresh_short_circuits_when_fresh(self, authcode_auth):
        provider = OAuthAuthCodeProvider()
        future = isoformat(now_utc() + timedelta(hours=1))
        bundle = {"access_token": "T", "refresh_token": "R", "expiry_iso": future}
        assert await provider.refresh(authcode_auth, "g", bundle) == bundle

    @pytest.mark.asyncio
    async def test_refresh_uses_token_url(self, monkeypatch, authcode_auth):
        provider = OAuthAuthCodeProvider()
        past = isoformat(now_utc() - timedelta(hours=1))
        bundle = {"access_token": "old", "refresh_token": "R", "expiry_iso": past}

        captured_urls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_urls.append(str(request.url))
            return httpx.Response(200, json={
                "access_token": "fresh",
                "expires_in": 3600,
                "refresh_token": "R2",
                "token_type": "Bearer",
            })

        _patch_httpx(monkeypatch, "backend.auth.oauth_authcode", handler)
        result = await provider.refresh(authcode_auth, "g", bundle)
        assert result is not None
        assert result["access_token"] == "fresh"
        # When the issuer DID rotate the refresh token, we adopt the new one.
        assert result["refresh_token"] == "R2"
        assert captured_urls == ["https://example.test/token"]


# ---------------------------------------------------------------------------
# Browser capture — refresh has no silent path
# ---------------------------------------------------------------------------


class TestBrowserCaptureRefresh:
    @pytest.mark.asyncio
    async def test_returns_bundle_when_fresh(self):
        provider = BrowserCaptureProvider()
        auth = MCPAuthConfig(kind="browser_capture", landing_url="https://x.test",
                             allowed_hosts=["x.test"],
                             env_mapping={"T": "access_token"})
        future = isoformat(now_utc() + timedelta(hours=1))
        bundle = {"access_token": "tok-with-enough-len-1234567", "expiry_iso": future}
        assert await provider.refresh(auth, "vendor", bundle) == bundle

    @pytest.mark.asyncio
    async def test_returns_none_when_expired(self):
        provider = BrowserCaptureProvider()
        auth = MCPAuthConfig(kind="browser_capture", landing_url="https://x.test",
                             allowed_hosts=["x.test"],
                             env_mapping={"T": "access_token"})
        past = isoformat(now_utc() - timedelta(hours=1))
        bundle = {"access_token": "tok-with-enough-len-1234567", "expiry_iso": past}
        # No refresh path for browser-captured tokens — surface as a
        # signal that NeedsLoginError should be raised upstream.
        assert await provider.refresh(auth, "vendor", bundle) is None

    @pytest.mark.asyncio
    async def test_acquire_rejects_disallowed_landing_url(self):
        provider = BrowserCaptureProvider()
        auth = MCPAuthConfig(
            kind="browser_capture",
            landing_url="https://evil.test",
            allowed_hosts=["safe.test"],
            env_mapping={"T": "access_token"},
        )
        with pytest.raises(NeedsLoginError):
            await provider.acquire(auth, "vendor")
