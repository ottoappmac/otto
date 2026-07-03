"""Integration-style tests for the ``/api/mcp-servers/{id}/auth/...`` routes.

We spin up a FastAPI test client around just the MCP router so we
don't pay the full backend startup cost (which would also try to
contact the real keychain, scheduler, MCP processes, etc.).  Every
external dependency the routes touch is patched:

* ``backend.config.AppConfig.aload`` — returns a hand-built config
  with one ``oauth_device``-flavoured server.
* ``backend.credential_vault.vault`` — replaced with an in-memory
  fake (same shape as :mod:`tests.test_credential_vault_bundle`).
* The OAuth provider ``acquire`` — patched to return a canned bundle
  so the test never opens a browser.

Coverage:

* ``GET /auth/status`` returns a static-default for legacy servers
  and a populated payload for interactive ones.
* ``POST /auth/login`` calls ``provider.acquire`` and writes the
  bundle to the vault.
* ``POST /auth/login`` rejects static MCPs with a clear error.
* ``POST /auth/logout`` wipes the bundle.
* ``POST /start`` short-circuits with a structured ``needs_login``
  error when the bundle is missing for an interactive server.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.config import AppConfig, MCPAuthConfig, MCPServerConfig
from backend.routes.mcp import router as mcp_router


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


class _FakeVault:
    """In-memory replacement for :class:`backend.credential_vault.vault`.

    Implements just the surface the routes touch: ``get_bundle`` /
    ``set_bundle`` / ``has_bundle`` / ``delete_bundle`` / ``has`` /
    ``list_names``.  The route handlers are the only consumers.
    """

    def __init__(self) -> None:
        self.bundles: dict[str, dict] = {}
        self.statics: dict[tuple[str, str], str] = {}

    def get_bundle(self, sid: str) -> dict | None:
        return self.bundles.get(sid)

    def set_bundle(self, sid: str, bundle: dict) -> None:
        self.bundles[sid] = dict(bundle)

    def has_bundle(self, sid: str) -> bool:
        return sid in self.bundles

    def delete_bundle(self, sid: str) -> bool:
        return self.bundles.pop(sid, None) is not None

    def has(self, sid: str, name: str) -> bool:
        return (sid, name) in self.statics

    def get(self, sid: str, name: str) -> str | None:
        return self.statics.get((sid, name))

    def list_names(self, sid: str) -> list[str]:
        return [n for (s, n) in self.statics if s == sid]


def _interactive_srv() -> MCPServerConfig:
    return MCPServerConfig(
        id="github",
        name="GitHub",
        transport="stdio",
        command="/usr/bin/false",
        args=[],
        enabled=True,
        generated=True,
        auth=MCPAuthConfig(
            kind="oauth_device",
            client_id="cid",
            device_url="https://x.test/device",
            token_url="https://x.test/token",
            env_mapping={"GITHUB_TOKEN": "access_token"},
        ),
    )


def _static_srv() -> MCPServerConfig:
    return MCPServerConfig(
        id="stripe",
        name="Stripe",
        transport="stdio",
        command="/usr/bin/false",
        args=[],
        enabled=True,
        generated=True,
        required_secrets=["STRIPE_SECRET_KEY"],
    )


@pytest.fixture
def app(monkeypatch):
    """Build a minimal FastAPI app exposing the MCP router with all
    external dependencies stubbed.
    """
    fake_vault = _FakeVault()

    cfg = AppConfig()
    cfg.mcp_servers = [_interactive_srv(), _static_srv()]

    async def _aload() -> AppConfig:
        return cfg

    monkeypatch.setattr(AppConfig, "aload", classmethod(lambda cls: _aload()))
    monkeypatch.setattr(
        "backend.routes.mcp.AppConfig.aload",
        classmethod(lambda cls: _aload()),
    )

    # Patch BOTH import paths the route uses for the vault — one inside
    # ``_auth_status`` (lazy-imported) and one inside the login route.
    monkeypatch.setattr("backend.credential_vault.vault", fake_vault)

    # Stub mcp_mgr so list / status responses don't try to hit a real
    # MCPManager (it has live state across tests when shared).
    fake_mgr = MagicMock()
    fake_mgr.connections = {}
    fake_mgr.is_process_running.return_value = False
    monkeypatch.setattr("backend.routes.mcp.mcp_mgr", fake_mgr)

    application = FastAPI()
    application.include_router(mcp_router)
    application.state.fake_vault = fake_vault  # exposed for assertions
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# /auth/status
# ---------------------------------------------------------------------------


def test_auth_status_for_static_server(client: TestClient):
    r = client.get("/api/mcp-servers/stripe/auth/status")
    assert r.status_code == 200
    data = r.json()
    assert data == {
        "kind": "static",
        "has_bundle": False,
        "expired": False,
        "needs_login": False,
        "expiry_iso": None,
    }


def test_auth_status_for_interactive_server_without_bundle(client: TestClient):
    r = client.get("/api/mcp-servers/github/auth/status")
    assert r.status_code == 200
    data = r.json()
    assert data["kind"] == "oauth_device"
    assert data["has_bundle"] is False
    assert data["needs_login"] is True


def test_auth_status_with_bundle_present(app: FastAPI, client: TestClient):
    from datetime import timedelta

    from backend.auth.utils import isoformat, now_utc

    future = isoformat(now_utc() + timedelta(hours=1))
    app.state.fake_vault.set_bundle("github", {
        "access_token": "tok",
        "refresh_token": "ref",
        "expiry_iso": future,
    })
    r = client.get("/api/mcp-servers/github/auth/status")
    assert r.status_code == 200
    data = r.json()
    assert data["has_bundle"] is True
    assert data["expired"] is False
    assert data["needs_login"] is False
    assert data["expiry_iso"] == future


def test_auth_status_unknown_server(client: TestClient):
    r = client.get("/api/mcp-servers/nope/auth/status")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# /auth/login
# ---------------------------------------------------------------------------


def test_login_invokes_provider_and_persists_bundle(
    app: FastAPI, client: TestClient, monkeypatch,
):
    fake_provider = MagicMock()
    fake_provider.acquire = AsyncMock(return_value={
        "access_token": "freshly-minted",
        "refresh_token": "rt",
        "token_type": "Bearer",
        "expiry_iso": "2099-01-01T00:00:00+00:00",
    })

    def _get_provider(kind: str) -> Any:
        assert kind == "oauth_device"
        return fake_provider

    monkeypatch.setattr("backend.auth.get_provider", _get_provider)

    r = client.post("/api/mcp-servers/github/auth/login")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["auth"]["has_bundle"] is True

    persisted = app.state.fake_vault.get_bundle("github")
    assert persisted is not None
    assert persisted["access_token"] == "freshly-minted"


def test_login_for_static_server_returns_400(client: TestClient):
    r = client.post("/api/mcp-servers/stripe/auth/login")
    assert r.status_code == 400
    body = r.json()
    assert body["error"] == "static_auth"


# ---------------------------------------------------------------------------
# /auth/logout
# ---------------------------------------------------------------------------


def test_logout_wipes_bundle(app: FastAPI, client: TestClient):
    app.state.fake_vault.set_bundle("github", {"access_token": "T"})
    r = client.post("/api/mcp-servers/github/auth/logout")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "logged_out"
    assert app.state.fake_vault.get_bundle("github") is None


def test_logout_when_no_bundle(client: TestClient):
    r = client.post("/api/mcp-servers/github/auth/logout")
    assert r.status_code == 200
    assert r.json()["status"] == "no_bundle"


# ---------------------------------------------------------------------------
# /start short-circuit on needs_login
# ---------------------------------------------------------------------------


def test_start_returns_needs_login_when_bundle_missing(
    app: FastAPI, client: TestClient, monkeypatch,
):
    # No bundle in vault — /start MUST refuse with the structured
    # ``needs_login`` payload so the frontend renders a Login button
    # instead of trying to spawn the MCP and failing.
    monkeypatch.setattr("backend.routes.mcp._missing_secrets", lambda srv: [])
    r = client.post("/api/mcp-servers/github/start")
    assert r.status_code == 400
    body = r.json()
    assert body["error"] == "needs_login"
    assert body["auth_kind"] == "oauth_device"
    assert body["server_id"] == "github"
