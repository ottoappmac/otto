"""POST /start on a built-in that ships disabled turns it on (Start is the opt-in);
non-built-ins and hook servers stay refused."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.config import AppConfig, MCPServerConfig
from backend.routes.mcp import router as mcp_router


def _srv(sid: str, *, builtin: bool) -> MCPServerConfig:
    return MCPServerConfig(id=sid, name=sid, transport="stdio", command="/usr/bin/false",
                           args=[], enabled=False, builtin=builtin)


@pytest.fixture
def setup(monkeypatch):
    cfg = AppConfig()
    cfg.mcp_servers = [_srv("cad-tools", builtin=True), _srv("mine", builtin=False),
                       _srv("claude-eval-hook", builtin=True)]
    saves = []

    async def _aload():
        return cfg

    async def _asave(self):
        saves.append(True)

    monkeypatch.setattr("backend.routes.mcp.AppConfig.aload", classmethod(lambda cls: _aload()))
    monkeypatch.setattr(AppConfig, "asave", _asave)
    monkeypatch.setattr("backend.routes.mcp._missing_secrets", lambda srv: [])
    from backend.schemas import MCPAuthStatus
    monkeypatch.setattr("backend.routes.mcp._auth_status",
                        lambda srv: MCPAuthStatus(kind="static", needs_login=False))
    monkeypatch.setattr("backend.builtin_mcps.ensure_builtin_mcp_venvs", AsyncMock(return_value={}))
    mgr = MagicMock()
    mgr.connect = AsyncMock(return_value=MagicMock(connected=True, tools=[1, 2], error=None))
    mgr.is_process_running.return_value = True
    mgr.connections = {}
    monkeypatch.setattr("backend.routes.mcp.mcp_mgr", mgr)
    monkeypatch.setattr("backend.routes.mcp.session_mgr", MagicMock(refresh_tools=AsyncMock()))
    app = FastAPI()
    app.include_router(mcp_router)
    return TestClient(app), cfg, saves


def test_start_enables_disabled_builtin(setup):
    client, cfg, saves = setup
    r = client.post("/api/mcp-servers/cad-tools/start")
    assert r.status_code == 200, r.text
    assert r.json()["connected"] is True
    assert next(s for s in cfg.mcp_servers if s.id == "cad-tools").enabled is True
    assert saves


def test_start_still_refuses_user_server_and_hooks(setup):
    client, cfg, _ = setup
    assert client.post("/api/mcp-servers/mine/start").status_code == 400
    r = client.post("/api/mcp-servers/claude-eval-hook/start")
    assert r.status_code == 400 and "Settings" in r.json()["error"]
    assert not any(s.enabled for s in cfg.mcp_servers)


def test_connect_also_enables_disabled_builtin(setup):
    client, cfg, saves = setup
    r = client.post("/api/mcp-servers/cad-tools/connect")
    assert r.status_code == 200, r.text
    assert next(s for s in cfg.mcp_servers if s.id == "cad-tools").enabled is True and saves
    assert client.post("/api/mcp-servers/mine/connect").status_code == 400
