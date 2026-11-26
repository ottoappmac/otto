"""POST /api/sessions shows model-setup errors to the user instead of a generic 500."""

from __future__ import annotations

import asyncio
import json

import pytest

from backend.routes import sessions as routes
from backend.schemas import SessionCreateRequest


class _Cfg:
    class llm:  # noqa: N801 — mimics AppConfig.llm
        provider = "mlx"


def _create_with(monkeypatch, exc: Exception):
    async def aload():
        return _Cfg()

    async def create_session(**_kwargs):
        raise exc

    monkeypatch.setattr(routes.AppConfig, "aload", staticmethod(aload))
    monkeypatch.setattr(routes.session_mgr, "create_session", create_session)
    resp = asyncio.run(routes.api_create_session(SessionCreateRequest()))
    return resp.status_code, json.loads(resp.body)


@pytest.mark.parametrize("exc", [
    ValueError("mlx-community/Inkling-2-Mini-oQ4e-mtp (model_type 'inkling2_loop') runs only on Turbo (oMLX)"),
    FileNotFoundError("MLX model 'x/y' has no weight files in the local cache"),
])
def test_model_setup_errors_reach_the_user(monkeypatch, exc):
    status, body = _create_with(monkeypatch, exc)
    assert status == 400
    assert body["error"] == str(exc)
    assert body["error_code"] == "model_setup"


def test_unexpected_errors_stay_generic(monkeypatch):
    status, body = _create_with(monkeypatch, RuntimeError("internal detail /Users/someone/secret"))
    assert status == 500
    assert "secret" not in body["error"]
    assert body["error"].startswith("Session creation failed")
