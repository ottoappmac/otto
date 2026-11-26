"""Masked ``••••`` secrets sent back by the settings form must resolve to
the saved values before hitting a provider (regression: after an app
restart, Anthropic list-models / test-connection failed with
"'ascii' codec can't encode characters in position 0-3")."""

import asyncio
from types import SimpleNamespace

import pytest

from backend.routes import settings as settings_routes
from backend.schemas import TestConnectionRequest


def _fake_cfg():
    return SimpleNamespace(
        llm=SimpleNamespace(
            anthropic=SimpleNamespace(
                api_key="sk-ant-real",
                aws_access_key_id="AKIA",
                aws_secret_access_key="secret",
            ),
            openai=SimpleNamespace(api_key="sk-oa", azure_api_key="az-key"),
            google=SimpleNamespace(api_key="g-key"),
            mlx=SimpleNamespace(hf_token="hf"),
        )
    )


@pytest.fixture
def patched_load(monkeypatch):
    async def _aload():
        return _fake_cfg()

    monkeypatch.setattr(settings_routes.AppConfig, "aload", staticmethod(_aload))


P = settings_routes._REDACTED_PLACEHOLDER


@pytest.mark.parametrize(
    "kwargs,field,expected",
    [
        ({"provider": "anthropic", "api_key": P}, "api_key", "sk-ant-real"),
        ({"provider": "openai", "api_key": P}, "api_key", "sk-oa"),
        ({"provider": "openai", "openai_model_provider": "azure", "api_key": P}, "api_key", "az-key"),
        ({"provider": "google", "api_key": P}, "api_key", "g-key"),
        ({"provider": "anthropic", "model_provider": "bedrock", "aws_secret_access_key": P},
         "aws_secret_access_key", "secret"),
    ],
)
def test_placeholder_resolved(patched_load, kwargs, field, expected):
    req = TestConnectionRequest(**kwargs)
    asyncio.run(settings_routes._resolve_placeholder_secrets(req))
    assert getattr(req, field) == expected


def test_typed_key_is_kept(patched_load):
    req = TestConnectionRequest(provider="anthropic", api_key="sk-ant-new")
    asyncio.run(settings_routes._resolve_placeholder_secrets(req))
    assert req.api_key == "sk-ant-new"
