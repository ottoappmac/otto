"""GPT-5/GPT-6/o-series get temperature 1 and max_completion_tokens; older models keep theirs."""

from __future__ import annotations

import pytest

from deep_agent.model_factory import _is_modern_openai_model, _openai_temperature, _openai_token_kwargs


@pytest.mark.parametrize("name", ["gpt-5", "gpt-5-mini", "gpt-6", "gpt-6-astra", "gpt-6.1", "o1-preview", "o3-mini-2025-01-31"])
def test_modern_models_use_default_temperature_and_completion_tokens(name):
    assert _is_modern_openai_model(name)
    assert _openai_temperature(name, 0.0) == 1.0
    assert _openai_token_kwargs(name, 4096) == {"model_kwargs": {"max_completion_tokens": 4096}}


@pytest.mark.parametrize("name", ["gpt-4o", "gpt-4.1", "gpt-4o-mini", "gpt-60x", ""])
def test_older_models_keep_their_settings(name):
    assert not _is_modern_openai_model(name)
    assert _openai_temperature(name, 0.2) == 0.2
    assert _openai_token_kwargs(name, 4096) == {"max_tokens": 4096}


def test_azure_gpt6_deployment_gets_modern_params(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL_PROVIDER", "azure")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_AZURE_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("OPENAI_AZURE_DEPLOYMENT", "gpt-6-astra")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "gpt-6-astra")
    from deep_agent.model_factory import create_llm

    from langchain_core.messages import HumanMessage

    llm = create_llm("openai")
    payload = llm._get_request_payload([HumanMessage("hi")])  # what goes on the wire
    assert payload["temperature"] == 1.0
    assert payload.get("max_completion_tokens")
    assert "max_tokens" not in payload
