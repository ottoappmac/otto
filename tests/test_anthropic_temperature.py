"""Tests for Anthropic sampling-parameter routing in model_factory."""

from __future__ import annotations

from deep_agent.model_factory import (
    _anthropic_llm_kwargs,
    _anthropic_rejects_temperature,
)


def test_older_anthropic_models_accept_temperature():
    assert _anthropic_rejects_temperature("claude-sonnet-4-6") is False
    assert _anthropic_rejects_temperature("claude-opus-4-6") is False
    assert _anthropic_rejects_temperature("claude-haiku-4-5-20251001") is False


def test_modern_anthropic_models_reject_temperature():
    assert _anthropic_rejects_temperature("claude-opus-4-7") is True
    assert _anthropic_rejects_temperature("claude-opus-4-8") is True
    assert _anthropic_rejects_temperature("claude-opus-5") is True
    assert _anthropic_rejects_temperature("claude-sonnet-5") is True
    assert _anthropic_rejects_temperature("claude-fable-5") is True


def test_bedrock_anthropic_model_ids_reject_temperature():
    assert _anthropic_rejects_temperature(
        "us.anthropic.claude-opus-4-7-20260416-v1:0",
    ) is True


def test_anthropic_llm_kwargs_omit_temperature_for_modern_models():
    kwargs = _anthropic_llm_kwargs("claude-opus-4-8")
    assert kwargs == {"model": "claude-opus-4-8"}
    assert "temperature" not in kwargs


def test_anthropic_llm_kwargs_include_temperature_for_older_models():
    kwargs = _anthropic_llm_kwargs("claude-sonnet-4-6")
    assert kwargs == {"model": "claude-sonnet-4-6", "temperature": 0.0}
