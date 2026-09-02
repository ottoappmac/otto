"""Tests for Anthropic sampling-parameter routing in model_factory."""

from __future__ import annotations

from deep_agent.model_factory import (
    _anthropic_locks_sampling_params,
    _anthropic_temperature,
)


def test_older_anthropic_models_accept_temperature():
    assert _anthropic_locks_sampling_params("claude-sonnet-4-6") is False
    assert _anthropic_locks_sampling_params("claude-opus-4-6") is False
    assert _anthropic_locks_sampling_params("claude-haiku-4-5-20251001") is False


def test_modern_anthropic_models_lock_sampling_params():
    assert _anthropic_locks_sampling_params("claude-opus-4-7") is True
    assert _anthropic_locks_sampling_params("claude-opus-4-8") is True
    assert _anthropic_locks_sampling_params("claude-opus-5") is True
    assert _anthropic_locks_sampling_params("claude-sonnet-5") is True


def test_bedrock_anthropic_model_ids_lock_sampling_params():
    assert _anthropic_locks_sampling_params(
        "us.anthropic.claude-opus-4-7-20260416-v1:0",
    ) is True


def test_anthropic_temperature_omits_for_locked_models():
    assert _anthropic_temperature("claude-opus-4-8") is None


def test_anthropic_temperature_includes_for_older_models():
    assert _anthropic_temperature("claude-sonnet-4-6") == 0.0
