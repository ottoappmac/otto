"""Tests for VLM detection in :func:`supports_vision`.

Regression guard for oMLX/exo multimodal models (e.g. the Qwen3.x MoE VLMs)
that declare a ``vision_config`` but carry no "vl"/"vision" token in their
repo name — the name regex alone misclassified them as text-only.
"""

from __future__ import annotations

import json
from pathlib import Path

from deep_agent.model_factory import supports_vision


def test_cloud_providers_are_vision():
    assert supports_vision("anthropic", "claude-sonnet-4-6") is True
    assert supports_vision("openai", "gpt-4o") is True


def test_omlx_name_regex_still_matches():
    assert supports_vision("omlx", "mlx-community/Qwen2.5-VL-7B-Instruct-4bit") is True


def test_omlx_unknown_provider_is_false():
    assert supports_vision("ollama", "whatever") is False


def test_omlx_config_with_vision_config_is_detected(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"model_type": "qwen3_5_moe", "vision_config": {}}))
    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda repo, fname: str(cfg),
    )
    # No "vl"/"vision" token in the name, so this must come from the config.
    assert supports_vision("omlx", "mlx-community/Qwen3.6-35B-A3B-4bit") is True


def test_omlx_config_with_image_token_is_detected(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"model_type": "qwen3_5_moe", "image_token_id": 151655}))
    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda repo, fname: str(cfg),
    )
    assert supports_vision("omlx", "mlx-community/Some-MoE-4bit") is True


def test_omlx_text_only_config_is_false(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"model_type": "qwen3", "hidden_size": 4096}))
    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda repo, fname: str(cfg),
    )
    assert supports_vision("omlx", "mlx-community/Qwen3-8B-4bit") is False


def test_omlx_uncached_does_not_download(monkeypatch):
    calls = {"n": 0}

    def _fake(repo, fname):
        calls["n"] += 1
        return None  # not cached

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", _fake)
    assert supports_vision("exo", "some/unknown-model") is False
    assert calls["n"] == 1
