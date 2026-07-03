"""Tests for wiring the macos-native read_screen vision combo to subagents.

Covers the provider/model derivation that drives the text→image-combo swap and
the swap itself. These are pure-Python helpers (no macOS / pyobjc needed).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.schemas import AgentSpec
from backend.session_manager import (
    _apply_macos_vision_variant,
    _provider_model_for_orchestrator,
    _provider_model_for_subagent,
    _split_provider_model,
)


def _config(provider: str = "omlx", omlx_model: str = "mlx-community/Qwen3.6-35B-A3B-4bit"):
    return SimpleNamespace(
        llm=SimpleNamespace(
            provider=provider,
            anthropic=SimpleNamespace(model_name="claude-sonnet-4-6"),
            openai=SimpleNamespace(model_name="gpt-4o"),
            mlx=SimpleNamespace(hf_llm_model_id="mlx-community/Qwen3-8B-4bit"),
        ),
        omlx=SimpleNamespace(model_name=omlx_model),
        exo=SimpleNamespace(model_name="mlx-community/Qwen2.5-VL-7B-Instruct-4bit"),
        orchestrator=SimpleNamespace(provider_override=None, llm_family="follow_main", mlx_model=""),
    )


def test_split_provider_model():
    assert _split_provider_model("anthropic:claude-sonnet-4-6") == ("anthropic", "claude-sonnet-4-6")
    assert _split_provider_model("gpt-4o") == ("", "gpt-4o")


def test_orchestrator_follow_main_uses_main():
    cfg = _config(provider="omlx")
    assert _provider_model_for_orchestrator(cfg) == ("omlx", "mlx-community/Qwen3.6-35B-A3B-4bit")


def test_orchestrator_frontier_is_anthropic():
    cfg = _config()
    cfg.orchestrator.llm_family = "frontier"
    assert _provider_model_for_orchestrator(cfg) == ("anthropic", "claude-sonnet-4-6")


def test_subagent_inherit_uses_parent():
    cfg = _config()
    spec = AgentSpec(name="a", description="d", subagent_llm_family="inherit")
    assert _provider_model_for_subagent(spec, cfg, "omlx", "some-model") == ("omlx", "some-model")


def test_subagent_none_with_override_splits():
    cfg = _config()
    spec = AgentSpec(name="a", description="d", model_override="anthropic:claude-sonnet-4-6")
    assert _provider_model_for_subagent(spec, cfg, "omlx", "x") == ("anthropic", "claude-sonnet-4-6")


def test_subagent_frontier_is_anthropic():
    cfg = _config()
    spec = AgentSpec(name="a", description="d", subagent_llm_family="frontier")
    assert _provider_model_for_subagent(spec, cfg, "omlx", "x") == ("anthropic", "claude-sonnet-4-6")


def test_subagent_exo_uses_exo_model():
    cfg = _config()
    spec = AgentSpec(name="a", description="d", subagent_llm_family="exo")
    prov, mid = _provider_model_for_subagent(spec, cfg, "omlx", "x")
    assert prov == "exo"
    assert "VL" in mid  # the exo model in the fake config is a VL model


class _FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name


def test_apply_macos_vision_variant_swaps_and_adds():
    text_read = _FakeTool("read_screen")
    vision_read = _FakeTool("read_screen")  # distinct object, same name
    capture = _FakeTool("capture_app_screenshot")
    other = _FakeTool("click_at")
    conn = SimpleNamespace(
        macos_native_vision={"read_screen": vision_read, "extra": [capture]},
    )

    out = _apply_macos_vision_variant([text_read, other], conn)

    # read_screen replaced by the vision object; capture appended; other kept.
    by_name = {t.name: t for t in out}
    assert by_name["read_screen"] is vision_read
    assert by_name["capture_app_screenshot"] is capture
    assert other in out
    assert len(out) == 3


def test_apply_macos_vision_variant_noop_without_stash():
    text_read = _FakeTool("read_screen")
    conn = SimpleNamespace(macos_native_vision=None)
    out = _apply_macos_vision_variant([text_read], conn)
    assert out == [text_read]


def test_apply_macos_vision_variant_no_duplicate_capture():
    # If capture is already present, it must not be duplicated.
    vision_read = _FakeTool("read_screen")
    capture = _FakeTool("capture_app_screenshot")
    existing_capture = _FakeTool("capture_app_screenshot")
    conn = SimpleNamespace(
        macos_native_vision={"read_screen": vision_read, "extra": [capture]},
    )
    out = _apply_macos_vision_variant([_FakeTool("read_screen"), existing_capture], conn)
    captures = [t for t in out if t.name == "capture_app_screenshot"]
    assert len(captures) == 1
