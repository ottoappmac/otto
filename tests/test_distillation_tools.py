"""Agent distillation tools, purpose patch, and spawn-on-LoRA helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.distillation.adapter_meta import write_adapter_meta
from backend.distillation.catalog import iter_trained_adapters, set_adapter_purpose
from backend.distillation_tools import (
    build_distillation_tools,
    build_distilled_adapters_prompt_block,
)
from backend.schemas import SessionInfo
from backend.session_manager import Session, _distill_llm_override


def _ready_adapter(tmp_path: Path, *, purpose: str = "mail triage") -> Path:
    ready = tmp_path / "adapters" / "ready"
    ready.mkdir(parents=True)
    (ready / "adapters.safetensors").write_bytes(b"lora")
    write_adapter_meta(
        ready,
        base_repo_id="mlx-community/Qwen3-8B-4bit",
        extra={
            "catalog_id": "otto-distill/ready",
            "display_name": "Ready distilled",
            "kind": "activity",
            "purpose": purpose,
            "teacher_model_id": "mlx-community/Qwen3-32B-4bit",
        },
    )
    return ready


def test_set_adapter_purpose_patches_sidecar(tmp_path: Path, monkeypatch):
    _ready_adapter(tmp_path, purpose="")
    monkeypatch.setattr(
        "backend.distillation.catalog.adapters_dir",
        lambda **k: tmp_path / "adapters",
    )
    rec = set_adapter_purpose("otto-distill/ready", "calendar from activity")
    assert rec is not None
    assert rec["purpose"] == "calendar from activity"
    listed = iter_trained_adapters()
    assert listed[0]["purpose"] == "calendar from activity"


def test_capability_ladder_mentions_distill_when_tools_bound():
    from langchain_core.tools import tool as lc_tool
    from deep_agent.prompt import build_orchestrator_prompt

    @lc_tool
    def use_distilled_model() -> str:
        """bind"""
        return ""

    prompt = build_orchestrator_prompt([use_distilled_model])
    assert "Distilled LoRA" in prompt
    assert "use_distilled_model" in prompt
    assert "NEW session" in prompt


def test_prompt_block_includes_purpose(tmp_path: Path, monkeypatch):
    _ready_adapter(tmp_path)
    monkeypatch.setattr(
        "backend.distillation.catalog.adapters_dir",
        lambda **k: tmp_path / "adapters",
    )
    block = build_distilled_adapters_prompt_block("otto-distill/ready")
    assert "<distilled_adapters>" in block
    assert "otto-distill/ready" in block
    assert "[BOUND]" in block
    assert "mail triage" in block
    assert "use_distilled_model" in block


def test_resolved_distill_catalog_id_from_mlx_settings(monkeypatch, tmp_path):
    from backend.session_manager import _resolved_distill_catalog_id

    _ready_adapter(tmp_path)
    monkeypatch.setattr(
        "backend.distillation.catalog.adapters_dir",
        lambda **k: tmp_path / "adapters",
    )
    mlx_cfg = SimpleNamespace(hf_llm_model_id="otto-distill/ready")
    cfg = SimpleNamespace(llm=SimpleNamespace(provider="mlx", mlx=mlx_cfg))
    assert _resolved_distill_catalog_id(None, cfg) == "otto-distill/ready"

    cfg.llm.provider = "omlx"
    assert _resolved_distill_catalog_id(None, cfg) is None
    assert _resolved_distill_catalog_id("otto-distill/ready", cfg) == "otto-distill/ready"
    assert _resolved_distill_catalog_id("otto-distill/missing", cfg) is None


def test_resolved_distill_catalog_id_from_omlx_fused(monkeypatch):
    from backend.session_manager import _omlx_input_budget, _omlx_native_context_window
    from backend.session_manager import _resolved_distill_catalog_id

    monkeypatch.setattr(
        "backend.distillation.catalog.catalog_id_for_omlx_model",
        lambda mid, **k: "otto-distill/ready" if mid == "Qwen3-fused" else None,
    )
    monkeypatch.setattr(
        "backend.distillation.catalog.resolve_catalog_id",
        lambda cid, **k: {"catalog_id": cid} if cid == "otto-distill/ready" else None,
    )
    cfg = SimpleNamespace(
        llm=SimpleNamespace(provider="omlx", mlx=SimpleNamespace(hf_llm_model_id="")),
        omlx=SimpleNamespace(model_name="Qwen3-fused", max_context_window=131072, max_tokens=10240),
    )
    assert _resolved_distill_catalog_id(None, cfg) == "otto-distill/ready"


def test_omlx_input_budget_uses_fused_window(tmp_path, monkeypatch):
    from backend.session_manager import _omlx_input_budget, _omlx_native_context_window

    fused = tmp_path / "fused" / "Qwen3-fused"
    fused.mkdir(parents=True)
    (fused / "config.json").write_text(
        '{"max_position_embeddings": 40960}', encoding="utf-8",
    )
    monkeypatch.setattr(
        "backend.distillation.paths.fused_model_path",
        lambda slug, **k: fused if slug.endswith("Qwen3-fused") else tmp_path / "missing",
    )
    assert _omlx_native_context_window("Qwen3-fused") == 40960
    cfg = SimpleNamespace(omlx=SimpleNamespace(max_context_window=131072, max_tokens=10240))
    budget = _omlx_input_budget(cfg, "Qwen3-fused")
    assert budget == 40960 - 10240


def test_distill_session_provider_uses_omlx_when_fused(monkeypatch):
    from backend.session_manager import _distill_session_provider, _fused_omlx_model_id

    monkeypatch.setattr(
        "backend.distillation.catalog.resolve_catalog_id",
        lambda cid, **k: {
            "catalog_id": cid,
            "omlx_model_id": "Qwen3-8B-4bit-activity-fused",
        },
    )
    cfg = SimpleNamespace(llm=SimpleNamespace(provider="omlx"))
    assert _fused_omlx_model_id("otto-distill/ready") == "Qwen3-8B-4bit-activity-fused"
    assert _distill_session_provider("otto-distill/ready", cfg) == "omlx"
    cfg.llm.provider = "mlx"
    assert _distill_session_provider("otto-distill/ready", cfg) == "mlx"
    monkeypatch.setattr(
        "backend.distillation.catalog.resolve_catalog_id",
        lambda cid, **k: {"catalog_id": cid, "omlx_model_id": ""},
    )
    cfg.llm.provider = "omlx"
    assert _distill_session_provider("otto-distill/ready", cfg) == "mlx"
    session = Session(
        "sid",
        None,
        graph=None,
        distill_catalog_id="otto-distill/ready",
        llm_provider="mlx",
    )
    info = session.to_info()
    assert info.distill_catalog_id == "otto-distill/ready"
    restored = SessionInfo.model_validate(json.loads(info.model_dump_json()))
    assert restored.distill_catalog_id == "otto-distill/ready"


def test_effective_distill_drops_when_omlx_selection_differs(monkeypatch):
    from backend.session_manager import _effective_distill_catalog_id

    monkeypatch.setattr(
        "backend.session_manager._fused_omlx_model_id",
        lambda cid: "Qwen3-fused" if cid == "otto-distill/ready" else None,
    )
    cfg = SimpleNamespace(
        llm=SimpleNamespace(provider="omlx"),
        omlx=SimpleNamespace(model_name="mlx-community--Qwen3.6-35B-A3B-4bit"),
    )
    assert _effective_distill_catalog_id("otto-distill/ready", cfg) is None
    cfg.omlx.model_name = "Qwen3-fused"
    assert _effective_distill_catalog_id("otto-distill/ready", cfg) == "otto-distill/ready"


def test_distill_env_override_sets_hf_id(monkeypatch):
    monkeypatch.setattr(
        "backend.distillation.catalog.resolve_catalog_id",
        lambda cid, **k: {
            "catalog_id": cid,
            "adapter_path": "/tmp/adapter",
            "base_repo_id": "mlx-community/Qwen3-8B-4bit",
        },
    )
    import os

    with _distill_llm_override("otto-distill/ready"):
        assert os.environ["HF_LLM_MODEL_ID"] == "otto-distill/ready"
        assert os.environ["MLX_ADAPTER_PATH"] == "/tmp/adapter"


def test_list_distilled_models_tool(tmp_path: Path, monkeypatch):
    _ready_adapter(tmp_path)
    monkeypatch.setattr(
        "backend.distillation.catalog.adapters_dir",
        lambda **k: tmp_path / "adapters",
    )
    fake_sess = SimpleNamespace(distill_catalog_id="otto-distill/ready")
    monkeypatch.setattr(
        "backend.state.session_mgr",
        SimpleNamespace(get_session=lambda _sid: fake_sess),
        raising=False,
    )
    tools = {t.name: t for t in build_distillation_tools("sid")}
    raw = tools["list_distilled_models"].invoke({})
    data = json.loads(raw)
    assert data["bound_catalog_id"] == "otto-distill/ready"
    assert data["adapters"][0]["purpose"] == "mail triage"
    assert data["adapters"][0]["bound_to_this_session"] is True


@pytest.mark.asyncio
async def test_start_distill_train_requires_purpose():
    tools = {t.name: t for t in build_distillation_tools("sid")}
    raw = await tools["start_distill_train"].ainvoke({"purpose": "  "})
    assert "purpose is required" in raw


@pytest.mark.asyncio
async def test_use_distilled_model_spawns_child(monkeypatch, tmp_path):
    _ready_adapter(tmp_path)
    monkeypatch.setattr(
        "backend.distillation.catalog.adapters_dir",
        lambda **k: tmp_path / "adapters",
    )
    child = SimpleNamespace(
        id="child-1",
        title="Do the task",
        agent_name=None,
        chain_depth=1,
    )
    spawn = AsyncMock(return_value=child)
    kick = AsyncMock()
    monkeypatch.setattr(
        "backend.state.session_mgr",
        SimpleNamespace(spawn_child_session=spawn, get_session=lambda _s: None),
        raising=False,
    )
    monkeypatch.setattr("backend.session_dispatch.kick_off_message", kick)

    tools = {t.name: t for t in build_distillation_tools("sid")}
    raw = await tools["use_distilled_model"].ainvoke({
        "catalog_id": "otto-distill/ready",
        "prompt": "Triage today's mail",
    })
    data = json.loads(raw)
    assert data["child_session_id"] == "child-1"
    assert data["catalog_id"] == "otto-distill/ready"
    spawn.assert_awaited_once()
    assert spawn.await_args.kwargs["distill_catalog_id"] == "otto-distill/ready"
    kick.assert_awaited_once_with("child-1", "Triage today's mail")


@pytest.mark.asyncio
async def test_use_distilled_model_rejects_unknown_and_empty_prompt(monkeypatch, tmp_path):
    _ready_adapter(tmp_path)
    monkeypatch.setattr(
        "backend.distillation.catalog.adapters_dir",
        lambda **k: tmp_path / "adapters",
    )
    spawn = AsyncMock()
    monkeypatch.setattr(
        "backend.state.session_mgr",
        SimpleNamespace(spawn_child_session=spawn, get_session=lambda _s: None),
        raising=False,
    )
    tools = {t.name: t for t in build_distillation_tools("sid")}
    unknown = await tools["use_distilled_model"].ainvoke({
        "catalog_id": "otto-distill/missing",
        "prompt": "go",
    })
    assert "unknown distilled catalog id" in unknown
    empty = await tools["use_distilled_model"].ainvoke({
        "catalog_id": "otto-distill/ready",
        "prompt": "  ",
    })
    assert "prompt must be a non-empty string" in empty
    spawn.assert_not_awaited()


@pytest.mark.asyncio
async def test_clear_distilled_model_spawns_unbound_child(monkeypatch):
    child = SimpleNamespace(
        id="child-2",
        title="Plain run",
        agent_name=None,
        chain_depth=1,
    )
    spawn = AsyncMock(return_value=child)
    kick = AsyncMock()
    monkeypatch.setattr(
        "backend.state.session_mgr",
        SimpleNamespace(spawn_child_session=spawn, get_session=lambda _s: None),
        raising=False,
    )
    monkeypatch.setattr("backend.session_dispatch.kick_off_message", kick)

    tools = {t.name: t for t in build_distillation_tools("sid")}
    raw = await tools["clear_distilled_model"].ainvoke({
        "prompt": "Use the default model",
    })
    data = json.loads(raw)
    assert data["child_session_id"] == "child-2"
    assert data["catalog_id"] is None
    spawn.assert_awaited_once()
    assert spawn.await_args.kwargs["distill_catalog_id"] == ""
    kick.assert_awaited_once_with("child-2", "Use the default model")


@pytest.mark.asyncio
async def test_spawn_child_session_distill_inherit_and_unbind(monkeypatch):
    from backend.session_manager import SessionManager

    mgr = SessionManager()
    parent = Session(
        "parent",
        None,
        graph=None,
        distill_catalog_id="otto-distill/ready",
    )
    monkeypatch.setattr(mgr, "_ensure_session", AsyncMock(return_value=parent))
    created: list[dict] = []

    async def fake_create(**kwargs):
        created.append(kwargs)
        child = Session(
            "child",
            None,
            graph=None,
            distill_catalog_id=kwargs.get("distill_catalog_id"),
            chain_depth=kwargs.get("chain_depth", 1),
        )
        child.save_meta_async = AsyncMock()  # type: ignore[method-assign]
        return child

    monkeypatch.setattr(mgr, "create_session", fake_create)
    monkeypatch.setattr(
        "backend.session_manager.AppConfig.aload",
        AsyncMock(return_value=SimpleNamespace()),
    )

    inherited = await mgr.spawn_child_session("parent", "keep lora")
    assert created[0]["distill_catalog_id"] == "otto-distill/ready"
    assert inherited.distill_catalog_id == "otto-distill/ready"

    created.clear()
    rebound = await mgr.spawn_child_session(
        "parent", "switch", distill_catalog_id="otto-distill/other",
    )
    assert created[0]["distill_catalog_id"] == "otto-distill/other"
    assert rebound.distill_catalog_id == "otto-distill/other"

    created.clear()
    unbound = await mgr.spawn_child_session(
        "parent", "plain", distill_catalog_id="",
    )
    assert created[0]["distill_catalog_id"] is None
    assert unbound.distill_catalog_id is None
