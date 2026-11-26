"""Inkling 2 Mini: thinking off over oMLX, Turbo-only catalog row, Standard and Distill guards."""

from __future__ import annotations

import pytest

from backend import mlx_catalog
from backend.distillation.model_pair import ModelPairError, validate_pair
from deep_agent import model_factory

INKLING = "mlx-community/Inkling-2-Mini-oQ4e-mtp"
QWEN = "mlx-community/Qwen3-8B-4bit"


def _omlx_client(monkeypatch, *, model, model_type, thinking):
    monkeypatch.setenv("OMLX_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("OMLX_MODEL_NAME", model)
    monkeypatch.setenv("OMLX_THINKING", "true" if thinking else "false")
    monkeypatch.setattr(model_factory, "_resolve_omlx_model_id", lambda base, name: name.split("/")[-1])
    monkeypatch.setattr(model_factory, "_cached_model_type", lambda model_id: model_type)
    return model_factory.create_llm("omlx")


# ── thinking over oMLX ------------------------------------------------------


def test_inkling_thinking_off_sends_thinking_budget_zero(monkeypatch):
    llm = _omlx_client(monkeypatch, model=INKLING, model_type="inkling2_loop", thinking=False)
    assert llm.extra_body["thinking_budget"] == 0
    assert llm.extra_body["chat_template_kwargs"]["enable_thinking"] is False


def test_inkling_thinking_on_leaves_the_budget_alone(monkeypatch):
    llm = _omlx_client(monkeypatch, model=INKLING, model_type="inkling2_loop", thinking=True)
    assert "thinking_budget" not in llm.extra_body
    assert llm.extra_body["chat_template_kwargs"]["enable_thinking"] is True


def test_other_models_keep_enable_thinking_only(monkeypatch):
    llm = _omlx_client(monkeypatch, model=QWEN, model_type="qwen3", thinking=False)
    assert "thinking_budget" not in llm.extra_body


def test_inkling_uses_native_tool_calls_not_the_react_shim(monkeypatch):
    llm = _omlx_client(monkeypatch, model=INKLING, model_type="inkling2_loop", thinking=False)
    assert type(llm).__name__ == "_OMLXChat"


# ── Standard (in-process) engine refuses it ---------------------------------


def test_standard_engine_refuses_an_omlx_only_model(monkeypatch):
    monkeypatch.setattr(model_factory, "_cached_model_type", lambda model_id: "inkling2_loop")
    with pytest.raises(ValueError, match="runs only on Turbo"):
        model_factory._build_mlx_chat(
            model_id=INKLING, draft_model_id="", max_tokens=16, num_draft_tokens=0, thinking=False,
            enable_prompt_cache=False, enable_system_prompt_cache=False, kv_bits=None, kv_group_size=64,
        )


def test_standard_engine_guard_passes_other_models(monkeypatch):
    monkeypatch.setattr(model_factory, "_cached_model_type", lambda model_id: "qwen3")
    model_factory._refuse_omlx_only_model(QWEN)
    monkeypatch.setattr(model_factory, "_cached_model_type", lambda model_id: None)
    model_factory._refuse_omlx_only_model("not/cached")


# ── catalog ------------------------------------------------------------------


def test_inkling_catalog_row_is_turbo_only():
    row = next(r for r in mlx_catalog.CURATED if r.repo_id == INKLING)
    assert row.engines == ["omlx"]
    assert {"tools", "vlm"} <= set(row.role)


def test_other_curated_rows_run_on_both_engines():
    for row in mlx_catalog.CURATED:
        if row.repo_id != INKLING:
            assert row.engines == ["mlx", "omlx"], row.repo_id


def test_scored_rows_carry_engines_for_the_route_filter():
    scored = mlx_catalog.score_catalog(
        mlx_catalog.CURATED, ram_gb=128, wired_limit_gb=96, free_disk_gb=500, ctx_len=8192, kv_bits=None,
    )
    by_id = {r["repo_id"]: r for r in scored}
    assert by_id[INKLING]["engines"] == ["omlx"]
    assert by_id[QWEN]["engines"] == ["mlx", "omlx"]


def test_discovered_inkling_repo_is_turbo_only():
    row = mlx_catalog._infer_row_from_hub("someone/Inkling-2-Mini-26B-4bit")
    assert row is not None and row.family == "inkling2" and row.engines == ["omlx"]


def test_standard_engine_runs(monkeypatch):
    monkeypatch.setattr(mlx_catalog, "cached_model_type", lambda repo_id: None)
    assert mlx_catalog.standard_engine_runs(QWEN) is True
    assert mlx_catalog.standard_engine_runs(INKLING) is False  # curated row says Turbo only
    monkeypatch.setattr(mlx_catalog, "cached_model_type", lambda repo_id: "inkling2_loop")
    assert mlx_catalog.standard_engine_runs("someone/renamed-inkling") is False


# ── Distill ------------------------------------------------------------------


def test_inkling_is_refused_as_a_distill_student():
    with pytest.raises(ModelPairError, match="can't be a Distill student"):
        validate_pair(QWEN, INKLING, ram_gb=128)


def test_inkling_can_still_be_a_distill_teacher():
    result = validate_pair(INKLING, QWEN, ram_gb=128, allow_family_mismatch=True)
    assert result.teacher_id == INKLING
