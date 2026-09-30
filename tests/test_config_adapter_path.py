"""Distillation config reaches Environment through to_env_dict."""

from __future__ import annotations

from backend.config import AppConfig, DistillationConfig, MlxHfConfig
from utilities.environment import Environment


def test_adapter_path_emitted_in_env_dict():
    cfg = AppConfig(
        llm={"provider": "mlx", "mlx": MlxHfConfig(
            hf_llm_model_id="mlx-community/Qwen3-8B-4bit",
            adapter_path="/tmp/activity",
        )},
    )
    env = cfg.to_env_dict()
    assert env["MLX_ADAPTER_PATH"] == "/tmp/activity"
    assert env["HF_LLM_MODEL_ID"] == "mlx-community/Qwen3-8B-4bit"


def test_empty_adapter_path_in_env_dict():
    cfg = AppConfig(llm={"provider": "mlx", "mlx": MlxHfConfig(adapter_path="")})
    env = cfg.to_env_dict()
    assert env.get("MLX_ADAPTER_PATH") == ""


def test_get_mlx_adapter_path(monkeypatch):
    monkeypatch.delenv("MLX_ADAPTER_PATH", raising=False)
    monkeypatch.delenv("HF_LLM_MODEL_ID", raising=False)
    assert Environment.get_mlx_adapter_path() is None
    monkeypatch.setenv("MLX_ADAPTER_PATH", " /tmp/lora ")
    assert Environment.get_mlx_adapter_path() == "/tmp/lora"


def test_resolve_otto_distill_catalog_id(monkeypatch, tmp_path):
    from backend.distillation.adapter_meta import write_adapter_meta

    adapter = tmp_path / "Qwen3-8B-4bit-activity-20260922-112615"
    adapter.mkdir()
    (adapter / "adapters.safetensors").write_bytes(b"x")
    write_adapter_meta(
        adapter,
        base_repo_id="mlx-community/Qwen3-8B-4bit",
        extra={
            "catalog_id": "otto-distill/Qwen3-8B-4bit-activity-20260922-112615",
            "display_name": "Qwen3-8B-4bit distilled (activity · 22 Sep 11:26)",
        },
    )
    monkeypatch.setattr(
        "backend.distillation.catalog.adapters_dir",
        lambda **k: tmp_path,
    )
    monkeypatch.setenv(
        "HF_LLM_MODEL_ID",
        "otto-distill/Qwen3-8B-4bit-activity-20260922-112615",
    )
    monkeypatch.delenv("MLX_ADAPTER_PATH", raising=False)
    assert Environment.get_hf_llm_model_id() == "mlx-community/Qwen3-8B-4bit"
    assert Environment.get_mlx_adapter_path() == str(adapter)


def test_distillation_defaults():
    cfg = AppConfig()
    assert cfg.distillation.enabled is False
    assert cfg.distillation.teacher_model_id.endswith("Qwen3-32B-4bit")
    assert cfg.distillation.student_model_id.endswith("Qwen3-8B-4bit")
    assert cfg.distillation.filter_sessions_by_teacher is False


def test_distillation_round_trip():
    cfg = AppConfig(
        distillation=DistillationConfig(
            enabled=True,
            teacher_model_id="mlx-community/Qwen3-14B-4bit",
            student_model_id="mlx-community/Qwen3-4B-4bit",
            filter_sessions_by_teacher=True,
            min_tool_calls=3,
        ),
    )
    dumped = cfg.model_dump()
    again = AppConfig.model_validate(dumped)
    assert again.distillation.teacher_model_id == "mlx-community/Qwen3-14B-4bit"
    assert again.distillation.student_model_id == "mlx-community/Qwen3-4B-4bit"
    assert again.distillation.filter_sessions_by_teacher is True
    assert again.distillation.min_tool_calls == 3
