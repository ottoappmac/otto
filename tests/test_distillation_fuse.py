"""Fuse LoRA into a standalone MLX model for oMLX.  No GPU."""

from __future__ import annotations

from pathlib import Path

from backend.distillation.adapter_meta import read_adapter_meta, write_adapter_meta
from backend.distillation.fuse import (
    build_fuse_argv,
    fuse_lora,
    fused_ready,
    omlx_model_id_for,
)


def test_build_fuse_argv_points_at_mlx_lm_fuse():
    argv = build_fuse_argv(
        student_id="mlx-community/Qwen3-8B-4bit",
        adapter_path=Path("/tmp/ad"),
        save_path=Path("/tmp/fused/slug"),
    )
    assert "-m" in argv
    assert argv[argv.index("-m") + 1] == "mlx_lm.fuse"
    assert argv[argv.index("--model") + 1] == "mlx-community/Qwen3-8B-4bit"
    assert argv[argv.index("--adapter-path") + 1] == "/tmp/ad"
    assert argv[argv.index("--save-path") + 1] == "/tmp/fused/slug"


def test_fused_ready_requires_config_and_weights(tmp_path: Path):
    dest = tmp_path / "model"
    dest.mkdir()
    assert fused_ready(dest) is False
    (dest / "config.json").write_text("{}", encoding="utf-8")
    assert fused_ready(dest) is False
    (dest / "model.safetensors").write_bytes(b"w")
    assert fused_ready(dest) is True


def test_fused_ready_rejects_adapter_only(tmp_path: Path):
    dest = tmp_path / "ad"
    dest.mkdir()
    (dest / "adapter_config.json").write_text("{}", encoding="utf-8")
    (dest / "adapters.safetensors").write_bytes(b"lora")
    assert fused_ready(dest) is False


def test_fuse_lora_dry_run_and_skip(tmp_path: Path):
    adapter = tmp_path / "Qwen3-8B-4bit-activity"
    adapter.mkdir()
    (adapter / "adapters.safetensors").write_bytes(b"lora")
    write_adapter_meta(adapter, base_repo_id="mlx-community/Qwen3-8B-4bit")
    dest = tmp_path / "fused" / adapter.name
    dest.mkdir(parents=True)
    (dest / "config.json").write_text("{}", encoding="utf-8")
    (dest / "model.safetensors").write_bytes(b"w")
    result = fuse_lora(
        adapter_path=adapter,
        save_path=dest,
        run=True,
        runner=lambda argv: 99,
    )
    assert result["skipped"] is True
    assert result["exit_code"] == 0
    assert result["omlx_model_id"] == adapter.name
    assert omlx_model_id_for(adapter) == adapter.name
    meta = read_adapter_meta(adapter)
    assert meta is not None
    assert meta["fused_path"] == str(dest)


def test_fuse_lora_dry_run_argv(tmp_path: Path):
    adapter = tmp_path / "activity"
    adapter.mkdir()
    write_adapter_meta(adapter, base_repo_id="mlx-community/Qwen3-8B-4bit")
    dest = tmp_path / "out"
    result = fuse_lora(adapter_path=adapter, save_path=dest, run=False)
    assert result["skipped"] is False
    assert result["exit_code"] is None
    assert "--save-path" in result["argv"]
