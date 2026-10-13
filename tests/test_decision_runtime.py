"""Sidecar env and process teardown. Does not install or download OpenJev."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from backend.config import DecisionConfig
from backend.decision_runtime import (
    _terminate,
    build_process_env,
    bundled_runtime_dir,
    installed_ref,
    local_api_key,
    log_path,
    runtime_current,
    snapshot_dir,
    venv_python,
)
from backend.routes.mlx import _DEFAULT_ALLOW_PATTERNS


def test_jev_download_keeps_the_chat_template() -> None:
    assert "*.jinja" in _DEFAULT_ALLOW_PATTERNS


def test_snapshot_is_incomplete_without_the_chat_template(tmp_path, monkeypatch) -> None:
    repo = "mlx-community/diffusiongemma-26B-A4B-it-4bit"
    root = tmp_path / "models--mlx-community--diffusiongemma-26B-A4B-it-4bit"
    snap = root / "snapshots" / "rev"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}", encoding="utf-8")
    (snap / "model.safetensors").write_text("", encoding="utf-8")
    (root / "refs").mkdir()
    (root / "refs" / "main").write_text("rev\n", encoding="utf-8")
    monkeypatch.setattr("backend.decision_runtime.hf_repo_dir", lambda _repo_id: root)

    assert snapshot_dir(repo) is None
    (snap / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    assert snapshot_dir(repo) == snap


def test_process_env_is_loopback_and_offline() -> None:
    env = build_process_env(
        DecisionConfig(),
        Path("/tmp/openjev-snapshot"),
        api_key="test-key",
    )
    assert env["OPENJEV_HOST"] == "127.0.0.1"
    assert env["OPENJEV_PORT"] == "52416"
    assert env["OPENJEV_MLX_MODEL"] == "/tmp/openjev-snapshot"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["OPENJEV_API_KEY"] == "test-key"
    assert env["OPENJEV_BACKEND"] == "mlx"


def test_runtime_current_matches_the_pin(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("backend.decision_runtime.runtime_dir", lambda: tmp_path)
    cfg = DecisionConfig()
    assert runtime_current(cfg) is False
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
    (tmp_path / "INSTALLED_REF").write_text(cfg.runtime_ref + "\n", encoding="utf-8")
    assert runtime_current(cfg) is True
    (tmp_path / "INSTALLED_REF").write_text("other\n", encoding="utf-8")
    assert runtime_current(cfg) is False


def test_packaged_app_prefers_the_bundled_venv(tmp_path, monkeypatch) -> None:
    bundle = tmp_path / "Resources" / "openjev"
    py = bundle / "venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("", encoding="utf-8")
    (bundle / "INSTALLED_REF").write_text("pinned\n", encoding="utf-8")
    exe = tmp_path / "Resources" / "backend" / "otto-backend"
    exe.parent.mkdir(parents=True)
    exe.write_text("", encoding="utf-8")
    app_data = tmp_path / "appdata"
    monkeypatch.setattr(sys, "executable", str(exe))
    monkeypatch.setattr("backend.decision_runtime.runtime_dir", lambda: app_data)

    assert bundled_runtime_dir() == bundle
    assert venv_python() == py
    assert installed_ref() == "pinned"
    assert runtime_current(DecisionConfig(runtime_ref="pinned")) is True
    assert log_path() == app_data / "server.log"


def test_local_api_key_stays_put_when_the_vault_is_empty(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("backend.decision_runtime.runtime_dir", lambda: tmp_path)
    monkeypatch.setattr("backend.decision_runtime._vault_get", lambda account: None)
    first = local_api_key()
    second = local_api_key()
    assert first == second
    assert first.startswith("ojv_")
    assert (tmp_path / "local_api_key").read_text(encoding="utf-8").strip() == first


def test_terminate_does_not_leave_the_process() -> None:
    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        _terminate(proc)
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()
