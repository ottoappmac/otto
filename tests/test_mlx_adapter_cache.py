"""MLX process-wide cache must isolate LoRA adapters from the bare model."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from chat_models.mlx._shared import (
    _LOADED_MODELS,
    cache_key,
    effective_adapter_path,
    evict_all_mlx_models,
    _load_or_reuse,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    evict_all_mlx_models()
    yield
    evict_all_mlx_models()


def test_mlx_lm_load_accepts_adapter_path():
    inspect = pytest.importorskip("inspect")
    mlx_lm = pytest.importorskip("mlx_lm")
    from mlx_lm import load

    assert "adapter_path" in inspect.signature(load).parameters


def test_cache_keys_include_adapter_path():
    bare = cache_key("mlx-community/Qwen3-8B-4bit")
    adapted = cache_key("mlx-community/Qwen3-8B-4bit", None, "/tmp/activity")
    assert bare != adapted
    assert bare[2] is None
    assert adapted[2] == "/tmp/activity"


def test_mismatched_adapter_meta_is_dropped(tmp_path: Path):
    adapter = tmp_path / "blender"
    adapter.mkdir()
    (adapter / "adapter_meta.json").write_text(
        json.dumps({"base_repo_id": "mlx-community/Qwen3-8B-4bit"}),
        encoding="utf-8",
    )
    assert effective_adapter_path("mlx-community/Qwen3-8B-4bit", str(adapter)) == str(adapter)
    assert effective_adapter_path("mlx-community/Qwen3-14B-4bit", str(adapter)) is None


def test_missing_meta_is_compatible(tmp_path: Path):
    adapter = tmp_path / "third-party"
    adapter.mkdir()
    assert effective_adapter_path("mlx-community/Qwen3-8B-4bit", str(adapter)) == str(adapter)


def test_load_or_reuse_does_not_share_bare_and_adapted(monkeypatch, tmp_path: Path):
    adapter = tmp_path / "activity"
    adapter.mkdir()
    (adapter / "adapter_meta.json").write_text(
        json.dumps({"base_repo_id": "local-model"}),
        encoding="utf-8",
    )

    loads: list[dict] = []

    def fake_load(path, adapter_path=None, **kwargs):
        loads.append({"path": path, "adapter_path": adapter_path})
        model = MagicMock(name=f"model-{len(loads)}")
        tok = MagicMock(name=f"tok-{len(loads)}")
        tok.vocab_size = 32000
        return model, tok

    monkeypatch.setattr(
        "chat_models.mlx._shared._resolve_local_path", lambda p: f"/resolved/{p}",
    )
    monkeypatch.setattr("mlx_lm.load", fake_load)

    bare, fresh_bare = _load_or_reuse("local-model", None, None)
    adapted, fresh_adapted = _load_or_reuse("local-model", None, str(adapter))

    assert fresh_bare is True
    assert fresh_adapted is True
    assert bare[0] is not adapted[0]
    assert len(_LOADED_MODELS) == 2
    assert loads[0]["adapter_path"] is None
    assert loads[1]["adapter_path"] == str(adapter)

    bare2, fresh_bare2 = _load_or_reuse("local-model", None, None)
    assert fresh_bare2 is False
    assert bare2[0] is bare[0]


def test_adapter_disables_draft(monkeypatch, tmp_path: Path):
    adapter = tmp_path / "activity"
    adapter.mkdir()

    def fake_load(path, adapter_path=None, **kwargs):
        model = MagicMock()
        tok = MagicMock()
        tok.vocab_size = 32
        return model, tok

    monkeypatch.setattr(
        "chat_models.mlx._shared._resolve_local_path", lambda p: f"/resolved/{p}",
    )
    monkeypatch.setattr("mlx_lm.load", fake_load)

    triple, _ = _load_or_reuse("local-model", "draft-model", str(adapter))
    assert triple[2] is None  # no draft
    key = next(iter(_LOADED_MODELS))
    assert key[1] is None
