"""Unique otto-distill catalog ids for trained LoRA adapters."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from backend.distillation.adapter_meta import write_adapter_meta
from backend.distillation.catalog import (
    catalog_rows_from_adapters,
    is_distill_catalog_id,
    iter_trained_adapters,
    make_identity,
    resolve_catalog_id,
    student_short,
)


def test_student_short_strips_org():
    assert student_short("mlx-community/Qwen3-8B-4bit") == "Qwen3-8B-4bit"
    assert student_short("Qwen3-8B-4bit") == "Qwen3-8B-4bit"


def test_make_identity_unique_differs_by_time_and_sha():
    t0 = datetime(2026, 9, 22, 11, 26, 15, tzinfo=timezone.utc)
    t1 = datetime(2026, 9, 22, 11, 26, 16, tzinfo=timezone.utc)
    a = make_identity(
        student_id="mlx-community/Qwen3-8B-4bit",
        kind="activity",
        dataset_sha="abcd1234ffff",
        now=t0,
    )
    b = make_identity(
        student_id="mlx-community/Qwen3-8B-4bit",
        kind="activity",
        dataset_sha="abcd1234ffff",
        now=t1,
    )
    c = make_identity(
        student_id="mlx-community/Qwen3-8B-4bit",
        kind="activity",
        dataset_sha="deadbeef0000",
        now=t0,
    )
    assert a["catalog_id"].startswith("otto-distill/")
    assert a["catalog_id"] != b["catalog_id"]
    assert a["catalog_id"] != c["catalog_id"]
    assert "Qwen3-8B-4bit distilled" in a["display_name"]
    assert is_distill_catalog_id(a["catalog_id"])


def test_make_identity_stable_when_not_unique():
    t0 = datetime(2026, 9, 22, 11, 26, 15, tzinfo=timezone.utc)
    a = make_identity(
        student_id="mlx-community/Qwen3-8B-4bit",
        kind="activity",
        unique=False,
        now=t0,
    )
    b = make_identity(
        student_id="mlx-community/Qwen3-8B-4bit",
        kind="activity",
        unique=False,
        now=t0,
    )
    assert a["slug"] == "activity"
    assert a["catalog_id"] == b["catalog_id"] == "otto-distill/activity"


def test_iter_trained_requires_weights(tmp_path: Path, monkeypatch):
    adapters = tmp_path / "adapters"
    ghost = adapters / "ghost"
    ghost.mkdir(parents=True)
    write_adapter_meta(
        ghost,
        base_repo_id="mlx-community/Qwen3-8B-4bit",
        extra={"catalog_id": "otto-distill/ghost", "display_name": "Ghost"},
    )
    ready = adapters / "ready"
    ready.mkdir()
    (ready / "adapters.safetensors").write_bytes(b"lora")
    write_adapter_meta(
        ready,
        base_repo_id="mlx-community/Qwen3-8B-4bit",
        extra={
            "catalog_id": "otto-distill/Qwen3-8B-4bit-activity-20260922-112615",
            "display_name": "Qwen3-8B-4bit distilled (activity · 22 Sep 11:26)",
            "teacher_model_id": "mlx-community/Qwen3-32B-4bit",
            "kind": "activity",
            "purpose": "mail triage from activity traces",
        },
    )
    monkeypatch.setattr(
        "backend.distillation.catalog.adapters_dir",
        lambda **k: adapters,
    )
    recs = iter_trained_adapters()
    assert [r["catalog_id"] for r in recs] == [
        "otto-distill/Qwen3-8B-4bit-activity-20260922-112615",
    ]
    found = resolve_catalog_id("otto-distill/Qwen3-8B-4bit-activity-20260922-112615")
    assert found is not None
    assert found["base_repo_id"].endswith("Qwen3-8B-4bit")
    assert found["purpose"] == "mail triage from activity traces"
    assert found["kind"] == "activity"
    assert found.get("fused_path") == ""
    assert found.get("omlx_model_id") == ""
    assert resolve_catalog_id("mlx-community/Qwen3-8B-4bit") is None
    rows = catalog_rows_from_adapters()
    assert len(rows) == 1
    assert rows[0].source == "distill"
    assert rows[0].featured is True
    assert rows[0].repo_id.startswith("otto-distill/")
    assert rows[0].base_repo_id.endswith("Qwen3-8B-4bit")
    assert rows[0].adapter_path == str(ready)


def test_catalog_id_for_omlx_model(monkeypatch):
    from backend.distillation.catalog import catalog_id_for_omlx_model

    monkeypatch.setattr(
        "backend.distillation.catalog.iter_trained_adapters",
        lambda **k: [{
            "catalog_id": "otto-distill/ready",
            "omlx_model_id": "Qwen3-8B-4bit-activity-fused",
        }],
    )
    assert catalog_id_for_omlx_model("Qwen3-8B-4bit-activity-fused") == "otto-distill/ready"
    assert catalog_id_for_omlx_model("missing") is None
    assert catalog_id_for_omlx_model("") is None
