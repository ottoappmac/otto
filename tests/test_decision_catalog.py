"""Fit labels for the decision-model catalog. No network."""

from __future__ import annotations

from backend.decision_catalog import get_model, score_model, scored_catalog


def test_four_bit_does_not_fit_a_16gb_mac() -> None:
    row = get_model("mlx-community/diffusiongemma-26B-A4B-it-4bit")
    assert row is not None
    scored = score_model(row, ram_gb=16, wired_limit_gb=12, free_disk_gb=40)
    assert scored["fits"] == "over"


def test_four_bit_is_comfortable_on_128gb() -> None:
    row = get_model("mlx-community/diffusiongemma-26B-A4B-it-4bit")
    assert row is not None
    scored = score_model(row, ram_gb=128, wired_limit_gb=100, free_disk_gb=200)
    assert scored["fits"] == "comfortable"


def test_catalog_rows_carry_cache_flags(tmp_path) -> None:
    rows = scored_catalog(str(tmp_path))
    by_id = {row["repo_id"]: row for row in rows}
    four = by_id["mlx-community/diffusiongemma-26B-A4B-it-4bit"]
    assert four["featured"] is True
    assert isinstance(four["already_cached"], bool)
    assert isinstance(four["cache_complete"], bool)
    if four["cache_complete"]:
        assert four["already_cached"] is True
    assert set(by_id) == {
        "mlx-community/diffusiongemma-26B-A4B-it-4bit",
        "mlx-community/diffusiongemma-26B-A4B-it-8bit",
        "mlx-community/diffusiongemma-26B-A4B-it-bf16",
    }


def test_disk_warning_is_separate_from_ram() -> None:
    row = get_model("mlx-community/diffusiongemma-26B-A4B-it-bf16")
    assert row is not None
    scored = score_model(row, ram_gb=128, wired_limit_gb=100, free_disk_gb=10)
    assert scored["disk_ok"] is False
