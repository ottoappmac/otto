"""Offline tests for the oMLX install path selection.

Validates that :func:`backend.omlx_provisioner.ainstall` prefers the
Homebrew path when brew is present and falls back to the official
GitHub-release download when it is absent, plus the release-asset
picker and the no-asset error path.
"""

from __future__ import annotations

import asyncio

import pytest

from backend import omlx_provisioner as op
from backend.config import OmlxConfig


async def _drain(job, timeout: float = 2.0) -> None:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while job.status in ("pending", "running"):
        if loop.time() > deadline:
            raise AssertionError(f"job did not finish; log={job.log_lines[-3:]}")
        await asyncio.sleep(0.01)


# ── path selection --------------------------------------------------------


async def test_install_prefers_brew_when_present(monkeypatch):
    called = {"brew": False, "release": False}

    async def fake_brew(job, cfg):
        called["brew"] = True

    async def fake_release(job, cfg):
        called["release"] = True

    monkeypatch.setattr(op, "has_homebrew", lambda: True)
    monkeypatch.setattr(op, "_install_via_brew", fake_brew)
    monkeypatch.setattr(op, "_install_via_release", fake_release)

    job = await op.ainstall(OmlxConfig())
    await _drain(job)

    assert job.status == "done"
    assert called == {"brew": True, "release": False}


async def test_install_uses_releases_when_no_homebrew(monkeypatch):
    called = {"brew": False, "release": False}

    async def fake_brew(job, cfg):
        called["brew"] = True

    async def fake_release(job, cfg):
        called["release"] = True

    monkeypatch.setattr(op, "has_homebrew", lambda: False)
    monkeypatch.setattr(op, "_install_via_brew", fake_brew)
    monkeypatch.setattr(op, "_install_via_release", fake_release)

    job = await op.ainstall(OmlxConfig())
    await _drain(job)

    assert job.status == "done"
    assert called == {"brew": False, "release": True}


# ── release asset picker --------------------------------------------------


def test_pick_release_asset_prefers_arch_archive(monkeypatch):
    monkeypatch.setattr(op.platform, "machine", lambda: "arm64")
    assets = [
        {"name": "omlx-x64.tar.gz", "browser_download_url": "u1"},
        {"name": "omlx-arm64.tar.gz", "browser_download_url": "u2"},
        {"name": "oMLX-arm64.dmg", "browser_download_url": "u3"},
    ]
    picked = op._pick_release_asset(assets)
    assert picked["name"] == "omlx-arm64.tar.gz"


def test_pick_release_asset_falls_back_to_dmg(monkeypatch):
    monkeypatch.setattr(op.platform, "machine", lambda: "arm64")
    assets = [{"name": "oMLX-arm64.dmg", "browser_download_url": "u"}]
    picked = op._pick_release_asset(assets)
    assert picked["name"] == "oMLX-arm64.dmg"


def test_pick_release_asset_none_when_empty():
    assert op._pick_release_asset([]) is None


async def test_install_via_release_errors_without_asset(monkeypatch):
    async def no_assets():
        return None, []

    monkeypatch.setattr(op, "aget_latest_release_assets", no_assets)

    job = op._new_job("install")
    with pytest.raises(RuntimeError, match="No suitable oMLX release asset"):
        await op._install_via_release(job, OmlxConfig())


# ── detection regression --------------------------------------------------


def test_find_cli_detects_path_binary(monkeypatch):
    monkeypatch.setattr(op.shutil, "which", lambda b: "/Users/me/.local/bin/omlx")
    found = op.find_cli("")
    assert found is not None
    assert str(found).endswith("omlx")


# ── admin key auto-adopt --------------------------------------------------


def test_adopt_admin_key_from_server_settings(monkeypatch):
    """When Otto's key is blank, adopt the one in oMLX settings.json."""
    persisted: dict[str, str] = {}
    monkeypatch.setattr(
        op, "_read_omlx_settings", lambda: {"auth": {"api_key": "otto-existing"}},
    )
    monkeypatch.setattr(
        op, "_persist_admin_key_to_config",
        lambda key: persisted.update(key=key),
    )

    cfg = OmlxConfig(admin_api_key="")
    assert op.adopt_existing_admin_key(cfg) is True
    assert cfg.admin_api_key == "otto-existing"
    assert persisted == {"key": "otto-existing"}


def test_adopt_admin_key_noop_when_otto_already_has_one(monkeypatch):
    """Never overwrite a key Otto already holds."""
    monkeypatch.setattr(
        op, "_read_omlx_settings", lambda: {"auth": {"api_key": "otto-server"}},
    )
    monkeypatch.setattr(
        op, "_persist_admin_key_to_config",
        lambda key: pytest.fail("must not persist when Otto already has a key"),
    )

    cfg = OmlxConfig(admin_api_key="otto-mine")
    assert op.adopt_existing_admin_key(cfg) is False
    assert cfg.admin_api_key == "otto-mine"


def test_adopt_admin_key_noop_when_server_has_none(monkeypatch):
    """No key on the server ⇒ nothing to adopt, no persistence."""
    monkeypatch.setattr(op, "_read_omlx_settings", lambda: {"auth": {}})
    monkeypatch.setattr(
        op, "_persist_admin_key_to_config",
        lambda key: pytest.fail("must not persist when server has no key"),
    )

    cfg = OmlxConfig(admin_api_key="")
    assert op.adopt_existing_admin_key(cfg) is False
    assert cfg.admin_api_key == ""


def test_status_loaded_ids_ignores_catalog_when_admin_reports_empty():
    """Registered catalog ids are not a substitute for GPU-resident ids."""
    status = {
        "models": [{"id": "Qwen3-8B-4bit-activity-fused"}],
        "loaded_models": [],
    }
    assert op.status_loaded_ids(status) == []


def test_status_loaded_ids_reads_loaded_models():
    status = {
        "models": [{"id": "catalog-only"}, {"id": "resident"}],
        "loaded_models": [{"id": "resident"}],
    }
    assert op.status_loaded_ids(status) == ["resident"]


# ── minimum oMLX version per model architecture ---------------------------


def _cached_model(monkeypatch, tmp_path, model_type):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text(f'{{"model_type": "{model_type}"}}')
    monkeypatch.setattr(op, "_find_hf_cache_snapshot", lambda repo_id: snap)


@pytest.mark.parametrize("raw,want", [
    ("0.8.3", "0.8.3"), ("omlx 0.3.10", "0.3.10"), ("v0.4.2", "0.4.2"), (None, None), ("", None),
])
def test_normalize_version(raw, want):
    assert op.normalize_version(raw) == want


def test_version_at_least():
    assert op.version_at_least("0.8.3", "0.8.3") is True
    assert op.version_at_least("0.10.0", "0.8.3") is True
    assert op.version_at_least("0.8.0rc1", "0.8.3") is False
    assert op.version_at_least("not-a-version", "0.8.3") is None


def test_inkling_on_old_omlx_is_refused_with_upgrade_hint(monkeypatch, tmp_path):
    _cached_model(monkeypatch, tmp_path, "inkling2_loop")
    monkeypatch.setattr(op, "find_app_bundle", lambda: None)
    msg = op.check_model_version(OmlxConfig(), "mlx-community/Inkling-2-Mini-oQ4e-mtp", installed="0.6.3")
    assert msg is not None
    assert "Inkling 2 Mini" in msg and "0.8.3" in msg and "0.6.3" in msg
    assert "brew update && brew upgrade omlx" in msg


def test_inkling_on_new_enough_omlx_passes(monkeypatch, tmp_path):
    _cached_model(monkeypatch, tmp_path, "inkling2_loop")
    assert op.check_model_version(OmlxConfig(), "x/Inkling", installed="0.8.3") is None
    assert op.check_model_version(OmlxConfig(), "x/Inkling", installed="omlx 0.9.0") is None


def test_unlisted_model_type_is_never_blocked(monkeypatch, tmp_path):
    _cached_model(monkeypatch, tmp_path, "qwen3_5_moe")
    assert op.check_model_version(OmlxConfig(), "x/Qwen", installed="0.1.0") is None


def test_uncached_model_or_unknown_version_is_not_blocked(monkeypatch, tmp_path):
    monkeypatch.setattr(op, "_find_hf_cache_snapshot", lambda repo_id: None)
    assert op.check_model_version(OmlxConfig(), "x/Inkling", installed="0.6.3") is None
    _cached_model(monkeypatch, tmp_path, "inkling2_loop")
    monkeypatch.setattr(op, "installed_version", lambda cfg: None)
    assert op.check_model_version(OmlxConfig(), "x/Inkling") is None


def test_dmg_only_install_is_told_to_update_the_app(monkeypatch, tmp_path):
    _cached_model(monkeypatch, tmp_path, "inkling2_loop")
    monkeypatch.setattr(op, "find_app_bundle", lambda: tmp_path / "oMLX.app")
    monkeypatch.setattr(op, "has_homebrew", lambda: False)
    msg = op.check_model_version(OmlxConfig(), "x/Inkling", installed="0.6.3")
    assert msg is not None and "/Applications" in msg and "brew" not in msg


async def test_load_refuses_before_asking_omlx_when_too_old(monkeypatch):
    async def reachable(cfg):
        return {"reachable": True, "models": []}

    async def must_not_resolve(cfg, model_id):
        raise AssertionError("oMLX was asked to resolve a model it is too old for")

    monkeypatch.setattr(op, "afetch_status", reachable)
    monkeypatch.setattr(op, "_resolve_omlx_model_id", must_not_resolve)
    monkeypatch.setattr(op, "check_model_version", lambda cfg, model_id: "needs oMLX 0.8.3")

    job = await op.aload_model(OmlxConfig(), "x/Inkling")
    await _drain(job)

    assert job.status == "error"
    assert job.error == "needs oMLX 0.8.3"


async def test_upgrade_refreshes_the_tap_before_upgrading(monkeypatch):
    cmds = []

    async def fake_run(cmd, *, job, timeout=600.0):
        cmds.append(cmd[1:])
        return 0

    monkeypatch.setattr(op, "has_homebrew", lambda: True)
    monkeypatch.setattr(op, "_brew_bin", lambda: "brew")
    monkeypatch.setattr(op, "_brew_service_state", lambda formula: "stopped")
    monkeypatch.setattr(op, "_run_streaming", fake_run)
    monkeypatch.setattr(op, "detect", lambda cfg: op.Detection(None, None, True, "stopped"))

    job = await op.aupgrade(OmlxConfig())
    await _drain(job)

    assert job.status == "done", job.error
    assert cmds == [["update"], ["upgrade", OmlxConfig().brew_formula]]


async def test_upgrade_continues_when_brew_update_fails(monkeypatch):
    cmds = []

    async def fake_run(cmd, *, job, timeout=600.0):
        cmds.append(cmd[1:])
        return 1 if cmd[1] == "update" else 0

    monkeypatch.setattr(op, "has_homebrew", lambda: True)
    monkeypatch.setattr(op, "_brew_bin", lambda: "brew")
    monkeypatch.setattr(op, "_brew_service_state", lambda formula: "stopped")
    monkeypatch.setattr(op, "_run_streaming", fake_run)
    monkeypatch.setattr(op, "detect", lambda cfg: op.Detection(None, None, True, "stopped"))

    job = await op.aupgrade(OmlxConfig())
    await _drain(job)

    assert job.status == "done", job.error
    assert cmds[-1] == ["upgrade", OmlxConfig().brew_formula]
