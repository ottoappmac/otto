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
