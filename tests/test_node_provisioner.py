"""Offline unit tests for the brew-free Node.js provisioner.

Validates both install paths of :mod:`backend.node_provisioner`:

* the Homebrew fast path (``brew install node``), and
* the portable-tarball fallback used when Homebrew is absent or the
  brew install does not yield a usable ``node``.

Plus presence detection and arch-specific tarball URL selection.
"""

from __future__ import annotations

import asyncio

import pytest

from backend import node_provisioner as nodep
from backend import tool_provisioner as tp


@pytest.fixture()
def isolated_tools_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tp, "get_app_data_dir", lambda: tmp_path)
    return tmp_path


async def _drain(job, timeout: float = 2.0) -> None:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while job.status in ("pending", "running"):
        if loop.time() > deadline:
            raise AssertionError(f"job did not finish; log={job.log_lines[-3:]}")
        await asyncio.sleep(0.01)


# ── presence detection ----------------------------------------------------


def test_node_is_present_detects_path_binary(monkeypatch):
    monkeypatch.setattr(nodep.shutil, "which", lambda b: f"/usr/bin/{b}")
    assert nodep.node_is_present() is True


def test_node_is_present_false_when_npx_missing_from_path(monkeypatch):
    monkeypatch.setattr(nodep.shutil, "which", lambda b: "/usr/bin/node" if b == "node" else None)
    assert nodep.node_is_present() is False


def test_node_is_present_false_when_node_npx_in_different_dirs(monkeypatch):
    def _which(b):
        return "/opt/playwright/node" if b == "node" else "/usr/bin/npx"
    monkeypatch.setattr(nodep.shutil, "which", _which)
    assert nodep.node_is_present() is False


def test_node_is_present_detects_app_data_binary(isolated_tools_dir, monkeypatch):
    monkeypatch.setattr(nodep.shutil, "which", lambda b: None)
    bin_dir = nodep.node_bin_dir()
    bin_dir.mkdir(parents=True, exist_ok=True)
    (bin_dir / "node").write_text("x", encoding="utf-8")
    (bin_dir / "npx").write_text("x", encoding="utf-8")
    assert nodep.node_is_present() is True


def test_node_is_present_false_when_absent(isolated_tools_dir, monkeypatch):
    monkeypatch.setattr(nodep.shutil, "which", lambda b: None)
    assert nodep.node_is_present() is False


# ── arch selection --------------------------------------------------------


def test_arch_selects_arm64_url(monkeypatch):
    monkeypatch.setattr(nodep.platform, "machine", lambda: "arm64")
    assert "darwin-arm64" in nodep._tarball_url()


def test_arch_selects_x64_url(monkeypatch):
    monkeypatch.setattr(nodep.platform, "machine", lambda: "x86_64")
    assert "darwin-x64" in nodep._tarball_url()


# ── install: brew fast path -----------------------------------------------


async def test_install_uses_brew_when_homebrew_present(isolated_tools_dir, monkeypatch):
    state = {"installed": False}
    runs = []

    async def fake_run(cmd, *, job, timeout=600.0, env=None):
        runs.append(cmd)
        state["installed"] = True
        return 0

    def fake_which(_b):
        return "/opt/homebrew/bin/node" if state["installed"] else None

    portable_called = {"v": False}

    async def fake_portable(job):
        portable_called["v"] = True

    monkeypatch.setattr(nodep.shutil, "which", fake_which)
    monkeypatch.setattr("backend.omlx_provisioner.has_homebrew", lambda: True)
    monkeypatch.setattr("backend.omlx_provisioner._brew_bin", lambda: "/opt/homebrew/bin/brew")
    monkeypatch.setattr(nodep, "run_streaming", fake_run)
    monkeypatch.setattr(nodep, "_install_portable", fake_portable)

    job = await nodep.ainstall_node()
    await _drain(job)

    assert job.status == "done"
    assert runs == [["/opt/homebrew/bin/brew", "install", "node"]]
    assert portable_called["v"] is False


# ── install: portable fallback --------------------------------------------


async def test_install_falls_back_to_tarball_without_homebrew(isolated_tools_dir, monkeypatch):
    monkeypatch.setattr(nodep.shutil, "which", lambda b: None)
    monkeypatch.setattr("backend.omlx_provisioner.has_homebrew", lambda: False)

    async def fake_portable(job):
        bin_dir = nodep.node_bin_dir()
        bin_dir.mkdir(parents=True, exist_ok=True)
        (bin_dir / "node").write_text("x", encoding="utf-8")
        job.append("portable install done")

    runs = []

    async def fake_run(cmd, *, job, timeout=600.0, env=None):
        runs.append(cmd)
        return 0

    monkeypatch.setattr(nodep, "_install_portable", fake_portable)
    monkeypatch.setattr(nodep, "run_streaming", fake_run)

    job = await nodep.ainstall_node()
    await _drain(job)

    assert job.status == "done"
    assert runs == []  # brew never invoked
    assert (nodep.node_bin_dir() / "node").exists()


async def test_install_falls_back_to_tarball_when_brew_fails(isolated_tools_dir, monkeypatch):
    monkeypatch.setattr(nodep.shutil, "which", lambda b: None)
    monkeypatch.setattr("backend.omlx_provisioner.has_homebrew", lambda: True)
    monkeypatch.setattr("backend.omlx_provisioner._brew_bin", lambda: "/opt/homebrew/bin/brew")

    async def fake_run(cmd, *, job, timeout=600.0, env=None):
        return 1  # brew install fails

    portable_called = {"v": False}

    async def fake_portable(job):
        portable_called["v"] = True

    monkeypatch.setattr(nodep, "run_streaming", fake_run)
    monkeypatch.setattr(nodep, "_install_portable", fake_portable)

    job = await nodep.ainstall_node()
    await _drain(job)

    assert job.status == "done"
    assert portable_called["v"] is True


async def test_install_errors_when_both_paths_fail(isolated_tools_dir, monkeypatch):
    monkeypatch.setattr(nodep.shutil, "which", lambda b: None)
    monkeypatch.setattr("backend.omlx_provisioner.has_homebrew", lambda: False)

    async def boom(job):
        raise RuntimeError("download failed")

    monkeypatch.setattr(nodep, "_install_portable", boom)

    job = await nodep.ainstall_node()
    await _drain(job)

    assert job.status == "error"
    assert "download failed" in (job.error or "")


async def test_install_noop_when_already_present(isolated_tools_dir, monkeypatch):
    monkeypatch.setattr(nodep.shutil, "which", lambda b: f"/usr/bin/{b}")

    async def fake_portable(job):
        raise AssertionError("should not install when already present")

    monkeypatch.setattr(nodep, "_install_portable", fake_portable)

    job = await nodep.ainstall_node()
    await _drain(job)

    assert job.status == "done"
    assert any("already available" in line for line in job.log_lines)


async def test_force_bypasses_presence_check(isolated_tools_dir, monkeypatch):
    monkeypatch.setattr(nodep.shutil, "which", lambda b: f"/usr/bin/{b}")
    monkeypatch.setattr("backend.omlx_provisioner.has_homebrew", lambda: False)

    portable_called = {"v": False}

    async def fake_portable(job):
        portable_called["v"] = True

    monkeypatch.setattr(nodep, "_install_portable", fake_portable)

    job = await nodep.ainstall_node(force=True)
    await _drain(job)

    assert portable_called["v"] is True
