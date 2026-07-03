"""Offline unit tests for the shared brew-free install primitives.

Covers :mod:`backend.tool_provisioner`: app-data tools dir, tarball
extraction (top-level strip), SHA-256 verification, the streaming
``httpx`` download (with a fake client), the ``uv`` installer's path
selection (brew vs official installer), and ToolJob serialization.

No network, no real ``brew``/``curl``, no real downloads.
"""

from __future__ import annotations

import asyncio
import hashlib
import tarfile

import pytest

from backend import tool_provisioner as tp


@pytest.fixture()
def isolated_tools_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tp, "get_app_data_dir", lambda: tmp_path)
    return tmp_path


async def _drain(job, timeout: float = 2.0) -> None:
    """Wait for a background install job to leave the running state."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while job.status in ("pending", "running"):
        if loop.time() > deadline:
            raise AssertionError(f"job did not finish; last log={job.log_lines[-3:]}")
        await asyncio.sleep(0.01)


def _make_node_tarball(path, top="node-v22.14.0-darwin-arm64"):
    """Build a fixture tarball wrapping bin/node in a single top dir."""
    payload = path.parent / "node"
    payload.write_text("#!/bin/sh\necho node", encoding="utf-8")
    with tarfile.open(path, "w:gz") as tf:
        tf.add(payload, arcname=f"{top}/bin/node")
    payload.unlink()


# ── tools_dir -------------------------------------------------------------


def test_tools_dir_under_app_data(isolated_tools_dir):
    d = tp.tools_dir()
    assert d == isolated_tools_dir / "tools"
    assert d.is_dir()


# ── extract_tarball -------------------------------------------------------


def test_extract_tarball_strips_top_level_dir(isolated_tools_dir, tmp_path):
    archive = tmp_path / "node.tar.gz"
    _make_node_tarball(archive)

    dest = tp.tools_dir() / "node"
    tp.extract_tarball(archive, dest, strip_top=True)

    assert (dest / "bin" / "node").exists()
    # The wrapping node-v.../ directory must NOT survive.
    assert not (dest / "node-v22.14.0-darwin-arm64").exists()


def test_extract_tarball_without_strip_keeps_top(isolated_tools_dir, tmp_path):
    archive = tmp_path / "node.tar.gz"
    _make_node_tarball(archive)

    dest = tp.tools_dir() / "raw"
    tp.extract_tarball(archive, dest, strip_top=False)
    assert (dest / "node-v22.14.0-darwin-arm64" / "bin" / "node").exists()


# ── verify_sha256 ---------------------------------------------------------


def test_verify_sha256_accepts_matching_digest(tmp_path):
    f = tmp_path / "blob"
    f.write_bytes(b"hello otto")
    digest = hashlib.sha256(b"hello otto").hexdigest()
    tp.verify_sha256(f, digest)  # must not raise


def test_verify_sha256_rejects_mismatch(tmp_path):
    f = tmp_path / "blob"
    f.write_bytes(b"hello otto")
    with pytest.raises(ValueError):
        tp.verify_sha256(f, "deadbeef")


# ── download_file ---------------------------------------------------------


class _FakeStreamResp:
    def __init__(self, data: bytes):
        self._data = data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def raise_for_status(self):
        return None

    async def aiter_bytes(self, chunk_size: int):
        for i in range(0, len(self._data), chunk_size):
            yield self._data[i:i + chunk_size]


class _FakeClient:
    def __init__(self, data: bytes):
        self._data = data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def stream(self, method: str, url: str):
        return _FakeStreamResp(self._data)


async def test_download_file_streams_to_dest(isolated_tools_dir, tmp_path, monkeypatch):
    payload = b"x" * (3 * 1024 * 1024)
    monkeypatch.setattr(
        tp.httpx, "AsyncClient", lambda *a, **k: _FakeClient(payload)
    )
    job = tp.new_job("test")
    dest = tmp_path / "out.bin"

    out = await tp.download_file("https://example/test", dest, job=job)

    assert out == dest
    assert dest.read_bytes() == payload
    assert any("Downloaded" in line for line in job.log_lines)


# ── ToolJob ---------------------------------------------------------------


def test_tool_job_to_dict_and_log_tail():
    job = tp.ToolJob(id="abc", kind="install-node")
    for i in range(tp.ToolJob._LOG_TAIL + 50):
        job.append(f"line {i}")
    assert len(job.log_lines) == tp.ToolJob._LOG_TAIL
    d = job.to_dict()
    assert d["id"] == "abc" and d["kind"] == "install-node"
    assert d["log_lines"][-1] == f"line {tp.ToolJob._LOG_TAIL + 49}"


def test_job_registry_roundtrip():
    job = tp.new_job("install-node")
    assert tp.get_job(job.id) is job
    assert job in tp.list_jobs()


# ── ainstall_uv (brew vs official installer) ------------------------------


async def test_uv_uses_brew_when_present(isolated_tools_dir, monkeypatch):
    calls = []

    async def fake_run(cmd, *, job, timeout=600.0, env=None):
        calls.append(cmd)
        return 0

    monkeypatch.setattr("backend.omlx_provisioner.has_homebrew", lambda: True)
    monkeypatch.setattr("backend.omlx_provisioner._brew_bin", lambda: "/opt/homebrew/bin/brew")
    # brew install yields a usable uv on PATH → no fallthrough.
    monkeypatch.setattr(tp.shutil, "which", lambda b: "/opt/homebrew/bin/uv")
    monkeypatch.setattr(tp, "run_streaming", fake_run)

    job = await tp.ainstall_uv()
    await _drain(job)

    assert job.status == "done"
    assert calls == [["/opt/homebrew/bin/brew", "install", "uv"]]


async def test_uv_uses_curl_installer_without_brew(isolated_tools_dir, monkeypatch, tmp_path):
    calls = []
    local_bin = tmp_path / "localbin"
    local_bin.mkdir()
    monkeypatch.setattr(tp, "_local_bin_dir", lambda: local_bin)

    async def fake_run(cmd, *, job, timeout=600.0, env=None):
        calls.append(cmd)
        (local_bin / "uv").write_text("x", encoding="utf-8")  # installer drops uv
        return 0

    monkeypatch.setattr("backend.omlx_provisioner.has_homebrew", lambda: False)
    # curl + bash available so the official installer path is taken.
    monkeypatch.setattr(tp.shutil, "which", lambda b: f"/usr/bin/{b}")
    monkeypatch.setattr(tp, "run_streaming", fake_run)

    job = await tp.ainstall_uv()
    await _drain(job)

    assert job.status == "done"
    assert len(calls) == 1
    assert calls[0][0] == "bash"
    assert "astral.sh/uv/install.sh" in calls[0][-1]
    assert "brew" not in " ".join(calls[0])


async def test_uv_falls_back_to_portable_when_no_curl(isolated_tools_dir, monkeypatch):
    """No brew, no curl/bash → portable GitHub-release binary download."""
    monkeypatch.setattr("backend.omlx_provisioner.has_homebrew", lambda: False)
    monkeypatch.setattr(tp.shutil, "which", lambda b: None)

    portable = {"v": False}

    async def fake_portable(job):
        portable["v"] = True

    monkeypatch.setattr(tp, "_install_uv_portable", fake_portable)

    job = await tp.ainstall_uv()
    await _drain(job)

    assert job.status == "done"
    assert portable["v"] is True


async def test_uv_falls_back_to_portable_when_curl_installer_fails(
    isolated_tools_dir, monkeypatch, tmp_path
):
    local_bin = tmp_path / "localbin"
    local_bin.mkdir()
    monkeypatch.setattr(tp, "_local_bin_dir", lambda: local_bin)
    monkeypatch.setattr("backend.omlx_provisioner.has_homebrew", lambda: False)
    monkeypatch.setattr(
        tp.shutil, "which",
        lambda b: f"/usr/bin/{b}" if b in ("curl", "bash") else None,
    )

    async def fake_run(cmd, *, job, timeout=600.0, env=None):
        return 1  # curl|sh installer fails / yields no uv

    monkeypatch.setattr(tp, "run_streaming", fake_run)

    portable = {"v": False}

    async def fake_portable(job):
        portable["v"] = True

    monkeypatch.setattr(tp, "_install_uv_portable", fake_portable)

    job = await tp.ainstall_uv()
    await _drain(job)

    assert job.status == "done"
    assert portable["v"] is True


def test_uv_arch_selects_aarch64_url(monkeypatch):
    monkeypatch.setattr(tp.platform, "machine", lambda: "arm64")
    assert "uv-aarch64-apple-darwin" in tp._uv_tarball_url()


def test_uv_arch_selects_x86_64_url(monkeypatch):
    monkeypatch.setattr(tp.platform, "machine", lambda: "x86_64")
    assert "uv-x86_64-apple-darwin" in tp._uv_tarball_url()
