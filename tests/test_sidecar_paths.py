"""Tests for PATH wiring of brew-free tool installs in the frozen app.

:func:`backend.server._ensure_sidecar_paths` must prepend the
app-data Node bin dir and ``~/.local/bin`` (uv) to ``PATH`` when running
as a frozen PyInstaller binary, and must be a no-op otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from backend import server


@pytest.fixture()
def restore_syspath():
    snapshot = list(sys.path)
    yield
    sys.path[:] = snapshot


def test_ensure_sidecar_paths_adds_tool_dirs(tmp_path, monkeypatch, restore_syspath):
    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr("sys._MEIPASS", str(tmp_path), raising=False)
    monkeypatch.setenv("PATH", "/usr/bin")

    server._ensure_sidecar_paths()

    path = __import__("os").environ["PATH"]
    node_bin = str(Path.home() / "Library" / "Application Support" / "Otto" / "tools" / "node" / "bin")
    local_bin = str(Path.home() / ".local" / "bin")
    assert node_bin in path
    assert local_bin in path


def test_ensure_sidecar_paths_noop_when_not_frozen(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    server._ensure_sidecar_paths()

    assert __import__("os").environ["PATH"] == "/usr/bin:/bin"
