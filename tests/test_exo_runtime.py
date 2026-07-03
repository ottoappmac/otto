"""Tests for the prebuilt exo runtime's ``is_installed`` health check.

Covers the fix for a class of bug where a runtime artifact has all the
right files on disk (console script + resolvable ``exo`` package) but its
bundled interpreter was accidentally linked against a build-host-only
path (e.g. a Homebrew ``Python.framework``) and dies with a dyld
"Library not loaded" error the instant it's invoked. See
``scripts/build_exo.py``'s ``uv_sync``/``verify_relocatable`` for the
build-time half of this fix.
"""

from __future__ import annotations

import stat
import subprocess
from pathlib import Path

from backend import exo_runtime


def _make_runtime(tmp_path: Path, *, working_python: bool) -> Path:
    """Build a minimal, filesystem-valid prebuilt runtime tree under ``tmp_path``.

    ``working_python=False`` writes a ``bin/python`` shim that always exits
    non-zero (standing in for an interpreter that dyld-fails to launch).
    """
    venv = tmp_path / ".venv"
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)

    site_packages = venv / "lib" / "python3.13" / "site-packages"
    site_packages.mkdir(parents=True)
    exo_pkg = site_packages / "exo"
    exo_pkg.mkdir()
    (exo_pkg / "__init__.py").write_text("")

    (bin_dir / "exo").write_text("#!/bin/sh\necho exo\n")
    (bin_dir / "exo").chmod(0o755)

    python = bin_dir / "python"
    if working_python:
        python.write_text("#!/bin/sh\nexit 0\n")
    else:
        python.write_text("#!/bin/sh\necho 'dyld: Library not loaded' >&2\nexit 1\n")
    python.chmod(python.stat().st_mode | stat.S_IEXEC)

    return tmp_path


def test_interpreter_launches_true_for_working_python(tmp_path, monkeypatch):
    _make_runtime(tmp_path, working_python=True)
    monkeypatch.setattr(exo_runtime, "runtime_dir", lambda: tmp_path)

    assert exo_runtime._interpreter_launches() is True


def test_interpreter_launches_false_for_broken_python(tmp_path, monkeypatch):
    _make_runtime(tmp_path, working_python=False)
    monkeypatch.setattr(exo_runtime, "runtime_dir", lambda: tmp_path)

    assert exo_runtime._interpreter_launches() is False


def test_interpreter_launches_false_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(exo_runtime, "runtime_dir", lambda: tmp_path)

    assert exo_runtime._interpreter_launches() is False


def test_is_installed_false_when_interpreter_is_broken(tmp_path, monkeypatch):
    """A runtime with valid files but a dyld-broken interpreter is unusable.

    Regression test: previously ``is_installed`` only checked the console
    script + ``.pth`` resolution, so a non-relocatable artifact (like the
    one produced before the ``UV_PYTHON_PREFERENCE=only-managed`` fix)
    reported ``installed=True`` forever and Otto never re-downloaded it —
    the daemon just crash-looped on every "Up" click.
    """
    _make_runtime(tmp_path, working_python=False)
    monkeypatch.setattr(exo_runtime, "runtime_dir", lambda: tmp_path)
    monkeypatch.setattr(exo_runtime, "load_runtime_state", lambda: exo_runtime.RuntimeState())

    assert exo_runtime.is_installed("") is False


def test_is_installed_true_when_everything_works(tmp_path, monkeypatch):
    _make_runtime(tmp_path, working_python=True)
    monkeypatch.setattr(exo_runtime, "runtime_dir", lambda: tmp_path)
    monkeypatch.setattr(exo_runtime, "load_runtime_state", lambda: exo_runtime.RuntimeState())

    assert exo_runtime.is_installed("") is True


def test_interpreter_launches_handles_timeout(tmp_path, monkeypatch):
    _make_runtime(tmp_path, working_python=True)
    monkeypatch.setattr(exo_runtime, "runtime_dir", lambda: tmp_path)

    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="python", timeout=10)

    monkeypatch.setattr(exo_runtime.subprocess, "run", _raise_timeout)

    assert exo_runtime._interpreter_launches() is False
