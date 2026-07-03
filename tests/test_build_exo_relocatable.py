"""Tests for the exo runtime build script's relocatability safety net.

``scripts/build_exo.py`` packs a prebuilt exo ``.venv`` that must run on
any user's Mac, not just the machine that built it. These tests cover
``_is_relocatable_dep`` (classifying dylib load paths) and
``verify_relocatable`` (the build-time check that fails loudly instead of
shipping a Homebrew/system-linked binary — see the postmortem in
``uv_sync``'s docstring for the bug this guards against).
"""

from __future__ import annotations

import importlib.util
import stat
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "build_exo", Path(__file__).resolve().parent.parent / "scripts" / "build_exo.py"
)
build_exo = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("build_exo", build_exo)
_SPEC.loader.exec_module(build_exo)


@pytest.mark.parametrize(
    "path,expected",
    [
        ("@rpath/libpython3.13.dylib", True),
        ("@loader_path/../lib/libpython3.13.dylib", True),
        ("@executable_path/python", True),
        ("/usr/lib/libSystem.B.dylib", True),
        ("/System/Library/Frameworks/Metal.framework/Metal", True),
        ("/opt/homebrew/Cellar/python@3.13/3.13.14/Frameworks/Python.framework/Versions/3.13/Python", False),
        ("/usr/local/Cellar/python@3.13/3.13.14/lib/libpython3.13.dylib", False),
        ("/Users/someone/.local/share/uv/python/cpython-3.13/lib/libpython3.13.dylib", False),
    ],
)
def test_is_relocatable_dep(path, expected):
    assert build_exo._is_relocatable_dep(path) is expected


def _write_fake_binary(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def test_verify_relocatable_raises_on_homebrew_dependency(tmp_path, monkeypatch):
    repo = tmp_path
    python_bin = repo / ".venv" / "bin" / "python"
    _write_fake_binary(python_bin)

    def _fake_otool_deps(binary: Path) -> list[str]:
        if binary == python_bin:
            return [
                "/opt/homebrew/Cellar/python@3.13/3.13.14/Frameworks/Python.framework/Versions/3.13/Python",
                "/usr/lib/libSystem.B.dylib",
            ]
        return []

    monkeypatch.setattr(build_exo, "_otool_deps", _fake_otool_deps)

    with pytest.raises(RuntimeError, match="NOT relocatable"):
        build_exo.verify_relocatable(repo)


def test_verify_relocatable_passes_when_clean(tmp_path, monkeypatch):
    repo = tmp_path
    python_bin = repo / ".venv" / "bin" / "python"
    _write_fake_binary(python_bin)

    monkeypatch.setattr(
        build_exo, "_otool_deps",
        lambda binary: ["@rpath/libpython3.13.dylib", "/usr/lib/libSystem.B.dylib"],
    )

    build_exo.verify_relocatable(repo)  # must not raise


def test_verify_relocatable_raises_when_no_venv(tmp_path):
    with pytest.raises(RuntimeError, match="no .venv found"):
        build_exo.verify_relocatable(tmp_path)


def test_verify_relocatable_ignores_own_install_name(tmp_path, monkeypatch):
    """A dylib's own install name (LC_ID_DYLIB) is not a dependency.

    ``otool -L`` lists it alongside real deps and it is frequently an
    absolute build-tree or delocate-sentinel path (e.g. ``/DLC/...``), but
    it never affects how the loader finds *other* libraries — so it must
    not be flagged. This is the exact false positive that broke CI.
    """
    repo = tmp_path
    dylib = repo / ".venv" / "lib" / "python3.13" / "site-packages" / "PIL" / ".dylibs" / "libavif.16.dylib"
    _write_fake_binary(dylib)

    install_name = "/DLC/PIL/.dylibs/libavif.16.dylib"
    monkeypatch.setattr(
        build_exo, "_otool_deps",
        lambda binary: [install_name, "/usr/lib/libSystem.B.dylib"],
    )
    monkeypatch.setattr(build_exo, "_install_name", lambda binary: install_name)

    build_exo.verify_relocatable(repo)  # must not raise


def test_verify_relocatable_ignores_self_reference(tmp_path, monkeypatch):
    """A binary that references its own absolute build path is not broken."""
    repo = tmp_path
    ext = repo / ".venv" / "lib" / "python3.13" / "site-packages" / "charset_normalizer" / "md.cpython-313-darwin.so"
    _write_fake_binary(ext)

    monkeypatch.setattr(build_exo, "_otool_deps", lambda binary: [str(binary)])
    monkeypatch.setattr(build_exo, "_install_name", lambda binary: None)

    build_exo.verify_relocatable(repo)  # must not raise


def test_verify_relocatable_ignores_relative_paths(tmp_path, monkeypatch):
    """Relative load commands (e.g. protobuf's ``bazel-out/...``) resolve via
    rpath on any machine and are not build-host-specific."""
    repo = tmp_path
    ext = repo / ".venv" / "lib" / "python3.13" / "site-packages" / "google" / "_upb" / "_message.abi3.so"
    _write_fake_binary(ext)

    monkeypatch.setattr(
        build_exo, "_otool_deps",
        lambda binary: [
            "bazel-out/osx-aarch_64-opt/bin/python/lib_message_binary.so",
            "/usr/lib/libSystem.B.dylib",
        ],
    )
    monkeypatch.setattr(build_exo, "_install_name", lambda binary: None)

    build_exo.verify_relocatable(repo)  # must not raise
