"""Offline tests for exo's brew-free prereq bootstrap.

Validates :func:`backend.exo_cli.install_prereqs` on macOS picks the
Homebrew fast path when brew is present and falls back to the official
uv installer + portable Node tarball when it is absent, and that the
manual-instructions message no longer leads with Homebrew.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend import exo_cli
from backend.exo_cli import Prereqs


def _which_factory(present: set[str]):
    def _which(binary: str, path: str | None = None):
        return f"/usr/bin/{binary}" if binary in present else None
    return _which


def _complete_prereqs() -> Prereqs:
    return Prereqs(
        brew="/brew", uv="/uv", node="/node", npm="/npm", git="/git",
        rustup="/r", cargo="/c", rust_nightly=True, platform="Darwin",
    )


def _detect_seq(monkeypatch, first: Prereqs):
    """Patch detect_prereqs to return ``first`` then an all-present snapshot.

    install_prereqs detects once up front (driving install branches) and
    again at the end to verify; the second call must report success.
    """
    calls = {"n": 0}

    def _detect():
        calls["n"] += 1
        return first if calls["n"] == 1 else _complete_prereqs()

    monkeypatch.setattr(exo_cli, "detect_prereqs", _detect)


# ── macOS: Homebrew present -----------------------------------------------


def test_macos_uses_brew_when_present(monkeypatch):
    _detect_seq(monkeypatch, Prereqs(
        brew="/opt/homebrew/bin/brew", uv=None, node=None, npm=None,
        git=None, rustup="/r", cargo="/c", rust_nightly=True, platform="Darwin",
    ))
    monkeypatch.setattr(exo_cli.shutil, "which", _which_factory({"rustup"}))
    run_streaming = MagicMock(return_value=0)
    monkeypatch.setattr(exo_cli, "run_streaming", run_streaming)

    exo_cli.install_prereqs(auto=True)

    cmds = [tuple(c.args) for c in run_streaming.call_args_list]
    assert ("brew", "install", "uv") in cmds
    assert ("brew", "install", "node") in cmds
    assert ("brew", "install", "git") in cmds


# ── macOS: no Homebrew ----------------------------------------------------


def test_macos_falls_back_to_portable_without_brew(monkeypatch):
    _detect_seq(monkeypatch, Prereqs(
        brew=None, uv=None, node=None, npm=None,
        git="/usr/bin/git", rustup="/r", cargo="/c",
        rust_nightly=True, platform="Darwin",
    ))
    monkeypatch.setattr(exo_cli.shutil, "which", _which_factory({"git", "rustup"}))
    run_streaming = MagicMock(return_value=0)
    monkeypatch.setattr(exo_cli, "run_streaming", run_streaming)
    portable = MagicMock()
    monkeypatch.setattr(exo_cli, "_install_node_portable_macos", portable)

    exo_cli.install_prereqs(auto=True)

    portable.assert_called_once()
    all_cmds = [" ".join(c.args) for c in run_streaming.call_args_list]
    # uv via official installer, never brew.
    assert any("astral.sh/uv/install.sh" in c for c in all_cmds)
    assert all("brew" not in c for c in all_cmds)


def test_rustup_installed_via_official_script_when_missing(monkeypatch):
    _detect_seq(monkeypatch, Prereqs(
        brew=None, uv="/uv", node="/node", npm="/npm",
        git="/git", rustup=None, cargo=None,
        rust_nightly=False, platform="Darwin",
    ))
    monkeypatch.setattr(exo_cli.shutil, "which", _which_factory({"uv", "node", "git"}))
    run_streaming = MagicMock(return_value=0)
    monkeypatch.setattr(exo_cli, "run_streaming", run_streaming)

    exo_cli.install_prereqs(auto=True)

    all_cmds = [" ".join(c.args) for c in run_streaming.call_args_list]
    assert any("sh.rustup.rs" in c for c in all_cmds)


# ── manual instructions ---------------------------------------------------


def test_manual_instructions_lead_with_brew_free(monkeypatch, capsys):
    prereqs = Prereqs(
        brew=None, uv=None, node=None, npm=None,
        git=None, rustup=None, cargo=None,
        rust_nightly=False, platform="Darwin",
    )
    monkeypatch.setattr(exo_cli, "detect_prereqs", lambda: prereqs)

    with pytest.raises(SystemExit):
        exo_cli.install_prereqs(auto=False)

    err = capsys.readouterr().err
    assert "astral.sh/uv/install.sh" in err
    assert "brew install uv node" not in err
