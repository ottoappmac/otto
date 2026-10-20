"""Seatbelt re-opens the session files directory so output can be written."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from backend.exec_sandbox import ShellResult, render_profile


def test_session_files_reopen_after_otto_deny(tmp_path: Path):
    otto = tmp_path / "Library" / "Application Support" / "Otto"
    session = otto / "sessions" / "abc" / "files"
    profile = render_profile(
        write_paths=[session],
        sensitive_paths=[otto],
        session_paths=[session],
    )
    otto_real = str(otto.resolve())
    session_real = str(session.resolve())
    deny_at = profile.index(f'(deny file-read* (subpath "{otto_real}"))')
    allow_at = profile.rindex(f'(allow file-read* (subpath "{session_real}"))')
    write_at = profile.rindex(f'(allow file-write* (subpath "{session_real}"))')
    assert allow_at > deny_at
    assert write_at > deny_at


@pytest.mark.skipif(sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").exists(), reason="Seatbelt")
def test_shell_can_write_output_and_read_uploads(tmp_path: Path):
    import subprocess

    otto = tmp_path / "Library" / "Application Support" / "Otto"
    (otto / "sessions").mkdir(parents=True)
    (otto / "config.json").write_text("SECRET", encoding="utf-8")
    session = otto / "sessions" / "abc" / "files"
    (session / "uploads").mkdir(parents=True)
    (session / "uploads" / "note.txt").write_text("hello", encoding="utf-8")
    profile = tmp_path / "test.sb"
    profile.write_text(
        render_profile(
            write_paths=[session],
            sensitive_paths=[otto],
            session_paths=[session],
        ),
        encoding="utf-8",
    )
    allowed = subprocess.run(
        [
            "/usr/bin/sandbox-exec", "-f", str(profile),
            "/bin/sh", "-c",
            "cat uploads/note.txt && mkdir -p output && echo hi > output/note.txt",
        ],
        cwd=session,
        capture_output=True, text=True, check=False,
    )
    assert allowed.returncode == 0, allowed.stderr
    assert "hello" in allowed.stdout
    assert (session / "output" / "note.txt").read_text(encoding="utf-8").strip() == "hi"
    blocked = subprocess.run(
        ["/usr/bin/sandbox-exec", "-f", str(profile), "/bin/cat", str(otto / "config.json")],
        cwd=session,
        capture_output=True, text=True, check=False,
    )
    assert blocked.returncode != 0
    assert "SECRET" not in (blocked.stdout or "")


def test_execute_keeps_session_files_env(tmp_path: Path, monkeypatch):
    from backend.session_manager import _make_backend

    files = tmp_path / "files"
    files.mkdir()
    project = tmp_path / "proj"
    project.mkdir()
    captured: dict = {}

    def _fake_run(command: str, **kwargs):
        captured["env"] = kwargs["env"]
        captured["session_paths"] = kwargs["session_paths"]
        captured["session_write"] = kwargs["session_write"]
        return ShellResult(stdout="ok", returncode=0)

    monkeypatch.setattr("backend.exec_sandbox.run_shell", _fake_run)
    monkeypatch.setattr("backend.exec_sandbox.sandbox_settings", lambda: (True, False, [], []))
    monkeypatch.setattr("backend.exec_sandbox.passthrough_names", lambda: [])
    backend = _make_backend(files, project_root=project)
    result = backend.execute("echo hi")
    assert result.exit_code == 0
    assert captured["env"]["SESSION_FILES"] == str(files)
    assert captured["env"]["PROJECT_ROOT"] == str(project)
    assert captured["session_paths"] == [files]
    assert captured["session_write"] is True
