"""Allowed-folder grants, path policy, and Seatbelt deny ordering."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from backend.exec_sandbox import ShellResult, render_profile
from backend.workspace_grants import list_grants, revoke_grant, upsert_grant
from backend.workspace_policy import GrantRejected, validate_grantable


@pytest.fixture
def grant_store(tmp_path, monkeypatch):
    path = tmp_path / "workspace_grants.json"
    monkeypatch.setattr("backend.workspace_grants.grants_path", lambda: path)
    return path


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
def test_shell_can_read_session_files_but_not_otto_config(tmp_path: Path):
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
            "/bin/sh", "-c", "pwd && cat uploads/note.txt && echo logged > .live_test.log",
        ],
        cwd=session,
        capture_output=True, text=True, check=False,
    )
    assert allowed.returncode == 0, allowed.stderr
    assert "hello" in allowed.stdout
    assert "Operation not permitted" not in (allowed.stderr or "")
    assert (session / ".live_test.log").read_text(encoding="utf-8").strip() == "logged"
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


def test_sensitive_denies_follow_write_allows(tmp_path: Path):
    home = tmp_path / "home"
    ssh = home / ".ssh"
    profile = render_profile(write_paths=[home], sensitive_paths=[ssh])
    allow_at = profile.index(f'(allow file* (subpath "{home}"))')
    deny_at = profile.index(f'(deny file-read* (subpath "{ssh}"))')
    assert deny_at > allow_at


@pytest.mark.skipif(sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").exists(), reason="Seatbelt")
def test_write_grant_on_parent_does_not_unlock_sensitive_child(tmp_path: Path):
    import subprocess

    home = tmp_path / "home"
    secret = home / ".ssh"
    secret.mkdir(parents=True)
    (secret / "id").write_text("SECRET", encoding="utf-8")
    project = home / "proj"
    project.mkdir()
    profile = tmp_path / "test.sb"
    profile.write_text(
        render_profile(write_paths=[home], sensitive_paths=[secret]),
        encoding="utf-8",
    )
    blocked = subprocess.run(
        ["/usr/bin/sandbox-exec", "-f", str(profile), "/bin/cat", str(secret / "id")],
        capture_output=True, text=True, check=False,
    )
    assert blocked.returncode != 0
    assert "SECRET" not in (blocked.stdout or "")
    allowed = subprocess.run(
        ["/usr/bin/sandbox-exec", "-f", str(profile), "/usr/bin/touch", str(project / "ok")],
        capture_output=True, text=True, check=False,
    )
    assert allowed.returncode == 0
    assert (project / "ok").is_file()


def test_rejects_home_root_and_sensitive_parent(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    ssh = home / ".ssh"
    ssh.mkdir(parents=True)
    project = home / "proj"
    project.mkdir()
    monkeypatch.setattr("backend.workspace_policy.sensitive_read_denies", lambda: [ssh])
    monkeypatch.setattr("backend.workspace_policy._blocked_roots", lambda: {str(home)})
    monkeypatch.setattr(
        "backend.workspace_policy._sensitive_paths",
        lambda: [ssh],
    )
    with pytest.raises(GrantRejected):
        validate_grantable(home)
    with pytest.raises(GrantRejected):
        validate_grantable(ssh)
    assert validate_grantable(project) == project.resolve()


def test_grant_round_trip(tmp_path: Path, grant_store, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.setattr("backend.workspace_policy.validate_grantable", lambda path: Path(path).resolve())
    grant = upsert_grant(project, mode="read", source="user")
    assert grant.mode == "read"
    assert list_grants()[0].id == grant.id
    again = upsert_grant(project, mode="write", source="user")
    assert again.id == grant.id
    assert again.mode == "write"
    assert len(list_grants()) == 1
    assert revoke_grant(grant.id) is not None
    assert list_grants() == []
    assert grant_store.is_file()
