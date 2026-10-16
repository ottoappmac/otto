"""Allowed-folder grants, path policy, and Seatbelt deny ordering."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from backend.exec_sandbox import render_profile
from backend.workspace_grants import list_grants, revoke_grant, upsert_grant
from backend.workspace_policy import GrantRejected, validate_grantable


@pytest.fixture
def grant_store(tmp_path, monkeypatch):
    path = tmp_path / "workspace_grants.json"
    monkeypatch.setattr("backend.workspace_grants.grants_path", lambda: path)
    return path


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
