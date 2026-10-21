"""Allowed-folder grants, path policy, and Seatbelt deny ordering."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from backend.exec_sandbox import ShellResult, render_profile
from backend.workspace_grants import list_grants, revoke_grant, upsert_grant
from backend.workspace_policy import GrantRejected, protected_workspace_reason, validate_grantable


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


def test_concurrent_upserts_keep_every_grant(tmp_path: Path, grant_store):
    import threading

    folders = []
    for index in range(8):
        folder = tmp_path / f"p{index}"
        folder.mkdir()
        folders.append(folder)
    errors: list[BaseException] = []

    def worker(folder: Path) -> None:
        try:
            upsert_grant(folder, mode="write")
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(folder,)) for folder in folders]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(list_grants()) == len(folders)


def test_bound_session_scan_leaves_the_event_loop(tmp_path: Path, monkeypatch):
    import asyncio
    import threading

    from backend.routes import workspaces as routes

    root = tmp_path / "sessions"
    root.mkdir()
    monkeypatch.setattr("backend.session_manager._sessions_dir", lambda: root)
    seen: dict[str, int] = {}
    real = routes._bound

    def wrapped(grant_id: str):
        seen["thread"] = threading.get_ident()
        return real(grant_id)

    monkeypatch.setattr(routes, "_bound", wrapped)

    async def run() -> None:
        seen["loop"] = threading.get_ident()
        await routes._refresh_bound_sessions("missing", "read")

    asyncio.run(run())
    assert seen["thread"] != seen["loop"]


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


def test_protected_workspace_paths():
    assert protected_workspace_reason(".git/config")
    assert protected_workspace_reason(".GIT/hooks/pre-commit")
    assert protected_workspace_reason("src/.gitattributes")
    assert protected_workspace_reason("Pkg/.GitAttributes") is not None
    assert protected_workspace_reason(".vscode/tasks.json")
    assert protected_workspace_reason(".idea/workspace.xml")
    assert protected_workspace_reason(".envrc")
    assert protected_workspace_reason("AGENTS.md")
    assert protected_workspace_reason("CLAUDE.md")
    assert protected_workspace_reason("src/main.py") is None
    assert protected_workspace_reason("not.git/main.py") is None


def test_profile_denies_git_metadata_after_write_allow(tmp_path: Path):
    project = tmp_path / "proj"
    profile = render_profile(write_paths=[project])
    real = str(project.resolve())
    allow_at = profile.index(f'(allow file* (subpath "{real}"))')
    deny_at = profile.index("[.][Gg][Ii][Tt]")
    attr_at = profile.index("[Aa][Tt][Tt][Rr]")
    assert deny_at > allow_at
    assert attr_at > allow_at


@pytest.mark.skipif(sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").exists(), reason="Seatbelt")
def test_sandbox_cannot_write_git_metadata(tmp_path: Path):
    import subprocess

    project = tmp_path / "proj"
    (project / ".git" / "hooks").mkdir(parents=True)
    (project / "src").mkdir()
    profile = tmp_path / "test.sb"
    profile.write_text(render_profile(write_paths=[project]), encoding="utf-8")

    def run(script: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["/usr/bin/sandbox-exec", "-f", str(profile), "/bin/sh", "-c", script],
            cwd=project,
            capture_output=True,
            text=True,
            check=False,
        )

    assert run("echo pwned > .git/config").returncode != 0
    assert not (project / ".git" / "config").exists()
    assert run("echo pwned > .gitattributes").returncode != 0
    assert not (project / ".gitattributes").exists()
    assert run("echo pwned > src/.gitattributes").returncode != 0
    assert not (project / "src" / ".gitattributes").exists()
    assert run("mkdir -p .GIT/hooks && echo pwned > .GIT/hooks/pre-commit").returncode != 0
    assert not (project / ".git" / "hooks" / "pre-commit").exists()
    assert run("echo ok > ok.txt").returncode == 0
    assert (project / "ok.txt").read_text(encoding="utf-8").strip() == "ok"


def test_file_tools_refuse_git_metadata(tmp_path: Path):
    from backend.session_manager import _make_backend

    project = tmp_path / "proj"
    (project / ".git").mkdir(parents=True)
    (project / ".gitattributes").write_text("old\n", encoding="utf-8")
    files = tmp_path / "files"
    (files / "links").mkdir(parents=True)
    (files / "links" / "proj").symlink_to(project, target_is_directory=True)
    backend = _make_backend(files, project_root=project, grant_roots=[project])
    written = backend.write("/links/proj/.git/config", "[core]\n")
    assert written.error
    edited = backend.edit("/links/proj/.gitattributes", "old", "new")
    assert edited.error
    assert (project / ".gitattributes").read_text(encoding="utf-8") == "old\n"
    allowed = backend.write("/links/proj/main.py", "print(1)\n")
    assert not getattr(allowed, "error", None)
    assert (project / "main.py").read_text(encoding="utf-8") == "print(1)\n"
