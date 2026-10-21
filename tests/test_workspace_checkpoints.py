"""Git checkpoints do not touch the user's branch, and revert respects hashes."""

from __future__ import annotations

import os
import stat
import subprocess
import tempfile
from pathlib import Path

import pytest

from backend.exec_sandbox import ShellResult
from backend.schemas import SessionWorkspace
from backend.workspace_git import (
    capture_preimage,
    forget_session,
    list_changes,
    record_shell_changes,
    record_write,
    revert,
    revert_hunk,
    shell_baseline,
    snapshot_before_turn,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "otto@example.com")
    _git(root, "config", "user.name", "Otto")
    (root / "main.py").write_text("print(1)\n", encoding="utf-8")
    _git(root, "add", "main.py")
    _git(root, "commit", "-m", "init")
    return root


def test_snapshot_leaves_head_and_status(repo: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "backend.workspace_git._state_path",
        lambda session_id: tmp_path / f"{session_id}.workspace.json",
    )
    head_before = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    branch_before = subprocess.check_output(["git", "-C", str(repo), "branch", "--show-current"], text=True).strip()
    workspace = SessionWorkspace(host_path=str(repo), virtual_path="/links/proj", name="proj", grant_id="g", mode="write")
    result = snapshot_before_turn(workspace, "sess")
    assert result["ok"] is True
    head_after = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    branch_after = subprocess.check_output(["git", "-C", str(repo), "branch", "--show-current"], text=True).strip()
    assert head_after == head_before
    assert branch_after == branch_before
    ref = result["checkpoint"]["ref"]
    subprocess.check_output(["git", "-C", str(repo), "rev-parse", ref], text=True)


def _patch_state(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "backend.workspace_git._state_path",
        lambda session_id: tmp_path / f"{session_id}.workspace.json",
    )


def _workspace(repo: Path) -> SessionWorkspace:
    return SessionWorkspace(host_path=str(repo), virtual_path="/links/proj", name="proj", grant_id="g", mode="write")


def test_revert_refuses_when_file_changed_after_agent(repo: Path, tmp_path: Path, monkeypatch):
    _patch_state(monkeypatch, tmp_path)
    workspace = _workspace(repo)
    snap = snapshot_before_turn(workspace, "sess")
    assert snap["ok"]
    (repo / "main.py").write_text("print(2)\n", encoding="utf-8")
    record_write("sess", "main.py", b"print(1)\n", b"print(2)\n")
    (repo / "main.py").write_text("print(3)\n", encoding="utf-8")
    result = revert(workspace, "sess", paths=["main.py"])
    assert result["conflicts"]
    assert (repo / "main.py").read_text(encoding="utf-8") == "print(3)\n"


def test_revert_restores_agent_edit(repo: Path, tmp_path: Path, monkeypatch):
    _patch_state(monkeypatch, tmp_path)
    workspace = _workspace(repo)
    snapshot_before_turn(workspace, "sess")
    (repo / "main.py").write_text("print(2)\n", encoding="utf-8")
    record_write("sess", "main.py", b"print(1)\n", b"print(2)\n")
    result = revert(workspace, "sess", paths=["main.py"])
    assert result["restored"] == ["main.py"]
    assert (repo / "main.py").read_text(encoding="utf-8") == "print(1)\n"


def test_changes_ignore_preexisting_untracked_and_quote_paths(repo: Path, tmp_path: Path, monkeypatch):
    _patch_state(monkeypatch, tmp_path)
    (repo / "notes café.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "already.txt").write_text("keep\n", encoding="utf-8")
    workspace = _workspace(repo)
    assert snapshot_before_turn(workspace, "sess")["ok"]
    (repo / "notes café.py").write_text("x = 2\n", encoding="utf-8")
    (repo / "brand new.py").write_text("print('hi')\n", encoding="utf-8")
    (repo / "main.py").unlink()
    changes = list_changes(workspace, "sess")
    by_path = {item["path"]: item for item in changes["files"]}
    assert "already.txt" not in by_path
    assert by_path["notes café.py"]["status"] == "modified"
    assert by_path["brand new.py"]["status"] == "added"
    assert by_path["main.py"]["status"] == "deleted"
    assert by_path["notes café.py"]["added"] == 1
    assert by_path["notes café.py"]["deleted"] == 1


def test_later_shell_edit_does_not_block_reverting_that_turn(repo: Path, tmp_path: Path, monkeypatch):
    _patch_state(monkeypatch, tmp_path)
    workspace = _workspace(repo)
    snapshot_before_turn(workspace, "sess")
    (repo / "main.py").write_text("print(2)\n", encoding="utf-8")
    record_write("sess", "main.py", b"print(1)\n", b"print(2)\n")
    snapshot_before_turn(workspace, "sess")
    (repo / "main.py").write_text("print(3)\n", encoding="utf-8")
    record_write("sess", "main.py", b"print(2)\n", b"print(3)\n")
    result = revert(workspace, "sess", paths=["main.py"])
    assert result["conflicts"] == []
    assert result["restored"] == ["main.py"]
    assert (repo / "main.py").read_text(encoding="utf-8") == "print(2)\n"


def test_revert_hunk_undoes_one_change(repo: Path, tmp_path: Path, monkeypatch):
    _patch_state(monkeypatch, tmp_path)
    (repo / "main.py").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    _git(repo, "add", "main.py")
    _git(repo, "commit", "-m", "three lines")
    workspace = _workspace(repo)
    snapshot_before_turn(workspace, "sess")
    (repo / "main.py").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
    record_write("sess", "main.py", b"alpha\nbeta\ngamma\n", b"alpha\nBETA\ngamma\n")
    result = revert_hunk(workspace, "sess", path="main.py", hunk=0)
    assert result["conflicts"] == []
    assert (repo / "main.py").read_text(encoding="utf-8") == "alpha\nbeta\ngamma\n"


def test_nongit_revert_deletes_created_file_and_keeps_per_turn_copy(tmp_path: Path, monkeypatch):
    _patch_state(monkeypatch, tmp_path)
    folder = tmp_path / "plain"
    folder.mkdir()
    (folder / "keep.txt").write_text("one\n", encoding="utf-8")
    workspace = _workspace(folder)
    snapshot_before_turn(workspace, "sess")
    capture_preimage("sess", folder, folder / "keep.txt")
    (folder / "keep.txt").write_text("two\n", encoding="utf-8")
    record_write("sess", "keep.txt", b"one\n", b"two\n")
    (folder / "new.txt").write_text("fresh\n", encoding="utf-8")
    record_write("sess", "new.txt", None, b"fresh\n")
    result = revert(workspace, "sess")
    assert set(result["restored"]) == {"keep.txt", "new.txt"}
    assert (folder / "keep.txt").read_text(encoding="utf-8") == "one\n"
    assert not (folder / "new.txt").exists()


def test_snapshot_does_not_run_filters_or_hooks(repo: Path, tmp_path: Path, monkeypatch):
    _patch_state(monkeypatch, tmp_path)
    marker = tmp_path / "escaped"
    script = tmp_path / "pwn.sh"
    script.write_text(
        f"#!/bin/sh\necho ran >> {marker}\necho FILTERED\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    _git(repo, "config", "filter.pwn.clean", str(script))
    _git(repo, "config", "filter.pwn.required", "true")
    _git(repo, "config", "core.fsmonitor", str(script))
    (repo / ".gitattributes").write_text("* filter=pwn\n", encoding="utf-8")
    hook = repo / ".git" / "hooks" / "reference-transaction"
    hook.write_text(f"#!/bin/sh\necho hook >> {marker}\n", encoding="utf-8")
    hook.chmod(0o755)
    old_hooks = Path(tempfile.gettempdir()) / "otto-no-hooks"
    old_hooks.mkdir(exist_ok=True)
    planted = old_hooks / "reference-transaction"
    planted.write_text(f"#!/bin/sh\necho oldhook >> {marker}\n", encoding="utf-8")
    planted.chmod(0o755)
    try:
        result = snapshot_before_turn(_workspace(repo), "sess")
        assert result["ok"] is True, result
        assert not marker.exists()
        commit = result["checkpoint"]["commit"]
        blob = subprocess.check_output(
            ["git", "-C", str(repo), "cat-file", "-p", f"{commit}:main.py"],
            text=True,
        )
        assert blob == "print(1)\n"
    finally:
        planted.unlink(missing_ok=True)


def test_revert_leaves_user_edits_and_restores_agent_edits(repo: Path, tmp_path: Path, monkeypatch):
    _patch_state(monkeypatch, tmp_path)
    workspace = _workspace(repo)
    snapshot_before_turn(workspace, "sess")
    (repo / "main.py").write_text("print(2)\n", encoding="utf-8")
    record_write("sess", "main.py", b"print(1)\n", b"print(2)\n")
    (repo / "mine.py").write_text("user\n", encoding="utf-8")
    changes = list_changes(workspace, "sess")
    by_path = {item["path"]: item for item in changes["files"]}
    assert by_path["main.py"]["origin"] == "otto"
    assert by_path["mine.py"]["origin"] == "outside"
    result = revert(workspace, "sess")
    assert result["restored"] == ["main.py"]
    assert result["conflicts"] == []
    assert (repo / "main.py").read_text(encoding="utf-8") == "print(1)\n"
    assert (repo / "mine.py").read_text(encoding="utf-8") == "user\n"
    refused = revert(workspace, "sess", paths=["mine.py"])
    assert refused["conflicts"]
    assert "outside" in refused["conflicts"][0]["reason"]
    assert (repo / "mine.py").read_text(encoding="utf-8") == "user\n"


def test_subfolder_grant_reverts_inside_the_grant_only(repo: Path, tmp_path: Path, monkeypatch):
    _patch_state(monkeypatch, tmp_path)
    pkg = repo / "pkg"
    pkg.mkdir()
    (pkg / "a.py").write_text("a = 1\n", encoding="utf-8")
    other = repo / "other"
    other.mkdir()
    (other / "b.py").write_text("b = 1\n", encoding="utf-8")
    _git(repo, "add", "pkg/a.py", "other/b.py")
    _git(repo, "commit", "-m", "split")
    workspace = _workspace(pkg)
    assert snapshot_before_turn(workspace, "sess")["ok"]
    (pkg / "a.py").write_text("a = 2\n", encoding="utf-8")
    record_write("sess", "a.py", b"a = 1\n", b"a = 2\n")
    (other / "b.py").write_text("b = 2\n", encoding="utf-8")
    changes = list_changes(workspace, "sess")
    assert {item["path"] for item in changes["files"]} == {"a.py"}
    result = revert(workspace, "sess")
    assert result["restored"] == ["a.py"]
    assert (pkg / "a.py").read_text(encoding="utf-8") == "a = 1\n"
    assert not (pkg / "pkg").exists()
    assert not (pkg / "other").exists()
    assert (other / "b.py").read_text(encoding="utf-8") == "b = 2\n"


def test_revert_restores_symlink_and_executable_bit(repo: Path, tmp_path: Path, monkeypatch):
    _patch_state(monkeypatch, tmp_path)
    link = repo / "link"
    link.symlink_to("main.py")
    script = repo / "run.sh"
    script.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
    script.chmod(0o755)
    _git(repo, "add", "link", "run.sh")
    _git(repo, "commit", "-m", "modes")
    workspace = _workspace(repo)
    assert snapshot_before_turn(workspace, "sess")["ok"]
    link.unlink()
    link.write_text("not a link\n", encoding="utf-8")
    script.write_text("#!/bin/sh\necho no\n", encoding="utf-8")
    script.chmod(0o644)
    record_write("sess", "link", b"main.py", b"not a link\n")
    record_write("sess", "run.sh", b"#!/bin/sh\necho hi\n", b"#!/bin/sh\necho no\n")
    result = revert(workspace, "sess")
    assert set(result["restored"]) == {"link", "run.sh"}
    assert link.is_symlink()
    assert os.readlink(link) == "main.py"
    assert script.read_text(encoding="utf-8") == "#!/bin/sh\necho hi\n"
    assert script.stat().st_mode & stat.S_IXUSR


def test_shell_bookend_attributes_only_files_the_command_changed(repo: Path, tmp_path: Path, monkeypatch):
    _patch_state(monkeypatch, tmp_path)
    workspace = _workspace(repo)
    assert snapshot_before_turn(workspace, "sess")["ok"]
    (repo / "mine.py").write_text("user\n", encoding="utf-8")
    before = shell_baseline(repo)
    assert before
    (repo / "main.py").write_text("print(9)\n", encoding="utf-8")
    record_shell_changes("sess", repo, before)
    changes = list_changes(workspace, "sess")
    by_path = {item["path"]: item for item in changes["files"]}
    assert by_path["main.py"]["origin"] == "otto"
    assert by_path["mine.py"]["origin"] == "outside"
    result = revert(workspace, "sess")
    assert result["restored"] == ["main.py"]
    assert (repo / "main.py").read_text(encoding="utf-8") == "print(1)\n"
    assert (repo / "mine.py").read_text(encoding="utf-8") == "user\n"


def test_execute_records_files_the_command_changed(repo: Path, tmp_path: Path, monkeypatch):
    _patch_state(monkeypatch, tmp_path)
    from backend.session_manager import _make_backend

    files = tmp_path / "files"
    files.mkdir()
    workspace = _workspace(repo)
    assert snapshot_before_turn(workspace, "sess")["ok"]
    (repo / "mine.py").write_text("user\n", encoding="utf-8")

    def _fake_run(command: str, **kwargs):
        (repo / "main.py").write_text("print(9)\n", encoding="utf-8")
        return ShellResult(stdout="ok", returncode=0)

    monkeypatch.setattr("backend.exec_sandbox.run_shell", _fake_run)
    monkeypatch.setattr("backend.exec_sandbox.sandbox_settings", lambda: (True, False, [], []))
    monkeypatch.setattr("backend.exec_sandbox.passthrough_names", lambda: [])
    backend = _make_backend(files, project_root=repo, session_id="sess")
    result = backend.execute("echo hi")
    assert result.exit_code == 0
    changes = list_changes(workspace, "sess")
    by_path = {item["path"]: item for item in changes["files"]}
    assert by_path["main.py"]["origin"] == "otto"
    assert by_path["mine.py"]["origin"] == "outside"


def test_commit_worktree_does_not_run_hooks(repo: Path):
    from backend.project_tools import commit_worktree

    hooks = repo / ".git" / "hooks"
    hooks.mkdir(parents=True, exist_ok=True)
    (hooks / "pre-commit").write_text("#!/bin/sh\ntouch \"$GIT_DIR/../hook-ran\"\n", encoding="utf-8")
    (hooks / "pre-commit").chmod(0o755)
    (repo / "main.py").write_text("print(2)\n", encoding="utf-8")
    result = commit_worktree(repo, "why")
    assert result["ok"], result
    assert not (repo / "hook-ran").exists()
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    assert head == result["sha"]
    message = subprocess.check_output(["git", "-C", str(repo), "log", "-1", "--format=%s"], text=True).strip()
    assert message == "why"


def test_index_tree_stays_bounded_on_a_wide_tree(tmp_path: Path):
    import time
    from backend.workspace_git import _index_tree

    root = tmp_path / "wide"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "otto@example.com")
    _git(root, "config", "user.name", "Otto")
    count = 3000
    for index in range(count):
        folder = root / f"d{index // 100}"
        folder.mkdir(exist_ok=True)
        (folder / f"f{index}.py").write_text("x = 1\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "wide")
    started = time.monotonic()
    tree, err = _index_tree(root)
    elapsed = time.monotonic() - started
    assert tree, err
    assert elapsed < 30


def test_forget_session_drops_checkpoint_ref(repo: Path, tmp_path: Path, monkeypatch):
    _patch_state(monkeypatch, tmp_path)
    workspace = _workspace(repo)
    snap = snapshot_before_turn(workspace, "sess")
    ref = snap["checkpoint"]["ref"]
    subprocess.check_output(["git", "-C", str(repo), "rev-parse", ref], text=True)
    forget_session("sess", repo)
    listed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", ref],
        capture_output=True, text=True,
    )
    assert listed.returncode != 0
    assert not (tmp_path / "sess.workspace.json").exists()
