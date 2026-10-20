"""Git checkpoints do not touch the user's branch, and revert respects hashes."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from backend.schemas import SessionWorkspace
from backend.workspace_git import (
    capture_preimage,
    forget_session,
    list_changes,
    record_write,
    revert,
    revert_hunk,
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
