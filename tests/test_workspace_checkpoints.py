"""Git checkpoints do not touch the user's branch, and revert respects hashes."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from backend.schemas import SessionWorkspace
from backend.workspace_git import revert, snapshot_before_turn


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


def test_revert_refuses_when_file_changed_after_agent(repo: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "backend.workspace_git._state_path",
        lambda session_id: tmp_path / f"{session_id}.workspace.json",
    )
    workspace = SessionWorkspace(host_path=str(repo), virtual_path="/links/proj", name="proj", grant_id="g", mode="write")
    snap = snapshot_before_turn(workspace, "sess")
    assert snap["ok"]
    from backend.workspace_git import record_write
    (repo / "main.py").write_text("print(2)\n", encoding="utf-8")
    record_write("sess", "main.py", b"print(1)\n", b"print(2)\n")
    (repo / "main.py").write_text("print(3)\n", encoding="utf-8")
    result = revert(workspace, "sess", paths=["main.py"])
    assert result["conflicts"]
    assert (repo / "main.py").read_text(encoding="utf-8") == "print(3)\n"


def test_revert_restores_agent_edit(repo: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "backend.workspace_git._state_path",
        lambda session_id: tmp_path / f"{session_id}.workspace.json",
    )
    workspace = SessionWorkspace(host_path=str(repo), virtual_path="/links/proj", name="proj", grant_id="g", mode="write")
    snapshot_before_turn(workspace, "sess")
    from backend.workspace_git import record_write
    (repo / "main.py").write_text("print(2)\n", encoding="utf-8")
    record_write("sess", "main.py", b"print(1)\n", b"print(2)\n")
    result = revert(workspace, "sess", paths=["main.py"])
    assert result["restored"] == ["main.py"]
    assert (repo / "main.py").read_text(encoding="utf-8") == "print(1)\n"
