"""Tests for mapped-folder workspace helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.schemas import SessionWorkspace
from backend.workspace import (
    ensure_workspace_link,
    list_tree,
    read_file,
    resolve_workspace_child,
    skip_dir_name,
    virtual_to_rel,
)


def _ws(tmp_path: Path, name: str = "proj") -> tuple[Path, SessionWorkspace]:
    host = tmp_path / name
    host.mkdir()
    (host / "README.md").write_text("# hello\n", encoding="utf-8")
    src = host / "src"
    src.mkdir()
    (src / "main.py").write_text("print(1)\n", encoding="utf-8")
    (host / "node_modules").mkdir()
    (host / "node_modules" / "pkg.js").write_text("x", encoding="utf-8")
    (host / ".git").mkdir()
    (host / ".gitignore").write_text("secret.env\n", encoding="utf-8")
    (host / "secret.env").write_text("KEY=1\n", encoding="utf-8")
    return host, SessionWorkspace(
        host_path=str(host),
        virtual_path=f"/links/{name}",
        name=name,
    )


def test_skip_dir_name():
    assert skip_dir_name("node_modules")
    assert skip_dir_name(".git")
    assert skip_dir_name(".venv")
    assert not skip_dir_name(".github")
    assert not skip_dir_name("src")


def test_list_tree_skips_heavy_and_gitignored(tmp_path: Path):
    host, ws = _ws(tmp_path)
    result = list_tree(ws, "")
    names = {e["name"] for e in result["entries"]}
    assert "README.md" in names
    assert "src" in names
    assert "node_modules" not in names
    assert ".git" not in names
    assert "secret.env" not in names
    src = next(e for e in result["entries"] if e["name"] == "src")
    assert src["is_dir"] is True

    nested = list_tree(ws, "src")
    assert [e["name"] for e in nested["entries"]] == ["main.py"]


def test_resolve_rejects_traversal(tmp_path: Path):
    _, ws = _ws(tmp_path)
    with pytest.raises(ValueError):
        resolve_workspace_child(ws, "../etc/passwd")
    with pytest.raises(ValueError):
        resolve_workspace_child(ws, "src/../../etc")


def test_read_file(tmp_path: Path):
    _, ws = _ws(tmp_path)
    result = read_file(ws, "src/main.py")
    assert "print(1)" in result["content"]
    assert result["truncated"] is False


def test_ensure_workspace_link_stable_name(tmp_path: Path):
    host, _ = _ws(tmp_path, "otto")
    files_dir = tmp_path / "session-files"
    files_dir.mkdir()
    ws = ensure_workspace_link(files_dir, host)
    assert ws.name == "otto"
    assert ws.virtual_path == "/links/otto"
    link = files_dir / "links" / "otto"
    assert link.is_symlink()
    assert link.resolve() == host.resolve()
    # Remapping the same folder is a no-op.
    ws2 = ensure_workspace_link(files_dir, host)
    assert ws2.virtual_path == ws.virtual_path


def test_virtual_to_rel():
    ws = SessionWorkspace(host_path="/tmp/p", virtual_path="/links/otto", name="otto")
    assert virtual_to_rel(ws, "/links/otto/src/main.py") == "src/main.py"
    assert virtual_to_rel(ws, "/links/otto") == ""
    assert virtual_to_rel(ws, "/output/report.md") is None
