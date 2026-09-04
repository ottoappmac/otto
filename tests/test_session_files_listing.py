"""Regression tests for ``GET /api/sessions/{id}/files``.

The endpoint walks the session's files dir (including anything symlinked
into ``files/links/`` by drag-drop/paste-a-folder) to list agent-visible
files. It's polled every few seconds per open session, so it must never
turn into an unbounded filesystem walk — that starves the shared
``asyncio.to_thread`` pool and makes the whole app feel hung the moment a
real project folder (with ``node_modules``/``.venv``/``.git``/etc.) gets
attached.
"""

from __future__ import annotations

import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.routes.sessions as sessions_routes
from backend.routes.sessions import router as sessions_router


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(sessions_router)
    return TestClient(app)


def _session_id() -> str:
    return str(uuid.uuid4())


def test_lists_plain_files(tmp_path, monkeypatch):
    sid = _session_id()
    files_dir = tmp_path / sid / "files"
    files_dir.mkdir(parents=True)
    (files_dir / "notes.txt").write_text("hi")
    sub = files_dir / "sub"
    sub.mkdir()
    (sub / "a.txt").write_text("a")

    monkeypatch.setattr(sessions_routes, "_session_files_dir", lambda s: files_dir)

    resp = _client().get(f"/api/sessions/{sid}/files")
    assert resp.status_code == 200
    paths = sorted(f["path"] for f in resp.json())
    assert paths == ["notes.txt", "sub/a.txt"]


def test_prunes_huge_project_dirs_in_a_linked_folder(tmp_path, monkeypatch):
    """A folder dropped/pasted into chat is symlinked under ``links/<name>``.
    If that folder is a real project, walking straight into its
    ``node_modules``/``.venv``/``.git`` must be skipped rather than
    recursed into.
    """
    sid = _session_id()
    files_dir = tmp_path / sid / "files"
    links_dir = files_dir / "links"
    links_dir.mkdir(parents=True)

    project = tmp_path / "project"
    (project / "node_modules" / "some-pkg").mkdir(parents=True)
    (project / "node_modules" / "some-pkg" / "index.js").write_text("x")
    (project / ".venv" / "lib").mkdir(parents=True)
    (project / ".venv" / "lib" / "site.py").write_text("x")
    (project / ".git" / "objects").mkdir(parents=True)
    (project / ".git" / "objects" / "pack").write_text("x")
    (project / "src").mkdir(parents=True)
    (project / "src" / "main.py").write_text("print(1)")

    (links_dir / "project").symlink_to(project, target_is_directory=True)

    monkeypatch.setattr(sessions_routes, "_session_files_dir", lambda s: files_dir)

    resp = _client().get(f"/api/sessions/{sid}/files")
    assert resp.status_code == 200
    paths = {f["path"] for f in resp.json()}
    assert "links/project/src/main.py" in paths
    assert not any("node_modules" in p for p in paths)
    assert not any(".venv" in p for p in paths)
    assert not any(".git" in p for p in paths)


def test_caps_total_entries_for_pathological_trees(tmp_path, monkeypatch):
    sid = _session_id()
    files_dir = tmp_path / sid / "files"
    files_dir.mkdir(parents=True)

    monkeypatch.setattr(sessions_routes, "_session_files_dir", lambda s: files_dir)
    monkeypatch.setattr(sessions_routes, "_SESSION_FILES_MAX_ENTRIES", 25)
    for i in range(100):
        (files_dir / f"f{i}.txt").write_text("x")

    resp = _client().get(f"/api/sessions/{sid}/files")
    assert resp.status_code == 200
    assert len(resp.json()) == 25


def test_missing_files_dir_returns_empty_list(tmp_path, monkeypatch):
    sid = _session_id()
    files_dir = tmp_path / sid / "files"  # never created

    monkeypatch.setattr(sessions_routes, "_session_files_dir", lambda s: files_dir)

    resp = _client().get(f"/api/sessions/{sid}/files")
    assert resp.status_code == 200
    assert resp.json() == []
