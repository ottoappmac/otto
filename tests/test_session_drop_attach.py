"""Drag-and-drop attach: files are copied, folders stay as links."""

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


def test_dropped_file_is_copied_into_uploads(tmp_path, monkeypatch):
    sid = str(uuid.uuid4())
    files_dir = tmp_path / "session" / "files"
    files_dir.mkdir(parents=True)
    original = tmp_path / "notes.txt"
    original.write_text("hello from disk")

    monkeypatch.setattr(sessions_routes, "_session_files_dir", lambda _sid: files_dir)

    async def _no_index(_path):
        return None

    monkeypatch.setattr(sessions_routes, "_maybe_index_file", _no_index)

    resp = _client().post(f"/api/sessions/{sid}/links", json={"source": str(original)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["path"] == "/uploads/notes.txt"
    assert body["is_dir"] is False

    copied = files_dir / "uploads" / "notes.txt"
    assert copied.is_file()
    assert not copied.is_symlink()
    assert copied.read_text() == "hello from disk"
    copied.write_text("edited copy")
    assert original.read_text() == "hello from disk"


def test_second_drop_of_the_same_name_does_not_overwrite(tmp_path, monkeypatch):
    sid = str(uuid.uuid4())
    files_dir = tmp_path / "session" / "files"
    uploads = files_dir / "uploads"
    uploads.mkdir(parents=True)
    (uploads / "notes.txt").write_text("already there")
    original = tmp_path / "notes.txt"
    original.write_text("new drop")

    monkeypatch.setattr(sessions_routes, "_session_files_dir", lambda _sid: files_dir)

    async def _no_index(_path):
        return None

    monkeypatch.setattr(sessions_routes, "_maybe_index_file", _no_index)

    resp = _client().post(f"/api/sessions/{sid}/links", json={"source": str(original)})
    assert resp.status_code == 200
    assert resp.json()["path"] == "/uploads/notes-1.txt"
    assert (uploads / "notes.txt").read_text() == "already there"
    assert (uploads / "notes-1.txt").read_text() == "new drop"


def test_dropped_folder_stays_a_link(tmp_path, monkeypatch):
    sid = str(uuid.uuid4())
    files_dir = tmp_path / "session" / "files"
    files_dir.mkdir(parents=True)
    folder = tmp_path / "project"
    folder.mkdir()
    (folder / "main.py").write_text("print(1)")

    monkeypatch.setattr(sessions_routes, "_session_files_dir", lambda _sid: files_dir)

    resp = _client().post(f"/api/sessions/{sid}/links", json={"source": str(folder)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["path"] == "/links/project"
    assert body["is_dir"] is True

    link = files_dir / "links" / "project"
    assert link.is_symlink()
    assert (link / "main.py").read_text() == "print(1)"
    assert not (files_dir / "uploads").exists()
