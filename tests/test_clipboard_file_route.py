"""Regression tests for ``GET /api/clipboard/file``.

This endpoint exists to close a WebView gap: when Finder copies a *folder*,
the paste event the chat input receives has no usable ``file://`` entry and
``getAsFile()`` yields a 0-byte ``File`` named after the folder. Without a
real path the UI silently attached that empty stand-in instead of linking
the folder, so pasting a folder behaved nothing like dragging one in.

The frontend keys off ``path`` / ``items`` to decide whether to attach at
all, so the "nothing on the clipboard" case must come back as an empty
path rather than an error, and a folder path must arrive without its
AppleScript trailing slash so it matches what Tauri's drag-drop hands over
for the same folder.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.routes.clipboard as clipboard_routes
from backend.routes.clipboard import router as clipboard_router


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(clipboard_router)
    return TestClient(app)


def _fake_clipboard(monkeypatch, paths: list[str]) -> None:
    """Stub the osascript pasteboard read to return *paths*."""

    async def _fake() -> list[str]:
        return paths

    monkeypatch.setattr(clipboard_routes, "_clipboard_paths", _fake)


def test_reports_a_copied_folder_as_a_directory(client, monkeypatch, tmp_path):
    folder = tmp_path / "my project"
    folder.mkdir()
    # AppleScript hands back folder paths with a trailing slash.
    _fake_clipboard(monkeypatch, [f"{folder}/"])

    body = client.get("/api/clipboard/file").json()

    assert body["path"] == str(folder)
    assert body["is_dir"] is True
    assert body["items"] == [{"path": str(folder), "is_dir": True}]


def test_reports_a_copied_file_as_a_file(client, monkeypatch, tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hi")
    _fake_clipboard(monkeypatch, [str(f)])

    body = client.get("/api/clipboard/file").json()

    assert body["path"] == str(f)
    assert body["is_dir"] is False
    assert body["items"] == [{"path": str(f), "is_dir": False}]


def test_reports_multiple_copied_items(client, monkeypatch, tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    f = tmp_path / "notes.txt"
    f.write_text("hi")
    _fake_clipboard(monkeypatch, [str(folder), str(f)])

    body = client.get("/api/clipboard/file").json()

    assert body["path"] == str(folder)
    assert body["is_dir"] is True
    assert body["items"] == [
        {"path": str(folder), "is_dir": True},
        {"path": str(f), "is_dir": False},
    ]


def test_empty_file_is_not_mistaken_for_a_folder(client, monkeypatch, tmp_path):
    """The 0-byte blob heuristic that routes here also fires for real empty
    files, so those must still resolve to a plain file path to attach."""
    f = tmp_path / "empty.log"
    f.touch()
    _fake_clipboard(monkeypatch, [str(f)])

    body = client.get("/api/clipboard/file").json()

    assert body == {
        "path": str(f),
        "is_dir": False,
        "items": [{"path": str(f), "is_dir": False}],
    }


def test_no_file_reference_returns_empty_path(client, monkeypatch):
    """Plain text / an image / an empty clipboard is the common case, and the
    caller treats an empty path as "nothing to attach" — not an error."""
    _fake_clipboard(monkeypatch, [])

    res = client.get("/api/clipboard/file")

    assert res.status_code == 200
    assert res.json() == {"path": "", "is_dir": False, "items": []}


def test_missing_path_does_not_error(client, monkeypatch, tmp_path):
    """A stale clipboard reference (file moved/deleted since Cmd+C) still has
    to answer cleanly so the paste degrades instead of throwing."""
    _fake_clipboard(monkeypatch, [str(tmp_path / "gone")])

    body = client.get("/api/clipboard/file").json()

    assert body["is_dir"] is False


def test_non_darwin_platform_reads_nothing(monkeypatch):
    """osascript is macOS-only; elsewhere the read is skipped entirely rather
    than spawning a process that can't exist."""
    monkeypatch.setattr(clipboard_routes.sys, "platform", "linux")

    def _explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("must not spawn osascript off macOS")

    monkeypatch.setattr(clipboard_routes.asyncio, "create_subprocess_exec", _explode)

    assert asyncio.run(clipboard_routes._clipboard_paths()) == []
