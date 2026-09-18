"""GET /api/sessions/{id}/status must report awaiting_input for HITL pauses.

The chat poll used ``running`` (whether the stream task is alive) as the only
signal.  Execute approval pauses the graph and pops that task, so the UI
treated the turn as finished and stopped polling before the approval card
arrived.  ``awaiting_input`` keeps the poll alive until the card is in the
transcript.
"""

from __future__ import annotations

import json
import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.routes.sessions as sessions_routes
from backend.routes.sessions import router as sessions_router


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(sessions_router)
    return TestClient(app)


class _FakeSession:
    def __init__(self, status: str):
        self.status = status


class _FakeMgr:
    def __init__(self, session=None):
        self._session = session

    def get_session(self, _sid):
        return self._session


def test_status_reports_awaiting_input_from_live_session(tmp_path, monkeypatch):
    sid = str(uuid.uuid4())
    monkeypatch.setattr(sessions_routes, "session_mgr", _FakeMgr(_FakeSession("awaiting_input")))
    monkeypatch.setattr(sessions_routes, "running_tasks", {})
    monkeypatch.setattr(sessions_routes, "_sessions_dir", lambda: tmp_path)

    resp = _client().get(f"/api/sessions/{sid}/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["running"] is False
    assert body["awaiting_input"] is True
    assert body["active"] is True


def test_status_reads_awaiting_input_from_saved_meta(tmp_path, monkeypatch):
    sid = str(uuid.uuid4())
    (tmp_path / f"{sid}.json").write_text(
        json.dumps({"status": "awaiting_input"}), encoding="utf-8",
    )
    monkeypatch.setattr(sessions_routes, "session_mgr", _FakeMgr(None))
    monkeypatch.setattr(sessions_routes, "running_tasks", {})
    monkeypatch.setattr(sessions_routes, "_sessions_dir", lambda: tmp_path)

    resp = _client().get(f"/api/sessions/{sid}/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["active"] is True
    assert body["running"] is False
    assert body["awaiting_input"] is True


def test_status_idle_completed_session_is_not_awaiting_input(tmp_path, monkeypatch):
    sid = str(uuid.uuid4())
    monkeypatch.setattr(sessions_routes, "session_mgr", _FakeMgr(_FakeSession("completed")))
    monkeypatch.setattr(sessions_routes, "running_tasks", {})
    monkeypatch.setattr(sessions_routes, "_sessions_dir", lambda: tmp_path)

    resp = _client().get(f"/api/sessions/{sid}/status")
    assert resp.json()["awaiting_input"] is False
    assert resp.json()["running"] is False
