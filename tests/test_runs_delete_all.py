"""Regression tests for ``DELETE /api/runs`` (Runs page delete-all).

The unified runs list merges session meta files with leftover schedule and
trigger ``run.json`` records. Clearing only ``/api/sessions`` used to leave
those records on disk, so they reappeared on the next poll as orphan rows.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routes.runs import router as runs_router
from backend.routes.sessions import router as sessions_router
from backend.schemas import ScheduleRun, SessionInfo, TriggerRun


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(sessions_router)
    app.include_router(runs_router)
    return TestClient(app)


def _redirect_app_data(tmp_path, monkeypatch):
    sessions = tmp_path / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)

    def _sessions_dir():
        sessions.mkdir(parents=True, exist_ok=True)
        return sessions

    monkeypatch.setattr("backend.config.get_app_data_dir", lambda: tmp_path)
    monkeypatch.setattr("backend.session_manager._sessions_dir", _sessions_dir)
    monkeypatch.setattr("backend.routes.sessions._sessions_dir", _sessions_dir)
    monkeypatch.setattr("backend.routes.runs._sessions_dir", _sessions_dir)
    monkeypatch.setattr("backend.routes.runs.get_app_data_dir", lambda: tmp_path)
    monkeypatch.setattr("backend.scheduler.get_app_data_dir", lambda: tmp_path)
    monkeypatch.setattr("backend.trigger_manager.get_app_data_dir", lambda: tmp_path)
    monkeypatch.setattr("backend.session_transcript.get_app_data_dir", lambda: tmp_path)

    from backend.state import session_mgr
    session_mgr._active.clear()
    return sessions


def _write_session(sessions_dir, title: str = "Chat run") -> str:
    sid = str(uuid.uuid4())
    info = SessionInfo(id=sid, title=title, status="completed", message_count=2)
    (sessions_dir / f"{sid}.json").write_text(info.model_dump_json(), encoding="utf-8")
    (sessions_dir / f"{sid}.messages.json").write_text("", encoding="utf-8")
    return sid


def _write_schedule_run(tmp_path, session_id: str | None, schedule_id: str = "daily") -> str:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-sched"
    run = ScheduleRun(
        id=run_id,
        schedule_id=schedule_id,
        status="success",
        started_at=datetime.now(timezone.utc),
        session_id=session_id,
    )
    sched_dir = tmp_path / "schedules" / schedule_id
    run_dir = sched_dir / "runs" / run_id
    run_dir.mkdir(parents=True)
    (sched_dir / "schedule.json").write_text("{}", encoding="utf-8")
    (run_dir / "run.json").write_text(run.model_dump_json(), encoding="utf-8")
    return run_id


def _write_trigger_run(tmp_path, session_id: str | None, trigger_id: str = "inbox") -> str:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-trig"
    run = TriggerRun(
        id=run_id,
        trigger_id=trigger_id,
        status="success",
        started_at=datetime.now(timezone.utc),
        session_id=session_id,
    )
    trig_dir = tmp_path / "triggers" / trigger_id
    run_dir = trig_dir / "runs" / run_id
    run_dir.mkdir(parents=True)
    (trig_dir / "trigger.json").write_text("{}", encoding="utf-8")
    (run_dir / "run.json").write_text(run.model_dump_json(), encoding="utf-8")
    return run_id


def test_delete_all_runs_wipes_sessions_and_schedule_trigger_history(tmp_path, monkeypatch):
    sessions = _redirect_app_data(tmp_path, monkeypatch)
    sid = _write_session(sessions)
    _write_schedule_run(tmp_path, sid)
    _write_trigger_run(tmp_path, sid)

    client = _client()
    listed = client.get("/api/runs").json()
    assert listed["total"] >= 1

    resp = client.delete("/api/runs")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "deleted"
    assert body["sessions"] == 1
    assert body["schedule_runs"] == 1
    assert body["trigger_runs"] == 1

    leftover = list(sessions.glob("*.json"))
    assert leftover == []
    assert not (tmp_path / "schedules" / "daily" / "runs").exists()
    assert (tmp_path / "schedules" / "daily" / "schedule.json").exists()
    assert not (tmp_path / "triggers" / "inbox" / "runs").exists()
    assert (tmp_path / "triggers" / "inbox" / "trigger.json").exists()

    after = client.get("/api/runs").json()
    assert after["total"] == 0
    assert after["runs"] == []


def test_delete_all_runs_removes_orphaned_schedule_runs_without_sessions(tmp_path, monkeypatch):
    _redirect_app_data(tmp_path, monkeypatch)
    _write_schedule_run(tmp_path, session_id=None)

    client = _client()
    listed = client.get("/api/runs").json()
    assert listed["total"] == 1
    assert listed["runs"][0]["kind"] == "schedule_run"

    resp = client.delete("/api/runs")
    assert resp.status_code == 200, resp.text
    assert resp.json()["schedule_runs"] == 1

    after = client.get("/api/runs").json()
    assert after["total"] == 0


def test_delete_all_sessions_skips_non_uuid_json(tmp_path, monkeypatch):
    sessions = _redirect_app_data(tmp_path, monkeypatch)
    sid = _write_session(sessions)
    (sessions / "not-a-session.json").write_text("{}", encoding="utf-8")

    client = _client()
    resp = client.delete("/api/sessions")
    assert resp.status_code == 200, resp.text
    assert resp.json()["count"] == 1
    assert not (sessions / f"{sid}.json").exists()
    assert (sessions / "not-a-session.json").exists()
