"""Tests for the ``/api/node`` routes.

Spins up a FastAPI test client around just the node router (avoiding
full backend startup) and patches the provisioner so no install runs.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import tool_provisioner as tp
from backend.routes.node import router as node_router


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(node_router)
    return TestClient(app)


def test_status_reports_presence(monkeypatch):
    monkeypatch.setattr(
        "backend.routes.node.nodep.status_snapshot",
        lambda: {"present": True, "pinned_version": "v22.14.0"},
    )
    resp = _client().get("/api/node/status")
    assert resp.status_code == 200
    assert resp.json()["present"] is True


def test_install_returns_job_id(monkeypatch):
    job = tp.ToolJob(id="job123", kind="install-node", status="running")

    async def fake_install():
        return job

    monkeypatch.setattr("backend.routes.node.nodep.ainstall_node", fake_install)
    resp = _client().post("/api/node/install")
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == "job123"
    assert body["status"] == "running"


def test_get_job_returns_known_job(monkeypatch):
    job = tp.new_job("install-node")
    resp = _client().get(f"/api/node/jobs/{job.id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == job.id


def test_get_job_404_for_unknown_id():
    resp = _client().get("/api/node/jobs/does-not-exist")
    assert resp.status_code == 404
