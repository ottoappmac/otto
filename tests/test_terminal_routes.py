import json
import sys
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import terminal, terminal_context
from backend.routes.terminals import router, ws_router

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="PTY needs a POSIX system")


@pytest.fixture
def client(tmp_path, monkeypatch):
    def fake_context(session_id):
        return terminal_context.TerminalContext(cwd=tmp_path, sandbox=False, network=True), None

    monkeypatch.setattr(terminal_context, "build_context", fake_context)
    app = FastAPI()
    app.include_router(router)
    app.include_router(ws_router)
    yield TestClient(app)
    terminal.close_session("sid")


def _read_until(ws, needle: bytes, timeout=8.0):
    seen = b""
    end = time.time() + timeout
    while time.time() < end:
        message = ws.receive()
        if message.get("bytes"):
            seen += message["bytes"]
            if needle in seen:
                return seen
        elif message.get("text") and json.loads(message["text"]).get("type") == "exit":
            break
    return seen


def test_open_list_and_close(client):
    info = client.post("/api/sessions/sid/terminals", json={"rows": 20, "cols": 80}).json()
    assert info["kind"] == "shell" and info["running"] is True
    assert info["ports"] == []
    listed = client.get("/api/sessions/sid/terminals").json()["terminals"]
    assert [t["id"] for t in listed] == [info["id"]]
    assert client.delete(f"/api/sessions/sid/terminals/{info['id']}").json() == {"status": "closed"}
    assert client.delete(f"/api/sessions/sid/terminals/{info['id']}").status_code == 404


def test_websocket_round_trip_and_replay(client):
    info = client.post("/api/sessions/sid/terminals", json={}).json()
    path = f"/ws/terminal/sid/{info['id']}"
    with client.websocket_connect(path) as ws:
        time.sleep(1.0)
        ws.send_text(json.dumps({"type": "input", "data": "echo otto-$((20+22))\n"}))
        assert b"otto-42" in _read_until(ws, b"otto-42\r\n")
    # A second connection gets the scrollback it missed.
    with client.websocket_connect(path) as ws:
        assert b"otto-42" in _read_until(ws, b"otto-42")


def test_open_refused_returns_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(terminal_context, "build_context", lambda sid: (None, "No project folder is mapped."))
    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).post("/api/sessions/none/terminals", json={})
    assert response.status_code == 409
    assert response.json()["error"] == "No project folder is mapped."


def test_process_exit_is_reported(client):
    from backend.terminal_context import open_terminal

    result = open_terminal("sid", kind="process", command="echo bye; exit 2")
    term = result["terminal"]
    with client.websocket_connect(f"/ws/terminal/sid/{term.id}") as ws:
        got_exit = None
        end = time.time() + 8
        while time.time() < end:
            message = ws.receive()
            if message.get("text"):
                data = json.loads(message["text"])
                if data.get("type") == "exit":
                    got_exit = data["code"]
                    break
        assert got_exit == 2


def test_dev_start_is_idempotent_and_respects_modes(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from backend import session_manager, workspace_grants

    grant = SimpleNamespace(dev_command="sleep 30", dev_autostart=True, id="g1")
    monkeypatch.setattr(session_manager, "_read_workspace_meta", lambda sid: SimpleNamespace(grant_id="g1", host_path=str(tmp_path), mode="write"))
    monkeypatch.setattr(workspace_grants, "get_grant", lambda gid: grant)
    monkeypatch.setattr(
        terminal_context, "build_context",
        lambda sid: (terminal_context.TerminalContext(cwd=tmp_path, sandbox=False, network=True), None),
    )
    import backend.routes.terminals as routes

    monkeypatch.setattr(routes, "build_context", terminal_context.build_context)
    monkeypatch.setattr(routes, "session_mode", lambda sid: "agent")
    app = FastAPI()
    app.include_router(router)
    c = TestClient(app)
    try:
        state = c.get("/api/sessions/dev1/dev").json()
        assert state["command"] == "sleep 30" and state["autostart"] is True and state["running_id"] is None
        first = c.post("/api/sessions/dev1/dev/start").json()
        assert first["started"] is True and first["running_id"]
        second = c.post("/api/sessions/dev1/dev/start").json()
        assert second["started"] is False and second["running_id"] == first["running_id"]
        monkeypatch.setattr(routes, "session_mode", lambda sid: "plan")
        terminal.close_session("dev1")
        refused = c.post("/api/sessions/dev1/dev/start")
        assert refused.status_code == 409 and "Plan" in refused.json()["error"]
    finally:
        terminal.close_session("dev1")
