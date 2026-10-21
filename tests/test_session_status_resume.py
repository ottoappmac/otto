"""A session that was waiting on the user must report ``running`` once answered.

``_check_interrupts_or_done`` persists ``awaiting_input`` when a run pauses for
a HITL / ask_user card. Resuming used to leave that status in place until the
whole resume ended, so the Runs page said "awaiting input" for a run that was
busy, and the chat's 2s poll treated the run as blocked (the Stop button turned
back into Send for a moment).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.routes import sessions as sessions_routes
from backend.session_manager import Session, SessionManager


def _session(status: str) -> Session:
    session = Session(session_id="s-1", agent_name="coding-agent", graph=SimpleNamespace())
    session.status = status
    return session


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch):
    """A manager whose heavy collaborators are replaced with recorders."""
    mgr = SessionManager()
    seen: dict[str, object] = {"saved": 0}

    async def _no_privacy_error(_session):
        return None

    async def _no_stream(*_args, **_kwargs):
        return
        yield  # pragma: no cover - makes this an async generator

    async def _record_status(_session, _session_id):
        seen["status_during_resume"] = _session.status
        yield {"type": "done", "content": ""}

    async def _noop(_session):
        return None

    class _Graph:
        async def aget_state(self, _config):
            return SimpleNamespace(values={"messages": []})

        def astream(self, *_args, **_kwargs):
            return object()

    session = _session("awaiting_input")
    session.graph = _Graph()

    async def _save():
        seen["saved"] = int(seen["saved"]) + 1

    session.save_meta_async = _save  # type: ignore[method-assign]

    async def _ensure(_session_id):
        return session

    monkeypatch.setattr(mgr, "_ensure_session", _ensure)
    monkeypatch.setattr(mgr, "_check_privacy_lock", _no_privacy_error)
    monkeypatch.setattr(mgr, "_do_stream", _no_stream)
    monkeypatch.setattr(mgr, "_check_interrupts_or_done", _record_status)
    monkeypatch.setattr(mgr, "_maybe_consolidate_memory", _noop)
    monkeypatch.setattr(mgr, "_maybe_index_transcript", _noop)
    monkeypatch.setattr(mgr, "_maybe_ambient_sweep", _noop)
    return mgr, session, seen


async def test_resume_marks_an_awaiting_session_running(manager):
    mgr, session, seen = manager

    events = [e async for e in mgr.stream_resume("s-1", [{"type": "approve"}])]

    assert seen["status_during_resume"] == "running"
    assert seen["saved"] == 1, "the new status must be persisted, not just held in memory"
    assert events[-1]["type"] == "done"
    assert session.status == "running"  # the finalizer (stubbed here) owns the end state


async def test_resume_leaves_a_running_session_alone(manager):
    mgr, session, seen = manager
    session.status = "running"

    [e async for e in mgr.stream_resume("s-1", [{"type": "approve"}])]

    assert seen["saved"] == 0


async def _status_for(monkeypatch, *, session_status: str, task_running: bool):
    session = _session(session_status)
    monkeypatch.setattr(sessions_routes.session_mgr, "get_session", lambda _sid: session)
    task = SimpleNamespace(done=lambda: not task_running)
    monkeypatch.setattr(sessions_routes, "running_tasks", {"s-1": task})
    return await sessions_routes.api_session_status("s-1")


async def test_status_is_awaiting_only_when_nothing_is_running(monkeypatch):
    paused = await _status_for(monkeypatch, session_status="awaiting_input", task_running=False)
    assert paused["awaiting_input"] is True
    assert paused["running"] is False


async def test_status_is_not_awaiting_while_a_resume_is_running(monkeypatch):
    # Stale "awaiting_input" with a live task: the run is executing.
    resumed = await _status_for(monkeypatch, session_status="awaiting_input", task_running=True)
    assert resumed["running"] is True
    assert resumed["awaiting_input"] is False
