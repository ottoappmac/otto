"""Chat auto-eval waits out the idle delay. One-shot runs do not."""

from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace

import pytest

from backend.eval_runner import cancel_evaluation, launch_evaluation
from backend.session_manager import SessionManager


def _session(trigger_source):
    return SimpleNamespace(
        id=str(uuid.uuid4()),
        trigger_source=trigger_source,
    )


def _manager() -> SessionManager:
    manager = SessionManager.__new__(SessionManager)
    manager._eval_idle = {}
    manager._active = {}
    return manager


async def _enable(monkeypatch, auto_evaluate: bool = True) -> None:
    class _Cfg:
        evaluation = SimpleNamespace(auto_evaluate=auto_evaluate)

    async def _aload():
        return _Cfg()

    monkeypatch.setattr("backend.session_manager.AppConfig.aload", _aload)


async def test_chat_waits_for_idle_before_evaluating(monkeypatch):
    launched: list[str] = []

    def _launch(session_id: str, manual: bool = False) -> None:
        launched.append(session_id)

    monkeypatch.setattr("backend.eval_runner.launch_evaluation", _launch)
    monkeypatch.setattr("backend.session_manager._EVAL_IDLE_SECS", 0.02)
    await _enable(monkeypatch)
    manager = _manager()
    session = _session(None)

    await manager._maybe_evaluate_run(session)
    assert launched == []
    await asyncio.sleep(0.08)
    assert launched == [session.id]


@pytest.mark.parametrize("source", ["schedule", "trigger", "spawn"])
async def test_oneshot_sources_evaluate_immediately(monkeypatch, source):
    launched: list[str] = []

    def _launch(session_id: str, manual: bool = False) -> None:
        launched.append(session_id)

    monkeypatch.setattr("backend.eval_runner.launch_evaluation", _launch)
    monkeypatch.setattr("backend.session_manager._EVAL_IDLE_SECS", 30)
    await _enable(monkeypatch)
    manager = _manager()
    session = _session(source)

    await manager._maybe_evaluate_run(session)
    assert launched == [session.id]
    assert manager._eval_idle == {}


async def _arm_and_start(monkeypatch, manager: SessionManager, session_id: str):
    async def _hang(_session_id: str, manual: bool = False) -> None:
        await asyncio.sleep(30)

    monkeypatch.setattr("backend.eval_runner.evaluate_session", _hang)
    timer = asyncio.create_task(asyncio.sleep(30))
    manager._eval_idle[session_id] = timer
    task = launch_evaluation(session_id)
    await asyncio.sleep(0)
    return timer, task


@pytest.mark.parametrize("method", ["stream_message", "stream_resume", "stream_edit"])
async def test_new_turn_cancels_timer_and_started_eval(monkeypatch, method):
    manager = _manager()
    session_id = str(uuid.uuid4())
    timer, task = await _arm_and_start(monkeypatch, manager, session_id)

    async def _missing(_session_id: str):
        return None

    monkeypatch.setattr(manager, "_ensure_session", _missing)
    stream = {
        "stream_message": manager.stream_message(session_id, "hi"),
        "stream_resume": manager.stream_resume(session_id, []),
        "stream_edit": manager.stream_edit(session_id, 0, "hi"),
    }[method]
    async for _item in stream:
        pass

    with pytest.raises(asyncio.CancelledError):
        await timer
    assert timer.cancelled()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()


async def test_delete_session_cancels_timer_and_started_eval(monkeypatch, tmp_path):
    manager = _manager()
    session_id = str(uuid.uuid4())
    timer, task = await _arm_and_start(monkeypatch, manager, session_id)
    monkeypatch.setattr("backend.session_manager._sessions_dir", lambda: tmp_path)

    async def _no_db():
        raise RuntimeError("no checkpoint db in this test")

    monkeypatch.setattr("backend.session_manager._connect_checkpoint_db", _no_db)
    monkeypatch.setattr("backend.session_transcript.delete_transcript", lambda _sid: False)
    monkeypatch.setattr("backend.workspace_git.forget_session", lambda *_a, **_k: None)

    await manager.delete_session(session_id)

    with pytest.raises(asyncio.CancelledError):
        await timer
    assert timer.cancelled()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()


async def test_cancelling_a_started_eval_drops_the_running_sidecar(monkeypatch, tmp_path):
    session_id = str(uuid.uuid4())
    monkeypatch.setattr("backend.session_manager._sessions_dir", lambda: tmp_path)

    async def _slow(_session_id: str):
        await asyncio.sleep(30)
        return []

    monkeypatch.setattr("backend.session_manager._load_messages_async", _slow)
    task = launch_evaluation(session_id)
    for _ in range(50):
        sidecar = tmp_path / f"{session_id}.eval.json"
        if sidecar.is_file():
            break
        await asyncio.sleep(0.01)
    assert json.loads(sidecar.read_text())["status"] == "running"

    cancel_evaluation(session_id)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not sidecar.exists()


def test_cancel_leaves_a_finished_eval(monkeypatch, tmp_path):
    session_id = str(uuid.uuid4())
    monkeypatch.setattr("backend.session_manager._sessions_dir", lambda: tmp_path)
    sidecar = tmp_path / f"{session_id}.eval.json"
    sidecar.write_text(json.dumps({"status": "done", "results": []}), encoding="utf-8")

    cancel_evaluation(session_id)

    assert json.loads(sidecar.read_text())["status"] == "done"
