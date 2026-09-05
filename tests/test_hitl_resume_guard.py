"""HITL resume must not cancel an in-flight execute on duplicate approvals.

The 2s streaming poll used to wipe client-side ``resolved``, Always-allow
re-sent ``hitl_response``, and the WS handler cancelled the running resume
(SIGTERM -15).  These tests lock the decision table and the transcript stamp
that stop that loop.
"""

from __future__ import annotations

import json
import uuid

from backend.routes.sessions import _hitl_resume_action
from backend.session_manager import (
    load_messages,
    mark_last_interrupt_resolved,
)


def test_duplicate_response_while_resume_running_is_ignored():
    assert _hitl_resume_action(running=True, resume_inflight=True) == "ignore"


def test_first_response_while_stream_is_finishing_waits():
    assert _hitl_resume_action(running=True, resume_inflight=False) == "wait_then_start"


def test_response_with_nothing_in_flight_starts():
    assert _hitl_resume_action(running=False, resume_inflight=False) == "start"


def test_stale_inflight_flag_without_a_running_task_starts():
    # A leaked flag must not block the next real approval.
    assert _hitl_resume_action(running=False, resume_inflight=True) == "start"


def _write_messages(tmp_path, monkeypatch, sid: str, messages: list[dict]) -> None:
    monkeypatch.setattr("backend.session_manager._sessions_dir", lambda: tmp_path)
    p = tmp_path / f"{sid}.messages.json"
    p.write_text("".join(json.dumps(m) + "\n" for m in messages), encoding="utf-8")


def test_mark_last_interrupt_resolved_stamps_the_open_hitl(tmp_path, monkeypatch):
    sid = str(uuid.uuid4())
    _write_messages(tmp_path, monkeypatch, sid, [
        {"type": "user", "content": "review packages"},
        {
            "type": "hitl_request",
            "content": "Tool execution requires approval",
            "metadata": {"action_requests": [{"name": "execute", "args": {"command": "pip list"}}]},
        },
    ])
    decisions = [{"type": "approve"}]
    assert mark_last_interrupt_resolved(sid, decisions) is True
    msgs = load_messages(sid)
    meta = msgs[-1]["metadata"]
    assert meta["resolved"] is True
    assert meta["decisions"] == decisions


def test_mark_last_interrupt_resolved_is_idempotent(tmp_path, monkeypatch):
    sid = str(uuid.uuid4())
    _write_messages(tmp_path, monkeypatch, sid, [
        {
            "type": "hitl_request",
            "content": "Tool execution requires approval",
            "metadata": {"resolved": True, "decisions": [{"type": "approve"}]},
        },
    ])
    assert mark_last_interrupt_resolved(sid, [{"type": "reject"}]) is False
    msgs = load_messages(sid)
    assert msgs[-1]["metadata"]["decisions"] == [{"type": "approve"}]


def test_mark_last_interrupt_resolved_does_not_touch_older_interrupts(tmp_path, monkeypatch):
    sid = str(uuid.uuid4())
    _write_messages(tmp_path, monkeypatch, sid, [
        {
            "type": "hitl_request",
            "content": "first",
            "metadata": {"resolved": True, "decisions": [{"type": "approve"}]},
        },
        {"type": "tool_result", "content": "ok", "metadata": {"name": "execute"}},
        {
            "type": "ask_user",
            "content": "which one?",
            "metadata": {"question": "which one?"},
        },
    ])
    decisions = [{"type": "ask_user_answer", "answer": "the first"}]
    assert mark_last_interrupt_resolved(sid, decisions) is True
    msgs = load_messages(sid)
    assert msgs[0]["metadata"]["decisions"] == [{"type": "approve"}]
    assert msgs[-1]["metadata"]["resolved"] is True
    assert msgs[-1]["metadata"]["decisions"] == decisions
