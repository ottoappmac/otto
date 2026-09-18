"""HITL interrupt payloads must become JSON-safe hitl_request / ask_user events."""

from __future__ import annotations

from types import SimpleNamespace

from backend.session_manager import (
    SessionManager,
    _interrupts_from_state,
    _normalize_interrupt_value,
)


class _Dumpable:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


def test_normalize_unwraps_list_and_pydantic_dump():
    inner = _Dumpable({
        "action_requests": [{"name": "execute", "args": {"command": "echo hi"}}],
    })
    out = _normalize_interrupt_value([inner])
    assert out["action_requests"][0]["name"] == "execute"


def test_hitl_request_carries_action_requests():
    msg = SessionManager._interrupt_value_to_message({
        "action_requests": [{"name": "execute", "args": {"command": "pip list"}}],
        "review_configs": [{"action_name": "execute"}],
    })
    assert msg["type"] == "hitl_request"
    assert msg["content"] == "Tool execution requires approval"
    assert msg["metadata"]["action_requests"][0]["args"]["command"] == "pip list"


def test_ask_user_interrupt_keeps_question_and_options():
    msg = SessionManager._interrupt_value_to_message({
        "type": "ask_user",
        "question": "Which file?",
        "options": ["a.py", "b.py"],
    })
    assert msg["type"] == "ask_user"
    assert msg["content"] == "Which file?"
    assert msg["metadata"]["options"] == ["a.py", "b.py"]
    assert "question" not in msg["metadata"]


def test_interrupts_from_state_reads_nested_tasks():
    state = SimpleNamespace(tasks=[
        SimpleNamespace(interrupts=[]),
        SimpleNamespace(interrupts=[SimpleNamespace(value={"action_requests": []})]),
    ])
    pending = _interrupts_from_state(state)
    assert len(pending) == 1
    assert pending[0].value == {"action_requests": []}
