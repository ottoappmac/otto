"""Thought/preamble text on a tool-calling turn must be persisted as an agent event.

The chat UI streams that text via agent_delta, then used to drop it when the
tool_call row arrived.  Keeping an ``agent`` event in the transcript is what
lets thoughts survive both live tool execution and a later Refresh.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from backend.session_manager import _ws_events_from_ai_message


def test_tool_calling_turn_emits_preamble_then_tool_call():
    msg = AIMessage(
        content="I'll list the packages, then summarise.",
        tool_calls=[{
            "name": "execute",
            "args": {"command": "pip list"},
            "id": "tc-1",
            "type": "tool_call",
        }],
    )
    events = _ws_events_from_ai_message(msg, stats={"generation_tokens": 12})

    assert [e["type"] for e in events] == ["agent", "tool_call"]
    assert events[0]["content"] == "I'll list the packages, then summarise."
    assert "stats" not in (events[0].get("metadata") or {})
    assert events[1]["content"] == "execute"
    assert events[1]["metadata"]["tool_call_id"] == "tc-1"
    assert events[1]["metadata"]["args"] == {"command": "pip list"}
    assert events[1]["metadata"]["stats"] == {"generation_tokens": 12}


def test_text_only_turn_attaches_stats_and_thought_to_agent():
    msg = AIMessage(
        content="Done.",
        additional_kwargs={"thought": "greeting acknowledged"},
    )
    events = _ws_events_from_ai_message(
        msg, stats={"generation_tps": 40.0}, memory_topics=["foo.md"],
    )

    assert len(events) == 1
    assert events[0]["type"] == "agent"
    assert events[0]["metadata"]["thought"] == "greeting acknowledged"
    assert events[0]["metadata"]["stats"] == {"generation_tps": 40.0}
    assert events[0]["metadata"]["memory_topics"] == ["foo.md"]


def test_tool_call_without_preamble_emits_only_tool_call():
    msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "read_file",
            "args": {"path": "/tmp/a"},
            "id": "tc-2",
            "type": "tool_call",
        }],
    )
    events = _ws_events_from_ai_message(msg)

    assert [e["type"] for e in events] == ["tool_call"]
    assert events[0]["content"] == "read_file"


def test_multiple_tool_calls_keep_stats_on_first_row_only():
    msg = AIMessage(
        content="Running both.",
        tool_calls=[
            {"name": "execute", "args": {"command": "ls"}, "id": "a", "type": "tool_call"},
            {"name": "execute", "args": {"command": "pwd"}, "id": "b", "type": "tool_call"},
        ],
    )
    events = _ws_events_from_ai_message(msg, stats={"generation_tokens": 3})

    assert [e["type"] for e in events] == ["agent", "tool_call", "tool_call"]
    assert events[1]["metadata"]["stats"] == {"generation_tokens": 3}
    assert "stats" not in events[2]["metadata"]
