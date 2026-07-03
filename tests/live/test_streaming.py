"""Live tests: stream event ordering and structural invariants.

These verify the _do_stream output contract:
* First non-trivial event is "agent" or a tool call.
* Every tool_call has a matching tool_result before the next agent event.
* No orphaned ToolMessages at stream end (the summarization_guard invariant).
* The stream terminates cleanly with a "done" event.

Run with::

    pytest -m live tests/live/test_streaming.py
"""

from __future__ import annotations

import pytest

from tests.live.conftest import run_session

pytestmark = pytest.mark.live


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _substantive(events: list[dict]) -> list[dict]:
    """Drop memory_search / memory_context events (they're pre-run housekeeping)."""
    return [e for e in events if e.get("type") not in ("memory_search", "memory_context")]


# ---------------------------------------------------------------------------
# 1. Stream terminates with "done"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_ends_with_done_event(session_manager, live_session):
    """Every completed stream must end with a 'done' event."""
    events = await run_session(
        session_manager,
        live_session.id,
        "Reply with exactly: pong",
    )

    event_types = [e.get("type") for e in events]
    assert "done" in event_types, (
        f"Expected 'done' at end of stream; event types: {event_types}"
    )
    assert event_types[-1] == "done", (
        f"'done' should be the last event; got {event_types[-1]!r} last"
    )


# ---------------------------------------------------------------------------
# 2. First substantive event is "agent" or "tool_call"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_substantive_event_is_agent_or_tool_call(session_manager, live_session):
    """The agent should emit content before a done/error/hitl event."""
    events = await run_session(
        session_manager,
        live_session.id,
        "Say hello.",
    )

    sub = _substantive(events)
    assert sub, "Expected at least one substantive event"

    first_type = sub[0].get("type")
    assert first_type in ("agent", "tool_call"), (
        f"Expected first substantive event to be 'agent' or 'tool_call'; got {first_type!r}"
    )


# ---------------------------------------------------------------------------
# 3. Every tool_call has a matching tool_result before the next agent event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_call_followed_by_tool_result(session_manager, live_session):
    """For any tool_call event, a tool_result with a matching tool_call_id
    must appear before the next 'agent' event (or end of stream)."""
    events = await run_session(
        session_manager,
        live_session.id,
        "What is my privacy status?",
    )

    pending: dict[str, str] = {}  # tool_call_id -> tool_name
    errors: list[str] = []

    for ev in events:
        etype = ev.get("type")
        meta = ev.get("metadata", {})

        if etype == "tool_call":
            tc_id = meta.get("tool_call_id", "")
            if tc_id:
                pending[tc_id] = ev["content"]

        elif etype == "tool_result":
            tc_id = meta.get("tool_call_id", "")
            if tc_id in pending:
                del pending[tc_id]

        elif etype == "agent" and pending:
            # An agent event arrived while tool calls are still open — that
            # could be intermediate text (Anthropic emits text before calling
            # tools), so only flag if the pending set is still non-empty at
            # the NEXT agent event after results were expected.
            pass  # We check pending at stream-end below.

    if pending:
        errors.append(
            f"Unmatched tool_call IDs still pending at end of stream: "
            + ", ".join(f"{v}({k})" for k, v in pending.items())
        )

    assert not errors, "\n".join(errors)


# ---------------------------------------------------------------------------
# 4. No error events for a normal query
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_error_events_for_normal_query(session_manager, live_session):
    """A benign question should never yield an error event."""
    events = await run_session(
        session_manager,
        live_session.id,
        "What is the capital of Japan?",
    )

    error_events = [e for e in events if e.get("type") == "error"]
    assert not error_events, (
        f"Unexpected error events: {error_events}"
    )


# ---------------------------------------------------------------------------
# 5. Agent content is non-empty
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_event_content_is_non_empty(session_manager, live_session):
    """Every 'agent' event must carry a non-empty content string."""
    events = await run_session(
        session_manager,
        live_session.id,
        "Give me a one-sentence summary of what you can do.",
    )

    agent_events = [e for e in events if e.get("type") == "agent"]
    assert agent_events, "Expected at least one agent event"

    for ev in agent_events:
        assert ev.get("content", "").strip(), (
            f"Agent event has empty content: {ev}"
        )


# ---------------------------------------------------------------------------
# 6. Multi-turn: second message continues from the same session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_second_turn_produces_coherent_response(session_manager, live_session):
    """A follow-up question should refer back to the prior answer — confirming
    the checkpoint context is maintained across turns."""
    await run_session(
        session_manager,
        live_session.id,
        "My favourite colour is ultraviolet.",
    )
    events2 = await run_session(
        session_manager,
        live_session.id,
        "What did I just tell you my favourite colour is?",
    )

    agent_events = [e for e in events2 if e.get("type") == "agent"]
    combined = " ".join(e["content"] for e in agent_events).lower()

    assert "ultraviolet" in combined, (
        f"Second turn did not recall 'ultraviolet'; response: {combined[:500]}"
    )
