"""Live tests: HITL (Human-in-the-Loop) approval and resume.

``execute`` is gated by the action-policy middleware, not by
``interrupt_on``. The stream yields a ``hitl_request`` and keeps running
until that request's future is resolved. ``stream_resume`` is only for
graph interrupts such as ``ask_user``.

These tests verify:

* The stream emits an approval card before the command runs.
* Approving that card lets the same stream finish.
* The checkpoint still has the conversation after the card.

Run with::

    pytest -m live tests/live/test_hitl.py
"""

from __future__ import annotations

import pytest

from tests.live.conftest import _probe_all_llms, collect_stream, run_session

pytestmark = pytest.mark.live


def _hitl_events(events: list[dict]) -> list[dict]:
    return [e for e in events if e.get("type") in ("hitl_request", "ask_user")]


def _done_events(events: list[dict]) -> list[dict]:
    return [e for e in events if e.get("type") == "done"]


async def _drive(session_manager, session_id: str, prompt: str, decision: dict) -> list[dict]:
    """Collect the stream, resolving the first action-policy card."""
    from backend.action_policy.runtime import resolve

    events: list[dict] = []
    answered = False
    async for event in session_manager.stream_message(session_id, prompt):
        events.append(event)
        if answered or event.get("type") != "hitl_request":
            continue
        request_id = (event.get("metadata") or {}).get("request_id")
        if request_id:
            resolve(session_id, str(request_id), [decision])
            answered = True
    return events


# ---------------------------------------------------------------------------
# 1. Execute-triggered HITL pauses the session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_triggers_hitl_interrupt(session_manager, live_session):
    """A prompt that asks the agent to run a shell command should pause with
    a hitl_request event and NOT immediately yield a 'done' event."""
    try:
        events = await _drive(
            session_manager,
            live_session.id,
            "Run the shell command: echo hello-from-test",
            {"type": "reject", "message": "live test"},
        )
    except Exception as exc:
        msg = str(exc)
        if "authentication method" in msg or "api_key" in msg or "auth_token" in msg:
            pytest.skip(f"LLM auth error during stream (provider not configured in test env): {exc}")
        raise

    hitl = _hitl_events(events)

    if not hitl:
        # Some providers/prompts may answer without calling execute at all.
        # Skip rather than fail so this test only fails when execute IS called
        # but the interrupt was somehow bypassed.
        execute_calls = [
            e for e in events
            if e.get("type") == "tool_call" and e.get("content") == "execute"
        ]
        if not execute_calls:
            pytest.skip("Agent did not call execute for this prompt; HITL cannot be tested")
        pytest.fail(
            "Agent called execute but no hitl_request was emitted — "
            "the action policy did not ask"
        )

    assert len(hitl) >= 1, f"Expected at least one HITL event; got: {events}"


# ---------------------------------------------------------------------------
# 2. Resuming with approval lets the agent continue to completion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_after_approval_completes(session_manager, live_session):
    """After a HITL pause, stream_resume with approval should yield a 'done'."""
    events = await _drive(
        session_manager,
        live_session.id,
        "Run the shell command: echo resume-test",
        {"type": "approve"},
    )

    hitl = _hitl_events(events)
    if not hitl:
        pytest.skip("No HITL triggered; cannot test resume")

    done = _done_events(events)
    assert done, (
        f"Expected 'done' on the same stream after approving the card.\n"
        f"Events: {events}"
    )


# ---------------------------------------------------------------------------
# 3. Checkpoint is preserved across the interrupt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_checkpoint_preserved_across_interrupt(session_manager, live_app_dir):
    """After a HITL pause the graph checkpoint must survive so that the session
    can be resumed with the correct message history."""
    from backend.config import AppConfig

    cfg = AppConfig.load()
    await _probe_all_llms(cfg)
    try:
        session = await session_manager.create_session(cfg)
    except Exception as exc:
        pytest.skip(f"Could not create session: {exc}")

    events = await _drive(
        session_manager,
        session.id,
        "Execute: echo checkpoint-test",
        {"type": "reject", "message": "live test"},
    )

    hitl = _hitl_events(events)
    if not hitl:
        pytest.skip("No HITL triggered; cannot test checkpoint")

    # Inspect the LangGraph checkpoint state — it must have messages.
    run_config = {"configurable": {"thread_id": session.id}}
    state = await session.graph.aget_state(run_config)
    messages = state.values.get("messages", [])

    assert len(messages) >= 1, (
        "Expected at least the user message to be in the checkpoint after a HITL pause"
    )

    await session_manager.close_session(session.id)


# ---------------------------------------------------------------------------
# 4. ask_user tool emits an ask_user event (not a hitl_request)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_user_emits_ask_user_event(session_manager, live_session):
    """Prompt that is intentionally vague should trigger ask_user rather than
    hallucinating an action."""
    events = await run_session(
        session_manager,
        live_session.id,
        "Do the thing.",
    )

    ask_events = [e for e in events if e.get("type") == "ask_user"]

    if not ask_events:
        # The model may choose to ask for clarification differently, or just
        # give a direct reply — this is a soft assertion.
        pytest.skip(
            "Agent did not emit ask_user for this ambiguous prompt. "
            "This is provider-dependent and not a hard failure."
        )

    assert ask_events[0].get("content"), "ask_user event must carry a question"
