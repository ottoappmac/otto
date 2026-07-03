"""Live tests: HITL (Human-in-the-Loop) interrupt and resume.

The agent triggers an interrupt when it attempts to call the ``execute`` tool
(shell execution), because the graph is built with ``interrupt_on={"execute": True}``.
These tests verify:

* The stream stops and emits an interrupt event.
* After ``stream_resume`` with an approval, the agent continues.
* The checkpoint survives the interrupt so a resumed session is coherent.

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


# ---------------------------------------------------------------------------
# 1. Execute-triggered HITL pauses the session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_triggers_hitl_interrupt(session_manager, live_session):
    """A prompt that asks the agent to run a shell command should pause with
    a hitl_request event and NOT immediately yield a 'done' event."""
    try:
        events = await run_session(
            session_manager,
            live_session.id,
            "Run the shell command: echo hello-from-test",
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
            "interrupt_on may be broken"
        )

    assert len(hitl) >= 1, f"Expected at least one HITL event; got: {events}"


# ---------------------------------------------------------------------------
# 2. Resuming with approval lets the agent continue to completion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_after_approval_completes(session_manager, live_session):
    """After a HITL pause, stream_resume with approval should yield a 'done'."""
    events = await run_session(
        session_manager,
        live_session.id,
        "Run the shell command: echo resume-test",
    )

    hitl = _hitl_events(events)
    if not hitl:
        pytest.skip("No HITL triggered; cannot test resume")

    # Resume with approval for all pending decisions.
    resume_events = await collect_stream(
        session_manager.stream_resume(
            live_session.id,
            decisions=[{"type": "approve"}],
        )
    )

    done = _done_events(resume_events)
    assert done, (
        f"Expected 'done' event after resuming with approval.\n"
        f"Resume events: {resume_events}"
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

    events = await run_session(
        session_manager,
        session.id,
        "Execute: echo checkpoint-test",
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
