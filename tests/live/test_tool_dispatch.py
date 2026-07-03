"""Live tests: agent tool selection and dispatch.

Tests that the LLM correctly decides which tool to call (or not to call),
passes valid arguments, and incorporates tool results into its reply.

Run with::

    pytest -m live tests/live/test_tool_dispatch.py
"""

from __future__ import annotations

import pytest

from tests.live.conftest import run_session

pytestmark = pytest.mark.live


# ---------------------------------------------------------------------------
# 1. The agent calls the right tool for an explicit request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_calls_privacy_status_on_request(session_manager, live_session):
    """Prompt explicitly asks about privacy status; the agent should invoke
    privacy_status rather than hallucinate the answer."""
    events = await run_session(
        session_manager,
        live_session.id,
        "What is my current privacy lock status?",
    )

    tool_calls = [e for e in events if e.get("type") == "tool_call"]
    tool_names = [e["content"] for e in tool_calls]

    assert "privacy_status" in tool_names, (
        f"Expected privacy_status tool call; got tool calls: {tool_names}\n"
        f"All events: {events}"
    )


@pytest.mark.asyncio
async def test_agent_calls_web_research_for_explicit_search(session_manager, live_session):
    """Prompt explicitly asks to search the web; agent should call web_research."""
    events = await run_session(
        session_manager,
        live_session.id,
        "Search the web: what is the current version of Python?",
    )

    tool_calls = [e for e in events if e.get("type") == "tool_call"]
    tool_names = [e["content"] for e in tool_calls]

    assert "web_research" in tool_names, (
        f"Expected web_research call; got: {tool_names}"
    )


# ---------------------------------------------------------------------------
# 2. The agent does NOT call tools for a simple factual Q&A
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_does_not_call_tools_for_simple_qa(session_manager, live_session):
    """A simple well-known fact should be answered directly without any tool."""
    events = await run_session(
        session_manager,
        live_session.id,
        "What is 2 + 2? Reply with just the number.",
    )

    tool_calls = [e for e in events if e.get("type") == "tool_call"]
    agent_events = [e for e in events if e.get("type") == "agent"]

    assert not tool_calls, (
        f"Expected no tool calls for trivial QA; got: {[e['content'] for e in tool_calls]}"
    )
    assert agent_events, "Expected at least one agent response event"
    combined = " ".join(e["content"] for e in agent_events)
    assert "4" in combined, f"Expected '4' in response; got: {combined}"


# ---------------------------------------------------------------------------
# 3. Tool args are non-empty and valid
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_web_research_tool_call_has_non_empty_query(session_manager, live_session):
    """The query argument passed to web_research must be a non-empty string."""
    events = await run_session(
        session_manager,
        live_session.id,
        "Search the web for the capital city of Australia.",
    )

    tool_calls = [
        e for e in events
        if e.get("type") == "tool_call" and e.get("content") == "web_research"
    ]

    assert tool_calls, "Expected at least one web_research tool call"
    args = tool_calls[0].get("metadata", {}).get("args", {})
    query = args.get("query", "")
    assert isinstance(query, str) and query.strip(), (
        f"Expected non-empty query string; got args: {args}"
    )


# ---------------------------------------------------------------------------
# 4. Tool result is incorporated into the final response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_result_is_reflected_in_agent_response(session_manager, live_session):
    """After privacy_status returns, the agent's reply must reference the
    information — confirming it read the tool result, not the empty string."""
    events = await run_session(
        session_manager,
        live_session.id,
        "Tell me whether the privacy lock is currently engaged or not.",
    )

    tool_calls = [e for e in events if e.get("type") == "tool_call"]
    tool_results = [e for e in events if e.get("type") == "tool_result"]
    agent_events = [e for e in events if e.get("type") == "agent"]

    assert tool_calls, "Expected at least one tool call"
    assert tool_results, "Expected at least one tool result"

    combined = " ".join(e["content"] for e in agent_events).lower()
    # The agent's answer should mention some form of the lock status.
    assert any(kw in combined for kw in ("engaged", "disengaged", "enabled", "disabled", "privacy", "lock")), (
        f"Agent response doesn't mention lock status: {combined[:500]}"
    )


# ---------------------------------------------------------------------------
# 5. Single-tool dispatch — no spurious repeated calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_privacy_status_called_at_most_once(session_manager, live_session):
    """privacy_status is idempotent; the agent should call it once, not loop."""
    events = await run_session(
        session_manager,
        live_session.id,
        "Is the privacy lock on?",
    )

    status_calls = [
        e for e in events
        if e.get("type") == "tool_call" and e.get("content") == "privacy_status"
    ]
    assert len(status_calls) <= 1, (
        f"privacy_status called {len(status_calls)} times; expected at most 1"
    )
