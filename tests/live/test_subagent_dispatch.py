"""Live tests: orchestrator subagent dispatch via the task() tool.

The orchestrator decides when to delegate work to a subagent by calling the
``task`` tool with a ``subagent_type`` and ``description``.  These tests verify
that the decision is made correctly and that the subagent spec (description,
LLM family) is well-formed.

Run with::

    pytest -m live tests/live/test_subagent_dispatch.py
"""

from __future__ import annotations

import pytest

from tests.live.conftest import run_session

pytestmark = pytest.mark.live


# ---------------------------------------------------------------------------
# 1. Orchestrator delegates a research task to the general-purpose subagent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_dispatches_for_multi_step_research(session_manager, live_session):
    """A prompt that clearly requires research and synthesis should trigger at
    least one ``task`` tool call at the orchestrator level."""
    events = await run_session(
        session_manager,
        live_session.id,
        (
            "Research the top 3 open-source LLM inference frameworks in 2025 and "
            "summarise their key differences. Save the summary to /output/llm-frameworks.md."
        ),
    )

    task_calls = [
        e for e in events
        if e.get("type") == "tool_call" and e.get("content") == "task"
    ]

    assert task_calls, (
        "Expected at least one 'task' tool call for a multi-step research prompt.\n"
        f"All tool calls: {[e['content'] for e in events if e.get('type') == 'tool_call']}"
    )


# ---------------------------------------------------------------------------
# 2. The task description is coherent and non-empty
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task_description_is_coherent(session_manager, live_session):
    """The description argument the orchestrator passes to task() must be a
    non-empty string plausibly related to the user's request."""
    events = await run_session(
        session_manager,
        live_session.id,
        "Search the web for the latest Python release notes and give me a bullet summary.",
    )

    task_calls = [
        e for e in events
        if e.get("type") == "tool_call" and e.get("content") == "task"
    ]

    if not task_calls:
        pytest.skip("No task() call emitted for this prompt/provider combination")

    for tc in task_calls:
        args = tc.get("metadata", {}).get("args", {})
        desc = args.get("description", "")
        assert isinstance(desc, str) and len(desc.strip()) > 10, (
            f"task() description is too short or missing: {desc!r}"
        )
        # The description should at least mention the domain.
        desc_lower = desc.lower()
        assert any(kw in desc_lower for kw in ("python", "search", "web", "release", "notes", "research")), (
            f"task() description seems unrelated to the prompt: {desc!r}"
        )


# ---------------------------------------------------------------------------
# 3. Subagent LLM family "inherit" uses the same provider as the orchestrator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_subagent_model_inherit_uses_parent_model():
    """_resolve_subagent_model with 'inherit' must return the parent model
    unchanged — no new LLM is created."""
    from backend.config import AppConfig
    from backend.schemas import AgentSpec
    from backend.session_manager import _resolve_subagent_model
    from deep_agent.model_factory import create_llm

    cfg = AppConfig.load()
    try:
        parent = create_llm(cfg.llm.provider)
    except Exception as exc:
        pytest.skip(f"Provider '{cfg.llm.provider}' not configured: {exc}")
    spec = AgentSpec(
        name="test-inherit-agent",
        description="Test agent for inherit family resolution",
        subagent_llm_family="inherit",
    )

    resolved = _resolve_subagent_model(spec, parent)
    assert resolved is parent, (
        "Expected _resolve_subagent_model('inherit') to return the parent model object itself"
    )


@pytest.mark.asyncio
async def test_resolve_subagent_model_none_family_uses_parent():
    """When subagent_llm_family is None/empty, the parent model is also used."""
    from backend.config import AppConfig
    from backend.schemas import AgentSpec
    from backend.session_manager import _resolve_subagent_model
    from deep_agent.model_factory import create_llm

    cfg = AppConfig.load()
    try:
        parent = create_llm(cfg.llm.provider)
    except Exception as exc:
        pytest.skip(f"Provider '{cfg.llm.provider}' not configured: {exc}")
    spec = AgentSpec(name="test-default-agent", description="Test default agent")  # no subagent_llm_family

    resolved = _resolve_subagent_model(spec, parent)
    assert resolved is parent
