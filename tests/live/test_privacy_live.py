"""Live tests: privacy lock enforcement via the agent and the factory.

Tests that:
* The agent can engage the privacy lock via the engage_privacy_lock tool.
* After engagement, create_llm() refuses cloud providers.
* privacy_status tool accurately reflects the current lock state.

Run with::

    pytest -m live tests/live/test_privacy_live.py
"""

from __future__ import annotations

import pytest

from tests.live.conftest import run_session

pytestmark = pytest.mark.live


# ---------------------------------------------------------------------------
# 1. Agent invokes engage_privacy_lock when asked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_engages_privacy_lock_via_tool(session_manager, live_session, live_app_dir):
    """Prompt: 'engage the privacy lock'. The agent must call engage_privacy_lock."""
    events = await run_session(
        session_manager,
        live_session.id,
        "Please engage the privacy lock right now.",
    )

    tool_names = [e["content"] for e in events if e.get("type") == "tool_call"]

    assert "engage_privacy_lock" in tool_names, (
        f"Expected engage_privacy_lock tool call; got: {tool_names}"
    )

    # Disengagement cleanup — prevent bleed into other tests.
    from backend.config import AppConfig
    from backend import privacy_lock as pl

    cfg = AppConfig.load()
    pl.disengage(cfg)
    cfg.save()
    cfg.apply_to_environ()


# ---------------------------------------------------------------------------
# 2. create_llm refuses cloud providers when the lock is programmatically engaged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_llm_blocked_after_lock_engaged(live_app_dir):
    """After engaging the lock, create_llm('anthropic') must raise PrivacyLockActive."""
    from backend import privacy_lock as pl
    from backend.config import AppConfig
    from deep_agent.model_factory import create_llm

    cfg = AppConfig.load()
    pl.engage(cfg)
    cfg.save()
    cfg.apply_to_environ()

    try:
        with pytest.raises(pl.PrivacyLockActive):
            create_llm("anthropic")
    finally:
        pl.disengage(cfg)
        cfg.save()
        cfg.apply_to_environ()


@pytest.mark.asyncio
async def test_create_llm_blocked_for_openai_when_locked(live_app_dir):
    """Same guard applies to OpenAI when the lock is engaged."""
    from backend import privacy_lock as pl
    from backend.config import AppConfig
    from deep_agent.model_factory import create_llm

    cfg = AppConfig.load()
    pl.engage(cfg)
    cfg.save()
    cfg.apply_to_environ()

    try:
        with pytest.raises(pl.PrivacyLockActive):
            create_llm("openai")
    finally:
        pl.disengage(cfg)
        cfg.save()
        cfg.apply_to_environ()


# ---------------------------------------------------------------------------
# 3. Local providers are still allowed when the lock is engaged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_providers_allowed_when_locked(live_app_dir):
    """mlx, omlx, exo must not be blocked by the privacy lock (they're local)."""
    from backend import privacy_lock as pl
    from backend.config import AppConfig

    cfg = AppConfig.load()
    pl.engage(cfg)
    cfg.save()
    cfg.apply_to_environ()

    try:
        for provider in ("mlx", "omlx", "exo"):
            pl.enforce_provider_allowed(provider, cfg)  # must not raise
    finally:
        pl.disengage(cfg)
        cfg.save()
        cfg.apply_to_environ()


# ---------------------------------------------------------------------------
# 4. privacy_status tool reflects actual lock state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_privacy_status_tool_returns_current_state(session_manager, live_session):
    """The agent's privacy_status tool result must match the actual config state."""
    from backend.config import AppConfig
    from backend import privacy_lock as pl

    cfg = AppConfig.load()
    expected_engaged = pl.is_engaged(cfg)

    events = await run_session(
        session_manager,
        live_session.id,
        "What is my current privacy lock status?",
    )

    tool_results = [
        e for e in events
        if e.get("type") == "tool_result"
        and e.get("metadata", {}).get("name") == "privacy_status"
    ]

    assert tool_results, "Expected a tool_result event for privacy_status"

    result_text = tool_results[0]["content"].lower()
    if expected_engaged:
        assert "engaged" in result_text, (
            f"Lock is engaged but tool reported: {result_text}"
        )
    else:
        assert "disengaged" in result_text or "not engaged" in result_text or "disengaged" in result_text, (
            f"Lock is not engaged but tool reported: {result_text}"
        )


# ---------------------------------------------------------------------------
# 5. Per-turn privacy check blocks cloud provider mid-session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_turn_lock_blocks_cloud_session(session_manager, live_app_dir):
    """If the session uses a cloud provider and the lock is engaged mid-run,
    the next stream_message call must return an error event."""
    from backend.config import AppConfig
    from backend import privacy_lock as pl
    from backend.session_manager import SessionManager

    cfg = AppConfig.load()
    # Only makes sense to test if a cloud provider is currently configured.
    if cfg.llm.provider not in ("anthropic", "openai", "cohere", "bedrock"):
        pytest.skip(f"Provider '{cfg.llm.provider}' is not a cloud provider — test not applicable")

    mgr = SessionManager()
    try:
        session = await mgr.create_session(cfg)

        # Engage the lock AFTER the session is created (simulating mid-session lock).
        pl.engage(cfg)
        cfg.save()
        cfg.apply_to_environ()

        events = []
        try:
            async for ev in mgr.stream_message(session.id, "hello"):
                events.append(ev)
        finally:
            pl.disengage(cfg)
            cfg.save()
            cfg.apply_to_environ()

        error_events = [e for e in events if e.get("type") == "error"]
        assert error_events, (
            "Expected an error event when a cloud-provider session is used while locked"
        )
        assert "privacy" in error_events[0]["content"].lower(), (
            f"Error should mention privacy: {error_events[0]}"
        )
    finally:
        await mgr.close_all()
