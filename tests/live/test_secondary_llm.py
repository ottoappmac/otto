"""Live tests: secondary LLM calls (title generation, agent config generation,
connection probe).

These are lightweight one-shot invocations that don't go through the full
session pipeline.  They verify that the LLM can be reached and that the
helper functions produce well-formed outputs.

Run with::

    pytest -m live tests/live/test_secondary_llm.py
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.live


# ---------------------------------------------------------------------------
# 1. Title generation produces a short, non-empty string
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_title_generation_returns_short_string(session_manager, live_session):
    """_generate_title must set session.title to a non-empty string ≤ 80 chars."""
    from backend.session_manager import SessionManager

    mgr = session_manager

    # _generate_title is triggered automatically on the first message; we
    # also call it directly to test in isolation.
    await mgr._generate_title(
        live_session,
        live_session.id,
        "Summarise the history of machine learning in three sentences.",
    )

    title = live_session.title
    if title == "New Session":
        pytest.skip(
            "_generate_title returned the default — LLM call likely failed "
            "silently (check provider credentials)"
        )
    assert title, "Expected a non-empty title"
    assert len(title) <= 80, f"Title too long ({len(title)} chars): {title!r}"


@pytest.mark.asyncio
async def test_title_generation_is_related_to_prompt():
    """The generated title should contain at least one word from the prompt."""
    from backend.config import AppConfig
    from backend.session_manager import SessionManager

    mgr = SessionManager()
    cfg = AppConfig.load()
    try:
        session = await mgr.create_session(cfg)
    except Exception as exc:
        pytest.skip(f"Could not create session: {exc}")

    prompt = "Explain the Rust ownership model"
    await mgr._generate_title(session, session.id, prompt)

    if session.title == "New Session":
        pytest.skip(
            "_generate_title returned the default — LLM call likely failed "
            "silently (check provider credentials)"
        )

    title_lower = session.title.lower()
    prompt_words = {w.lower() for w in prompt.split() if len(w) > 3}

    overlap = prompt_words & set(title_lower.split())
    assert overlap, (
        f"Title {session.title!r} shares no words with prompt {prompt!r}"
    )

    await mgr.close_all()


# ---------------------------------------------------------------------------
# 2. Connection probe returns success for the configured provider
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connection_probe_succeeds_for_configured_provider():
    """create_llm + ainvoke('Reply with OK') must succeed for the active provider."""
    from backend.config import AppConfig
    from deep_agent.model_factory import create_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    cfg = AppConfig.load()
    try:
        llm = create_llm(cfg.llm.provider)
    except Exception as exc:
        pytest.skip(f"Could not create LLM for provider '{cfg.llm.provider}': {exc}")

    try:
        resp = await llm.ainvoke([
            SystemMessage(content="You are a test assistant. Follow instructions exactly."),
            HumanMessage(content="Reply with exactly the word: OK"),
        ])
    except Exception as exc:
        pytest.skip(f"LLM ainvoke failed (provider may not be ready or model not loaded): {exc}")

    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    assert content.strip(), "Expected a non-empty response from the LLM"


# ---------------------------------------------------------------------------
# 3. Agent config generation produces a valid AgentSpec
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_config_generation_returns_valid_spec():
    """The agent generation prompt + LLM must produce JSON that validates as
    an AgentSpec (Pydantic model)."""
    import json

    from backend.config import AppConfig
    from backend.prompts import AGENT_GENERATION_PROMPT
    from backend.schemas import AgentSpec
    from deep_agent.model_factory import create_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    cfg = AppConfig.load()
    try:
        llm = create_llm(cfg.llm.provider)
    except Exception as exc:
        pytest.skip(f"Could not create LLM: {exc}")

    user_description = "A research assistant that searches the web and summarises findings"

    try:
        resp = await llm.ainvoke([
            SystemMessage(content=AGENT_GENERATION_PROMPT),
            HumanMessage(content=user_description),
        ])
    except Exception as exc:
        pytest.skip(f"LLM ainvoke failed (provider may not be ready or model not loaded): {exc}")

    raw = resp.content if isinstance(resp.content, str) else str(resp.content)

    # Extract JSON block if wrapped in markdown fences.
    import re

    json_match = re.search(r"```(?:json)?\s*(\{[\s\S]+?\})\s*```", raw)
    json_str = json_match.group(1) if json_match else raw.strip()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        pytest.fail(f"Agent generation response is not valid JSON: {exc}\nRaw: {raw[:500]}")

    try:
        spec = AgentSpec.model_validate(data)
    except Exception as exc:
        pytest.fail(f"Generated JSON does not validate as AgentSpec: {exc}\nData: {data}")

    assert spec.name, "AgentSpec.name must be non-empty"


# ---------------------------------------------------------------------------
# 4. MLX model connection probe (only when MLX is the configured provider)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mlx_ainvoke_returns_text():
    """When the configured provider is 'mlx', a simple ainvoke must return
    an AIMessage with non-empty text content."""
    from backend.config import AppConfig

    cfg = AppConfig.load()
    if cfg.llm.provider != "mlx":
        pytest.skip("Provider is not 'mlx'; skipping MLX-specific probe")

    from deep_agent.model_factory import create_llm
    from langchain_core.messages import HumanMessage

    llm = create_llm("mlx")
    resp = await llm.ainvoke([HumanMessage(content="Say hello.")])

    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    assert content.strip(), "Expected non-empty response from MLX model"
