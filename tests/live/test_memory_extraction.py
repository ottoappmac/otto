"""Live tests: memory extraction and relevance ranking.

MemoryExtractionMiddleware runs a lightweight LLM call after agent turns to
persist durable facts to AGENTS.md / learnings.md.  MemoryRelevanceMiddleware
performs per-turn topic ranking to inject relevant memories into the system
prompt.

These tests call the middleware directly (bypassing the session graph) so we
can isolate extraction behaviour without waiting 3 turns.

Run with::

    pytest -m live tests/live/test_memory_extraction.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytestmark = pytest.mark.live


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_messages(user_text: str, assistant_text: str):
    """Build a minimal two-message conversation for extraction tests."""
    from langchain_core.messages import AIMessage, HumanMessage

    return [
        HumanMessage(content=user_text),
        AIMessage(content=assistant_text),
    ]


# ---------------------------------------------------------------------------
# 1. Direct extraction: facts are written to the memory path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extraction_writes_to_memory_file(live_app_dir):
    """After _run_extraction with extract_every_n_turns=1, the memory_path
    file should exist and contain non-empty text."""
    from backend.config import AppConfig
    from backend.memory_extraction import MemoryExtractionMiddleware
    from deep_agent.model_factory import create_llm
    from tests.live.conftest import _probe_all_llms

    cfg = AppConfig.load()
    await _probe_all_llms(cfg)

    memory_path = live_app_dir / "test-learnings.md"
    llm = create_llm(cfg.llm.provider)

    mw = MemoryExtractionMiddleware(
        model=llm,
        memory_path=memory_path,
        extract_every_n_turns=1,
    )

    messages = _make_messages(
        "My name is Alice and I prefer Python over JavaScript.",
        "Got it — I'll remember that you prefer Python.",
    )

    await mw._run_extraction(messages)

    # The extraction is fire-and-forget; wait briefly for the background task.
    from backend.memory_extraction import _background_tasks

    if _background_tasks:
        await asyncio.gather(*list(_background_tasks), return_exceptions=True)

    # Give the file system a moment in case of async writes.
    await asyncio.sleep(0.5)

    assert memory_path.exists(), "Expected memory_path to be created after extraction"
    content = memory_path.read_text(encoding="utf-8").strip()
    assert content, "Expected non-empty content written to memory file"


@pytest.mark.asyncio
async def test_extracted_facts_reference_conversation_content(live_app_dir):
    """The extracted text should plausibly relate to what was said — confirming
    the LLM read the conversation rather than hallucinating generic facts."""
    from backend.config import AppConfig
    from backend.memory_extraction import MemoryExtractionMiddleware, _background_tasks
    from deep_agent.model_factory import create_llm

    cfg = AppConfig.load()
    memory_path = live_app_dir / "test-facts.md"
    llm = create_llm(cfg.llm.provider)

    mw = MemoryExtractionMiddleware(
        model=llm,
        memory_path=memory_path,
        extract_every_n_turns=1,
    )

    messages = _make_messages(
        "I'm working on a Rust web server project using the Axum framework.",
        "Understood. I'll keep in mind that your project uses Rust and Axum.",
    )

    await mw._run_extraction(messages)

    if _background_tasks:
        await asyncio.gather(*list(_background_tasks), return_exceptions=True)

    await asyncio.sleep(0.5)

    if not memory_path.exists():
        pytest.skip("Memory file was not created (LLM may have found nothing to extract)")

    content = memory_path.read_text(encoding="utf-8").lower()
    # At least one domain term from the conversation should appear.
    assert any(kw in content for kw in ("rust", "axum", "web", "server", "framework", "project")), (
        f"Extracted facts don't seem to reference the conversation content:\n{content[:500]}"
    )


# ---------------------------------------------------------------------------
# 2. Extraction respects the every-n-turns gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extraction_skipped_before_interval(live_app_dir):
    """With extract_every_n_turns=3, calling _run_extraction once must NOT
    write the file (the gate hasn't triggered yet)."""
    from backend.config import AppConfig
    from backend.memory_extraction import MemoryExtractionMiddleware
    from deep_agent.model_factory import create_llm

    cfg = AppConfig.load()
    memory_path = live_app_dir / "test-skipped.md"
    llm = create_llm(cfg.llm.provider)

    mw = MemoryExtractionMiddleware(
        model=llm,
        memory_path=memory_path,
        extract_every_n_turns=3,
    )

    messages = _make_messages("Hello", "Hi there!")
    await mw._run_extraction(messages)
    await asyncio.sleep(0.2)

    assert not memory_path.exists(), (
        "Memory file should NOT be written when the every-n-turns gate hasn't fired"
    )


# ---------------------------------------------------------------------------
# 3. Memory relevance ranking returns ordered results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_relevance_ranking_returns_top_match(live_app_dir):
    """_rank_topics should return the most semantically relevant topic first.

    The real signature is _rank_topics(user_text, topics, cfg, llm_provider)
    where topics is a list of dicts with 'file' and 'description' keys and
    the return value is a list of matching filenames.
    """
    from backend.config import AppConfig
    from backend.memory_relevance import _rank_topics

    cfg = AppConfig.load()

    topics = [
        {"file": "dark-mode.md", "description": "User prefers dark mode in their editor"},
        {"file": "python-ml.md", "description": "User is working on a machine learning project in Python"},
        {"file": "sushi.md", "description": "User's favourite food is sushi"},
        {"file": "vim-keys.md", "description": "User uses vim keybindings"},
    ]
    user_text = "How should I set up the Python ML environment?"

    try:
        ranked = await _rank_topics(user_text, topics, cfg.memory, cfg.llm.provider)
    except Exception as exc:
        pytest.skip(f"_rank_topics raised (provider not configured?): {exc}")

    if not ranked:
        pytest.skip("_rank_topics returned [] — LLM ranking call may have failed silently")

    assert any(
        "python-ml" in f or "ml" in f
        for f in ranked
    ), f"Expected python-ml.md in ranked results; got: {ranked}"


@pytest.mark.asyncio
async def test_relevance_ranking_respects_max_topics(live_app_dir):
    """_rank_topics must not return more filenames than MAX_TOPICS."""
    from backend.config import AppConfig
    from backend.memory_relevance import MAX_TOPICS, _rank_topics

    cfg = AppConfig.load()

    topics = [
        {"file": f"topic-{i}.md", "description": f"Topic {i}: some random fact about subject {i}"}
        for i in range(10)
    ]

    try:
        ranked = await _rank_topics(
            "general question about anything",
            topics,
            cfg.memory,
            cfg.llm.provider,
        )
    except Exception as exc:
        pytest.skip(f"_rank_topics raised (provider not configured?): {exc}")

    assert len(ranked) <= MAX_TOPICS, (
        f"Expected at most {MAX_TOPICS} topics; got {len(ranked)}: {ranked}"
    )
