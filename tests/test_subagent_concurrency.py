"""Tests for the subagent (``task``) concurrency cap."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from middleware.subagent_concurrency import (
    SubagentConcurrencyMiddleware,
    maybe_for_environment,
)
from utilities.environment import Environment


def _req(name: str):
    return SimpleNamespace(tool_call={"name": name, "args": {}, "id": name}, tool=None)


# ── async ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_caps_concurrent_task_calls():
    mw = SubagentConcurrencyMiddleware(2)
    running = peak = 0

    async def handler(_req):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.02)
        running -= 1
        return ToolMessage(content="ok", tool_call_id="t")

    await asyncio.gather(*(mw.awrap_tool_call(_req("task"), handler) for _ in range(6)))
    assert peak == 2


@pytest.mark.asyncio
async def test_async_other_tools_are_not_limited():
    mw = SubagentConcurrencyMiddleware(1)
    running = peak = 0

    async def handler(_req):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.02)
        running -= 1
        return ToolMessage(content="ok", tool_call_id="t")

    await asyncio.gather(*(mw.awrap_tool_call(_req("web_research"), handler) for _ in range(4)))
    assert peak == 4


@pytest.mark.asyncio
async def test_zero_disables_limit():
    mw = SubagentConcurrencyMiddleware(0)
    running = peak = 0

    async def handler(_req):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.02)
        running -= 1
        return ToolMessage(content="ok", tool_call_id="t")

    await asyncio.gather(*(mw.awrap_tool_call(_req("task"), handler) for _ in range(5)))
    assert peak == 5


# ── sync ─────────────────────────────────────────────────────────────────


def test_sync_caps_concurrent_task_calls():
    mw = SubagentConcurrencyMiddleware(2)
    lock = threading.Lock()
    running = peak = 0

    def handler(_req):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.03)
        with lock:
            running -= 1
        return ToolMessage(content="ok", tool_call_id="t")

    threads = [threading.Thread(target=mw.wrap_tool_call, args=(_req("task"), handler)) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert peak == 2


# ── Environment resolution ───────────────────────────────────────────────


def test_env_auto_local_vs_hosted(monkeypatch):
    monkeypatch.setenv("MAX_PARALLEL_SUBAGENTS", "auto")
    monkeypatch.delenv("DEEP_AGENT_LLM_PROVIDER", raising=False)
    for provider in ("omlx", "exo", "mlx"):
        monkeypatch.setenv("LLM_PROVIDER", provider)
        assert Environment.get_max_parallel_subagents() == 2
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert Environment.get_max_parallel_subagents() == 0
    assert maybe_for_environment() is None


def test_env_explicit_and_invalid(monkeypatch):
    monkeypatch.delenv("DEEP_AGENT_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "omlx")
    monkeypatch.setenv("MAX_PARALLEL_SUBAGENTS", "4")
    assert Environment.get_max_parallel_subagents() == 4
    monkeypatch.setenv("MAX_PARALLEL_SUBAGENTS", "0")
    assert Environment.get_max_parallel_subagents() == 0
    monkeypatch.setenv("MAX_PARALLEL_SUBAGENTS", "garbage")
    assert Environment.get_max_parallel_subagents() == 2  # falls back to auto
    monkeypatch.setenv("MAX_PARALLEL_SUBAGENTS", "-3")
    assert Environment.get_max_parallel_subagents() == 2


def test_orchestrator_provider_override_wins(monkeypatch):
    monkeypatch.setenv("MAX_PARALLEL_SUBAGENTS", "auto")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("DEEP_AGENT_LLM_PROVIDER", "omlx")
    assert Environment.get_max_parallel_subagents() == 2
