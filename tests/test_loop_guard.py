"""Unit tests for the universal tool loop guard (``tools.loop_guard``).

These cover the failure modes that motivated the universal guard:

* a model that alternates between two distinct *successful* calls that each
  keep returning the same result (the carpet ``doc_research`` loop) — caught
  by the success-loop and no-progress detectors;
* "different args, same useless result" — caught by the no-progress detector;
* read-only observation tools that legitimately repeat — must stay exempt;
* a model that keeps looping past the escalation limit — must trigger the
  terminal directive and the ``on_escalate`` abort callback exactly once;
* idempotency — a tool already guarded by a per-connection loader must not be
  re-wrapped by the universal ``guard_all_tools`` pass.
"""

from __future__ import annotations

import pytest
from langchain_core.tools import StructuredTool

from tools.loop_guard import (
    DEFAULT_HIGH_COST_TOOLS,
    DEFAULT_MAX_HIGH_COST_REPEATS,
    OBSERVATION_TOOLS,
    ToolLoopDetected,
    ToolLoopGuard,
    build_default_guard,
    guard_all_tools,
    wrap_with_loop_guard,
)


def _make_tool(name: str, impl):
    """Build an async-only StructuredTool wrapping *impl*."""
    return StructuredTool.from_function(
        coroutine=impl, name=name, description=f"{name} tool",
    )


async def test_no_progress_trips_on_same_result_different_args():
    """Distinct arguments that always return the same result must trip the
    no-progress guard even though no two calls are identical."""
    guard = build_default_guard(
        max_identical_success=None,  # isolate the no-progress detector
        max_no_progress=4,
        window=8,
    )

    async def impl(q: str) -> str:
        return "ALWAYS THE SAME BORING PAGE"

    tool = _make_tool("doc_research", impl)
    wrap_with_loop_guard(tool, guard)

    with pytest.raises(ToolLoopDetected):
        for i in range(10):
            await tool.coroutine(q=f"distinct-query-{i}")


async def test_alternating_two_call_loop_trips_success_guard():
    """The carpet pattern: alternating between two distinct calls that each
    return their own stable result trips the (per-key) success-loop guard."""
    guard = build_default_guard(window=8)

    async def impl(url: str) -> str:
        return f"stable-content-for::{url}"

    tool = _make_tool("doc_research", impl)
    wrap_with_loop_guard(tool, guard)

    urls = ["victoriacarpets/ambur", "choices/kadota"]
    with pytest.raises(ToolLoopDetected):
        for i in range(12):
            await tool.coroutine(url=urls[i % 2])


async def test_observation_tools_are_exempt():
    """A read-only observation tool may repeat identical calls/results without
    tripping the success or no-progress detectors."""
    assert "get_screen_controls" in OBSERVATION_TOOLS
    guard = build_default_guard(max_no_progress=3, window=8)

    async def impl(app_name: str) -> str:
        return "same control tree every time"

    tool = _make_tool("get_screen_controls", impl)
    wrap_with_loop_guard(tool, guard)

    # Many identical calls with identical results: must NOT raise.
    for _ in range(12):
        out = await tool.coroutine(app_name="Slack")
    assert out == "same control tree every time"


async def test_failure_loop_still_trips():
    """The classic identical-args failure loop must still trip."""
    guard = build_default_guard(max_identical=3, window=8)

    async def impl(x: int) -> str:
        raise RuntimeError("boom")

    tool = _make_tool("flaky_tool", impl)
    wrap_with_loop_guard(tool, guard)

    # First three calls surface the underlying error (and record a failure).
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await tool.coroutine(x=1)

    # Once the failure count reaches the threshold the guard raises
    # ToolLoopDetected before the tool even runs.
    with pytest.raises(ToolLoopDetected):
        await tool.coroutine(x=1)


async def test_escalation_fires_once_and_stops():
    """After ``max_escalations`` trips the guard emits the terminal directive
    and calls ``on_escalate`` exactly once."""
    reasons: list[str] = []
    guard = build_default_guard(
        max_identical=3,
        max_identical_success=None,
        max_no_progress=None,
        max_escalations=3,
        on_escalate=reasons.append,
    )

    # Seed three identical failures so check_before trips on the failure path.
    for _ in range(3):
        guard.record_result("t", {"a": 1}, ok=False)

    # Trip 1 and 2: normal corrective message, no escalation yet.
    for _ in range(2):
        with pytest.raises(ToolLoopDetected) as ei:
            guard.check_before("t", {"a": 1})
        assert "STOP NOW" not in str(ei.value)
    assert reasons == []

    # Trip 3 reaches the escalation limit: terminal directive + one callback.
    with pytest.raises(ToolLoopDetected) as ei:
        guard.check_before("t", {"a": 1})
    assert "STOP NOW" in str(ei.value)
    assert len(reasons) == 1

    # Subsequent trips keep emitting the terminal directive but never fire the
    # callback again.
    with pytest.raises(ToolLoopDetected):
        guard.check_before("t", {"a": 1})
    assert len(reasons) == 1


async def test_guard_all_tools_does_not_double_wrap():
    """A tool already guarded by a per-connection loader keeps that guard; the
    universal pass must skip it idempotently."""
    async def impl(q: str) -> str:
        return "x"

    tool = _make_tool("web_research", impl)

    g1 = build_default_guard()
    wrap_with_loop_guard(tool, g1)
    first_coro = tool.coroutine

    g2 = build_default_guard()
    guard_all_tools([tool], guard=g2)

    # Idempotent: the coroutine object is unchanged (not re-wrapped by g2).
    assert tool.coroutine is first_coro


async def test_guard_all_tools_wraps_unguarded_tools():
    """The universal pass wraps a previously-unguarded tool so it now trips."""
    async def impl(q: str) -> str:
        return "constant"

    tool = _make_tool("doc_research", impl)
    guard_all_tools([tool], max_no_progress=4, max_identical_success=None, window=8)

    with pytest.raises(ToolLoopDetected):
        for i in range(10):
            await tool.coroutine(q=f"q-{i}")


# ── High-cost cumulative ceiling ──────────────────────────────────────────


async def test_high_cost_ceiling_trips_on_repeated_same_url():
    """Dozens of identical high-cost calls trip the cumulative per-run ceiling,
    even though each individual call *succeeds* with a varying-looking result
    (so the small-window detectors would otherwise miss them)."""
    guard = ToolLoopGuard(
        # Disable the window detectors to isolate the high-cost ceiling.
        max_identical_success=None,
        max_no_progress=None,
        window=8,
        high_cost_tools=frozenset({"browser_navigate"}),
        max_high_cost_repeats=5,
    )

    async def impl(url: str) -> str:
        return f"ok::{url}"

    tool = _make_tool("browser_navigate", impl)
    wrap_with_loop_guard(tool, guard)

    with pytest.raises(ToolLoopDetected):
        for _ in range(20):
            await tool.coroutine(url="https://example.com")


async def test_high_cost_ceiling_allows_distinct_targets():
    """Distinct high-cost calls (different args) are NOT curbed by the per-key
    ceiling — only redundant repeats of the *same* call are."""
    guard = ToolLoopGuard(
        max_identical_success=None,
        max_no_progress=None,
        window=8,
        high_cost_tools=frozenset({"browser_navigate"}),
        max_high_cost_repeats=5,
    )

    async def impl(url: str) -> str:
        return f"ok::{url}"

    tool = _make_tool("browser_navigate", impl)
    wrap_with_loop_guard(tool, guard)

    # 20 distinct URLs, each visited once: under the per-key ceiling.
    for i in range(20):
        await tool.coroutine(url=f"https://example.com/page-{i}")


async def test_high_cost_ceiling_off_by_default_without_repeats():
    """When ``max_high_cost_repeats`` is None the ceiling is inert."""
    guard = ToolLoopGuard(
        max_identical_success=None,
        max_no_progress=None,
        window=8,
        high_cost_tools=frozenset({"browser_navigate"}),
        max_high_cost_repeats=None,
    )

    async def impl(url: str) -> str:
        return "ok"

    tool = _make_tool("browser_navigate", impl)
    wrap_with_loop_guard(tool, guard)

    for _ in range(30):
        await tool.coroutine(url="https://example.com")


def test_default_high_cost_set_and_ceiling_exposed():
    """The default high-cost set + ceiling are sane and wired into the default
    guard so the universal pass curbs redundant research churn."""
    assert "web_research" in DEFAULT_HIGH_COST_TOOLS
    assert "browser_navigate" in DEFAULT_HIGH_COST_TOOLS
    assert DEFAULT_MAX_HIGH_COST_REPEATS >= 1
    guard = build_default_guard()
    assert guard._max_high_cost_repeats == DEFAULT_MAX_HIGH_COST_REPEATS
    assert guard._high_cost_tools == DEFAULT_HIGH_COST_TOOLS
