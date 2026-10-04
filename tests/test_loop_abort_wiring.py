"""Unit tests for the loop-guard escalation → cooperative-abort wiring.

Covers the fixes that reconnect a tripped loop guard to the run-abort
machinery so a model that ignores corrective messages is forcibly unwound:

* fix #1 — per-connection MCP guards now request escalation (``max_escalations``);
* fix #2 — and fire ``request_loop_abort_current``, which resolves the looping
  invocation from run-scoped contextvars (per-connection guards are shared
  across sessions and never see a session id at construction time);
* fix #4 — subagent universal guards install that same callback so an
  escalation flags only the bound invocation, not every subagent on the session.

The abort itself is a cooperative flag in ``backend.state.loop_abort_requested``
(session id → invocation id → reason) that the matching subagent stream checks
at each step boundary.
"""

from __future__ import annotations

import pytest
from langchain_core.tools import StructuredTool

from tools.loop_guard import ToolLoopDetected, ToolLoopGuard
from utilities.environment import Environment

from backend import state
from backend.mcp_manager import _loop_recovery_kwargs
from backend.session_manager import _apply_universal_loop_guard
from backend.streaming_subagent import (
    request_loop_abort_current,
    reset_current_invocation,
    reset_current_session,
    set_current_invocation,
    set_current_session,
)


@pytest.fixture(autouse=True)
def _clean_abort_state():
    """Keep the module-global abort registry isolated per test."""
    state.loop_abort_requested.clear()
    yield
    state.loop_abort_requested.clear()


def _make_tool(name: str, impl):
    return StructuredTool.from_function(
        coroutine=impl, name=name, description=f"{name} tool",
    )


def _drive_to_escalation(guard: ToolLoopGuard, name: str, args: dict) -> int:
    """Seed an identical-args failure loop and trip the guard until it
    escalates.  Returns the number of trips performed."""
    max_identical = guard._max_identical  # noqa: SLF001 — white-box test
    for _ in range(max_identical):
        guard.record_result(name, args, ok=False)

    max_esc = guard._max_escalations or 0  # noqa: SLF001
    trips = 0
    for _ in range(max_esc):
        with pytest.raises(ToolLoopDetected):
            guard.check_before(name, args)
        trips += 1
    return trips


# ── contextvar + request_loop_abort_current (fix #2 mechanism) ──────────────

def test_request_loop_abort_current_uses_bound_invocation():
    session_token = set_current_session("sess-ctx")
    invocation_token = set_current_invocation("inv-a")
    try:
        request_loop_abort_current("looping on e127")
    finally:
        reset_current_invocation(invocation_token)
        reset_current_session(session_token)
    assert state.loop_abort_requested["sess-ctx"]["inv-a"] == "looping on e127"


def test_request_loop_abort_current_does_not_flag_sibling_invocation():
    """An escalation on one subagent leaves the other invocation unmarked."""
    session_token = set_current_session("sess-ctx")
    invocation_token = set_current_invocation("inv-a")
    try:
        request_loop_abort_current("looping on e127")
    finally:
        reset_current_invocation(invocation_token)
        reset_current_session(session_token)
    assert "inv-b" not in state.loop_abort_requested["sess-ctx"]
    assert state.loop_abort_requested["sess-ctx"]["inv-a"] == "looping on e127"


def test_request_loop_abort_current_is_noop_without_session():
    # No session bound to the context (default None) — must not raise or
    # pollute the registry with a None key.
    invocation_token = set_current_invocation("inv-a")
    try:
        request_loop_abort_current("no session here")
    finally:
        reset_current_invocation(invocation_token)
    assert state.loop_abort_requested == {}


def test_request_loop_abort_current_is_noop_without_invocation():
    """A session binding alone must not flag every subagent on that session."""
    session_token = set_current_session("sess-ctx")
    try:
        request_loop_abort_current("no invocation here")
    finally:
        reset_current_session(session_token)
    assert state.loop_abort_requested == {}


# ── _loop_recovery_kwargs arms escalation on every MCP guard (fix #1/#2) ─────

def test_loop_recovery_kwargs_arms_escalation():
    kw = _loop_recovery_kwargs()
    assert kw["max_escalations"] is not None and kw["max_escalations"] > 0
    # The shared abort callback that resolves the run from the contextvar.
    assert kw["on_escalate"] is request_loop_abort_current


def test_per_connection_guard_escalation_aborts_current_invocation():
    """A per-connection-style guard built from ``_loop_recovery_kwargs`` aborts
    only the invocation bound to the current context when it escalates."""
    guard = ToolLoopGuard(max_identical=3, **_loop_recovery_kwargs())

    session_token = set_current_session("sess-pw")
    invocation_token = set_current_invocation("inv-pw")
    try:
        _drive_to_escalation(guard, "browser_click", {"ref": "e127"})
    finally:
        reset_current_invocation(invocation_token)
        reset_current_session(session_token)

    assert state.loop_abort_requested["sess-pw"]["inv-pw"]
    assert "inv-other" not in state.loop_abort_requested["sess-pw"]


# ── _apply_universal_loop_guard threads session_id for subagents (fix #4) ────

def test_universal_guard_with_session_id_aborts_bound_invocation_only():
    async def impl(a: int) -> str:
        return "constant"

    tool = _make_tool("flaky", impl)
    guard = _apply_universal_loop_guard(
        [tool], scope="subagent:test", session_id="sess-uni",
    )
    assert guard is not None

    session_token = set_current_session("sess-uni")
    invocation_token = set_current_invocation("inv-uni")
    try:
        _drive_to_escalation(guard, "flaky", {"a": 1})
    finally:
        reset_current_invocation(invocation_token)
        reset_current_session(session_token)
    assert state.loop_abort_requested["sess-uni"]["inv-uni"]
    assert "inv-sibling" not in state.loop_abort_requested["sess-uni"]


def test_universal_guard_without_session_id_does_not_abort():
    """Regression marker: without a session id the guard still emits the
    terminal directive but cannot mark any run for abort (the gap fix #4
    closes for subagents)."""
    async def impl(a: int) -> str:
        return "constant"

    tool = _make_tool("flaky", impl)
    guard = _apply_universal_loop_guard([tool], scope="no-session")
    assert guard is not None

    _drive_to_escalation(guard, "flaky", {"a": 1})
    assert state.loop_abort_requested == {}


def test_universal_guard_escalation_limit_matches_environment():
    """The universal guard uses the configured escalation budget."""
    async def impl(a: int) -> str:
        return "constant"

    tool = _make_tool("flaky", impl)
    guard = _apply_universal_loop_guard(
        [tool], scope="env-check", session_id="sess-env",
    )
    assert guard._max_escalations == Environment.get_loop_guard_max_escalations()  # noqa: SLF001
