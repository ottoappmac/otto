"""Unit tests for ReAct-shim planning support.

Open-weights / text-only models run through the ReAct shim, which only
permits one tool call per turn.  These tests lock in the behaviour that
makes such models emit a ``write_todos`` plan:

* the injected tool-use section instructs the model to call ``write_todos``
  first for multi-step tasks;
* ``write_todos`` is never dropped when the tool list is capped for a
  small-context model;
* the ``force_action`` reminder (reasoning models) permits a ``write_todos``
  call as the first Action instead of forbidding any plan.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from middleware._react_core import (
    TOOL_SECTION_TEMPLATE,
    prioritise_planning_tools,
)
from middleware.react_wrapper import _FORCE_ACTION_REMINDER


class _StubTool(BaseTool):
    """Minimal BaseTool used only for ordering/rendering assertions."""

    name: str = "stub"
    description: str = "stub tool"

    def _run(self, *args, **kwargs):  # pragma: no cover - never invoked
        return ""


def _tool(name: str) -> _StubTool:
    return _StubTool(name=name, description=f"{name} description")


# ── ReAct tool-use template ────────────────────────────────────────────────

def test_react_template_instructs_write_todos_planning():
    assert "write_todos" in TOOL_SECTION_TEMPLATE
    assert "Planning Multi-Step Work" in TOOL_SECTION_TEMPLATE
    # The template renders via str.format(tool_descriptions=...); the literal
    # JSON example uses doubled braces so it survives formatting.
    rendered = TOOL_SECTION_TEMPLATE.format(tool_descriptions="(tools)")
    assert '"action": "write_todos"' in rendered
    assert "{{" not in rendered  # all escaped braces collapsed cleanly


# ── Planning-tool prioritisation under a cap ───────────────────────────────

def test_prioritise_planning_tools_floats_write_todos_to_front():
    tools = [_tool("a"), _tool("b"), _tool("write_todos"), _tool("c")]
    ordered = prioritise_planning_tools(tools)
    assert ordered[0].name == "write_todos"
    # Remaining order preserved.
    assert [t.name for t in ordered[1:]] == ["a", "b", "c"]


def test_prioritise_planning_tools_survives_a_tight_cap():
    """With max_tools=1, write_todos must be the surviving tool."""
    tools = [_tool("a"), _tool("b"), _tool("write_todos")]
    capped = prioritise_planning_tools(tools)[:1]
    assert [t.name for t in capped] == ["write_todos"]


def test_prioritise_planning_tools_noop_without_write_todos():
    tools = [_tool("a"), _tool("b")]
    ordered = prioritise_planning_tools(tools)
    assert [t.name for t in ordered] == ["a", "b"]


# ── force_action reminder ───────────────────────────────────────────────────

def test_force_action_reminder_allows_write_todos_plan():
    # Reasoning models were told "Do NOT write a plan" — that blocked todos.
    assert "write_todos" in _FORCE_ACTION_REMINDER
    assert "Do NOT write a plan," not in _FORCE_ACTION_REMINDER
    # It still forbids a *prose* plan / fabricated final answer.
    assert "prose plan" in _FORCE_ACTION_REMINDER
    assert "Final Answer" in _FORCE_ACTION_REMINDER
