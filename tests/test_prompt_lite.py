"""Unit tests for lite-mode orchestrator prompt assembly.

These tests lock in the contract that lite mode keeps:

* the four behaviour-critical rules (subagent dispatch shape, /output
  writes, execute path safety, action confirmation)
* every active subagent named in the delegation hints
* a substantial size reduction over the full prompt

and that full mode emits the capability ladder when meta-tools are bound.
"""

from __future__ import annotations

import os

import pytest

from deep_agent.prompt import (
    _CAPABILITY_GAP_TRIGGER_TOOLS,
    build_orchestrator_prompt,
)


class _FakeTool:
    """Minimal stand-in for a ``BaseTool`` — only ``name`` is consulted by
    :func:`build_orchestrator_prompt`."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"Stub tool {name}"


@pytest.fixture
def gp_tool() -> _FakeTool:
    return _FakeTool("write_file")


@pytest.fixture
def meta_tool() -> _FakeTool:
    """A tool whose presence triggers the capability ladder in full mode."""
    return _FakeTool("create_mcp_server")


# ── Lite mode --------------------------------------------------------------

def test_lite_keeps_subagent_dispatch_shape(gp_tool):
    p = build_orchestrator_prompt([gp_tool], subagents=None, lite=True)
    assert 'task(subagent_type="<name>"' in p, (
        "Lite must keep the literal task() example — it's the most "
        "frequently mis-emitted call shape on local models."
    )
    assert "DIRECT TOOLS" in p
    assert "SUBAGENTS are NOT tools" in p


def test_lite_keeps_output_path_rule(gp_tool):
    p = build_orchestrator_prompt([gp_tool], subagents=None, lite=True)
    assert "/output/" in p
    assert "write_file" in p


def test_lite_keeps_execute_path_safety_rule(gp_tool):
    p = build_orchestrator_prompt([gp_tool], subagents=None, lite=True)
    # The single highest-risk prompt-only rule.
    assert "$SESSION_FILES" in p
    assert "execute" in p
    assert "host filesystem root" in p


def test_lite_keeps_action_confirmation_line(gp_tool):
    p = build_orchestrator_prompt([gp_tool], subagents=None, lite=True)
    assert "destructive" in p.lower()
    assert "confirm" in p.lower()


def test_lite_workflow_tells_model_to_plan_with_write_todos(gp_tool):
    """Local models won't emit write_todos unless told to; the lite
    workflow must positively instruct it for multi-step tasks."""
    p = build_orchestrator_prompt([gp_tool], subagents=None, lite=True)
    assert "write_todos" in p
    assert "3+ steps" in p or "3 or more" in p


def test_lite_does_not_forbid_standalone_planning_turn(gp_tool):
    """The old guidance ('Don't reply with only write_todos') is
    unsatisfiable under one-action-per-turn ReAct and suppressed planning.
    It must be gone."""
    p = build_orchestrator_prompt([gp_tool], subagents=None, lite=True)
    assert "only `write_todos`" not in p
    assert "only write_todos" not in p


def test_lite_drops_capability_ladder_block(gp_tool, meta_tool):
    """Even when meta-tools are bound, lite mode skips the long block."""
    p = build_orchestrator_prompt([gp_tool, meta_tool], subagents=None, lite=True)
    assert "<capability_ladder>" not in p
    assert "spawn_followup_session" not in p


def test_lite_drops_long_blocks(gp_tool):
    p = build_orchestrator_prompt([gp_tool], subagents=None, lite=True)
    for tag in (
        "<task_delegation>",
        "<tool_results>",
        "<capability_ladder>",
        "<action_safety>",
        "<ask_user>",
    ):
        assert tag not in p, f"Lite mode should drop the {tag} block"


def test_lite_includes_only_active_subagent_hints(gp_tool):
    p = build_orchestrator_prompt(
        [gp_tool],
        subagents=[{"name": "web-voyager"}],
        lite=True,
    )
    assert "web-voyager" in p
    # Inactive subagents stay out — keeps prompt size bounded.
    assert "computer-voyager" not in p


def test_lite_size_under_budget(gp_tool):
    """Soft cap at ~1000 tokens (≈4000 chars) for a typical agent.

    History:
    * 400 tok / 1600 chars: original lite contract.
    * +missing-tool / credentials / reuse bullets: ~550 tok / 2200 chars.
    * +<subagents> enumeration block + generalised macOS rule (with
      illustrative app list): ~750 tok / 3000 chars.
    * +<workflow> plan→act→check loop, no-invented-names guard, and full
      (un-truncated) delegation hints: ~1000 tok / 4000 chars.

    The <subagents> block's per-agent "use when" trigger is load-bearing
    on 4B-class models — without it the orchestrator asks the user
    whether tools exist instead of climbing the ladder.  See the Slack
    DM trace that motivated the change.  The <workflow> loop addresses
    the "directionless" failure mode: without an explicit
    plan→act→check→stop scaffold, small models repeat failing calls and
    keep calling tools after the answer is known.
    """
    p = build_orchestrator_prompt(
        [gp_tool],
        subagents=[{"name": "web-voyager"}, {"name": "computer-voyager"}],
        lite=True,
    )
    assert len(p) < 4000, (
        f"Lite prompt is {len(p)} chars (~{len(p)//4} tok) — "
        "exceeds the ~1000 tok soft cap; review additions."
    )


# ── Full mode parity --------------------------------------------------------

def test_full_keeps_full_prompt_when_meta_tools_present(gp_tool, meta_tool):
    p = build_orchestrator_prompt([gp_tool, meta_tool], subagents=None, lite=False)
    assert "<capability_ladder>" in p
    assert "<task_delegation>" in p
    assert "<tool_results>" in p
    assert "<action_safety>" in p
    assert "<ask_user>" in p


def test_full_drops_capability_ladder_when_meta_tools_absent(gp_tool):
    """Cleanup: don't pay ~370 tokens of advice the model can't act on."""
    p = build_orchestrator_prompt([gp_tool], subagents=None, lite=False)
    assert "<capability_ladder>" not in p
    # But the other long blocks are unchanged.
    assert "<task_delegation>" in p
    assert "<action_safety>" in p


def test_full_efficiency_positively_encourages_planning(gp_tool):
    """Full mode must tell the model to START multi-step work with
    write_todos, and must NOT carry the old unsatisfiable 'never reply
    with only write_todos' rule that suppressed planning on ReAct models."""
    p = build_orchestrator_prompt([gp_tool], subagents=None, lite=False)
    assert "write_todos" in p
    assert "START by calling write_todos" in p
    assert "NEVER make a response that ONLY calls write_todos" not in p
    # A one-action-per-turn model must be told a standalone planning turn is OK.
    assert "ONE action per turn" in p


def test_capability_gap_trigger_tools_set_is_non_empty():
    """Catch typos in the trigger set.  If this list shrinks to zero,
    the capability ladder would never fire in full mode."""
    assert _CAPABILITY_GAP_TRIGGER_TOOLS
    assert "create_mcp_server" in _CAPABILITY_GAP_TRIGGER_TOOLS


# ── Dynamic subagent substitution -------------------------------------------

def test_capability_ladder_uses_registered_browser_subagent_name(meta_tool):
    """Ladder must reference whichever browser subagent is actually
    registered — not a hardcoded name that doesn't exist in this app."""
    p = build_orchestrator_prompt(
        [meta_tool],
        subagents=[{"name": "browser-agent"}],
        lite=False,
    )
    assert "browser-agent" in p
    # Don't refer to subagents that aren't registered.
    assert "web-voyager" not in p
    assert "computer-voyager" not in p


def test_capability_ladder_drops_browser_rung_when_no_browser_subagent(meta_tool):
    """If no browser subagent is registered, the ladder must NOT name a
    placeholder subagent — drop the rung entirely."""
    p = build_orchestrator_prompt([meta_tool], subagents=None, lite=False)
    # Browser-fallback rung disappears; "Ask the user" advances to step 5.
    assert "web-voyager" not in p
    assert "browser-agent" not in p
    assert "Ask the user" in p


def test_capability_ladder_includes_auto_signup_narration_rule(meta_tool):
    """The agent should narrate when it skips auto-signup so the user
    knows why we're falling back to manual paste."""
    p = build_orchestrator_prompt(
        [meta_tool],
        subagents=[{"name": "browser-agent"}],
        lite=False,
    )
    assert "Skip auto-signup" in p
    assert "tell the user" in p or "say so" in p or "say why" in p


def test_lite_guidance_uses_registered_subagent_name(gp_tool):
    p = build_orchestrator_prompt(
        [gp_tool],
        subagents=[{"name": "browser-agent"}],
        lite=True,
    )
    assert "browser-agent" in p
    assert "web-voyager" not in p


# ── Lite is meaningfully smaller --------------------------------------------

def test_lite_is_dramatically_smaller_than_full(gp_tool, meta_tool):
    full = build_orchestrator_prompt(
        [gp_tool, meta_tool],
        subagents=[{"name": "web-voyager"}],
        lite=False,
    )
    lite = build_orchestrator_prompt(
        [gp_tool, meta_tool],
        subagents=[{"name": "web-voyager"}],
        lite=True,
    )
    # Expect at least a 3× reduction; today it's closer to 6×.
    assert len(lite) * 3 < len(full), (
        f"Lite ({len(lite)}) should be much smaller than full ({len(full)}); "
        "lite mode appears to have grown."
    )


# ── Environment helpers -----------------------------------------------------

def test_environment_helpers_resolve_modes(monkeypatch):
    from utilities.environment import Environment

    monkeypatch.setenv("LLM_PROVIDER", "mlx")
    monkeypatch.delenv("DEEP_AGENT_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("LOCAL_PROMPT_MODE", "auto")
    assert Environment.is_oss_local_provider("mlx") is True
    assert Environment.is_oss_local_provider("anthropic") is False
    assert Environment.is_orchestrator_oss_local() is True
    # auto → full everywhere, including OSS-local providers; lite is
    # opt-in via LOCAL_PROMPT_MODE=lite only.
    assert Environment.use_lite_orchestrator_prompt() is False

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("DEEP_AGENT_LLM_PROVIDER", raising=False)
    assert Environment.use_lite_orchestrator_prompt() is False

    # Force full on local model.
    monkeypatch.setenv("LLM_PROVIDER", "mlx")
    monkeypatch.setenv("LOCAL_PROMPT_MODE", "full")
    assert Environment.use_lite_orchestrator_prompt() is False

    # Force lite on frontier (benchmarking).
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LOCAL_PROMPT_MODE", "lite")
    assert Environment.use_lite_orchestrator_prompt() is True

    # Unknown mode falls back to auto.
    monkeypatch.setenv("LOCAL_PROMPT_MODE", "garbage")
    assert Environment.get_local_prompt_mode() == "auto"


# ── AppleScript-first preference -------------------------------------------
#
# These tests lock in the contract that, when both
# ``macos-applescript-agent`` and ``macos-desktop-agent`` are registered as
# subagents, the orchestrator prompt prefers AppleScript and uses UI
# automation only as a fallback.  AppleScript is deterministic and
# idempotent whenever the target app exposes the verb; UI automation is
# non-deterministic and token-expensive, so the ladder must reflect that
# ordering — both in the long capability_ladder block (full mode) and in
# the slim guidance bullets (lite mode).


def test_full_ladder_orders_applescript_before_desktop(meta_tool):
    p = build_orchestrator_prompt(
        [meta_tool],
        subagents=[
            {"name": "macos-applescript-agent"},
            {"name": "macos-desktop-agent"},
        ],
        lite=False,
    )
    apple_idx = p.find("macos-applescript-agent")
    desktop_idx = p.find("macos-desktop-agent")
    assert apple_idx != -1 and desktop_idx != -1
    assert apple_idx < desktop_idx, (
        "AppleScript rung must come before the UI-automation rung in the "
        "capability ladder — UI is the fallback when AppleScript can't "
        "reach the verb."
    )
    # The fallback marker should be present so the orchestrator knows
    # what error pattern to look for when handing off.
    assert "FALLBACK:" in p or "fallback" in p.lower()


def test_full_ladder_keeps_only_applescript_when_desktop_absent(meta_tool):
    p = build_orchestrator_prompt(
        [meta_tool],
        subagents=[{"name": "macos-applescript-agent"}],
        lite=False,
    )
    assert "macos-applescript-agent" in p
    # The desktop-agent delegation hint only emits when the desktop
    # agent is registered; its absence is the reliable signal that the
    # ladder rung was dropped (the AppleScript-preference rule body
    # still references the desktop agent as a fallback target).
    assert "For macos-desktop-agent:" not in p


def test_full_ladder_handles_desktop_only(meta_tool):
    """Linux / Windows hosts won't seed the AppleScript agent — desktop
    rung must still render correctly in that scenario."""
    p = build_orchestrator_prompt(
        [meta_tool],
        subagents=[{"name": "macos-desktop-agent"}],
        lite=False,
    )
    assert "For macos-desktop-agent:" in p
    # AppleScript hints / preference rule are gated on the AppleScript
    # agent being registered.
    assert "For macos-applescript-agent:" not in p
    assert "FIRST dispatch macos-applescript-agent" not in p


def test_full_rules_inject_applescript_preference_when_present(meta_tool):
    p = build_orchestrator_prompt(
        [meta_tool],
        subagents=[
            {"name": "macos-applescript-agent"},
            {"name": "macos-desktop-agent"},
        ],
        lite=False,
    )
    # The generalised rule begins with "FIRST dispatch …" and lists
    # illustrative apps so 4B-class models can pattern-match on lexical
    # surface instead of classifying intent.
    assert "FIRST dispatch macos-applescript-agent" in p
    # Illustrative apps must include both dictionary apps (Mail) and
    # Electron / Catalyst apps (Slack, Cursor) — the rule explicitly
    # covers ANY GUI app, not just dictionary-rich ones.
    for app in ("Slack", "Cursor", "Mail", "Notes"):
        assert app in p, f"rule should list {app} as illustrative"
    # The rule explicitly mentions TCC as the one case where you do NOT
    # fall back.
    assert "-1743" in p


def test_full_rules_skip_applescript_preference_when_absent(meta_tool):
    """No AppleScript agent → no AppleScript-preference rule (the rule
    has no actionable fallback in that environment)."""
    p = build_orchestrator_prompt(
        [meta_tool],
        subagents=[{"name": "macos-desktop-agent"}],
        lite=False,
    )
    assert "FIRST dispatch macos-applescript-agent" not in p


def test_lite_guidance_orders_applescript_before_desktop(gp_tool):
    p = build_orchestrator_prompt(
        [gp_tool],
        subagents=[
            {"name": "macos-applescript-agent"},
            {"name": "macos-desktop-agent"},
        ],
        lite=True,
    )
    # Both must be referenced in the fallback chain.  Use the
    # AppleScript-tagged occurrence as the anchor (the literal
    # "macos-applescript-agent (AppleScript)" appears in the lite
    # guidance fallback chain) so the assertion is robust to other
    # places the agent name shows up.
    apple_chain_idx = p.find("macos-applescript-agent (AppleScript)")
    desktop_chain_idx = p.find("macos-desktop-agent (macOS UI)")
    assert apple_chain_idx != -1 and desktop_chain_idx != -1
    assert apple_chain_idx < desktop_chain_idx


def test_lite_rules_inject_applescript_preference_when_present(gp_tool):
    p = build_orchestrator_prompt(
        [gp_tool],
        subagents=[
            {"name": "macos-applescript-agent"},
            {"name": "macos-desktop-agent"},
        ],
        lite=True,
    )
    assert "FIRST dispatch macos-applescript-agent" in p
    assert "macos-desktop-agent" in p
    # Illustrative app list must travel in lite mode too — that's the
    # whole point of the generalisation for small models.
    for app in ("Slack", "Cursor"):
        assert app in p, f"lite rule should list {app}"


def test_lite_rules_skip_applescript_preference_when_absent(gp_tool):
    p = build_orchestrator_prompt(
        [gp_tool],
        subagents=[{"name": "macos-desktop-agent"}],
        lite=True,
    )
    # The rule body only emits when the AppleScript agent is registered;
    # the <subagents> block's macos-desktop-agent entry references
    # macos-applescript-agent as the primary, so check the rule body
    # specifically rather than the substring globally.
    assert "FIRST dispatch macos-applescript-agent" not in p


def test_full_info_section_threads_applescript(meta_tool):
    """For information queries about native macOS app state (frontmost
    app, current track, …), the ``To get information`` clause should
    name the AppleScript agent before the browser fallback."""
    p = build_orchestrator_prompt(
        [meta_tool],
        subagents=[
            {"name": "macos-applescript-agent"},
            {"name": "browser-agent"},
        ],
        lite=False,
    )
    apple_idx = p.find("macos-applescript-agent")
    browser_idx = p.find("browser-agent")
    assert apple_idx != -1 and browser_idx != -1
    assert apple_idx < browser_idx


# ── <subagents> enumeration block (lite mode) -------------------------------
#
# Motivated by the Slack DM trace where the Qwen3.5-4B orchestrator asked
# "Do you have a Playwright browser agent or another web automation tool
# available?" — i.e. it didn't see its subagents at decision time.  The
# enumeration block surfaces every registered subagent with a one-line
# "use when" trigger so small models pattern-match on lexical surface
# instead of doing intent classification.


def test_lite_emits_subagents_block_when_subagents_registered(gp_tool):
    p = build_orchestrator_prompt(
        [gp_tool],
        subagents=[
            {"name": "macos-applescript-agent"},
            {"name": "browser-agent"},
        ],
        lite=True,
    )
    assert "<subagents>" in p
    assert "</subagents>" in p
    assert 'task(subagent_type="<name>"' in p


def test_lite_omits_subagents_block_when_no_subagents(gp_tool):
    p = build_orchestrator_prompt([gp_tool], subagents=None, lite=True)
    assert "<subagents>" not in p


def test_lite_subagents_block_includes_apps_in_macos_trigger(gp_tool):
    """The macos-applescript-agent ``use when`` trigger MUST list
    illustrative apps (Slack, Messages, Mail, …) so a 4B-class model
    pattern-matches on the user's literal app name instead of having to
    derive 'Slack is a macOS app'."""
    p = build_orchestrator_prompt(
        [gp_tool],
        subagents=[{"name": "macos-applescript-agent"}],
        lite=True,
    )
    # Find the line for macos-applescript-agent inside <subagents>.
    block_start = p.index("<subagents>")
    block_end = p.index("</subagents>")
    block = p[block_start:block_end]
    assert "macos-applescript-agent" in block
    # At least these representative apps must appear in the trigger.
    for app in ("Slack", "Messages", "Mail", "Notes"):
        assert app in block, f"trigger should mention {app}"


def test_lite_subagents_falls_back_to_description_for_unknown_agent(gp_tool):
    """Custom (user-authored) subagents not in the static use-when map
    should still be enumerated, falling back to the agent's
    ``description`` so the block stays useful."""
    p = build_orchestrator_prompt(
        [gp_tool],
        subagents=[
            {
                "name": "custom-data-agent",
                "description": "Custom agent that ingests CSVs from /input.",
            }
        ],
        lite=True,
    )
    assert "custom-data-agent" in p
    assert "Custom agent that ingests CSVs" in p


def test_lite_subagents_enumerates_all_registered(gp_tool):
    p = build_orchestrator_prompt(
        [gp_tool],
        subagents=[
            {"name": "macos-applescript-agent"},
            {"name": "macos-desktop-agent"},
            {"name": "browser-agent"},
            {"name": "trigger-builder-agent"},
        ],
        lite=True,
    )
    block_start = p.index("<subagents>")
    block_end = p.index("</subagents>")
    block = p[block_start:block_end]
    for name in (
        "macos-applescript-agent",
        "macos-desktop-agent",
        "browser-agent",
        "trigger-builder-agent",
    ):
        assert name in block, f"{name} missing from <subagents>"


# ── Activity-vs-AppleScript temporal split ---------------------------------


def test_lite_guidance_distinguishes_activity_past_from_applescript_now(gp_tool):
    """When activity tools AND macos-applescript-agent are both active,
    the lite guidance must explicitly split temporal scope: PAST →
    activity, RIGHT NOW → applescript.  Without this, the orchestrator
    in the Slack trace picked search_screen_history for a
    'what's the latest message NOW' query."""

    class _ActivityTool:
        def __init__(self, name: str) -> None:
            self.name = name
            self.description = f"Stub {name}"

    p = build_orchestrator_prompt(
        [gp_tool, _ActivityTool("search_screen_history")],
        subagents=[{"name": "macos-applescript-agent"}],
        lite=True,
    )
    assert "PAST" in p
    assert "RIGHT NOW" in p
    # Both surfaces named in the same paragraph.
    assert "activity tools" in p
    assert "macos-applescript-agent" in p
