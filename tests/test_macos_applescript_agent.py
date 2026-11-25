"""Hermetic tests for the combined macOS agent's built-in seeding.

``macos-desktop-agent`` carries both the AppleScript tools
(``macos-osascript``) and the UI tools (``macos-native``).  It must:

* Materialise on macOS hosts with both tool sets and both skills
  (``macos-desktop``, ``macos-applescript``) so the orchestrator's capability
  ladder can address it via ``task(subagent_type="macos-desktop-agent", ...)``.
* Replace the old ``macos-applescript-agent``, which seeding deletes.
* Be hidden from ``list_agents`` / ``get_agent`` / ``list_skills`` /
  ``get_skill`` on non-macOS hosts.

Tests redirect ``get_app_data_dir`` to a tmp dir and monkey-patch
``platform_label`` so they can run on any host without touching the
user's real Otto profile.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def isolated_app_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Pin the app-data dir to a tmp path so seeding is hermetic."""
    from backend import agent_library, config

    monkeypatch.setattr(config, "get_app_data_dir", lambda: tmp_path)
    monkeypatch.setattr(agent_library, "get_app_data_dir", lambda: tmp_path)
    return tmp_path


def _force_platform(monkeypatch: pytest.MonkeyPatch, label: str) -> None:
    """Force the platform-label gate used by the macOS-only built-ins."""
    from backend import agent_library

    monkeypatch.setattr(agent_library, "platform_label", lambda: label)


def test_seed_creates_combined_macos_agent_on_macos(
    isolated_app_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_platform(monkeypatch, "macos")

    from backend import agent_library

    agent_library.seed_defaults()

    spec = agent_library.get_agent("macos-desktop-agent")
    assert spec is not None
    assert spec.tools == ["macos-native", "macos-osascript"]
    assert spec.skills == ["macos-desktop", "macos-applescript", "decision-making"]
    assert spec.builtin is True
    p = spec.system_prompt
    # Both routes are in the one prompt, and the hand-off is internal: no
    # FALLBACK contract that would need the orchestrator to re-dispatch.
    assert "## Script path" in p and "## UI path" in p
    assert "run_osascript" in p and "get_screen_controls" in p
    assert "FALLBACK" not in p
    assert "switch to the UI path" in p
    # The script path comes first.
    assert p.index("## Script path") < p.index("## UI path")


def test_seed_removes_old_applescript_agent(
    isolated_app_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An install that had the separate AppleScript agent loses it on the
    next seed — its job now belongs to ``macos-desktop-agent``."""
    _force_platform(monkeypatch, "macos")

    from backend import agent_library
    from backend.schemas import AgentSpec

    agent_library.save_agent(AgentSpec(name="macos-applescript-agent", description="old",
                                       system_prompt="old", tools=["macos-osascript"]))
    assert agent_library.get_agent("macos-applescript-agent") is not None
    agent_library.seed_defaults()
    assert agent_library.get_agent("macos-applescript-agent") is None


def test_seed_creates_macos_applescript_skill_on_macos(
    isolated_app_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_platform(monkeypatch, "macos")

    from backend import agent_library

    agent_library.seed_defaults()

    skill = agent_library.get_skill("macos-applescript")
    assert skill is not None
    assert skill.builtin is True
    # Cheat-sheet idioms are load-bearing — losing them silently would
    # regress the agent's success rate.
    assert "tell application" in skill.content
    assert "run_osascript" in skill.content
    # Generalised, introspect-then-act procedure must reference both
    # probe tools — the whole point of the v3 skill is replacing
    # LLM-side priors with system probes.
    assert "inspect_app_dictionary" in skill.content
    assert "dump_ax_tree" in skill.content
    # Universal 2-failure budget is the load-bearing rule that prevents
    # the trial-and-error loop seen in the Slack DM trace.  The hand-off is
    # now to the agent's own UI tools, not another agent.
    assert "switch to the UI tools" in skill.content
    assert "FALLBACK" not in skill.content
    # Syntax cheat-sheet (one-liner vs block tell) is what catches the
    # `-2741` syntax errors the 4B model kept producing.
    assert "One-liner" in skill.content or "one-liner" in skill.content
    assert "end tell" in skill.content
    # The probe now returns per-class properties — the skill must steer
    # the agent to read them instead of guessing field names (the root
    # cause of the Outlook `-2741 "found class name"` failures).
    assert "properties" in skill.content
    assert "found class name" in skill.content
    # query_mail_store is Apple Mail only — must not be applied to Outlook.
    assert "Apple Mail ONLY" in skill.content
    # Anti-loop rule: a successful empty/zero result is an answer, not a
    # reason to re-run the identical script.
    assert "Never re-run an identical script" in skill.content


def test_seed_macos_agent_prompt_has_decision_tree(
    isolated_app_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Agent prompt must encode the introspect-then-act decision tree
    (probe → branch → dictionary OR ax_tree) and the universal 2-failure
    budget so a 4B-class model has a flat checklist to follow."""
    _force_platform(monkeypatch, "macos")

    from backend import agent_library

    agent_library.seed_defaults()

    spec = agent_library.get_agent("macos-desktop-agent")
    assert spec is not None
    p = spec.system_prompt

    assert "inspect_app_dictionary" in p
    assert "dump_ax_tree" in p
    # Universal retry budget — both error codes must be referenced so
    # the agent recognises them.
    assert "-2741" in p
    assert "-1743" in p
    # The retry budget ends in the agent's own UI route.
    assert "After 2 failures on the same goal" in p
    # Property-driven scripting: the agent must read the probe's property
    # list and recognise the "found class name" diagnosis instead of
    # reusing another app's field names (the Outlook failure mode).
    assert "properties" in p
    assert "found class name" in p


def test_macos_agent_and_skills_hidden_off_macos(
    isolated_app_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Linux / Windows the macOS agent + skills must not surface even if
    the on-disk seed somehow exists."""
    _force_platform(monkeypatch, "macos")

    from backend import agent_library

    agent_library.seed_defaults()
    assert agent_library.get_agent("macos-desktop-agent") is not None
    assert agent_library.get_skill("macos-applescript") is not None

    # Now flip the gate; the seed files still exist on disk, but
    # ``list_agents`` / ``get_agent`` should hide them.
    _force_platform(monkeypatch, "linux")

    assert agent_library.get_agent("macos-desktop-agent") is None
    assert agent_library.get_skill("macos-applescript") is None
    names = [a.name for a in agent_library.list_agents()]
    assert "macos-desktop-agent" not in names
    skill_names = [s.name for s in agent_library.list_skills()]
    assert "macos-applescript" not in skill_names
    assert "macos-desktop" not in skill_names


def test_macos_agent_in_builtin_versions() -> None:
    """The version table is the source of truth for ``is_builtin_agent``
    — without the entry, users could delete a built-in via the REST API
    and break the orchestrator's ladder."""
    from backend import agent_library

    assert agent_library.is_builtin_agent("macos-desktop-agent")
    assert not agent_library.is_builtin_agent("macos-applescript-agent")
    assert agent_library.is_builtin_skill("macos-applescript")
