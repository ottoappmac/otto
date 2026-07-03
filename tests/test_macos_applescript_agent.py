"""Hermetic tests for the macos-applescript-agent built-in seeding.

The new built-in agent + skill must:

* Materialise on macOS hosts with ``tools=["macos-osascript"]`` and
  ``skills=["macos-applescript"]`` so the orchestrator's capability ladder
  can address it via ``task(subagent_type="macos-applescript-agent", ...)``.
* Be hidden from ``list_agents`` / ``get_agent`` / ``list_skills`` /
  ``get_skill`` on non-macOS hosts, the same gate that applies to the
  pre-existing ``macos-desktop-agent``.

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


def test_seed_creates_macos_applescript_agent_on_macos(
    isolated_app_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_platform(monkeypatch, "macos")

    from backend import agent_library

    agent_library.seed_defaults()

    spec = agent_library.get_agent("macos-applescript-agent")
    assert spec is not None
    assert spec.tools == ["macos-osascript"]
    assert spec.skills == ["macos-applescript"]
    assert spec.builtin is True
    # The agent's prompt must reference the deterministic AppleScript path
    # and the structured fallback contract — both are load-bearing for
    # orchestrator routing.
    assert "AppleScript" in spec.system_prompt
    assert "FALLBACK:" in spec.system_prompt
    assert "macos-desktop-agent" in spec.system_prompt


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
    # the trial-and-error loop seen in the Slack DM trace.
    assert "FALLBACK" in skill.content
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


def test_seed_macos_applescript_agent_prompt_has_decision_tree(
    isolated_app_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Agent prompt must encode the introspect-then-act decision tree
    (probe → branch → dictionary OR ax_tree) and the universal 2-failure
    budget so a 4B-class model has a flat checklist to follow."""
    _force_platform(monkeypatch, "macos")

    from backend import agent_library

    agent_library.seed_defaults()

    spec = agent_library.get_agent("macos-applescript-agent")
    assert spec is not None
    p = spec.system_prompt

    assert "inspect_app_dictionary" in p
    assert "dump_ax_tree" in p
    # Universal retry budget — both error codes must be referenced so
    # the agent recognises them.
    assert "-2741" in p
    assert "-1743" in p
    # Fallback contract shape (so the orchestrator can route on it).
    assert "FALLBACK: macos-desktop-agent" in p
    # Property-driven scripting: the agent must read the probe's property
    # list and recognise the "found class name" diagnosis instead of
    # reusing another app's field names (the Outlook failure mode).
    assert "properties" in p
    assert "found class name" in p


def test_macos_applescript_pair_hidden_off_macos(
    isolated_app_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Linux / Windows the AppleScript agent + skill must not surface
    even if the on-disk seed somehow exists — same gating as
    ``macos-desktop-agent``."""
    _force_platform(monkeypatch, "macos")

    from backend import agent_library

    agent_library.seed_defaults()
    assert agent_library.get_agent("macos-applescript-agent") is not None
    assert agent_library.get_skill("macos-applescript") is not None

    # Now flip the gate; the seed files still exist on disk, but
    # ``list_agents`` / ``get_agent`` should hide them.
    _force_platform(monkeypatch, "linux")

    assert agent_library.get_agent("macos-applescript-agent") is None
    assert agent_library.get_skill("macos-applescript") is None
    names = [a.name for a in agent_library.list_agents()]
    assert "macos-applescript-agent" not in names
    assert "macos-desktop-agent" not in names
    skill_names = [s.name for s in agent_library.list_skills()]
    assert "macos-applescript" not in skill_names
    assert "macos-desktop" not in skill_names


def test_macos_applescript_agent_in_builtin_versions() -> None:
    """The version table is the source of truth for ``is_builtin_agent``
    — without the entry, users could delete a built-in via the REST API
    and break the orchestrator's ladder."""
    from backend import agent_library

    assert agent_library.is_builtin_agent("macos-applescript-agent")
    assert agent_library.is_builtin_skill("macos-applescript")
