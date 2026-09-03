"""Hermetic tests for the coding-agent library subagent.

Locks in:

* ``coding-agent`` is registered for ``task()`` even with empty MCP tools.
* Other empty-MCP agents (e.g. ``schedule-builder-agent``) stay skipped.
* Seeded prompt/skill stay light: no ``write_todos``, ``edit_file`` + mapped path.
* Lite orchestrator routing lists ``coding-agent`` with a repo-work trigger.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.schemas import AgentSpec, SessionWorkspace
from backend.session_manager import (
    _FILESYSTEM_SUBAGENT_NAMES,
    _build_subagents_from_library,
    _mapped_project_prompt_block,
)
from deep_agent.prompt import _SUBAGENT_USE_WHEN, build_orchestrator_prompt


class _FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"Stub tool {name}"


@pytest.fixture
def isolated_app_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from backend import agent_library, config

    monkeypatch.setattr(config, "get_app_data_dir", lambda: tmp_path)
    monkeypatch.setattr(agent_library, "get_app_data_dir", lambda: tmp_path)
    return tmp_path


class _EmptyMcpMgr:
    connections: dict = {}


def _spec(name: str, tools: list[str]) -> AgentSpec:
    return AgentSpec(
        name=name,
        description=f"{name} description",
        tools=tools,
    )


def test_filesystem_subagent_allowlist_is_coding_only():
    assert _FILESYSTEM_SUBAGENT_NAMES == frozenset({"coding-agent"})


def test_build_subagents_registers_coding_agent_without_mcp(monkeypatch):
    specs = [
        _spec("coding-agent", []),
        _spec("schedule-builder-agent", []),
        _spec("browser-agent", ["playwright-mcp"]),
    ]
    monkeypatch.setattr("backend.agent_library.list_agents", lambda: specs)

    subagents, claimed = _build_subagents_from_library(
        _EmptyMcpMgr(), [], model=None, backend=None,
    )
    names = [s["name"] for s in (subagents or [])]
    assert "coding-agent" in names
    assert "schedule-builder-agent" not in names
    assert "browser-agent" not in names
    assert claimed == set()


def test_build_subagents_does_not_claim_mcp_ids_for_coding_agent(monkeypatch):
    mcp = SimpleNamespace(connections={
        "playwright-mcp": SimpleNamespace(connected=True, tools=[]),
    })
    specs = [_spec("coding-agent", [])]
    monkeypatch.setattr("backend.agent_library.list_agents", lambda: specs)

    subagents, claimed = _build_subagents_from_library(
        mcp, [], model=None, backend=None,
    )
    assert [s["name"] for s in (subagents or [])] == ["coding-agent"]
    assert claimed == set()


def test_mapped_project_prompt_block_includes_virtual_path():
    ws = SessionWorkspace(
        host_path="/Users/me/proj",
        virtual_path="/links/proj",
        name="proj",
    )
    block = _mapped_project_prompt_block(ws)
    assert "/links/proj" in block
    assert "$PROJECT_ROOT" in block
    assert "edit_file" in block


def test_mapped_project_prompt_block_unmapped_only_when_requested():
    assert _mapped_project_prompt_block(None) == ""
    missing = _mapped_project_prompt_block(None, include_unmapped=True)
    assert "No project folder is mapped" in missing


def test_seeded_coding_prompt_and_skill_are_light(isolated_app_data: Path):
    from backend import agent_library

    agent_library.seed_defaults()

    spec = agent_library.get_agent("coding-agent")
    assert spec is not None
    assert spec.tools == []
    assert spec.skills == ["coding"]
    assert "Do not use `write_todos`" in spec.system_prompt
    assert "edit_file" in spec.system_prompt
    assert "/links/" in spec.system_prompt
    assert "general-purpose" in spec.description

    skill = agent_library.get_skill("coding")
    assert skill is not None
    assert "Do not use `write_todos`" in skill.content
    assert "1. `write_todos`" not in skill.content
    assert "edit_file" in skill.content
    assert "/links/" in skill.content


def test_subagent_use_when_routes_coding_agent():
    assert "coding-agent" in _SUBAGENT_USE_WHEN
    trigger = _SUBAGENT_USE_WHEN["coding-agent"]
    assert "edit_file" in trigger
    assert "general-purpose" in trigger


def test_lite_prompt_lists_coding_agent_trigger():
    p = build_orchestrator_prompt(
        [_FakeTool("write_file")],
        subagents=[{"name": "coding-agent"}],
        lite=True,
    )
    block_start = p.index("<subagents>")
    block_end = p.index("</subagents>")
    block = p[block_start:block_end]
    assert "coding-agent" in block
    assert "edit_file" in block
    assert "general-purpose" in block
