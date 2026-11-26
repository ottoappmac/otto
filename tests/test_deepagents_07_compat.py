"""deepagents 0.7: the new ``delete`` tool is guarded, otto keeps its 0.5 defaults, large-result grep works."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.action_policy.catalog import _BARE_TOOLS
from backend.session_manager import _make_backend
from deep_agent.deepagents_compat import with_legacy_defaults


@pytest.fixture
def project_env(tmp_path: Path):
    files_dir = tmp_path / "files"
    (files_dir / "links").mkdir(parents=True)
    project = tmp_path / "proj"
    (project / "sub").mkdir(parents=True)
    (project / "a.txt").write_text("one\n")
    (project / "sub" / "b.txt").write_text("two\n")
    (project / "AGENTS.md").write_text("rules\n")
    (files_dir / "links" / "proj").symlink_to(project)
    return files_dir, project.resolve()


def _backend(files_dir: Path, project: Path, mode: str = "agent", grant_mode: str = "write"):
    return _make_backend(
        files_dir, project_root=project, grant_mode=grant_mode,
        grant_roots=[project], mode_getter=lambda: mode,
    )


# ── delete ─────────────────────────────────────────────────────────────────


def test_delete_removes_a_file_in_agent_mode(project_env):
    files_dir, project = project_env
    result = _backend(files_dir, project).delete("/links/proj/a.txt")
    assert not result.error
    assert not (project / "a.txt").exists()


@pytest.mark.parametrize("mode", ["plan", "ask"])
def test_delete_is_blocked_in_read_only_modes(project_env, mode):
    files_dir, project = project_env
    result = _backend(files_dir, project, mode=mode).delete("/links/proj/a.txt")
    assert result.error and "read-only" in result.error
    assert (project / "a.txt").exists()


def test_delete_is_blocked_on_a_read_only_grant(project_env):
    files_dir, project = project_env
    result = _backend(files_dir, project, grant_mode="read").delete("/links/proj/a.txt")
    assert result.error
    assert (project / "a.txt").exists()


def test_delete_refuses_protected_paths(project_env):
    files_dir, project = project_env
    result = _backend(files_dir, project).delete("/links/proj/AGENTS.md")
    assert result.error
    assert (project / "AGENTS.md").exists()


def test_delete_refuses_folders(project_env):
    files_dir, project = project_env
    backend = _backend(files_dir, project)
    for target in ("/links/proj/sub", "/links/proj"):
        result = backend.delete(target)
        assert result.error and "folder" in result.error
    assert (project / "sub" / "b.txt").exists()


def test_delete_goes_through_the_action_policy():
    assert _BARE_TOOLS["delete"] == "workspace_write"


# ── 0.5 defaults ──────────────────────────────────────────────────────────


def test_legacy_defaults_restore_todos_and_base_prompt():
    from langchain.agents.middleware import TodoListMiddleware

    prompt, middleware = with_legacy_defaults("OTTO", [])
    assert prompt.startswith("OTTO\n\n") and "You are a deep agent" in prompt
    assert isinstance(middleware[0], TodoListMiddleware)


def test_legacy_defaults_are_idempotent():
    from langchain.agents.middleware import TodoListMiddleware

    prompt, middleware = with_legacy_defaults("OTTO", [])
    again_prompt, again_mw = with_legacy_defaults(prompt, middleware)
    assert again_prompt == prompt
    assert sum(isinstance(m, TodoListMiddleware) for m in again_mw) == 1


def test_main_agent_is_offered_write_todos():
    import tempfile

    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    class Fake(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self

    seen: dict = {}

    class Rec(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            seen["tools"] = {getattr(t, "name", None) or t.get("name") for t in request.tools}
            return handler(request)

    prompt, middleware = with_legacy_defaults("OTTO", [Rec()])
    agent = create_deep_agent(
        model=Fake(messages=iter([AIMessage(content="hi")])), tools=[], system_prompt=prompt,
        middleware=middleware, backend=FilesystemBackend(root_dir=tempfile.mkdtemp(), virtual_mode=True),
    )
    agent.invoke({"messages": [("user", "x")]})
    assert {"write_todos", "read_file", "write_file", "delete"} <= seen["tools"]


# ── large-result grep ─────────────────────────────────────────────────────


def test_large_results_grep_finds_offloaded_files(tmp_path):
    from deepagents.backends import FilesystemBackend

    from tools.large_results_grep import _list_candidate_paths, _resolve_backend

    (tmp_path / "large_tool_results").mkdir()
    (tmp_path / "large_tool_results" / "call_1").write_text('{"id": 1}')
    backend = _resolve_backend(FilesystemBackend(root_dir=tmp_path, virtual_mode=True), runtime=None)
    paths = _list_candidate_paths(backend, "/large_tool_results", None)
    assert any(p.endswith("call_1") for p in paths)


def test_large_results_grep_defaults_to_a_state_backend():
    from deepagents.backends import StateBackend

    from tools.large_results_grep import _resolve_backend

    assert isinstance(_resolve_backend(None, runtime=None), StateBackend)
    assert isinstance(_resolve_backend(StateBackend, runtime=None), StateBackend)
