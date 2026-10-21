"""Plan, Debug and Agent modes.

Plan is read-only in the project by construction. These tests check the
guards, not the prompt: they must hold when the model ignores its
instructions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from backend.session_mode import (
    DEBUG_MARKER,
    DEEP_MODES,
    count_debug_markers,
    effective_mode,
    normalize_mode,
)
from backend.session_manager import _make_backend


def _decide(name, args=None, *, server_id=None, session_mode="agent", attended=True):
    from backend.action_policy.catalog import balanced_categories, classify_tool
    from backend.action_policy.gate import decide

    return decide(
        classify_tool(name, server_id=server_id, args=args or {}),
        categories=balanced_categories(),
        macos_automation_enabled=True,
        attended=attended,
        allowlist=frozenset(),
        session_grants=frozenset(),
        invocation_grants=frozenset(),
        tainted=False,
        rate_count=0,
        rate_limit=0,
        session_mode=session_mode,
    )


@pytest.fixture
def project_env(tmp_path: Path):
    files_dir = tmp_path / "files"
    (files_dir / "links").mkdir(parents=True)
    (files_dir / "output").mkdir()
    project = tmp_path / "proj"
    project.mkdir()
    (project / "a.txt").write_text("one\n")
    (files_dir / "links" / "proj").symlink_to(project)
    return files_dir, project.resolve()


def _backend(files_dir: Path, project: Path, mode: list[str]):
    return _make_backend(
        files_dir,
        project_root=project,
        grant_mode="write",
        grant_roots=[project],
        mode_getter=lambda: mode[0],
    )


def test_mode_helpers():
    assert normalize_mode("PLAN") == "plan"
    assert normalize_mode("nonsense") == "agent"
    assert normalize_mode(None) == "agent"
    assert effective_mode("plan", unattended=True) == "agent"
    assert normalize_mode("ASK") == "ask"
    assert effective_mode("ask", unattended=True) == "agent"
    assert effective_mode("debug") == "debug"
    assert DEEP_MODES == {"plan", "debug"}


def test_count_debug_markers():
    text = f"x = 1\nprint(x)  # {DEBUG_MARKER}\nprint(2)  # {DEBUG_MARKER}\n"
    assert count_debug_markers(text) == 2
    assert count_debug_markers("") == 0


def test_plan_blocks_project_writes_but_not_session_output(project_env):
    files_dir, project = project_env
    mode = ["plan"]
    backend = _backend(files_dir, project, mode)

    blocked = backend.write("/links/proj/new.txt", "hi")
    assert blocked.error and "Plan mode" in blocked.error
    assert not (project / "new.txt").exists()

    edit = backend.edit("/links/proj/a.txt", "one", "two")
    assert edit.error and "Plan mode" in edit.error
    assert (project / "a.txt").read_text() == "one\n"

    ok = backend.write("/output/plan.md", "# plan")
    assert not ok.error
    assert (files_dir / "output" / "plan.md").read_text() == "# plan"


def test_ask_blocks_every_write(project_env):
    files_dir, project = project_env
    backend = _backend(files_dir, project, ["ask"])

    blocked = backend.write("/links/proj/new.txt", "hi")
    assert blocked.error and "Ask mode" in blocked.error
    assert not (project / "new.txt").exists()

    edit = backend.edit("/links/proj/a.txt", "one", "two")
    assert edit.error and "Ask mode" in edit.error
    assert (project / "a.txt").read_text() == "one\n"

    output = backend.write("/output/notes.md", "hi")
    assert output.error and "Ask mode" in output.error
    assert not (files_dir / "output" / "notes.md").exists()


def test_ask_refuses_shell_when_sandbox_is_off(project_env, monkeypatch):
    files_dir, project = project_env
    monkeypatch.setattr(
        "backend.exec_sandbox.sandbox_settings", lambda: (False, False, [], []),
    )
    result = _backend(files_dir, project, ["ask"]).execute("touch from_ask.txt")
    assert result.exit_code == 1
    assert "Ask mode" in result.output
    assert not (project / "from_ask.txt").exists()


def test_ask_shell_is_given_no_write_paths(project_outside_temp, monkeypatch):
    files_dir, project = project_outside_temp
    seen: dict = {}

    class _Result:
        error = None
        stdout = "ok"
        stderr = ""
        returncode = 0

    def fake(command, **kwargs):
        seen["write_paths"] = list(kwargs.get("write_paths") or [])
        return _Result()

    monkeypatch.setattr("backend.exec_sandbox.sandbox_settings", lambda: (True, False, [], []))
    monkeypatch.setattr("backend.exec_sandbox.run_shell", fake)
    result = _backend(files_dir, project, ["ask"]).execute("ls")
    assert result.exit_code == 0, result.output
    assert seen["write_paths"] == []
    assert not (project / "from_ask.txt").exists()


def test_switching_mode_applies_to_the_next_call(project_env):
    files_dir, project = project_env
    mode = ["plan"]
    backend = _backend(files_dir, project, mode)
    assert backend.write("/links/proj/x.txt", "1").error
    mode[0] = "agent"
    assert not backend.write("/links/proj/x.txt", "1").error
    assert (project / "x.txt").read_text() == "1"
    mode[0] = "plan"
    assert backend.write("/links/proj/y.txt", "1").error
    assert not (project / "y.txt").exists()


def test_debug_mode_can_write(project_env):
    files_dir, project = project_env
    backend = _backend(files_dir, project, ["debug"])
    assert not backend.write("/links/proj/d.txt", "1").error


def test_unknown_session_is_plan_and_a_session_being_created_is_agent(project_env, tmp_path, monkeypatch):
    from backend.session_manager import SessionManager

    monkeypatch.setattr("backend.session_manager._sessions_dir", lambda: tmp_path)
    mgr = SessionManager()
    files_dir, project = project_env

    assert mgr._mode_for("missing") == "plan"
    blocked = _make_backend(
        files_dir,
        project_root=project,
        grant_roots=[project],
        mode_getter=lambda: mgr._mode_for("missing"),
    )
    assert blocked.write("/links/proj/z.txt", "1").error
    assert not (project / "z.txt").exists()

    mgr._opening.add("creating")
    assert mgr._mode_for("creating") == "agent"
    allowed = _make_backend(
        files_dir,
        project_root=project,
        grant_roots=[project],
        mode_getter=lambda: mgr._mode_for("creating"),
    )
    assert not allowed.write("/links/proj/ok.txt", "1").error
    assert (project / "ok.txt").read_text() == "1"

    (tmp_path / "resumed.json").write_text('{"mode": "plan"}', encoding="utf-8")
    mgr._opening.add("resumed")
    assert mgr._mode_for("resumed") == "plan"

    (tmp_path / "sched.json").write_text(
        '{"mode": "plan", "trigger_source": "schedule"}',
        encoding="utf-8",
    )
    mgr._opening.add("sched")
    assert mgr._mode_for("sched") == "agent"

    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    mgr._opening.add("broken")
    assert mgr._mode_for("broken") == "plan"

    class _Live:
        mode = "ask"
        trigger_source = None

    mgr._active["live"] = _Live()  # type: ignore[assignment]
    assert mgr._mode_for("live") == "ask"

    class _Unattended:
        mode = "plan"
        trigger_source = "trigger"

    mgr._active["unattended"] = _Unattended()  # type: ignore[assignment]
    assert mgr._mode_for("unattended") == "agent"


def test_set_mode_rejects_a_running_turn(monkeypatch):
    import asyncio

    from backend.session_manager import SessionManager

    mgr = SessionManager()

    class _Live:
        mode = "agent"
        trigger_source = None

    mgr._active["s"] = _Live()  # type: ignore[assignment]

    async def go():
        task = asyncio.create_task(asyncio.sleep(30))
        monkeypatch.setattr("backend.state.running_tasks", {"s": task})
        try:
            with pytest.raises(RuntimeError, match="while a run is in progress"):
                await mgr.set_mode("s", "plan")
        finally:
            task.cancel()

    asyncio.run(go())
    assert mgr._mode_for("s") == "agent"


def test_plan_fails_closed_when_mode_cannot_be_read(project_env):
    files_dir, project = project_env

    def boom() -> str:
        raise RuntimeError("no session")

    backend = _make_backend(
        files_dir, project_root=project, grant_roots=[project], mode_getter=boom,
    )
    assert backend.write("/links/proj/z.txt", "1").error
    assert not (project / "z.txt").exists()


def test_plan_refuses_shell_when_sandbox_is_off(project_env, monkeypatch):
    files_dir, project = project_env
    monkeypatch.setattr(
        "backend.exec_sandbox.sandbox_settings", lambda: (False, False, [], []),
    )
    backend = _backend(files_dir, project, ["plan"])
    result = backend.execute("touch from_shell.txt")
    assert result.exit_code == 1
    assert "sandbox" in result.output.lower()
    assert not (project / "from_shell.txt").exists()


@pytest.fixture
def project_outside_temp():
    """A project the sandbox does not treat as a temp folder."""
    import shutil
    import uuid

    base = Path(__file__).resolve().parent / f".tmp_mode_{uuid.uuid4().hex[:8]}"
    files_dir = base / "files"
    (files_dir / "links").mkdir(parents=True)
    (files_dir / "output").mkdir()
    project = base / "proj"
    project.mkdir()
    (files_dir / "links" / "proj").symlink_to(project)
    try:
        yield files_dir, project.resolve()
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_plan_shell_refuses_a_project_in_a_temp_folder(project_env, monkeypatch):
    # tmp_path is under /var/folders, which the profile always lets commands write.
    files_dir, project = project_env
    monkeypatch.setattr(
        "backend.exec_sandbox.sandbox_settings", lambda: (True, False, [], []),
    )
    backend = _backend(files_dir, project, ["plan"])
    result = backend.execute("touch from_plan.txt")
    assert result.exit_code == 1
    assert "read-only" in result.output
    assert not (project / "from_plan.txt").exists()


def test_extra_write_path_in_settings_also_defeats_the_shell_guard(project_env, monkeypatch):
    files_dir, project = project_env
    monkeypatch.setattr(
        "backend.exec_sandbox.sandbox_settings",
        lambda: (True, False, [], [str(project)]),
    )
    result = _backend(files_dir, project, ["plan"]).execute("touch x.txt")
    assert result.exit_code == 1
    assert not (project / "x.txt").exists()


@pytest.mark.skipif(
    sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").exists(),
    reason="Seatbelt",
)
def test_plan_shell_cannot_write_the_project_but_agent_can(project_outside_temp, monkeypatch):
    files_dir, project = project_outside_temp
    monkeypatch.setattr(
        "backend.exec_sandbox.sandbox_settings", lambda: (True, False, [], []),
    )
    mode = ["plan"]
    backend = _backend(files_dir, project, mode)

    backend.execute("touch from_plan.txt")
    assert not (project / "from_plan.txt").exists()

    mode[0] = "agent"
    result = backend.execute("touch from_agent.txt")
    assert result.exit_code == 0, result.output
    assert (project / "from_agent.txt").exists()


def test_plan_denies_actions_that_change_things_outside_the_project():
    send = _decide(
        "send_message", {"to": "a@b.c", "subject": "hi"},
        server_id="macos-mail", session_mode="plan",
    )
    assert send.effect == "deny"
    assert "Plan mode" in send.reason

    schedule = _decide("create_schedule", {"id": "x"}, session_mode="plan")
    assert schedule.effect == "deny"


def test_ask_denies_actions_that_change_things():
    send = _decide(
        "send_message", {"to": "a@b.c", "subject": "hi"},
        server_id="macos-mail", session_mode="ask",
    )
    assert send.effect == "deny"
    assert "Ask mode" in send.reason


def test_ask_prompt_addendum_is_added_once():
    from langchain_core.messages import SystemMessage

    from middleware.session_mode import SessionModeMiddleware

    class _Req:
        def __init__(self, text: str) -> None:
            self.system_message = SystemMessage(content=text)

        def override(self, *, system_message):
            clone = _Req("")
            clone.system_message = system_message
            return clone

    once = SessionModeMiddleware(lambda: "ask")._with_addendum(_Req("base"))
    assert "ASK mode" in once.system_message.content
    assert "cannot change anything" in once.system_message.content


def test_plan_leaves_shell_on_its_normal_rule():
    # Shell is guarded by the sandbox in Plan, not by the policy.
    plan = _decide("execute", {"command": "ls"}, session_mode="plan")
    agent = _decide("execute", {"command": "ls"}, session_mode="agent")
    assert plan.effect == agent.effect


def test_agent_and_debug_do_not_tighten_the_policy():
    for mode in ("agent", "debug"):
        decision = _decide(
            "send_message", {"to": "a@b.c", "subject": "hi"},
            server_id="macos-mail", session_mode=mode,
        )
        assert decision.effect != "deny" or "Plan mode" not in decision.reason


def test_plan_prompt_addendum_is_added_once():
    from langchain_core.messages import SystemMessage

    from middleware.session_mode import SessionModeMiddleware

    class _Req:
        def __init__(self, text: str) -> None:
            self.system_message = SystemMessage(content=text)

        def override(self, *, system_message):
            clone = _Req("")
            clone.system_message = system_message
            return clone

    mw = SessionModeMiddleware(lambda: "plan")
    once = mw._with_addendum(_Req("base"))
    assert "PLAN mode" in once.system_message.content
    assert "creates the file the first time and replaces it" in once.system_message.content
    twice = mw._with_addendum(once)
    assert twice.system_message.content.count("<session_mode>") == 1

    agent = SessionModeMiddleware(lambda: "agent")._with_addendum(_Req("base"))
    assert agent.system_message.content == "base"


_PLAN = """# Goal

Add a logout button.

## Files

- `app/src/pages/CodePage.tsx`

## Steps

1. Add the button.
2. Wire it to sign-out.

## Verify

Click logout and confirm the session ends.
"""

_REVISED = _PLAN + "\n## Risks\n\nThe button can be clicked twice.\n"


def test_plan_file_is_created_then_replaced(tmp_path: Path):
    from backend.session_mode import plan_markdown_path, store_plan_markdown

    files = tmp_path / "files"
    path = plan_markdown_path(files)
    assert store_plan_markdown(files, _PLAN) is True
    assert path.read_text(encoding="utf-8") == _PLAN
    assert store_plan_markdown(files, _PLAN) is False
    assert store_plan_markdown(files, f"```markdown\n{_REVISED}```") is True
    assert "clicked twice" in path.read_text(encoding="utf-8")
    assert not path.read_text(encoding="utf-8").lstrip().startswith("```")


def test_plan_file_ignores_questions_and_short_notes(tmp_path: Path):
    from backend.session_mode import plan_markdown_path, store_plan_markdown

    files = tmp_path / "files"
    path = plan_markdown_path(files)
    assert store_plan_markdown(files, "Which page should the button go on?") is False
    assert not path.exists()

    store_plan_markdown(files, _PLAN)
    before = path.read_text(encoding="utf-8")
    assert store_plan_markdown(files, "Saved the plan.") is False
    assert store_plan_markdown(files, "Should the button live in the header?") is False
    assert path.read_text(encoding="utf-8") == before


def test_finished_plan_turn_writes_the_file(tmp_path: Path, monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage

    from backend.session_manager import SessionManager

    files = tmp_path / "files"
    files.mkdir()
    monkeypatch.setattr("backend.session_manager._session_files_dir", lambda _sid: files)

    class _Session:
        mode = "plan"
        trigger_source = None

    class _State:
        values = {
            "messages": [
                HumanMessage(content="plan a logout button"),
                AIMessage(content="Reading the page.", tool_calls=[{"name": "read_file", "args": {}, "id": "1"}]),
                AIMessage(content=_PLAN),
            ]
        }

    manager = SessionManager.__new__(SessionManager)
    manager._active = {"s1": _Session()}
    manager._persist_plan("s1", _State())
    assert (files / "output" / "plan.md").read_text(encoding="utf-8").startswith("# Goal")

    manager._active["s1"].mode = "agent"
    (files / "output" / "plan.md").write_text("keep\n", encoding="utf-8")
    manager._persist_plan("s1", _State())
    assert (files / "output" / "plan.md").read_text(encoding="utf-8") == "keep\n"


def test_plan_file_keeps_this_turn_when_the_reply_is_not_a_plan(tmp_path: Path, monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage

    from backend.session_manager import SessionManager

    files = tmp_path / "files"
    (files / "output").mkdir(parents=True)
    revised = "# Goal\n\nRevised this turn.\n"
    (files / "output" / "plan.md").write_text(revised, encoding="utf-8")
    monkeypatch.setattr("backend.session_manager._session_files_dir", lambda _sid: files)

    class _Session:
        mode = "plan"
        trigger_source = None

    class _State:
        values = {
            "messages": [
                HumanMessage(content="plan a logout button"),
                AIMessage(content=_PLAN),
                HumanMessage(content="add a risk"),
                AIMessage(content="Updating the file.", tool_calls=[{"name": "read_file", "args": {}, "id": "2"}]),
            ]
        }

    manager = SessionManager.__new__(SessionManager)
    manager._active = {"s1": _Session()}
    manager._persist_plan("s1", _State())
    assert (files / "output" / "plan.md").read_text(encoding="utf-8") == revised


def test_list_changes_counts_leftover_debug_markers(tmp_path: Path, monkeypatch):
    from backend.schemas import SessionWorkspace
    from backend import workspace_git

    project = tmp_path / "p"
    project.mkdir()
    (project / "m.py").write_text(f"print(1)  # {DEBUG_MARKER}\nprint(2)\n")
    workspace = SessionWorkspace(
        host_path=str(project), name="p", virtual_path="/links/p", grant_id="g", mode="write",
    )
    monkeypatch.setattr(
        workspace_git,
        "_list_changes",
        lambda *_a, **_k: {"git": False, "files": [{"path": "m.py"}], "message": ""},
    )
    result = workspace_git.list_changes(workspace, "sid")
    assert result["debug_markers"] == 1
    assert result["files"][0]["debug_markers"] == 1
