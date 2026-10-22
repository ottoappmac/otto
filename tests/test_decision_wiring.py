"""jev is on the shared tool list and in the prompt only when enabled.

Also covers how an open chat follows the setting: it is rebuilt at the start of
its next user turn, never while a turn is running.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from backend.config import AppConfig
from backend.privacy_lock import PrivacyLockActive, enforce_decision_endpoint
from backend.session_manager import (
    SessionManager,
    _bind_decision_flags,
    _build_standard_tools,
    _insert_repeated_thought_guard,
    _insert_safety_guard,
)
from middleware.repeated_thought_guard import RepeatedThoughtGuardMiddleware
from middleware.safety_guard import SafetyGuardMiddleware
from deep_agent.prompt import build_orchestrator_prompt


class _Tool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = "stub"


def test_standard_tools_omit_jev_when_disabled(monkeypatch) -> None:
    cfg = AppConfig()
    cfg.decision.enabled = False
    monkeypatch.setattr("backend.config.AppConfig.load", lambda: cfg)
    names = [tool.name for tool in _build_standard_tools()]
    assert "jev" not in names


def test_standard_tools_include_jev_when_enabled(monkeypatch) -> None:
    cfg = AppConfig()
    cfg.decision.enabled = True
    monkeypatch.setattr("backend.config.AppConfig.load", lambda: cfg)
    names = [tool.name for tool in _build_standard_tools(session_id="wire")]
    assert "jev" in names


def test_standard_tools_follow_the_argument_not_the_saved_config(monkeypatch) -> None:
    """The caller's value wins, so the tool and the prompt share one snapshot."""
    cfg = AppConfig()
    cfg.decision.enabled = False
    monkeypatch.setattr("backend.config.AppConfig.load", lambda: cfg)
    on = [tool.name for tool in _build_standard_tools(session_id="wire", decide_enabled=True)]
    assert "jev" in on

    cfg.decision.enabled = True
    off = [tool.name for tool in _build_standard_tools(session_id="wire", decide_enabled=False)]
    assert "jev" not in off


def test_prompt_mentions_jev_only_when_the_tool_is_bound() -> None:
    full = build_orchestrator_prompt([_Tool("jev")], [])
    assert "<jev>" in full
    assert "ask_user" in full
    assert "<jev>" not in build_orchestrator_prompt([_Tool("web_research")], [])
    lite = build_orchestrator_prompt([_Tool("jev")], [], lite=True)
    assert "call `jev`" in lite
    assert "call `jev`" not in build_orchestrator_prompt([_Tool("web_research")], [], lite=True)


def test_prompt_flag_adds_the_guidance_with_an_empty_tool_list() -> None:
    assert "<jev>" in build_orchestrator_prompt([], [], decide=True)
    assert "call `jev`" in build_orchestrator_prompt([], [], lite=True, decide=True)


def test_prompt_flag_off_removes_the_guidance_even_if_the_tool_is_listed() -> None:
    assert "<jev>" not in build_orchestrator_prompt([_Tool("jev")], [], decide=False)
    assert "call `jev`" not in build_orchestrator_prompt(
        [_Tool("jev")], [], lite=True, decide=False,
    )


def test_prompt_flag_does_not_switch_on_unrelated_sections() -> None:
    """Passing the setting must not change anything but the jev block."""
    off = build_orchestrator_prompt([], [], decide=False)
    on = build_orchestrator_prompt([], [], decide=True)
    assert on != off
    assert on.replace(on[on.index("<jev>"):on.index("</jev>") + len("</jev>")] + "\n\n", "") == off


def test_privacy_lock_refuses_a_remote_endpoint(monkeypatch) -> None:
    monkeypatch.setattr("backend.privacy_lock.audit_event", lambda *a, **k: None)
    cfg = AppConfig()
    cfg.privacy.enabled = True
    enforce_decision_endpoint("", cfg)
    enforce_decision_endpoint("http://127.0.0.1:52416", cfg)
    try:
        enforce_decision_endpoint("https://api.typesafe.ai", cfg)
    except PrivacyLockActive as exc:
        assert "off this machine" in str(exc)
    else:
        raise AssertionError("remote endpoint was allowed under the privacy lock")


# ── Open chats follow the setting ────────────────────────────────────────────


def _manager(monkeypatch, *, enabled: bool, guard: bool = False):
    """A real SessionManager whose rebuild is recorded instead of run."""
    cfg = AppConfig()
    cfg.decision.enabled = enabled
    cfg.decision.safety_guard_enabled = guard

    async def _aload():
        return cfg

    monkeypatch.setattr("backend.session_manager.AppConfig.aload", _aload)
    mgr = SessionManager()
    rebuilt: list[object] = []

    async def _rebuild(session, config):
        rebuilt.append(session)
        _bind_decision_flags(session, config)

    monkeypatch.setattr(mgr, "_rebuild_one_session", _rebuild)
    return mgr, rebuilt


def _session(*, bound: bool, guard_bound: bool = False, status: str = "completed"):
    return SimpleNamespace(
        id="s1", decide_bound=bound, safety_bound=guard_bound, status=status,
    )


@pytest.mark.asyncio
async def test_enabling_rebuilds_an_open_chat_once(monkeypatch) -> None:
    mgr, rebuilt = _manager(monkeypatch, enabled=True)
    session = _session(bound=False)
    await mgr._reconcile_decide(session)
    assert rebuilt == [session]
    assert session.decide_bound is True
    await mgr._reconcile_decide(session)
    assert rebuilt == [session]


@pytest.mark.asyncio
async def test_changing_the_model_rebuilds_an_open_chat_once(monkeypatch) -> None:
    mgr, rebuilt = _manager(monkeypatch, enabled=False)
    session = _session(bound=False)
    session.model_bound = ("omlx", "old-model")
    await mgr._reconcile_decide(session)
    assert rebuilt == [session]
    # The rebuild re-stamps the signature, so the next turn leaves it alone.
    assert session.model_bound != ("omlx", "old-model")
    await mgr._reconcile_decide(session)
    assert rebuilt == [session]


@pytest.mark.asyncio
async def test_disabling_rebuilds_an_open_chat_once(monkeypatch) -> None:
    mgr, rebuilt = _manager(monkeypatch, enabled=False)
    session = _session(bound=True)
    await mgr._reconcile_decide(session)
    assert rebuilt == [session]
    assert session.decide_bound is False
    await mgr._reconcile_decide(session)
    assert rebuilt == [session]


@pytest.mark.asyncio
async def test_a_chat_that_already_matches_is_not_rebuilt(monkeypatch) -> None:
    for enabled in (True, False):
        mgr, rebuilt = _manager(monkeypatch, enabled=enabled)
        await mgr._reconcile_decide(_session(bound=enabled))
        assert rebuilt == []


@pytest.mark.asyncio
async def test_a_running_chat_is_left_alone_until_its_next_turn(monkeypatch) -> None:
    mgr, rebuilt = _manager(monkeypatch, enabled=True)
    session = _session(bound=False, status="running")
    await mgr._reconcile_decide(session)
    assert rebuilt == []
    assert session.decide_bound is False

    session.status = "completed"
    await mgr._reconcile_decide(session)
    assert rebuilt == [session]


@pytest.mark.asyncio
async def test_a_failed_rebuild_keeps_the_chat_usable(monkeypatch) -> None:
    mgr, _ = _manager(monkeypatch, enabled=True)

    async def _boom(session, config):
        raise RuntimeError("model not downloaded")

    monkeypatch.setattr(mgr, "_rebuild_one_session", _boom)
    session = _session(bound=False)
    await mgr._reconcile_decide(session)  # must not raise
    assert session.decide_bound is False


@pytest.mark.asyncio
async def test_a_new_message_reconciles_before_the_turn_runs(monkeypatch) -> None:
    mgr, _ = _manager(monkeypatch, enabled=True)
    order: list[str] = []
    session = _session(bound=False)

    async def _ensure(session_id):
        return session

    async def _reconcile(s):
        order.append("reconcile")

    async def _privacy(s):
        order.append("privacy")
        return {"type": "error", "content": "stop here"}

    monkeypatch.setattr(mgr, "_ensure_session", _ensure)
    monkeypatch.setattr(mgr, "_reconcile_decide", _reconcile)
    monkeypatch.setattr(mgr, "_check_privacy_lock", _privacy)

    events = [event async for event in mgr.stream_message("s1", "hello")]
    assert order == ["reconcile", "privacy"]
    assert events == [{"type": "error", "content": "stop here"}]


def test_toggle_state_is_independent_per_session(monkeypatch) -> None:
    async def _go():
        mgr, rebuilt = _manager(monkeypatch, enabled=True)
        a, b = _session(bound=False), _session(bound=True)
        await mgr._reconcile_decide(a)
        await mgr._reconcile_decide(b)
        return rebuilt, a, b

    rebuilt, a, b = asyncio.run(_go())
    assert rebuilt == [a]
    assert a.decide_bound is True and b.decide_bound is True


# ── Safety guardrail ─────────────────────────────────────────────────────────


def test_guardrail_is_attached_only_when_enabled() -> None:
    off: list = []
    _insert_safety_guard(off, scope="t", enabled=False, check_input=True)
    assert off == []

    on: list = []
    _insert_safety_guard(on, scope="t", enabled=True, check_input=True)
    assert len(on) == 1 and isinstance(on[0], SafetyGuardMiddleware)


def test_only_the_orchestrator_guard_checks_the_users_message() -> None:
    orchestrator: list = []
    subagent: list = []
    _insert_safety_guard(orchestrator, scope="o", enabled=True, check_input=True)
    _insert_safety_guard(subagent, scope="s", enabled=True, check_input=False)
    assert orchestrator[0]._check_input is True
    assert subagent[0]._check_input is False


def test_guardrail_sits_inside_the_other_guards() -> None:
    """It must see the model's raw reply, not a canned abort from an outer guard."""
    middleware: list = []
    _insert_safety_guard(middleware, scope="t", enabled=True, check_input=True)
    _insert_repeated_thought_guard(middleware, scope="t")
    kinds = [type(mw) for mw in middleware]
    assert kinds.index(RepeatedThoughtGuardMiddleware) < kinds.index(SafetyGuardMiddleware)


@pytest.mark.asyncio
async def test_enabling_only_the_guardrail_rebuilds_once(monkeypatch) -> None:
    mgr, rebuilt = _manager(monkeypatch, enabled=False, guard=True)
    session = _session(bound=False, guard_bound=False)
    assert await mgr._reconcile_decide(session) is None
    assert rebuilt == [session]
    assert session.safety_bound is True and session.decide_bound is False
    await mgr._reconcile_decide(session)
    assert rebuilt == [session]


@pytest.mark.asyncio
async def test_disabling_only_the_guardrail_rebuilds_once(monkeypatch) -> None:
    mgr, rebuilt = _manager(monkeypatch, enabled=True, guard=False)
    session = _session(bound=True, guard_bound=True)
    await mgr._reconcile_decide(session)
    assert rebuilt == [session]
    assert session.safety_bound is False and session.decide_bound is True
    await mgr._reconcile_decide(session)
    assert rebuilt == [session]


@pytest.mark.asyncio
async def test_a_running_chat_keeps_its_guardrail_state_until_the_next_turn(monkeypatch) -> None:
    mgr, rebuilt = _manager(monkeypatch, enabled=False, guard=True)
    session = _session(bound=False, status="running")
    assert await mgr._reconcile_decide(session) is None
    assert rebuilt == []
    assert session.safety_bound is False

    session.status = "completed"
    await mgr._reconcile_decide(session)
    assert rebuilt == [session]


@pytest.mark.asyncio
async def test_failed_rebuild_stops_the_turn_when_the_guardrail_is_wanted(monkeypatch) -> None:
    mgr, _ = _manager(monkeypatch, enabled=False, guard=True)

    async def _boom(session, config):
        raise RuntimeError("model not downloaded")

    monkeypatch.setattr(mgr, "_rebuild_one_session", _boom)
    session = _session(bound=False, guard_bound=False)
    error = await mgr._reconcile_decide(session)
    assert error is not None
    assert error["type"] == "error"
    assert error["metadata"]["error_code"] == "safety_guard_unavailable"
    assert "model not downloaded" in error["content"]
    assert session.safety_bound is False


@pytest.mark.asyncio
async def test_failed_rebuild_does_not_stop_a_turn_that_needs_no_guardrail(monkeypatch) -> None:
    mgr, _ = _manager(monkeypatch, enabled=True, guard=False)

    async def _boom(session, config):
        raise RuntimeError("model not downloaded")

    monkeypatch.setattr(mgr, "_rebuild_one_session", _boom)
    assert await mgr._reconcile_decide(_session(bound=False)) is None


@pytest.mark.asyncio
async def test_a_chat_already_guarded_is_not_stopped_by_a_failed_rebuild(monkeypatch) -> None:
    """Guardrail on and already bound: a rebuild for the tool flag must not block it."""
    mgr, _ = _manager(monkeypatch, enabled=True, guard=True)

    async def _boom(session, config):
        raise RuntimeError("boom")

    monkeypatch.setattr(mgr, "_rebuild_one_session", _boom)
    assert await mgr._reconcile_decide(_session(bound=False, guard_bound=True)) is None


@pytest.mark.asyncio
async def test_a_reconcile_error_stops_the_turn_before_the_privacy_check(monkeypatch) -> None:
    mgr, _ = _manager(monkeypatch, enabled=False, guard=True)
    order: list[str] = []
    session = _session(bound=False)
    error = {"type": "error", "content": "guardrail unavailable"}

    async def _ensure(session_id):
        return session

    async def _reconcile(s):
        order.append("reconcile")
        return error

    async def _privacy(s):
        order.append("privacy")
        return None

    monkeypatch.setattr(mgr, "_ensure_session", _ensure)
    monkeypatch.setattr(mgr, "_reconcile_decide", _reconcile)
    monkeypatch.setattr(mgr, "_check_privacy_lock", _privacy)

    events = [event async for event in mgr.stream_message("s1", "hello")]
    assert order == ["reconcile"]
    assert events == [error]


@pytest.mark.asyncio
async def test_an_edit_reconciles_before_the_turn_runs(monkeypatch) -> None:
    """Picking another model, then editing an earlier prompt, must use the new model."""
    mgr, _ = _manager(monkeypatch, enabled=True)
    order: list[str] = []
    session = _session(bound=False)

    async def _ensure(session_id):
        return session

    async def _reconcile(s):
        order.append("reconcile")

    async def _privacy(s):
        order.append("privacy")
        return {"type": "error", "content": "stop here"}

    monkeypatch.setattr(mgr, "_ensure_session", _ensure)
    monkeypatch.setattr(mgr, "_reconcile_decide", _reconcile)
    monkeypatch.setattr(mgr, "_check_privacy_lock", _privacy)

    events = [event async for event in mgr.stream_edit("s1", 0, "edited")]
    assert order == ["reconcile", "privacy"]
    assert events == [{"type": "error", "content": "stop here"}]


@pytest.mark.asyncio
async def test_a_failed_model_switch_stops_an_edit(monkeypatch) -> None:
    mgr, _ = _manager(monkeypatch, enabled=True)
    session = _session(bound=False)

    async def _ensure(session_id):
        return session

    async def _reconcile(s):
        return {"type": "error", "content": "could not switch", "metadata": {"error_code": "model_switch_failed"}}

    async def _privacy(s):
        raise AssertionError("the turn must not go on after a failed switch")

    monkeypatch.setattr(mgr, "_ensure_session", _ensure)
    monkeypatch.setattr(mgr, "_reconcile_decide", _reconcile)
    monkeypatch.setattr(mgr, "_check_privacy_lock", _privacy)

    events = [event async for event in mgr.stream_edit("s1", 0, "edited")]
    assert events[0]["metadata"]["error_code"] == "model_switch_failed"
