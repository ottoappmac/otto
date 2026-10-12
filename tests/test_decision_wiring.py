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
from backend.session_manager import SessionManager, _build_standard_tools
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


def _manager(monkeypatch, *, enabled: bool):
    """A real SessionManager whose rebuild is recorded instead of run."""
    cfg = AppConfig()
    cfg.decision.enabled = enabled

    async def _aload():
        return cfg

    monkeypatch.setattr("backend.session_manager.AppConfig.aload", _aload)
    mgr = SessionManager()
    rebuilt: list[object] = []

    async def _rebuild(session, config):
        rebuilt.append(session)
        session.decide_bound = bool(config.decision.enabled)

    monkeypatch.setattr(mgr, "_rebuild_one_session", _rebuild)
    return mgr, rebuilt


def _session(*, bound: bool, status: str = "completed"):
    return SimpleNamespace(id="s1", decide_bound=bound, status=status)


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
