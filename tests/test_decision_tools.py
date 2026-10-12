"""The jev tool's failure text, option coercion, and per-turn cap."""

from __future__ import annotations

from backend.config import AppConfig
from backend.decision_client import DecisionAnswer
from backend.decision_tools import build_decision_tools, reset_decision_turn


def _cfg(*, enabled: bool = True) -> AppConfig:
    cfg = AppConfig()
    cfg.decision.enabled = enabled
    cfg.decision.max_calls_per_turn = 2
    return cfg


def test_disabled_tells_the_agent_to_continue(monkeypatch) -> None:
    monkeypatch.setattr("backend.config.AppConfig.load", lambda: _cfg(enabled=False))
    tool = build_decision_tools("s1")[0]
    text = tool.invoke({"situation": "evidence", "question": "Go?"})
    assert "turned off" in text
    assert "Settings → Jev" in text


def test_options_json_string_is_accepted(monkeypatch) -> None:
    monkeypatch.setattr("backend.config.AppConfig.load", lambda: _cfg())
    monkeypatch.setattr("backend.decision_runtime.uses_local_sidecar", lambda cfg: True)
    monkeypatch.setattr("backend.decision_runtime.ensure_running", lambda cfg=None: "http://127.0.0.1:52416")
    monkeypatch.setattr("backend.decision_runtime.local_api_key", lambda: "k")
    monkeypatch.setattr("backend.decision_runtime.note_used", lambda: None)
    seen: dict = {}

    def fake_ask(state, question, options=None, scale=None, **kwargs):
        seen["options"] = options
        return DecisionAnswer(
            kind="choice",
            answer="left",
            confidence=0.9,
            margin=0.8,
            probabilities={"left": 0.9, "right": 0.1},
        )

    monkeypatch.setattr("backend.decision_client.ask", fake_ask)
    tool = build_decision_tools("s-json")[0]
    text = tool.invoke({
        "situation": "two files",
        "question": "which?",
        "options": '["left", "right"]',
    })
    assert seen["options"] == ["left", "right"]
    assert "CONFIDENT" in text


def test_per_turn_cap(monkeypatch) -> None:
    monkeypatch.setattr("backend.config.AppConfig.load", lambda: _cfg())
    monkeypatch.setattr("backend.decision_runtime.uses_local_sidecar", lambda cfg: True)
    monkeypatch.setattr("backend.decision_runtime.ensure_running", lambda cfg=None: "http://127.0.0.1:52416")
    monkeypatch.setattr("backend.decision_runtime.local_api_key", lambda: "k")
    monkeypatch.setattr("backend.decision_runtime.note_used", lambda: None)
    monkeypatch.setattr(
        "backend.decision_client.ask",
        lambda *a, **k: DecisionAnswer("noul", "yes", 0.9, 0.8, {"yes": 0.9, "no": 0.1}),
    )
    reset_decision_turn("cap")
    tool = build_decision_tools("cap")[0]
    assert "jev:" in tool.invoke({"situation": "a", "question": "q"})
    assert "jev:" in tool.invoke({"situation": "a", "question": "q"})
    third = tool.invoke({"situation": "a", "question": "q"})
    assert "call cap reached" in third
    reset_decision_turn("cap")
    assert "jev:" in tool.invoke({"situation": "a", "question": "q"})


def test_not_installed_is_plain_text(monkeypatch) -> None:
    monkeypatch.setattr("backend.config.AppConfig.load", lambda: _cfg())
    monkeypatch.setattr("backend.decision_runtime.uses_local_sidecar", lambda cfg: True)

    def boom(cfg=None):
        raise RuntimeError("The Jev runtime is not installed. Open Settings → Jev and install it.")

    monkeypatch.setattr("backend.decision_runtime.ensure_running", boom)
    tool = build_decision_tools("missing")[0]
    text = tool.invoke({"situation": "a", "question": "q"})
    assert "not installed" in text
    assert "Settings → Jev" in text
