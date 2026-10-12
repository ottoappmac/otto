"""Wire-format client for the jev tool. No model and no network."""

from __future__ import annotations

import json

import httpx
import pytest

from backend.decision_client import (
    DecisionClientError,
    ask,
    format_answer,
    model_name_for,
    option_map,
    parse_answer,
    sanitize_key,
    truncate_state,
    verdict_text,
)


def test_truncate_state_marks_the_cut() -> None:
    text = "a" * 50
    out = truncate_state(text, limit=20)
    assert out.endswith("[truncated]")
    assert len(out) <= 20


def test_sanitize_key_dedupes() -> None:
    used: set[str] = set()
    assert sanitize_key("Apply Patch!", used) == "apply_patch"
    assert sanitize_key("Apply Patch!", used) == "apply_patch_2"


def test_option_map_accepts_a_json_string() -> None:
    mapping = option_map('["Read file", "Ask user"]')
    assert mapping == {"read_file": "Read file", "ask_user": "Ask user"}


def test_model_name_never_pins_a_jev_version() -> None:
    assert model_name_for("https://api.typesafe.ai") == "jev-latest"
    assert model_name_for("http://127.0.0.1:52416") == "openjev-latest"
    assert model_name_for("https://api.codiv.ai") == "openjev-latest"
    assert "1.13" not in model_name_for("https://api.typesafe.ai")


def test_noul_without_confidence_uses_distance_from_half() -> None:
    answer = parse_answer({"answers": {"q": {"noul": 0.91}}})
    assert answer.answer == "yes"
    assert answer.confidence == pytest.approx(0.82)
    assert answer.probabilities["yes"] == pytest.approx(0.91)


def test_noul_keeps_an_explicit_confidence() -> None:
    answer = parse_answer({"answers": {"q": {"noul": 0.91, "confidence": 0.4}}})
    assert answer.confidence == pytest.approx(0.4)


def test_choice_request_shape_and_thresholds() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={
            "answers": {
                "q": {
                    "choice": "apply_patch",
                    "probabilities": {"apply_patch": 0.92, "ask_user": 0.06, "run_tests": 0.02},
                    "confidence": 0.91,
                }
            }
        })

    result = ask(
        "x" * 20_000,
        "Which next step?",
        {"apply_patch": "edit", "ask_user": "ask", "run_tests": "test"},
        base_url="http://127.0.0.1:52416",
        api_key="secret",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    body = seen["body"]
    assert body["model"] == "openjev-latest"
    assert list(body["questions"]) == ["q"]
    assert body["questions"]["q"]["type"] == "choice"
    assert body["state"].endswith("[truncated]")
    assert seen["auth"] == "Bearer secret"
    assert result.answer == "apply_patch"
    assert result.margin == pytest.approx(0.86)
    text = format_answer(
        result,
        confident_threshold=0.85,
        unsure_threshold=0.6,
        unattended=False,
    )
    assert "CONFIDENT. Proceed." in text
    assert "Advisory only" in text
    labeled = format_answer(
        result,
        confident_threshold=0.85,
        unsure_threshold=0.6,
        unattended=False,
        source="local (127.0.0.1)",
    )
    assert labeled.endswith("source: local (127.0.0.1)")


def test_verdict_bands() -> None:
    assert verdict_text(0.9, confident_threshold=0.85, unsure_threshold=0.6, unattended=False).startswith("CONFIDENT")
    assert verdict_text(0.7, confident_threshold=0.85, unsure_threshold=0.6, unattended=False).startswith("MODERATE")
    assert "ask_user" in verdict_text(0.2, confident_threshold=0.85, unsure_threshold=0.6, unattended=False)
    unsure = verdict_text(0.2, confident_threshold=0.85, unsure_threshold=0.6, unattended=True)
    assert "No user is available" in unsure
    assert "ask_user" not in unsure


def test_too_many_options_is_an_error() -> None:
    with pytest.raises(DecisionClientError):
        ask(
            "state",
            "which?",
            [f"opt{i}" for i in range(13)],
            base_url="http://127.0.0.1:9",
            api_key="",
            http_client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))),
        )
