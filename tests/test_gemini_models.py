"""Gemini defaults, retired-name migration, timeouts and friendly errors."""

from __future__ import annotations

import pytest
from google.genai import errors

from backend.gemini_models import (
    DEFAULT_GEMINI_LIVE_MODEL,
    DEFAULT_GEMINI_MODEL,
    GEMINI_CHAT_MAX_RETRIES,
    GEMINI_CHAT_TIMEOUT_S,
    current_gemini_live_model,
    current_gemini_model,
    gemini_error_message,
)


@pytest.mark.parametrize("name", ["", None, "gemini-2.5-flash", "models/gemini-2.0-flash", " gemini-1.5-flash "])
def test_empty_or_retired_names_use_the_default(name):
    assert current_gemini_model(name) == DEFAULT_GEMINI_MODEL


@pytest.mark.parametrize("name", ["gemini-3.8-flash", "gemini-2.5-pro", "gemini-flash-latest"])
def test_other_names_are_kept(name):
    assert current_gemini_model(name) == name


def test_live_model_default_and_migration():
    assert current_gemini_live_model("gemini-2.0-flash-live-001") == DEFAULT_GEMINI_LIVE_MODEL
    assert current_gemini_live_model("") == DEFAULT_GEMINI_LIVE_MODEL
    assert current_gemini_live_model("gemini-3.1-flash-live-preview") == "gemini-3.1-flash-live-preview"


def test_saved_config_with_retired_model_is_migrated():
    from backend.config import GoogleConfig

    assert GoogleConfig().model_name == DEFAULT_GEMINI_MODEL
    assert GoogleConfig(model_name="gemini-2.5-flash").model_name == DEFAULT_GEMINI_MODEL
    assert GoogleConfig(model_name="gemini-3.7-flash").model_name == "gemini-3.7-flash"


def test_environment_maps_retired_env_value(monkeypatch):
    from utilities.environment import Environment

    monkeypatch.setenv("GOOGLE_MODEL_NAME", "gemini-2.5-flash")
    assert Environment.get_google_model_name() == DEFAULT_GEMINI_MODEL


def test_chat_model_has_timeout_and_few_retries(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    monkeypatch.setenv("GOOGLE_MODEL_NAME", "gemini-3.8-flash")
    from deep_agent.model_factory import create_llm

    llm = create_llm("google")
    assert llm.model.endswith("gemini-3.8-flash")
    assert llm.timeout == GEMINI_CHAT_TIMEOUT_S
    assert llm.max_retries == GEMINI_CHAT_MAX_RETRIES


def test_video_client_has_timeout():
    from backend.video.gemini import _VIDEO_TIMEOUT_MS, _client

    opts = _client("test")._api_client._http_options
    assert opts.timeout == _VIDEO_TIMEOUT_MS
    assert opts.api_version == "v1beta"


def _busy():
    return errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE",
                                              "message": "This model is currently experiencing high demand."}})


def test_busy_gemini_gets_a_clear_message():
    msg, code = gemini_error_message(_busy())
    assert "Gemini is busy" in msg and code == "llm_overloaded"


def test_wrapped_busy_error_is_found_through_the_chain():
    try:
        try:
            raise _busy()
        except errors.ServerError as inner:
            raise RuntimeError("Error calling model") from inner
    except RuntimeError as outer:
        assert gemini_error_message(outer)[1] == "llm_overloaded"


def test_deadline_and_missing_model():
    timeout = errors.ServerError(504, {"error": {"code": 504, "status": "DEADLINE_EXCEEDED", "message": "Deadline expired"}})
    assert gemini_error_message(timeout)[1] == "timeout"
    missing = errors.ClientError(404, {"error": {"code": 404, "status": "NOT_FOUND",
                                                 "message": "models/gemini-2.5-flash is not found"}})
    msg, code = gemini_error_message(missing)
    assert code == "llm_model_not_found" and DEFAULT_GEMINI_MODEL in msg


def test_non_gemini_errors_are_left_alone():
    assert gemini_error_message(RuntimeError("503 Service Unavailable from proxy")) is None


def test_friendly_error_uses_gemini_message():
    from backend.routes.sessions import _friendly_error

    msg, code = _friendly_error(_busy())
    assert code == "llm_overloaded" and "Gemini is busy" in msg


def test_eval_runner_treats_gemini_failures_as_transient():
    from backend.eval_runner import classify_error_code, is_prompt_addressable

    msg, _ = gemini_error_message(_busy())
    assert classify_error_code(msg) == "llm_overloaded"
    assert not is_prompt_addressable("llm_model_not_found")
