"""Unit tests for the Slack MCP's pure-Python helpers.

The decorated tool entry points run in their own uv-provisioned venv at
runtime and can't be cleanly imported from the parent process's test
environment (FastMCP parameter introspection differs across versions).
All testable logic therefore lives in ``_helpers``, which is
pure-Python and free of MCP decoration — see
``tests/test_macos_osascript_introspection.py`` for the established
pattern this mirrors.
"""

from __future__ import annotations

import pytest

from backend.builtin_mcps.slack._helpers import (
    SlackAPIError,
    build_headers,
    clamp_limit,
    format_channel,
    format_message,
    format_user,
    next_cursor,
    normalize_emoji,
    parse_slack_payload,
)


def test_build_headers_includes_bearer_token():
    headers = build_headers("xoxb-fake-token")
    assert headers["Authorization"] == "Bearer xoxb-fake-token"
    assert "application/json" in headers["Content-Type"]


def test_parse_slack_payload_returns_payload_on_ok():
    payload = {"ok": True, "channels": [{"id": "C1"}]}
    assert parse_slack_payload(payload) == payload


def test_parse_slack_payload_raises_on_not_ok():
    with pytest.raises(SlackAPIError) as exc_info:
        parse_slack_payload({"ok": False, "error": "channel_not_found"})
    assert exc_info.value.error == "channel_not_found"


def test_parse_slack_payload_raises_on_missing_error_field():
    with pytest.raises(SlackAPIError) as exc_info:
        parse_slack_payload({"ok": False})
    assert exc_info.value.error == "unknown_error"


def test_parse_slack_payload_raises_on_malformed_response():
    with pytest.raises(SlackAPIError):
        parse_slack_payload(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "limit,default,max_limit,expected",
    [
        (50, 20, 200, 50),
        (0, 20, 200, 20),
        (-5, 20, 200, 20),
        (500, 20, 200, 200),
        ("not-a-number", 20, 200, 20),
        (None, 20, 200, 20),
    ],
)
def test_clamp_limit(limit, default, max_limit, expected):
    assert clamp_limit(limit, default=default, max_limit=max_limit) == expected


def test_next_cursor_extracts_pagination_token():
    payload = {"response_metadata": {"next_cursor": "dXNlcjpVMDYxTkZUVDI="}}
    assert next_cursor(payload) == "dXNlcjpVMDYxTkZUVDI="


def test_next_cursor_defaults_to_empty_string():
    assert next_cursor({}) == ""
    assert next_cursor({"response_metadata": {}}) == ""


def test_format_channel_extracts_expected_fields():
    raw = {
        "id": "C123",
        "name": "general",
        "is_channel": True,
        "is_private": False,
        "is_archived": False,
        "is_member": True,
        "num_members": 42,
        "topic": {"value": "general chat"},
        "purpose": {"value": "everything"},
    }
    formatted = format_channel(raw)
    assert formatted == {
        "id": "C123",
        "name": "general",
        "is_channel": True,
        "is_private": False,
        "is_archived": False,
        "is_member": True,
        "num_members": 42,
        "topic": "general chat",
        "purpose": "everything",
    }


def test_format_channel_handles_missing_topic_purpose():
    formatted = format_channel({"id": "C1", "name": "x"})
    assert formatted["topic"] == ""
    assert formatted["purpose"] == ""


def test_format_message_extracts_expected_fields():
    raw = {
        "ts": "1234.5678",
        "user": "U1",
        "text": "hello",
        "thread_ts": "1234.0000",
        "reply_count": 3,
        "reactions": [{"name": "thumbsup", "count": 2}],
    }
    formatted = format_message(raw)
    assert formatted["ts"] == "1234.5678"
    assert formatted["user"] == "U1"
    assert formatted["reply_count"] == 3
    assert formatted["reactions"] == [{"name": "thumbsup", "count": 2}]


def test_format_message_falls_back_to_bot_id():
    formatted = format_message({"ts": "1.0", "bot_id": "B1", "text": "hi"})
    assert formatted["user"] == "B1"


def test_format_user_extracts_expected_fields():
    raw = {
        "id": "U1",
        "name": "jdoe",
        "real_name": "Jane Doe",
        "is_bot": False,
        "is_admin": True,
        "deleted": False,
        "tz": "America/New_York",
        "profile": {"email": "jane@example.com"},
    }
    formatted = format_user(raw)
    assert formatted["email"] == "jane@example.com"
    assert formatted["real_name"] == "Jane Doe"
    assert formatted["is_admin"] is True


def test_format_user_falls_back_to_profile_real_name():
    raw = {"id": "U1", "name": "jdoe", "profile": {"real_name": "Jane Doe"}}
    assert format_user(raw)["real_name"] == "Jane Doe"


@pytest.mark.parametrize(
    "emoji,expected",
    [
        (":thumbsup:", "thumbsup"),
        ("thumbsup", "thumbsup"),
        ("  :wave:  ", "wave"),
        ("", ""),
    ],
)
def test_normalize_emoji(emoji, expected):
    assert normalize_emoji(emoji) == expected
