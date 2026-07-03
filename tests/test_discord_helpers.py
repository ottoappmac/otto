"""Unit tests for the Discord MCP's pure-Python helpers.

See ``tests/test_macos_osascript_introspection.py`` for the established
pattern of testing the ``_helpers`` module in isolation from the
``@mcp.tool()``-decorated, venv-only ``server.py``.
"""

from __future__ import annotations

import pytest

from backend.builtin_mcps.discord._helpers import (
    DiscordAPIError,
    build_headers,
    clamp_limit,
    classify_error,
    encode_emoji,
    find_matching_channels,
    format_channel,
    format_guild,
    format_member,
    format_message,
)


def test_build_headers_includes_bot_prefix():
    headers = build_headers("fake-token")
    assert headers["Authorization"] == "Bot fake-token"
    assert "Otto" in headers["User-Agent"]


def test_classify_error_extracts_message_and_retry_after():
    err = classify_error(429, {"message": "rate limited", "retry_after": 1.5})
    assert isinstance(err, DiscordAPIError)
    assert err.status_code == 429
    assert err.message == "rate limited"
    assert err.retry_after == 1.5


def test_classify_error_handles_missing_retry_after():
    err = classify_error(403, {"message": "Missing Permissions"})
    assert err.retry_after is None


def test_classify_error_handles_non_dict_body():
    err = classify_error(502, "<html>Bad Gateway</html>")
    assert err.status_code == 502
    assert err.message == "<html>Bad Gateway</html>"


def test_classify_error_handles_empty_body():
    err = classify_error(500, None)
    assert err.message == "request failed"


def test_classify_error_handles_malformed_retry_after():
    err = classify_error(429, {"message": "rate limited", "retry_after": "soon"})
    assert err.retry_after is None


@pytest.mark.parametrize(
    "limit,default,max_limit,expected",
    [
        (50, 20, 100, 50),
        (0, 20, 100, 20),
        (-1, 20, 100, 20),
        (500, 20, 100, 100),
        (None, 20, 100, 20),
    ],
)
def test_clamp_limit(limit, default, max_limit, expected):
    assert clamp_limit(limit, default=default, max_limit=max_limit) == expected


def test_format_guild_extracts_expected_fields():
    raw = {
        "id": "G1",
        "name": "My Server",
        "owner": True,
        "approximate_member_count": 150,
        "description": None,
    }
    formatted = format_guild(raw)
    assert formatted == {
        "id": "G1",
        "name": "My Server",
        "owner": True,
        "approximate_member_count": 150,
        "description": "",
    }


def test_format_channel_flags_text_channel_types():
    text_channel = format_channel({"id": "C1", "name": "general", "type": 0})
    voice_channel = format_channel({"id": "C2", "name": "voice", "type": 2})
    announcement = format_channel({"id": "C3", "name": "ann", "type": 5})
    assert text_channel["is_text_channel"] is True
    assert voice_channel["is_text_channel"] is False
    assert announcement["is_text_channel"] is True


def test_format_message_extracts_author_and_reactions():
    raw = {
        "id": "M1",
        "channel_id": "C1",
        "author": {"username": "jdoe", "id": "U1"},
        "content": "hello world",
        "timestamp": "2024-01-01T00:00:00Z",
        "reactions": [{"emoji": {"name": "👍"}, "count": 3}],
    }
    formatted = format_message(raw)
    assert formatted["author"] == "jdoe"
    assert formatted["author_id"] == "U1"
    assert formatted["reactions"] == [{"emoji": "👍", "count": 3}]


def test_format_message_handles_missing_author():
    formatted = format_message({"id": "M1", "content": "hi"})
    assert formatted["author"] == ""
    assert formatted["author_id"] == ""


def test_format_member_extracts_expected_fields():
    raw = {
        "user": {"id": "U1", "username": "jdoe"},
        "nick": "JD",
        "roles": ["R1", "R2"],
        "joined_at": "2024-01-01T00:00:00Z",
    }
    formatted = format_member(raw)
    assert formatted["user_id"] == "U1"
    assert formatted["username"] == "jdoe"
    assert formatted["nick"] == "JD"
    assert formatted["roles"] == ["R1", "R2"]


@pytest.mark.parametrize(
    "emoji,expected",
    [
        ("👍", "👍"),
        ("  thumbsup:123  ", "thumbsup:123"),
        ("", ""),
    ],
)
def test_encode_emoji_strips_whitespace(emoji, expected):
    assert encode_emoji(emoji) == expected


_CHANNELS = [
    {"id": "C1", "name": "general"},
    {"id": "C2", "name": "general-chat"},
    {"id": "C3", "name": "random"},
]


def test_find_matching_channels_exact_match_case_insensitive():
    matches = find_matching_channels(_CHANNELS, "General")
    assert [c["id"] for c in matches] == ["C1"]


def test_find_matching_channels_strips_hash_prefix():
    matches = find_matching_channels(_CHANNELS, "#random")
    assert [c["id"] for c in matches] == ["C3"]


def test_find_matching_channels_falls_back_to_substring():
    matches = find_matching_channels(_CHANNELS, "chat")
    assert [c["id"] for c in matches] == ["C2"]


def test_find_matching_channels_exact_match_wins_over_substring():
    # "general" is a substring of "general-chat" too, but the exact match
    # on "general" should be returned alone, not both.
    matches = find_matching_channels(_CHANNELS, "general")
    assert [c["id"] for c in matches] == ["C1"]


def test_find_matching_channels_no_match_returns_empty():
    assert find_matching_channels(_CHANNELS, "nonexistent") == []


def test_find_matching_channels_empty_name_returns_empty():
    assert find_matching_channels(_CHANNELS, "") == []
