"""Unit tests for the Microsoft Teams MCP's pure-Python helpers.

Covers the OAuth2 client-credentials token-cache logic (the part most
worth unit testing, since a bug there means every tool call either
re-authenticates on every request or — worse — serves an expired
token) plus response parsing/formatting. See
``tests/test_macos_osascript_introspection.py`` for the established
pattern of testing the ``_helpers`` module in isolation from the
``@mcp.tool()``-decorated, venv-only ``server.py``.
"""

from __future__ import annotations

import pytest

from backend.builtin_mcps.microsoft_teams._helpers import (
    TeamsAPIError,
    build_token_request,
    clamp_top,
    classify_error,
    format_channel,
    format_channel_message,
    format_member,
    format_team,
    format_user,
    parse_token_response,
    teams_list_params,
    token_is_valid,
)


def test_build_token_request_uses_client_credentials_grant():
    url, data = build_token_request("tenant-1", "client-1", "secret-1")
    assert url == "https://login.microsoftonline.com/tenant-1/oauth2/v2.0/token"
    assert data["grant_type"] == "client_credentials"
    assert data["client_id"] == "client-1"
    assert data["client_secret"] == "secret-1"
    assert data["scope"] == "https://graph.microsoft.com/.default"


def test_parse_token_response_computes_expiry():
    cache = parse_token_response({"access_token": "abc", "expires_in": 3600}, now=1000.0)
    assert cache == {"access_token": "abc", "expires_at": 4600.0}


def test_parse_token_response_defaults_expiry_when_missing():
    cache = parse_token_response({"access_token": "abc"}, now=1000.0)
    assert cache["expires_at"] == 1000.0 + 3600.0


def test_parse_token_response_raises_when_no_access_token():
    with pytest.raises(TeamsAPIError) as exc_info:
        parse_token_response({"error_description": "invalid_client"}, now=1000.0)
    assert exc_info.value.status_code == 400
    assert "invalid_client" in exc_info.value.message


def test_token_is_valid_true_when_well_within_expiry():
    cache = {"access_token": "abc", "expires_at": 2000.0}
    assert token_is_valid(cache, now=1000.0, leeway=300) is True


def test_token_is_valid_false_within_leeway_of_expiry():
    cache = {"access_token": "abc", "expires_at": 1200.0}
    assert token_is_valid(cache, now=1000.0, leeway=300) is False


def test_token_is_valid_false_when_already_expired():
    cache = {"access_token": "abc", "expires_at": 900.0}
    assert token_is_valid(cache, now=1000.0) is False


@pytest.mark.parametrize("cache", [None, {}, {"access_token": "abc"}, {"expires_at": 2000.0}])
def test_token_is_valid_false_for_missing_or_malformed_cache(cache):
    assert token_is_valid(cache, now=1000.0) is False


def test_token_is_valid_false_for_non_numeric_expiry():
    cache = {"access_token": "abc", "expires_at": "not-a-number"}
    assert token_is_valid(cache, now=1000.0) is False


def test_classify_error_extracts_nested_error_message():
    err = classify_error(403, {"error": {"code": "Forbidden", "message": "Access denied"}})
    assert isinstance(err, TeamsAPIError)
    assert err.status_code == 403
    assert err.message == "Access denied"


def test_classify_error_handles_string_error_field():
    err = classify_error(400, {"error": "invalid_request", "error_description": "bad scope"})
    assert err.message == "invalid_request"


def test_classify_error_handles_non_dict_body():
    err = classify_error(503, "Service Unavailable")
    assert err.message == "Service Unavailable"


@pytest.mark.parametrize(
    "value,default,max_top,expected",
    [
        (30, 20, 50, 30),
        (0, 20, 50, 20),
        (-1, 20, 50, 20),
        (999, 20, 50, 50),
        ("nope", 20, 50, 20),
    ],
)
def test_clamp_top(value, default, max_top, expected):
    assert clamp_top(value, default=default, max_top=max_top) == expected


def test_teams_list_params_filters_to_team_enabled_groups():
    params = teams_list_params()
    assert "resourceProvisioningOptions" in params["$filter"]
    assert "displayName" in params["$select"]


def test_format_team_extracts_expected_fields():
    formatted = format_team({"id": "T1", "displayName": "Engineering", "description": None})
    assert formatted == {"id": "T1", "display_name": "Engineering", "description": ""}


def test_format_channel_extracts_expected_fields():
    raw = {
        "id": "C1",
        "displayName": "General",
        "description": "Main channel",
        "membershipType": "standard",
    }
    assert format_channel(raw) == {
        "id": "C1",
        "display_name": "General",
        "description": "Main channel",
        "membership_type": "standard",
    }


def test_format_member_extracts_expected_fields():
    raw = {"id": "M1", "displayName": "Jane Doe", "roles": ["owner"], "email": "jane@example.com"}
    assert format_member(raw) == {
        "id": "M1",
        "display_name": "Jane Doe",
        "roles": ["owner"],
        "email": "jane@example.com",
    }


def test_format_user_prefers_mail_over_upn():
    raw = {"id": "U1", "displayName": "Jane", "mail": "jane@x.com", "userPrincipalName": "jane@upn.x.com"}
    assert format_user(raw)["mail"] == "jane@x.com"


def test_format_user_falls_back_to_upn_when_mail_missing():
    raw = {"id": "U1", "displayName": "Jane", "userPrincipalName": "jane@upn.x.com"}
    assert format_user(raw)["mail"] == "jane@upn.x.com"


def test_format_channel_message_extracts_sender_and_body():
    raw = {
        "id": "MSG1",
        "from": {"user": {"displayName": "Jane Doe"}},
        "createdDateTime": "2024-01-01T00:00:00Z",
        "subject": None,
        "body": {"content": "hello team", "contentType": "text"},
    }
    formatted = format_channel_message(raw)
    assert formatted["from"] == "Jane Doe"
    assert formatted["content"] == "hello team"
    assert formatted["subject"] == ""


def test_format_channel_message_handles_missing_sender():
    raw = {"id": "MSG1", "body": {"content": "x"}}
    formatted = format_channel_message(raw)
    assert formatted["from"] == ""
