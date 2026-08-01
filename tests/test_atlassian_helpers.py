"""Unit tests for the Atlassian (Jira + Confluence) MCP's pure-Python helpers.

See ``tests/test_macos_osascript_introspection.py`` for the established
pattern of testing the ``_helpers`` module in isolation from the
``@mcp.tool()``-decorated, venv-only ``server.py``. This module in
particular must stay importable without the ``atlassian`` package
installed — it's exercised straight from the main test environment.
"""

from __future__ import annotations

import pytest

from backend.builtin_mcps.atlassian._helpers import (
    CONFLUENCE_ACCESS_FORBIDDEN_HINT,
    AtlassianAPIError,
    adf_to_text,
    build_issue_create_fields,
    build_issue_update_fields,
    clamp_limit,
    classify_error,
    format_confluence_page,
    format_confluence_search_result,
    format_confluence_space,
    format_jira_comment,
    format_jira_issue,
    format_jira_project,
    is_confluence_access_forbidden,
)


def test_classify_error_builds_atlassian_api_error():
    err = classify_error(404, "errorMessages: Issue does not exist")
    assert isinstance(err, AtlassianAPIError)
    assert err.status_code == 404
    assert err.message == "errorMessages: Issue does not exist"


def test_classify_error_handles_missing_status_code():
    err = classify_error(None, "Connection refused")
    assert err.status_code is None
    assert err.message == "Connection refused"


def test_classify_error_defaults_empty_message():
    err = classify_error(500, "")
    assert err.message == "request failed"


def test_is_confluence_access_forbidden_matches_known_message():
    message = (
        'com.atlassian.confluence.mvc.rest.common.exception.StacklessResponseStatusException: '
        '403 FORBIDDEN "Request rejected because caller cannot access Confluence"'
    )
    assert is_confluence_access_forbidden(message) is True


def test_is_confluence_access_forbidden_is_case_insensitive():
    assert is_confluence_access_forbidden("CALLER CANNOT ACCESS CONFLUENCE") is True


def test_is_confluence_access_forbidden_ignores_unrelated_403s():
    assert is_confluence_access_forbidden("You do not have permission to view this page") is False


def test_is_confluence_access_forbidden_handles_none():
    assert is_confluence_access_forbidden(None) is False


def test_confluence_access_forbidden_hint_mentions_license_and_scoped_token():
    assert "license" in CONFLUENCE_ACCESS_FORBIDDEN_HINT.lower()
    assert "scoped" in CONFLUENCE_ACCESS_FORBIDDEN_HINT.lower()
    assert "id.atlassian.com" in CONFLUENCE_ACCESS_FORBIDDEN_HINT


@pytest.mark.parametrize(
    "limit,default,max_limit,expected",
    [
        (50, 25, 100, 50),
        (0, 25, 100, 25),
        (-5, 25, 100, 25),
        (500, 25, 100, 100),
        ("not-a-number", 25, 100, 25),
        (None, 25, 100, 25),
    ],
)
def test_clamp_limit(limit, default, max_limit, expected):
    assert clamp_limit(limit, default=default, max_limit=max_limit) == expected


def test_adf_to_text_passes_through_plain_string():
    assert adf_to_text("just plain text") == "just plain text"


def test_adf_to_text_handles_none():
    assert adf_to_text(None) == ""


def test_adf_to_text_extracts_single_paragraph():
    doc = {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": "Hello world"}],
            }
        ],
    }
    assert adf_to_text(doc) == "Hello world"


def test_adf_to_text_joins_multiple_paragraphs_with_newlines():
    doc = {
        "type": "doc",
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": "First line"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": "Second line"}]},
        ],
    }
    assert adf_to_text(doc) == "First line\nSecond line"


def test_adf_to_text_handles_hard_break_within_paragraph():
    doc = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": "line one"},
                    {"type": "hardBreak"},
                    {"type": "text", "text": "line two"},
                ],
            }
        ],
    }
    assert adf_to_text(doc) == "line one\nline two"


def test_adf_to_text_handles_nested_list_items():
    doc = {
        "type": "doc",
        "content": [
            {
                "type": "bulletList",
                "content": [
                    {
                        "type": "listItem",
                        "content": [
                            {"type": "paragraph", "content": [{"type": "text", "text": "item one"}]}
                        ],
                    },
                    {
                        "type": "listItem",
                        "content": [
                            {"type": "paragraph", "content": [{"type": "text", "text": "item two"}]}
                        ],
                    },
                ],
            }
        ],
    }
    assert adf_to_text(doc) == "item one\nitem two"


def test_format_confluence_space_extracts_expected_fields():
    raw = {"id": 123, "key": "ENG", "name": "Engineering", "type": "global", "status": "current"}
    assert format_confluence_space(raw) == {
        "id": 123,
        "key": "ENG",
        "name": "Engineering",
        "type": "global",
        "status": "current",
    }


def test_format_confluence_page_includes_body_by_default():
    raw = {
        "id": "456",
        "title": "Onboarding",
        "type": "page",
        "status": "current",
        "space": {"key": "ENG"},
        "version": {"number": 3},
        "body": {"storage": {"value": "<p>hello</p>"}},
        "_links": {"base": "https://x.atlassian.net/wiki", "webui": "/spaces/ENG/pages/456"},
    }
    formatted = format_confluence_page(raw)
    assert formatted["id"] == "456"
    assert formatted["space_key"] == "ENG"
    assert formatted["version"] == 3
    assert formatted["body_html"] == "<p>hello</p>"
    assert formatted["url"] == "https://x.atlassian.net/wiki/spaces/ENG/pages/456"


def test_format_confluence_page_omits_body_when_not_requested():
    raw = {"id": "456", "title": "Onboarding", "space": {}, "version": {}}
    formatted = format_confluence_page(raw, include_body=False)
    assert "body_html" not in formatted


def test_format_confluence_search_result_reads_nested_content():
    raw = {
        "content": {"id": "789", "type": "page", "space": {"key": "ENG"}},
        "title": "Runbook",
        "excerpt": "how to deploy...",
        "url": "/spaces/ENG/pages/789",
        "lastModified": "2026-01-01T00:00:00.000Z",
    }
    formatted = format_confluence_search_result(raw)
    assert formatted["id"] == "789"
    assert formatted["title"] == "Runbook"
    assert formatted["space_key"] == "ENG"


def test_format_jira_project_extracts_expected_fields():
    raw = {"id": "10000", "key": "ENG", "name": "Engineering", "projectTypeKey": "software"}
    assert format_jira_project(raw) == {
        "id": "10000",
        "key": "ENG",
        "name": "Engineering",
        "project_type_key": "software",
    }


def test_format_jira_issue_extracts_expected_fields_and_description():
    raw = {
        "key": "ENG-123",
        "fields": {
            "summary": "Fix the bug",
            "status": {"name": "In Progress"},
            "issuetype": {"name": "Bug"},
            "project": {"key": "ENG"},
            "assignee": {"displayName": "Jane Doe"},
            "reporter": {"displayName": "John Smith"},
            "created": "2026-01-01T00:00:00.000Z",
            "updated": "2026-01-02T00:00:00.000Z",
            "description": "Plain text description",
        },
    }
    formatted = format_jira_issue(raw)
    assert formatted["key"] == "ENG-123"
    assert formatted["status"] == "In Progress"
    assert formatted["assignee"] == "Jane Doe"
    assert formatted["description"] == "Plain text description"


def test_format_jira_issue_handles_unassigned_issue():
    raw = {"key": "ENG-1", "fields": {"summary": "x", "assignee": None, "reporter": None}}
    formatted = format_jira_issue(raw)
    assert formatted["assignee"] is None
    assert formatted["reporter"] is None


def test_format_jira_issue_can_omit_description():
    raw = {"key": "ENG-1", "fields": {"summary": "x", "description": "long text"}}
    formatted = format_jira_issue(raw, include_description=False)
    assert "description" not in formatted


def test_format_jira_comment_extracts_expected_fields():
    raw = {
        "id": "999",
        "author": {"displayName": "Jane Doe"},
        "body": "Looks good to me",
        "created": "2026-01-01T00:00:00.000Z",
        "updated": "2026-01-01T00:00:00.000Z",
    }
    formatted = format_jira_comment(raw)
    assert formatted["author"] == "Jane Doe"
    assert formatted["body"] == "Looks good to me"


def test_build_issue_create_fields_includes_description_when_present():
    fields = build_issue_create_fields("ENG", "Bug", "Fix it", "Steps to reproduce")
    assert fields == {
        "project": {"key": "ENG"},
        "issuetype": {"name": "Bug"},
        "summary": "Fix it",
        "description": "Steps to reproduce",
    }


def test_build_issue_create_fields_omits_description_when_blank():
    fields = build_issue_create_fields("ENG", "Task", "Do the thing")
    assert "description" not in fields


def test_build_issue_update_fields_only_includes_provided_values():
    assert build_issue_update_fields(summary="New summary") == {"summary": "New summary"}
    assert build_issue_update_fields(description="New description") == {
        "description": "New description"
    }
    assert build_issue_update_fields() == {}
