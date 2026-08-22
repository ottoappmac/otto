"""HTTP MCP header ingest/expand — Snowflake PAT and Cursor mcp.json."""

from __future__ import annotations

import pytest

from backend.config import MCPAuthConfig, MCPServerConfig
from backend.mcp_headers import (
    MissingHeaderSecret,
    expand_header_templates,
    fallback_auth_header,
    ingest_http_headers,
    shouty_id,
)
from backend.mcp_manager import _resolve_http_headers


def test_shouty_id():
    assert shouty_id("snowflake") == "SNOWFLAKE"
    assert shouty_id("my-mcp") == "MY_MCP"


def test_ingest_env_template_keeps_placeholder():
    templates, secrets, vault = ingest_http_headers("snowflake", {
        "headers": {"Authorization": "Bearer ${SNOWFLAKE_PAT_TOKEN}"},
    })
    assert templates == {"Authorization": "Bearer ${SNOWFLAKE_PAT_TOKEN}"}
    assert secrets == ["SNOWFLAKE_PAT_TOKEN"]
    assert vault == {}


def test_ingest_literal_bearer_extracts_to_vault():
    templates, secrets, vault = ingest_http_headers("snowflake", {
        "headers": {"Authorization": "Bearer tok_abc"},
    })
    assert templates == {"Authorization": "Bearer ${SNOWFLAKE_PAT_TOKEN}"}
    assert secrets == ["SNOWFLAKE_PAT_TOKEN"]
    assert vault == {"SNOWFLAKE_PAT_TOKEN": "tok_abc"}


def test_ingest_other_literal_header():
    templates, secrets, vault = ingest_http_headers("acme", {
        "headers": {"X-Api-Key": "k-secret"},
    })
    assert templates == {"X-Api-Key": "${ACME_X_API_KEY}"}
    assert secrets == ["ACME_X_API_KEY"]
    assert vault == {"ACME_X_API_KEY": "k-secret"}


def test_expand_header_templates():
    out = expand_header_templates(
        {"Authorization": "Bearer ${SNOWFLAKE_PAT_TOKEN}"},
        {"SNOWFLAKE_PAT_TOKEN": "tok_abc"},
    )
    assert out == {"Authorization": "Bearer tok_abc"}


def test_expand_raises_when_secret_missing():
    with pytest.raises(MissingHeaderSecret):
        expand_header_templates(
            {"Authorization": "Bearer ${SNOWFLAKE_PAT_TOKEN}"},
            {},
        )


def test_fallback_auth_header_prefixes_bearer():
    out = fallback_auth_header(
        "Authorization", "Bearer ",
        ["SNOWFLAKE_PAT_TOKEN"],
        {"SNOWFLAKE_PAT_TOKEN": "tok_abc"},
    )
    assert out == {"Authorization": "Bearer tok_abc"}


def test_fallback_does_not_double_prefix():
    out = fallback_auth_header(
        "Authorization", "Bearer ",
        ["SNOWFLAKE_PAT_TOKEN"],
        {"SNOWFLAKE_PAT_TOKEN": "Bearer tok_abc"},
    )
    assert out == {"Authorization": "Bearer tok_abc"}


def test_resolve_uses_templates(monkeypatch):
    cfg = MCPServerConfig(
        id="snowflake",
        name="Snowflake",
        transport="streamable_http",
        url="https://example.snowflakecomputing.com/mcp",
        headers={"Authorization": "Bearer ${SNOWFLAKE_PAT_TOKEN}"},
        required_secrets=["SNOWFLAKE_PAT_TOKEN"],
        auth=MCPAuthConfig(header_name="Authorization", token_prefix="Bearer "),
    )
    monkeypatch.setattr(
        "backend.mcp_manager._hydrate_secrets",
        lambda _c: {"SNOWFLAKE_PAT_TOKEN": "tok_abc"},
    )
    assert _resolve_http_headers(cfg) == {"Authorization": "Bearer tok_abc"}


def test_resolve_fallback_without_templates(monkeypatch):
    cfg = MCPServerConfig(
        id="snowflake",
        name="Snowflake",
        transport="streamable_http",
        url="https://example.snowflakecomputing.com/mcp",
        required_secrets=["SNOWFLAKE_PAT_TOKEN"],
        auth=MCPAuthConfig(header_name="Authorization", token_prefix="Bearer "),
    )
    monkeypatch.setattr(
        "backend.mcp_manager._hydrate_secrets",
        lambda _c: {"SNOWFLAKE_PAT_TOKEN": "tok_abc"},
    )
    assert _resolve_http_headers(cfg) == {"Authorization": "Bearer tok_abc"}
