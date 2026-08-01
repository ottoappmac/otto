"""Pure-Python helpers for the Atlassian (Jira + Confluence) MCP.

Lives in a separate module (no ``@mcp.tool()`` decoration, no
``atlassian``/``requests`` imports) so unit tests can exercise
response-parsing logic without the third-party client library
installed — mirrors the split used by
:mod:`backend.builtin_mcps.macos_osascript._helpers`. The actual
``atlassian.Confluence`` / ``atlassian.Jira`` clients only get
imported in ``server.py``, which runs inside this MCP's own
uv-provisioned venv.
"""

from __future__ import annotations

from typing import Any, Optional

DEFAULT_LIMIT = 25
MAX_LIMIT = 100

# Surfaced when Confluence returns its characteristic "caller cannot
# access Confluence" 403 -- a StacklessResponseStatusException that
# Atlassian's edge raises regardless of whether the request itself
# (auth header, CQL, endpoint) was correct. In practice this almost
# always means either (a) the ATLASSIAN_EMAIL account has no
# Confluence product license on this site (e.g. Jira-only access), or
# (b) the API token was created via Atlassian's newer *scoped* token
# flow without a Confluence scope selected. Both are account/token
# configuration problems the client library can't detect ahead of
# time -- see https://id.atlassian.com/manage-profile/security/api-tokens.
CONFLUENCE_ACCESS_FORBIDDEN_HINT = (
    "Confluence Cloud rejected this request with 'caller cannot access "
    "Confluence' (403). This is almost always one of two things: (1) the "
    "ATLASSIAN_EMAIL account has no Confluence product license on this "
    "site -- confirm by logging into the site in a browser with that "
    "account and opening Confluence directly; or (2) the API token was "
    "created with Atlassian's newer 'scoped' token flow without a "
    "Confluence scope selected -- create a classic (unscoped) API token "
    "instead at https://id.atlassian.com/manage-profile/security/api-tokens "
    "and use that."
)


def is_confluence_access_forbidden(message: Any) -> bool:
    """Whether an error message is Confluence's "cannot access" 403.

    Matched by substring rather than status code alone, since Atlassian
    raises this exact wording for the specific licensing/token-scope
    failure the hint addresses -- other 403s (e.g. real page
    permission restrictions) get no hint because it wouldn't apply.
    """
    return "cannot access confluence" in str(message or "").casefold()

# Atlassian Document Format node types that represent a line/block —
# used by :func:`adf_to_text` to decide where to insert a newline.
_ADF_BLOCK_TYPES = frozenset(
    {"paragraph", "heading", "listItem", "codeBlock", "blockquote", "tableRow"}
)


class AtlassianAPIError(RuntimeError):
    """Raised for a non-2xx Jira/Confluence Cloud REST response."""

    def __init__(self, status_code: Optional[int], message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Atlassian API error ({status_code}): {message}")


def classify_error(status_code: Optional[int], message: Any) -> AtlassianAPIError:
    """Build an :class:`AtlassianAPIError` from a status code + message.

    ``atlassian-python-api`` already extracts a readable message from
    Jira/Confluence's ``errorMessages``/``errors`` JSON shape and
    raises it as the ``str()`` of a ``requests.HTTPError`` — this just
    normalizes that (plus the "not an HTTP error at all", e.g. a
    connection failure) into one error shape for tool results.
    """
    return AtlassianAPIError(status_code, str(message) if message else "request failed")


def clamp_limit(limit: int, *, default: int = DEFAULT_LIMIT, max_limit: int = MAX_LIMIT) -> int:
    """Clamp a user-supplied page size into ``[1, max_limit]``.

    Falls back to *default* for non-positive or non-numeric input rather
    than raising — pagination knobs should never be the reason a tool call
    fails.
    """
    try:
        value = int(limit)
    except (TypeError, ValueError):
        return default
    if value <= 0:
        return default
    return min(value, max_limit)


def adf_to_text(value: Any) -> str:
    """Best-effort plain-text extraction from a Jira rich-text field.

    Jira Cloud always returns ``description``/comment ``body`` as
    Atlassian Document Format (ADF) — a nested JSON tree — even for
    the v2 API this MCP uses for writes (v2 still accepts a plain
    string on create/update; it's only reads that come back as ADF on
    Cloud). Jira Server/Data Center returns a plain string either way.
    This function accepts any of those shapes and always returns a
    flat string, so callers never need to branch on which one they got.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return str(value)
    parts: list[str] = []
    _adf_walk(value, parts)
    lines = [line.strip() for line in "".join(parts).split("\n")]
    return "\n".join(line for line in lines if line)


def _adf_walk(node: Any, parts: list[str]) -> None:
    if isinstance(node, dict):
        node_type = node.get("type")
        if node_type == "text":
            parts.append(str(node.get("text", "")))
        elif node_type == "hardBreak":
            parts.append("\n")
        for child in node.get("content", []) or []:
            _adf_walk(child, parts)
        if node_type in _ADF_BLOCK_TYPES:
            parts.append("\n")
    elif isinstance(node, list):
        for child in node:
            _adf_walk(child, parts)


def format_confluence_space(raw: dict[str, Any]) -> dict[str, Any]:
    """Shrink a Confluence space object down to the fields agents need."""
    return {
        "id": raw.get("id"),
        "key": raw.get("key", ""),
        "name": raw.get("name", ""),
        "type": raw.get("type", ""),
        "status": raw.get("status", ""),
    }


def _confluence_webui_url(raw: dict[str, Any]) -> str:
    links = raw.get("_links") or {}
    base = links.get("base", "")
    webui = links.get("webui", "")
    if base and webui:
        return f"{base}{webui}"
    return webui or ""


def format_confluence_page(raw: dict[str, Any], *, include_body: bool = True) -> dict[str, Any]:
    """Shrink a Confluence content object down to the fields agents need."""
    space = raw.get("space") or {}
    version = raw.get("version") or {}
    result: dict[str, Any] = {
        "id": raw.get("id", ""),
        "title": raw.get("title", ""),
        "type": raw.get("type", ""),
        "status": raw.get("status", ""),
        "space_key": space.get("key", ""),
        "version": version.get("number"),
        "url": _confluence_webui_url(raw),
    }
    if include_body:
        result["body_html"] = ((raw.get("body") or {}).get("storage") or {}).get("value", "")
    return result


def format_confluence_search_result(raw: dict[str, Any]) -> dict[str, Any]:
    """Shrink one ``cql()`` search hit down to the fields agents need."""
    content = raw.get("content") or {}
    space = content.get("space") or {}
    return {
        "id": content.get("id", ""),
        "title": raw.get("title") or content.get("title", ""),
        "type": content.get("type", ""),
        "space_key": space.get("key", ""),
        "excerpt": raw.get("excerpt", ""),
        "url": raw.get("url", ""),
        "last_modified": raw.get("lastModified", ""),
    }


def format_jira_project(raw: dict[str, Any]) -> dict[str, Any]:
    """Shrink a Jira project object down to the fields agents need."""
    return {
        "id": raw.get("id", ""),
        "key": raw.get("key", ""),
        "name": raw.get("name", ""),
        "project_type_key": raw.get("projectTypeKey", ""),
    }


def format_jira_issue(raw: dict[str, Any], *, include_description: bool = True) -> dict[str, Any]:
    """Shrink a Jira issue object down to the fields agents need."""
    fields = raw.get("fields") or {}
    status = fields.get("status") or {}
    issuetype = fields.get("issuetype") or {}
    project = fields.get("project") or {}
    assignee = fields.get("assignee") or {}
    reporter = fields.get("reporter") or {}
    result: dict[str, Any] = {
        "key": raw.get("key", ""),
        "summary": fields.get("summary", ""),
        "status": status.get("name", ""),
        "issue_type": issuetype.get("name", ""),
        "project_key": project.get("key", ""),
        "assignee": assignee.get("displayName") if assignee else None,
        "reporter": reporter.get("displayName") if reporter else None,
        "created": fields.get("created", ""),
        "updated": fields.get("updated", ""),
    }
    if include_description:
        result["description"] = adf_to_text(fields.get("description"))
    return result


def format_jira_comment(raw: dict[str, Any]) -> dict[str, Any]:
    """Shrink a Jira comment object down to the fields agents need."""
    author = raw.get("author") or {}
    return {
        "id": raw.get("id", ""),
        "author": author.get("displayName", ""),
        "body": adf_to_text(raw.get("body")),
        "created": raw.get("created", ""),
        "updated": raw.get("updated", ""),
    }


def build_issue_create_fields(
    project_key: str, issue_type: str, summary: str, description: str = "",
) -> dict[str, Any]:
    """Build the ``fields`` payload for ``Jira.create_issue``.

    Plain strings are fine for ``description`` here — this MCP's Jira
    client defaults to API v2 (see ``server.py``), which still accepts
    a plain string on write even though Jira Cloud renders it (and
    returns it on read) as Atlassian Document Format.
    """
    fields: dict[str, Any] = {
        "project": {"key": project_key},
        "issuetype": {"name": issue_type},
        "summary": summary,
    }
    if description:
        fields["description"] = description
    return fields


def build_issue_update_fields(summary: str = "", description: str = "") -> dict[str, Any]:
    """Build a partial ``fields`` payload for ``Jira.update_issue_field``.

    Only includes keys the caller actually supplied, so an update never
    clobbers a field the agent didn't mean to touch.
    """
    fields: dict[str, Any] = {}
    if summary:
        fields["summary"] = summary
    if description:
        fields["description"] = description
    return fields
