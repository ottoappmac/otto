#!/usr/bin/env python3
"""Built-in MCP server: Atlassian (Jira + Confluence).

Read + write tools over Atlassian Cloud's Jira and Confluence REST APIs,
via the ``atlassian-python-api`` client library:

Confluence:

* ``confluence_search`` / ``confluence_list_spaces``
* ``confluence_get_page`` / ``confluence_get_page_by_title``
* ``confluence_create_page`` / ``confluence_update_page`` / ``confluence_add_comment``

Jira:

* ``jira_search_issues`` / ``jira_get_issue`` / ``jira_list_projects``
* ``jira_create_issue`` / ``jira_update_issue`` / ``jira_add_comment``
* ``jira_get_transitions`` / ``jira_transition_issue``

Auth is a single Atlassian Cloud site + email + API token
(``ATLASSIAN_URL`` / ``ATLASSIAN_EMAIL`` / ``ATLASSIAN_API_TOKEN``) — the
same credentials work for both products because Jira and Confluence
share one site and one Atlassian account. The user generates an API
token at https://id.atlassian.com/manage-profile/security/api-tokens
and pastes it, along with their site URL (e.g.
``https://yourteam.atlassian.net``) and Atlassian account email, into
Otto's credentials dialog for this MCP.

This MCP targets **Atlassian Cloud only** — Atlassian retired Server
in February 2024, and Data Center's auth model (personal access
tokens, no Basic Auth with API tokens) is different enough that it
isn't wired up here.

This file is the canonical source for the ``atlassian`` builtin MCP.
The backend copies it into ``mcp_server/atlassian/`` on every startup
and runs it inside a uv-provisioned venv (``mcp[cli]`` +
``atlassian-python-api``) — see :mod:`backend.builtin_mcps.registry`.

Trust boundaries:

* Only the Atlassian Cloud site the user configured is reached.
* ``ATLASSIAN_API_TOKEN`` is read from the environment (hydrated from
  the OS keychain by the parent backend at spawn time) and never
  logged or echoed back in a tool result.
* Confluence page/comment bodies are exchanged in Confluence *storage
  format* (XHTML-like). Jira descriptions/comments are plain text —
  this client defaults to Jira's v2 API, which still accepts a plain
  string on write even though reads come back as Atlassian Document
  Format (see ``_helpers.adf_to_text``).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from atlassian import Confluence, Jira
from mcp.server.fastmcp import FastMCP

# Pure-Python helpers live in a sibling module so unit tests can import
# them without the ``atlassian`` package installed. Two import paths
# are supported because this file runs in two contexts: spawned directly
# as ``python <path-to-server.py>`` in production (script dir on
# sys.path, ``_helpers`` resolves as a top-level module) vs. imported as
# ``backend.builtin_mcps.atlassian.server`` from tests (relative import).
try:
    from ._helpers import (  # type: ignore[import-not-found]
        CONFLUENCE_ACCESS_FORBIDDEN_HINT,
        build_issue_create_fields,
        build_issue_update_fields,
        classify_error,
        clamp_limit,
        format_confluence_page,
        format_confluence_search_result,
        format_confluence_space,
        format_jira_comment,
        format_jira_issue,
        format_jira_project,
        is_confluence_access_forbidden,
    )
except ImportError:
    from _helpers import (  # type: ignore[no-redef]
        CONFLUENCE_ACCESS_FORBIDDEN_HINT,
        build_issue_create_fields,
        build_issue_update_fields,
        classify_error,
        clamp_limit,
        format_confluence_page,
        format_confluence_search_result,
        format_confluence_space,
        format_jira_comment,
        format_jira_issue,
        format_jira_project,
        is_confluence_access_forbidden,
    )

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("otto.mcp.atlassian")

mcp = FastMCP("Atlassian")

# Lazily built and cached for the life of the subprocess — credentials
# are only read from the environment once per spawn (see mcp_manager,
# which hydrates the vault into the environment at spawn time and never
# again mid-process).
_confluence_client: Optional[Confluence] = None
_jira_client: Optional[Jira] = None


def _credentials() -> tuple[str, str, str]:
    url = (os.environ.get("ATLASSIAN_URL") or "").strip().rstrip("/")
    email = (os.environ.get("ATLASSIAN_EMAIL") or "").strip()
    token = (os.environ.get("ATLASSIAN_API_TOKEN") or "").strip()
    if not (url and email and token):
        raise ValueError(
            "ATLASSIAN_URL / ATLASSIAN_EMAIL / ATLASSIAN_API_TOKEN are not "
            "all set. Configure them via Tools page → Atlassian → "
            "Credentials: your site URL (e.g. https://yourteam.atlassian.net), "
            "your Atlassian account email, and an API token from "
            "https://id.atlassian.com/manage-profile/security/api-tokens."
        )
    return url, email, token


def _confluence() -> Confluence:
    global _confluence_client
    if _confluence_client is None:
        url, email, token = _credentials()
        _confluence_client = Confluence(url=url, username=email, password=token, cloud=True)
    return _confluence_client


def _jira() -> Jira:
    global _jira_client
    if _jira_client is None:
        url, email, token = _credentials()
        _jira_client = Jira(url=url, username=email, password=token, cloud=True)
    return _jira_client


def _error(exc: Exception, *, confluence: bool = False) -> dict[str, Any]:
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    wrapped = classify_error(status_code, str(exc))
    result: dict[str, Any] = {"error": wrapped.message, "status_code": wrapped.status_code}
    # Confluence's "caller cannot access Confluence" 403 is a distinctive
    # account/token-configuration failure (no Confluence license, or a
    # scoped API token missing the Confluence scope) rather than a normal
    # permission error -- worth a pointed hint since the raw message alone
    # doesn't suggest either cause.
    if confluence and is_confluence_access_forbidden(wrapped.message):
        result["hint"] = CONFLUENCE_ACCESS_FORBIDDEN_HINT
    return result


# ---------------------------------------------------------------------------
# Confluence
# ---------------------------------------------------------------------------


@mcp.tool()
def confluence_search(cql: str, limit: int = 25) -> dict[str, Any]:
    """Search Confluence content with CQL (Confluence Query Language).

    Args:
        cql: A CQL query, e.g. ``type=page AND space=ENG AND text ~ "onboarding"``.
        limit: Max results (1-100).
    """
    try:
        # Without expand, search hits return a bare "content" stub with no
        # nested space info, so format_confluence_search_result would have
        # nothing to pull space_key from.
        payload = _confluence().cql(cql, limit=clamp_limit(limit), expand="content.space")
    except Exception as exc:  # noqa: BLE001 - surfaced as a structured error
        return _error(exc, confluence=True)
    results = [format_confluence_search_result(r) for r in (payload or {}).get("results", [])]
    return {"results": results}


@mcp.tool()
def confluence_list_spaces(limit: int = 25) -> dict[str, Any]:
    """List Confluence spaces visible to this account.

    Args:
        limit: Max spaces to return (1-100).
    """
    try:
        payload = _confluence().get_all_spaces(limit=clamp_limit(limit))
    except Exception as exc:
        return _error(exc, confluence=True)
    spaces = [format_confluence_space(s) for s in (payload or {}).get("results", [])]
    return {"spaces": spaces}


@mcp.tool()
def confluence_get_page(page_id: str, include_body: bool = True) -> dict[str, Any]:
    """Get a Confluence page by id.

    Args:
        page_id: Confluence content id.
        include_body: Whether to include the page body (storage/XHTML format).
    """
    try:
        expand = "space,version"
        if include_body:
            expand += ",body.storage"
        raw = _confluence().get_page_by_id(page_id, expand=expand)
    except Exception as exc:
        return _error(exc, confluence=True)
    return format_confluence_page(raw, include_body=include_body)


@mcp.tool()
def confluence_get_page_by_title(space_key: str, title: str) -> dict[str, Any]:
    """Look up a Confluence page by its space key and exact title.

    Args:
        space_key: Confluence space key (e.g. ``ENG``) — not the space name.
        title: Exact page title.
    """
    try:
        raw = _confluence().get_page_by_title(
            space_key, title, expand="space,version,body.storage",
        )
    except Exception as exc:
        return _error(exc, confluence=True)
    if not raw:
        return {"error": f"No page titled {title!r} found in space {space_key!r}."}
    return format_confluence_page(raw)


@mcp.tool()
def confluence_create_page(
    space_key: str, title: str, body_html: str, parent_id: str = "",
) -> dict[str, Any]:
    """Create a new Confluence page.

    Args:
        space_key: Confluence space key to create the page in.
        title: Page title.
        body_html: Page body in Confluence storage format (XHTML-like —
            plain tags like ``<p>``, ``<h1>``, ``<ul>`` work).
        parent_id: Optional parent page id to nest this page under.
    """
    try:
        raw = _confluence().create_page(
            space=space_key, title=title, body=body_html, parent_id=parent_id or None,
        )
    except Exception as exc:
        return _error(exc, confluence=True)
    return format_confluence_page(raw)


@mcp.tool()
def confluence_update_page(
    page_id: str, body_html: str, title: str = "", version_comment: str = "",
) -> dict[str, Any]:
    """Update an existing Confluence page's title and/or body.

    Args:
        page_id: Confluence content id to update.
        body_html: New page body in Confluence storage format. Replaces the
            existing body entirely.
        title: New title. Leave blank to keep the current title.
        version_comment: Optional edit summary shown in the page history.
    """
    try:
        client = _confluence()
        if not title:
            current = client.get_page_by_id(page_id)
            title = current.get("title", "")
        raw = client.update_page(
            page_id,
            title,
            body=body_html,
            version_comment=version_comment or None,
            # Suppress the library's own "is the body already identical"
            # check — it diffs our raw storage-format HTML against
            # Confluence's server-rendered copy, which reformats enough
            # that a genuine change can be misdetected as a no-op.
            always_update=True,
        )
    except Exception as exc:
        return _error(exc, confluence=True)
    return format_confluence_page(raw)


@mcp.tool()
def confluence_add_comment(page_id: str, comment_html: str) -> dict[str, Any]:
    """Add a comment to a Confluence page.

    Args:
        page_id: Confluence content id to comment on.
        comment_html: Comment body in Confluence storage format.
    """
    try:
        raw = _confluence().add_comment(page_id, comment_html)
    except Exception as exc:
        return _error(exc, confluence=True)
    return {"ok": True, "comment_id": (raw or {}).get("id", "")}


# ---------------------------------------------------------------------------
# Jira
# ---------------------------------------------------------------------------


@mcp.tool()
def jira_search_issues(jql: str, limit: int = 25) -> dict[str, Any]:
    """Search Jira issues with JQL (Jira Query Language).

    Args:
        jql: A JQL query, e.g. ``project = ENG AND status = "In Progress"``.
        limit: Max issues to return (1-100).
    """
    try:
        payload = _jira().jql(
            jql,
            fields=[
                "summary", "status", "issuetype", "project",
                "assignee", "reporter", "created", "updated",
            ],
            limit=clamp_limit(limit),
        )
    except Exception as exc:
        return _error(exc)
    issues = [
        format_jira_issue(i, include_description=False)
        for i in (payload or {}).get("issues", [])
    ]
    return {"issues": issues, "total": (payload or {}).get("total", len(issues))}


@mcp.tool()
def jira_get_issue(issue_key: str) -> dict[str, Any]:
    """Get full details for one Jira issue, including its description.

    Args:
        issue_key: Issue key (e.g. ``ENG-123``).
    """
    try:
        raw = _jira().issue(issue_key)
    except Exception as exc:
        return _error(exc)
    return format_jira_issue(raw)


@mcp.tool()
def jira_list_projects() -> dict[str, Any]:
    """List Jira projects visible to this account."""
    try:
        payload = _jira().projects()
    except Exception as exc:
        return _error(exc)
    return {"projects": [format_jira_project(p) for p in (payload or [])]}


@mcp.tool()
def jira_create_issue(
    project_key: str, issue_type: str, summary: str, description: str = "",
) -> dict[str, Any]:
    """Create a new Jira issue.

    Args:
        project_key: Jira project key (e.g. ``ENG``).
        issue_type: Issue type name (e.g. ``Task``, ``Bug``, ``Story``).
        summary: Issue summary/title.
        description: Optional plain-text description.
    """
    try:
        fields = build_issue_create_fields(project_key, issue_type, summary, description)
        raw = _jira().create_issue(fields=fields)
    except Exception as exc:
        return _error(exc)
    return {"key": raw.get("key", ""), "id": raw.get("id", "")}


@mcp.tool()
def jira_update_issue(issue_key: str, summary: str = "", description: str = "") -> dict[str, Any]:
    """Update an existing Jira issue's summary and/or description.

    Args:
        issue_key: Issue key (e.g. ``ENG-123``).
        summary: New summary. Leave blank to keep the current one.
        description: New plain-text description. Leave blank to keep the
            current one.
    """
    fields = build_issue_update_fields(summary, description)
    if not fields:
        return {"error": "Provide at least one of summary or description."}
    try:
        _jira().update_issue_field(issue_key, fields)
    except Exception as exc:
        return _error(exc)
    return {"ok": True, "key": issue_key}


@mcp.tool()
def jira_add_comment(issue_key: str, comment: str) -> dict[str, Any]:
    """Add a plain-text comment to a Jira issue.

    Args:
        issue_key: Issue key (e.g. ``ENG-123``).
        comment: Comment text.
    """
    try:
        raw = _jira().issue_add_comment(issue_key, comment)
    except Exception as exc:
        return _error(exc)
    return format_jira_comment(raw or {})


@mcp.tool()
def jira_get_transitions(issue_key: str) -> dict[str, Any]:
    """List the workflow transitions available for a Jira issue right now.

    Args:
        issue_key: Issue key (e.g. ``ENG-123``).

    Use the returned transition ``id`` with ``jira_transition_issue`` to
    change status — the set of valid transitions depends on the issue's
    current status and the project's workflow.
    """
    try:
        transitions = _jira().get_issue_transitions(issue_key)
    except Exception as exc:
        return _error(exc)
    return {"transitions": transitions}


@mcp.tool()
def jira_transition_issue(issue_key: str, transition_id: str) -> dict[str, Any]:
    """Move a Jira issue through a workflow transition (e.g. change its status).

    Args:
        issue_key: Issue key (e.g. ``ENG-123``).
        transition_id: A transition id from ``jira_get_transitions`` — not a
            status name.
    """
    try:
        _jira().set_issue_status_by_transition_id(issue_key, transition_id)
    except Exception as exc:
        return _error(exc)
    return {"ok": True, "key": issue_key}


if __name__ == "__main__":
    mcp.run()
