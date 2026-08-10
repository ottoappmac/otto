# Atlassian — Jira + Confluence (built-in MCP)

Read + write tools over Atlassian Cloud's Jira and Confluence REST APIs.

| Tool | Purpose |
|------|---------|
| `confluence_search` | Search Confluence content with CQL. |
| `confluence_list_spaces` | List spaces visible to this account. |
| `confluence_get_page` | Get a page by id (title, body, version, url). |
| `confluence_get_page_by_title` | Look up a page by space key + exact title. |
| `confluence_create_page` | Create a new page. |
| `confluence_update_page` | Update a page's title and/or body. |
| `confluence_add_comment` | Comment on a page. |
| `jira_search_issues` | Search issues with JQL. |
| `jira_get_issue` | Get full details for one issue. |
| `jira_list_projects` | List projects visible to this account. |
| `jira_create_issue` | Create a new issue. |
| `jira_update_issue` | Update an issue's summary and/or description. |
| `jira_add_comment` | Comment on an issue. |
| `jira_get_transitions` | List workflow transitions available right now. |
| `jira_transition_issue` | Move an issue through a transition (change status). |

## Required credentials

* `ATLASSIAN_URL` — your site URL, e.g. `https://yourteam.atlassian.net`.
* `ATLASSIAN_EMAIL` — the email of the Atlassian account the API token
  belongs to.
* `ATLASSIAN_API_TOKEN` — an API token for that account.

The same three values authenticate both Jira and Confluence, since they
live on the same Atlassian Cloud site and account.

**Cloud only.** Atlassian retired Server in February 2024, and Data
Center uses personal access tokens instead of Basic Auth with an API
token — a different enough auth model that it isn't wired up here.

## Setup walkthrough

### 1. Generate an API token

1. Go to <https://id.atlassian.com/manage-profile/security/api-tokens>
   while signed in to the Atlassian account you want Otto to act as.
2. Click **Create API token**, give it a label (e.g. "Otto"), and copy
   the token — you won't be able to see it again.

### 2. Find your site URL

This is the URL you use to open Jira or Confluence in a browser, e.g.
`https://yourteam.atlassian.net` (no trailing `/jira` or `/wiki`).

### 3. Connect it in Otto

1. In Otto, go to the **Agents** page → **Tools** tab.
2. Find the **Atlassian** card and click **Credentials**.
3. Fill in `ATLASSIAN_URL`, `ATLASSIAN_EMAIL` (the account from step 1),
   and `ATLASSIAN_API_TOKEN` (the token from step 1), then save.
4. Click **Start** (or toggle the server on) to connect it.

Once connected, the agent has all the tools above available, scoped to
whatever the account from step 1 can see in Jira and Confluence
(standard Atlassian permissions apply — the API token doesn't grant
anything the account itself can't already do).

## Known limitations

* **`403 ... "Request rejected because caller cannot access Confluence"`**
  on every Confluence tool (Jira tools working fine) means Confluence
  Cloud rejected the request before even checking space/page
  permissions. This is almost always an account or token configuration
  problem, not a bug in the request:
  * The `ATLASSIAN_EMAIL` account has no Confluence product license on
    this site (e.g. it only has Jira access). Confirm by logging into
    the site in a browser with that account and opening Confluence
    directly.
  * The API token was created with Atlassian's newer *scoped* token
    flow (at id.atlassian.com) without a Confluence scope selected.
    Create a classic (unscoped) API token instead and use that for
    `ATLASSIAN_API_TOKEN`.
* Confluence page/comment bodies are exchanged in Confluence *storage
  format* (an XHTML-like dialect) — the agent needs to write basic
  storage-format markup (`<p>`, `<h1>`–`<h6>`, `<ul>`/`<li>`, `<strong>`,
  `<a href="...">`, etc.), not Markdown.
* Jira descriptions and comments are read/written as plain text.
  Writes go through Jira's v2 API (which still accepts a plain string),
  but Jira Cloud always renders reads as Atlassian Document Format —
  this MCP's `_helpers.adf_to_text` flattens that back to plain text,
  so structure like tables or embedded images in an existing
  description won't round-trip.
* `jira_transition_issue` needs a transition *id*, not a status name —
  call `jira_get_transitions` first to see what's actually available for
  that issue's current status and workflow.
* No attachment upload/download, no Jira sprint/board tools, no
  Confluence page deletion — this MCP covers the common
  search/read/create/comment/status-change loop, not every endpoint
  either product exposes.

## Why is this a built-in MCP?

The canonical source lives at `backend/builtin_mcps/atlassian/server.py`
so the orchestrator gets Jira/Confluence tools out of the box without
the user having to author them via `mcp_builder`. On every backend
startup the file is copied into `mcp_server/atlassian/`, the venv is
provisioned with `uv` (installing `atlassian-python-api`), and the
registered `MCPServerConfig` points at
`<dir>/.venv/bin/python <dir>/server.py`.
