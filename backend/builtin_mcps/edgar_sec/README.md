# SEC EDGAR Filings (built-in MCP)

Read-only tools over the SEC's free EDGAR APIs:

| Tool | Purpose |
|------|---------|
| `search_filings` | Full-text search across 18M+ filings on `efts.sec.gov`. |
| `get_company_submissions` | Recent filing list and company profile. |
| `search_company_by_ticker` | Resolve ticker / company name → CIK. |
| `get_company_facts` | XBRL financial concepts (`us-gaap`, `ifrs-full`, `dei`). |
| `get_filing_document` | Document index for one accession number. |
| `get_xbrl_frames` | Cross-company snapshot of one concept in one period. |

## Required credentials

* `EDGAR_USER_AGENT` — a contact string the SEC uses to identify the
  caller, e.g. `"Your Name your-email@example.com"`. Stored in the OS
  keychain via the credential vault and only injected into the
  subprocess environment at spawn time. See
  <https://www.sec.gov/os/accessing-edgar-data>.

## Why is this a built-in MCP?

The canonical source lives at
`backend/builtin_mcps/edgar_sec/server.py` so the orchestrator gets
SEC tools out of the box without the user having to author them via
`mcp_builder`. On every backend startup the file is copied into the
per-MCP folder under `mcp_server/edgar-sec/`, the venv is provisioned
with `uv` (fast, isolated), and the registered `MCPServerConfig` points
at `<dir>/.venv/bin/python <dir>/server.py`.

To regenerate or modify, edit the source files in this folder and
restart the backend — the bootstrap layer overwrites the deployed copy
and reprovisions the venv when `requirements.txt` changes.
