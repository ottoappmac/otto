# macOS Reminders (built-in MCP)

Structured Create/Read/Update/Delete tools over Apple Reminders,
implemented by generating AppleScript against Reminders' own dictionary
and running it via `osascript`.

| Tool | Purpose |
|------|---------|
| `list_lists` | List every reminder list and its reminder count. |
| `list_reminders` | List reminders in one list (incomplete only, by default). |
| `get_reminder` | Fetch one reminder's full detail by id. |
| `create_reminder` | Create a reminder with optional notes, due/alert date, and priority. |
| `update_reminder` | Rename, re-note, re-date, re-prioritise, or complete/reopen a reminder. |
| `delete_reminder` | Permanently delete a reminder. |

## Why a dedicated Reminders MCP instead of `macos-osascript`'s `run_osascript`?

`macos-osascript` already exposes a generic `run_osascript`, but using it
for Reminders means the agent has to author correct AppleScript for every
call and knows things it usually gets wrong:

* **Priority is an integer scale**, not a word — Reminders stores
  `0 = none, 1 = high, 5 = medium, 9 = low`. This MCP takes
  `priority="high"` and maps it.
* **Dates must be locale-safe.** Building a date via `date "…"` depends on
  the host's regional format. This MCP parses your ISO string in Python
  and constructs the AppleScript `date` from components, avoiding that
  whole class of bugs.
* A due date alone doesn't raise an alert — Reminders notifies off
  `remind me date`. `create_reminder` mirrors your due date into the
  alert time by default so a reminder you create actually reminds you.

## Required permission: Automation (not Full Disk Access)

The first time any tool here talks to Reminders, macOS shows an
**Automation** permission prompt (per requesting app) — approve it. To
re-grant: **System Settings → Privacy & Security → Automation**, find the
app running Otto's backend, and make sure **Reminders** is checked under
it. No Full Disk Access is required.

## Known limitations

* **Deletion is permanent** — Reminders has no Trash. Prefer completing a
  reminder (`update_reminder(..., completed=True)`) unless you truly want
  it gone.
* `get_reminder`/`update_reminder`/`delete_reminder` accept an empty
  `list_name` and will then search every reminder in the app to find the
  id — correct, but slower. Pass the `list_name` a reminder was listed
  with whenever you have it.
* Read tools return dates as Reminders' own string form (locale
  dependent), matching the raw AppleScript value; date *inputs* are the
  locale-safe ISO strings described above.
* No support for subtasks, tags, or smart lists — those aren't exposed by
  the Reminders AppleScript dictionary.

## Why is this a built-in MCP?

The canonical source lives at
`backend/builtin_mcps/macos_reminders/server.py` so the orchestrator gets
these tools out of the box, macOS-only, with no setup beyond approving the
Automation prompt above. On every backend startup the file is copied into
`mcp_server/macos_reminders/`, the venv is provisioned with `uv`, and the
registered `MCPServerConfig` points at
`<dir>/.venv/bin/python <dir>/server.py`.
