# macOS Calendar (built-in MCP)

Structured Create/Read/Update/Delete tools over Apple Calendar,
implemented by generating AppleScript against Calendar's own dictionary
and running it via `osascript`.

| Tool | Purpose |
|------|---------|
| `list_calendars` | List every calendar (name, writable, description). |
| `list_events` | List events overlapping a date range, in one calendar or all. |
| `get_event` | Fetch one event's full detail by uid. |
| `create_event` | Create an event with optional location, notes, and all-day flag. |
| `update_event` | Change an event's title/time/location/notes/all-day flag. |
| `delete_event` | Permanently delete an event. |

## Why a dedicated Calendar MCP instead of `macos-osascript`'s `run_osascript`?

`macos-osascript` already exposes a generic `run_osascript`, but driving
Calendar through it means the agent has to author correct AppleScript and
gets bitten by the same things every time:

* **Dates must be locale-safe.** Building a date with `date "…"` depends on
  the host's regional format. This MCP parses your ISO string in Python
  and constructs the AppleScript `date` from components.
* **Range queries are non-obvious.** "Events this week" is an overlap
  test (`start date <= rangeEnd and end date >= rangeStart`), not a naive
  `start date` window. This MCP builds that predicate for you and returns
  each hit already addressable by `uid` + `calendar_name`.

## Required permission: Automation (not Full Disk Access)

The first time any tool here talks to Calendar, macOS shows an
**Automation** permission prompt (per requesting app) — approve it. To
re-grant: **System Settings → Privacy & Security → Automation**, find the
app running Otto's backend, and make sure **Calendar** is checked under
it. No Full Disk Access is required.

## Known limitations

* **Deletion is permanent** — Calendar has no per-event Trash via
  AppleScript.
* **Recurring events are matched by their master only.** The dictionary
  doesn't expand individual occurrences, so a weekly meeting appears once
  in a range, not once per week, and editing/deleting it affects the
  whole series.
* `list_events`' `whose` range filter can be slow on very large calendars
  — narrow with `start_date`/`end_date`, scope to one `calendar_name`, or
  raise `timeout_seconds` (up to 120).
* `create_event` requires an explicit `calendar_name`; the AppleScript
  dictionary has no notion of a "default" calendar.
* Attendees/invitations and alarms aren't exposed as writable typed
  arguments here — use `macos-osascript` for those advanced cases.

## Why is this a built-in MCP?

The canonical source lives at
`backend/builtin_mcps/macos_calendar/server.py` so the orchestrator gets
these tools out of the box, macOS-only, with no setup beyond approving the
Automation prompt above. On every backend startup the file is copied into
`mcp_server/macos_calendar/`, the venv is provisioned with `uv`, and the
registered `MCPServerConfig` points at
`<dir>/.venv/bin/python <dir>/server.py`.
