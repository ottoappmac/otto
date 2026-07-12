# macOS Messages (built-in MCP)

Typed tools over Apple Messages. Messages is unusual: its AppleScript
dictionary can **send** messages and enumerate services/buddies/chats, but
exposes **no readable message element** — so reading history is only
possible from the `chat.db` SQLite store. This MCP spans both surfaces.

| Tool | Surface | Purpose |
|------|---------|---------|
| `send_message` | Apple Events | Send an iMessage/SMS to a phone number or email handle. |
| `list_chats` | Apple Events | List existing conversations and their participants. |
| `list_buddies` | Apple Events | List known contacts on a service. |
| `read_messages` | SQLite | Read recent message history (requires Full Disk Access). |

There is intentionally **no update/delete tool**: Messages' dictionary
can't edit or delete sent messages, so those operations aren't possible.

## Why a dedicated Messages MCP instead of `macos-osascript`'s `run_osascript`?

* **Sending is fiddly.** The reliable idiom is `send "…" to buddy "…" of
  (1st service whose service type = iMessage)` — easy to get wrong (the
  service is an *enum constant*, not a string). This MCP builds it and
  maps `service="iMessage"|"SMS"` for you.
* **Reading isn't scriptable at all.** There's no `message` element to
  read, so `run_osascript` simply can't fetch history. `read_messages`
  reads `chat.db` directly, the same technique `macos-osascript`'s
  `query_mail_store` uses for Apple Mail — including the awkward bits:
  chat.db timestamps are Mac-absolute time in *nanoseconds* on modern
  macOS (seconds on older), and Ventura+ often leaves the `text` column
  NULL, stashing the body in a binary `attributedBody` blob that this MCP
  best-effort decodes.

## Required permissions

Two different macOS permissions, one per surface:

* **Automation** (for `send_message` / `list_chats` / `list_buddies`) —
  approve the prompt the first time, or grant it under System Settings →
  Privacy & Security → Automation → your backend app → **Messages**.
* **Full Disk Access** (for `read_messages` only) — `chat.db` lives under
  the protected `~/Library/Messages/` directory. Grant it under System
  Settings → Privacy & Security → Full Disk Access → your backend app.

## Known limitations

* **No edit or delete** — not supported by the Messages dictionary.
* `send_message` fails when the handle isn't reachable on the chosen
  service (e.g. iMessage to a number with no iMessage account); retry on
  the other service. `SMS` requires Text Message Forwarding with an
  iPhone.
* `read_messages`' `attributedBody` decoding is **best-effort** — there's
  no stable public parser for the archived `NSAttributedString`, so an
  occasional Ventura+ message may come back with empty `text`.
* `days_back` filtering happens after the DB fetch (chat.db's date units
  vary by macOS version), so a very old `days_back` window combined with a
  chatty recent history may return fewer than `limit` rows.
* `read_messages` reflects only what's in the local store — messages not
  yet synced to this Mac won't appear.

## Why is this a built-in MCP?

The canonical source lives at
`backend/builtin_mcps/macos_messages/server.py` so the orchestrator gets
these tools out of the box, macOS-only. On every backend startup the file
is copied into `mcp_server/macos_messages/`, the venv is provisioned with
`uv`, and the registered `MCPServerConfig` points at
`<dir>/.venv/bin/python <dir>/server.py`.
