# macOS Mail (built-in MCP)

Structured Create/Read/Update/Delete tools over Apple Mail, implemented by
generating AppleScript against Mail's own dictionary and running it via
`osascript`.

| Tool | Purpose |
|------|---------|
| `list_accounts` | List every configured Mail account. |
| `list_mailboxes` | List mailboxes (including nested folders), optionally scoped to one account. |
| `list_messages` | List the most recent messages in one mailbox, optionally filtered to unread/flagged. |
| `get_message` | Fetch one message's full content, headers, and attachment list. |
| `search_messages` | Free-text search across subject and sender (body content opt-in via `search_body=True`), across one mailbox or swept across every mailbox in every account. |
| `send_message` | Compose and immediately send a new email (with optional CC/BCC/attachments). |
| `create_draft` | Compose a new email and save it as a draft without sending. |
| `update_message` | Mark a message read/unread, flag/unflag it, and/or move it to another mailbox. |
| `delete_message` | Delete a message (Mail's standard soft-delete — moves it to Trash). |

## Why a dedicated Mail MCP instead of `macos-osascript`'s `run_osascript`?

The `macos-osascript` built-in MCP already exposes a generic `run_osascript`
tool the agent can use to drive any scriptable app, plus a read-only
`query_mail_store` that reads Mail's local SQLite index directly. Both are
useful, but:

* `run_osascript` requires the agent to *author* correct AppleScript for
  every operation — get a message reference wrong, or guess a property
  name that doesn't exist (`sender of msg` vs `sender` being a *class* in
  some mail clients), and the call fails with an opaque `-2741`/`-1728`.
  This MCP's tools take plain typed arguments (`to`, `subject`,
  `message_id`, …) and build the correct AppleScript internally.
* `query_mail_store` is fast (reads Mail's SQLite cache directly) but
  **requires Full Disk Access**, is **read-only**, and can't see live
  `read`/`flagged` status changes made through this MCP in the same
  request. This MCP's tools only need the **Automation** permission (see
  below) — no Full Disk Access — and can create/update/delete, not just
  read.

Use `query_mail_store` instead when you need to search a very large
mailbox by subject/sender only and don't need body-content matching or
any writes — it reads Mail's SQLite cache directly (no Apple Events at
all), so it's meaningfully faster for that narrow case, at the cost of
requiring Full Disk Access. Use this MCP for everything else:
composing/sending, marking/flagging/moving/deleting, and — with
`search_messages(search_body=True)` — search that needs to match
message body text.

## Required permission: Automation (not Full Disk Access)

The first time any tool here talks to Mail, macOS shows an **Automation**
permission prompt (per requesting app) — approve it. If you missed the
prompt or need to re-grant it:

1. Open **System Settings → Privacy & Security → Automation**.
2. Find the app that's running Otto's backend (e.g. Terminal, or the
   packaged Otto app) in the list.
3. Make sure **Mail** is checked underneath it.

No Full Disk Access is required for anything in this MCP.

## Known limitations

* `delete_message` is a soft-delete (moves to Trash), matching the Mail
  UI's Delete key. There's no "permanently delete" tool — call it again
  with `mailbox_name="trash"` if the account is configured to delete
  immediately on a second delete.
* `message_id`/`mailbox_name`/`account_name` together identify a message —
  always pass back the same `mailbox_name`/`account_name` a message was
  listed with.
* The special mailbox aliases (`inbox`/`sent`/`drafts`/`trash`/`junk`/
  `outbox`) only resolve when `account_name` is empty. Once you pass an
  `account_name` (to `list_messages`, `search_messages`,
  `update_message`'s `move_to_mailbox`, etc.), `mailbox_name` must be
  that account's *exact* mailbox name — e.g. Gmail accounts typically
  name their trash `"[Gmail]/Bin"` or `"[Gmail]/Trash"`, not `"trash"`.
  Use `list_mailboxes(account_name=...)` to look up the exact name.
* `search_messages` matches subject/sender only by default — cached
  metadata Mail can filter without touching any message body, so this
  is fast even on a full unscoped sweep. Pass `search_body=True` only
  if a subject/sender-only search comes back empty and the keyword
  might only appear in the body; it forces Mail to fetch/decode every
  candidate message and is the dominant cost on a large/IMAP mailbox.
  Narrow further with `mailbox_name`, `account_name`, and/or
  `days_back` whenever you know roughly where to look. On accounts
  with very large mailboxes (tens of thousands of messages), a
  body-content sweep can still hit Mail's own internal Apple Event
  timeout even when narrowed to one such mailbox — this is an inherent
  limit of driving Mail's `content contains` filter via AppleScript,
  not something a bigger `timeout_seconds` can fully paper over. Prefer
  `query_mail_store` for those mailboxes.
* `account_name` (everywhere it appears) accepts either an account's
  display name (`list_accounts`' `name` field) or one of its
  `email_addresses` — resolved internally, so either works.
* Mail can occasionally get its whole Apple-Event-handling main thread
  stuck — even on totally unrelated, previously-working requests —
  until it's quit and relaunched; this is an underlying Mail.app/macOS
  behavior, not something any of these tools can detect or repair from
  the outside. Every generated script runs inside AppleScript's own
  `with timeout of` block (a few seconds under the tool's own
  `timeout_seconds`), so a stuck request surfaces here as a fast, clean
  timeout error instead of hanging for the full outer timeout — but if
  *every* call starts timing out, including trivial ones like
  `list_accounts`, the fix is to quit and relaunch Mail.app.
* `send_message`/`create_draft`'s `account_name` sets the composing
  message's `sender` address (Mail's own account-selection heuristic —
  it picks the account whose configured address matches); there's no
  direct "send via account X" property in Mail's dictionary.
* No reply/forward "smart compose" helpers — only new outgoing messages.
  No GUI/Accessibility automation of the compose window; every tool here
  is a background Apple Event, so no desktop focus is ever taken.

## Why is this a built-in MCP?

The canonical source lives at `backend/builtin_mcps/macos_mail/server.py` so
the orchestrator gets these tools out of the box, macOS-only, with no setup
beyond approving the Automation prompt above. On every backend startup the
file is copied into `mcp_server/macos_mail/`, the venv is provisioned with
`uv`, and the registered `MCPServerConfig` points at
`<dir>/.venv/bin/python <dir>/server.py`.
