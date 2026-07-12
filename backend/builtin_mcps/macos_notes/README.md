# macOS Notes (built-in MCP)

Structured Create/Read/Update/Delete tools over Apple Notes, implemented
by generating AppleScript against Notes' own dictionary and running it via
`osascript`.

| Tool | Purpose |
|------|---------|
| `list_folders` | List note folders, optionally scoped to one account. |
| `list_notes` | List notes (metadata only) in one folder or across all. |
| `get_note` | Fetch one note's full content (plaintext + raw HTML) by id. |
| `search_notes` | Search notes by title (and optionally body text). |
| `create_note` | Create a note from a plain title + body. |
| `update_note` | Replace a note's content, or append text to the end. |
| `delete_note` | Delete a note (recoverable from "Recently Deleted"). |

## Why a dedicated Notes MCP instead of `macos-osascript`'s `run_osascript`?

`macos-osascript` already exposes a generic `run_osascript`, but Notes has
a sharp edge the agent keeps hitting: **a note's `body` is HTML, not plain
text.** Set it to raw text and you lose line breaks and can corrupt the
note when the text contains `<` or `&`. This MCP:

* Takes a plain `title` + `body`, HTML-escapes them, and builds the
  `<div>`-per-line HTML Notes expects (`create_note`, `update_note`).
* Returns both `plaintext` (clean, for reading) and the raw `body` HTML
  (`get_note`).
* Keeps `list_notes` metadata-only so listing a big folder doesn't force
  Notes to materialise every note's content.

## Required permission: Automation (not Full Disk Access)

The first time any tool here talks to Notes, macOS shows an **Automation**
permission prompt (per requesting app) — approve it. To re-grant:
**System Settings → Privacy & Security → Automation**, find the app
running Otto's backend, and make sure **Notes** is checked under it. No
Full Disk Access is required.

## Known limitations

* **`update_note` replaces the whole body** when you pass `title`/`body`
  (pass both — omitting one blanks it). Use `append_text` to add to the
  end without disturbing existing content.
* Rich formatting beyond line breaks (checklists, tables, inline images,
  attachments) isn't modelled — bodies round-trip as HTML/plaintext.
* `get_note`/`update_note`/`delete_note` with an empty `folder_name`
  search every note in the app to resolve the id — correct but slower;
  pass the `folder_name` a note was listed with when you have it.
* **Password-protected (locked) notes** can't be read or edited via
  AppleScript while locked.
* Body search (`search_notes(search_body=True)`) forces Notes to
  materialise each candidate note's text and is much slower than the
  default title-only search.

## Why is this a built-in MCP?

The canonical source lives at
`backend/builtin_mcps/macos_notes/server.py` so the orchestrator gets
these tools out of the box, macOS-only, with no setup beyond approving the
Automation prompt above. On every backend startup the file is copied into
`mcp_server/macos_notes/`, the venv is provisioned with `uv`, and the
registered `MCPServerConfig` points at
`<dir>/.venv/bin/python <dir>/server.py`.
