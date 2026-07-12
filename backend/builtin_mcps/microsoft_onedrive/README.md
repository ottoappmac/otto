# OneDrive / SharePoint (built-in MCP)

Browse/search OneDrive & SharePoint files through Microsoft Graph, scoped to
just the Files + SharePoint Sites tools of the third-party
[`microsoft365-mcp-server`](https://www.npmjs.com/package/microsoft365-mcp-server)
npm package, run locally via `npx`:

| Tool | Purpose |
|------|---------|
| `list_drive_items` | List files/folders (by `folder_id` or `folder_path`). |
| `get_drive_item` | Get file/folder metadata. |
| `search_files` | Search OneDrive/SharePoint by name/content. |
| `download_file` | Download a file (inline for text files under 100KB). |
| `create_folder` | Create a new folder. |
| `upload_file` | Upload a file (text or base64 binary, max ~4MB). |
| `list_sites` | List followed SharePoint sites, or search all sites. |
| `get_site` | Get SharePoint site details. |
| `list_site_drives` | List document libraries (drives) in a site. |
| `list_site_items` | List files/folders in a site drive. |
| `search_site_files` | Search files within a SharePoint site. |
| `get_auth_status` | Check whether you're currently signed in. |

Every other tool the upstream package ships (Mail, Calendar, Contacts,
Teams, Chats, Planner, OneNote, To Do, Users/Groups, and the raw
`graph_query` escape hatch) is hidden via `MS365_ENABLED_TOOLS` — see
[registry.py](../registry.py)'s `microsoft-onedrive` entry if you want to
widen that.

## Why a third-party package instead of Otto's own code?

This MCP used to be a hand-written FastMCP server calling Microsoft Graph
directly, authenticated via Otto's own OAuth device-code flow (type a code
at `microsoft.com/devicelogin`). It's now `microsoft365-mcp-server` run via
`npx`, authenticated with `MS365_AUTH_MODE=interactive`: `@azure/identity`
opens your actual browser to Microsoft's sign-in page with a loopback
redirect (no manual code-typing), the same class of flow as a normal "Sign
in with Microsoft" button.

The trade-off: sign-in is no longer wired into Otto's Login/Logout button
in the Credentials dialog. The Node subprocess manages its own MSAL token
cache (on disk at `TOKEN_STORAGE_PATH`, under this MCP's app-data
directory).

**Important correction:** the upstream package authenticates as part of
its own process bootstrap (`setupAuth()` runs before it starts serving MCP
traffic at all) — not lazily on first tool call as originally assumed.
Since Otto connects every *enabled* MCP eagerly (at backend startup, and
again when the first chat session is built), leaving this enabled by
default would pop an unprompted Microsoft sign-in window on every app
launch. To avoid that, **this MCP ships disabled by default** — it only
spawns (and therefore only prompts for sign-in) once you flip it on from
the Tools page yourself. See "Sign in" below.

## What this does *not* solve — read before relying on it

* **You still need *some* Microsoft-recognized identity.** A personal
  Microsoft account (free, can be created from any email including Gmail)
  or an Entra ID identity (work/school, or a B2B guest). There's no way to
  get a Graph token for a bare email address with zero Microsoft identity
  behind it.
* **The sharing organization's own tenant policies still apply.** If that
  org restricts guest/consumer sign-in to third-party apps, login will
  fail until *their* admin allows it.
* **Anonymous "Anyone with the link" shares need none of this.** A fully
  anonymous share is reachable with a plain unauthenticated HTTPS GET — no
  login, no MCP required.
* **OTP-only "guest access" links (`guestaccess.aspx?...&at=9`) won't
  work.** Those hand out a one-time emailed code with no durable identity
  behind it — there's no OAuth token to obtain. If that's what you have,
  open the link in a normal browser instead.
* **This is a small, third-party-maintained package** (MIT licensed,
  low install base at the time of writing), not something Microsoft ships
  or supports. It runs as a local subprocess with delegated access to your
  Microsoft account — review its source before trusting it with a real
  account, same as any other MCP you'd add via `npx`.

## Required credentials

One per-user secret: **`MS365_CLIENT_ID`** — the Application (client) ID
of an Entra app you (or whoever set this up) registered. This is a public
client identifier, not a secret in the traditional sense, but it's
still per-installer because each Entra app registration is independently
rate-limited/governed and tied to whoever registered it.

## Setup walkthrough

### 1. Register an app in Entra

1. Go to the [Entra admin center](https://entra.microsoft.com) →
   **Identity → Applications → App registrations** → **New registration**.
2. Name it (e.g. "Otto OneDrive"). Under **Supported account types**
   choose **"Accounts in any organizational directory and personal
   Microsoft accounts"** — this is what lets a Gmail-linked personal
   account sign in at all.
3. Under **Redirect URI**, pick platform **"Mobile and desktop
   applications"** and add `http://localhost` (this platform type is
   inherently a public client — you do *not* need to separately toggle
   "Allow public client flows", unlike the old device-code setup).
4. On the app's **Overview** page, copy the **Application (client) ID**
   (not the Directory/tenant ID — a common mix-up).

### 2. Add delegated API permissions

1. Go to **API permissions → Add a permission → Microsoft Graph →
   Delegated permissions** (not "Application" — this MCP uses a
   signed-in user's token).
2. Add: `User.Read`, `Files.ReadWrite`, `Sites.Read.All`,
   `Sites.ReadWrite.All`.
3. These are self-consentable by any signed-in user (personal account or
   guest) — no admin consent step is needed here. (A *different* org's
   tenant policy can still block consent for *their* users signing into
   this app — see limitations above.)

### 3. Paste the client ID into Otto

1. In Otto, go to the **Agents** page → **Tools** tab.
2. Find the **OneDrive / SharePoint** card and click **Credentials**.
3. Paste the Application (client) ID from step 1.4 into **`MS365_CLIENT_ID`**
   and save.

### 4. Turn it on and sign in

This MCP ships **disabled** — the Tools page will show it as inactive
until you explicitly turn it on:

1. Go to the **Agents** page → **Tools** tab → **OneDrive / SharePoint**
   card.
2. Click **Start** (or toggle it enabled). This is the moment the `npx`
   subprocess actually spawns, and the moment it opens your browser to
   sign in — not before.
3. Sign in with whichever Microsoft account the content was shared with
   (personal or guest) and approve.

Subsequent restarts of Otto will *not* re-prompt automatically, because
Otto only connects to servers you've left enabled — if you want it
available every session without re-clicking Start, leave it enabled after
the first sign-in and Otto will reconnect to it (silently reusing the
cached token) alongside your other tools going forward.

## Known limitations

* **No "Login" button.** Unlike Otto's `oauth_device`/`oauth_authcode`
  built-ins, this package's auth is opaque to Otto's backend — there's no
  server-side signal for "signed in" vs. "not signed in" beyond calling
  the `get_auth_status` tool.
* **Enabling it is what triggers sign-in, not a tool call.** Because the
  upstream package authenticates during its own process bootstrap, the
  sign-in prompt happens as soon as you enable/start the server — even
  before the agent has tried to use any tool. If the cached token has
  since expired, re-enabling (or restarting Otto) will prompt again.
* **Token storage isn't in Otto's OS-keychain vault.** It's a JSON file
  under `TOKEN_STORAGE_PATH` (defaults to a per-MCP folder under Otto's
  app-data directory, not the upstream package's `/tmp` default). Treat
  that file with the same care as a password.
* **No resumable uploads.** `upload_file` tops out around 4MB (Graph's
  simple-upload ceiling).

## Why is this a built-in MCP?

Unlike Otto's other built-ins, there's no repo-bundled `server.py` or
per-MCP `uv` venv here — the `microsoft-onedrive` entry in
[`registry.py`](../registry.py) has `runtime="node_npx"`, so
`builtin_mcp_config()` emits `command="npx"` with a pinned
`npx_package@npx_version` instead of a venv-python path, and
`sync_builtin_mcp_files()` / `ensure_builtin_mcp_venvs()` skip it entirely.
It's still registered as a built-in (rather than left for users to add via
`register_external_mcp_server`) so the Credentials dialog, required-secret
gating, and Tools page all work the same way they do for every other
built-in MCP.
