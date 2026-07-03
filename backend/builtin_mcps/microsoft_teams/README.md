# Microsoft Teams (built-in MCP)

**Read-only** tools over the Microsoft Graph API
(`graph.microsoft.com/v1.0`):

| Tool | Purpose |
|------|---------|
| `list_teams` | List Teams-enabled groups in the tenant. |
| `list_channels` | List channels in a team. |
| `get_channel_info` | Metadata for a single channel. |
| `list_channel_members` | List members of a channel. |
| `get_channel_messages` | Recent messages in a channel (see limitations below). |
| `list_users` | List users in the tenant. |
| `get_user_info` | Profile details for one user. |

## Why no `send_message`?

Microsoft Graph **does not support sending Teams channel or chat
messages with app-only (application-permission) tokens** — the
`POST .../messages` endpoint requires a delegated token tied to a signed-in
user. That's a fundamentally different auth model (interactive OAuth login
+ refresh tokens) than the client-credentials grant every other tool here
uses, so this MCP stays read-only rather than half-implementing a second
auth flow. If you need Otto to post into Teams, route that through an
Incoming Webhook connector on the channel instead (outside this MCP).

## Required credentials

* `TEAMS_TENANT_ID` — your Entra ID (Azure AD) tenant id.
* `TEAMS_CLIENT_ID` — the app registration's Application (client) id.
* `TEAMS_CLIENT_SECRET` — a client secret created for that app registration.

## Setup walkthrough

### 1. Get a Microsoft 365 tenant with Teams

Unlike Slack/Discord, Teams isn't something you can just sign up for with
a personal account — you need a Microsoft 365 tenant, which is also
where the app registration and admin consent in step 2-3 happen.

* **Already have a work/school Microsoft 365 tenant?** Use that — skip to
  step 2. You (or whoever holds Global Admin) will need admin rights for
  step 3.
* **Just testing?** Join the free
  [Microsoft 365 Developer Program](https://developer.microsoft.com/microsoft-365/dev-program)
  — it gives you your own sandbox tenant (90-day renewable, with sample
  users/teams data) where you're automatically the Global Admin, so you
  can do every step below yourself.

### 2. Register an app in Entra ID

1. Go to the [Entra admin center](https://entra.microsoft.com) →
   **Identity → Applications → App registrations** → **New registration**.
2. Give it a name (e.g. "Otto") and register it with default options.
3. On the app's **Overview** page, copy the **Directory (tenant) ID** and
   **Application (client) ID** — these are `TEAMS_TENANT_ID` and
   `TEAMS_CLIENT_ID`.
4. Go to **Certificates & secrets → Client secrets → New client secret**.
   Copy the secret's **Value** immediately — it's not shown again. This
   is `TEAMS_CLIENT_SECRET`.

### 3. Grant API permissions

1. On the app's page, go to **API permissions → Add a permission →
   Microsoft Graph → Application permissions** (not "Delegated" — this
   MCP uses app-only auth).
2. Add: `Team.ReadBasic.All`, `Channel.ReadBasic.All`,
   `ChannelMember.Read.All`, `User.Read.All`, and `ChannelMessage.Read.All`
   (only needed for `get_channel_messages` — see limitations below).
3. Click **Grant admin consent for `<tenant>`** at the top of the
   permissions list, and confirm. This requires Global Admin or
   Privileged Role Admin — if that's not you, send this step to whoever
   is.

### 4. Connect it in Otto

1. In Otto, go to the **Agents** page → **Tools** tab.
2. Find the **Microsoft Teams** card and click **Credentials**.
3. Paste in the three values from step 2.3-2.4: `TEAMS_TENANT_ID`,
   `TEAMS_CLIENT_ID`, `TEAMS_CLIENT_SECRET`. Save.
4. Click **Start** (or toggle the server on) to connect it.

Once connected, the agent has `list_teams`, `list_channels`,
`get_channel_info`, `list_channel_members`, `get_channel_messages`,
`list_users`, and `get_user_info` available — read-only, see below.

## Known limitations

* **No write tools at all.** See "Why no `send_message`?" above.
* **`get_channel_messages` needs more than admin consent.**
  `ChannelMessage.Read.All` is one of Microsoft's *Protected APIs* —
  granting the application permission and consenting in the admin
  center is necessary but **not sufficient**. The tenant must also
  submit Microsoft's Protected APIs request form (search Microsoft
  Learn for "Teams protected APIs") and wait for approval, which
  Microsoft reviews on a recurring (currently weekly) cadence. Until
  that's approved, this tool will return a 403 with a `hint` field
  explaining this.
* `list_teams` / `list_channels` / `list_channel_members` / `list_users`
  only need the application permissions + admin consent in step 3-4
  above and work immediately.
* The app-only access token is cached in-process per MCP subprocess and
  refreshed automatically a few minutes before it expires — there's no
  persistent token storage.

## Why is this a built-in MCP?

The canonical source lives at
`backend/builtin_mcps/microsoft_teams/server.py` so the orchestrator gets
Teams read access out of the box without the user having to author it via
`mcp_builder`. On every backend startup the file is copied into
`mcp_server/microsoft_teams/`, the venv is provisioned with `uv`, and the
registered `MCPServerConfig` points at
`<dir>/.venv/bin/python <dir>/server.py`.
