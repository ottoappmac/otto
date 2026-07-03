# Discord (built-in MCP)

Read + write tools over the Discord REST API (`discord.com/api/v10`):

| Tool | Purpose |
|------|---------|
| `list_guilds` | List servers the bot is a member of. |
| `get_guild_info` | Metadata for a single server. |
| `list_guild_members` | List members of a server. |
| `list_channels` | List channels in a server. |
| `get_channel_messages` | Recent messages in a channel. |
| `send_message` | Post a message to a channel. |
| `add_reaction` | Add an emoji reaction to a message. |

Discord's API only ever addresses channels by their numeric snowflake
id — there's no "send by channel name" endpoint. `get_channel_messages`,
`send_message`, and `add_reaction` accept a `channel_id`, but if you don't
have it handy you can instead pass `channel_name` (e.g. `"general"`, `#`
prefix optional, case-insensitive) and optionally `guild_id` to scope the
lookup to one server; the tool resolves it to an id internally by listing
channels. If the name matches channels in more than one server, pass
`guild_id` or `channel_id` to disambiguate.

## Required credentials

* `DISCORD_BOT_TOKEN` — a bot token from the Discord Developer Portal.

## Setup walkthrough

### 1. Get a Discord server

Discord calls a workspace a "server" (unrelated to client/server — it's
just what Discord calls a community). If you're connecting Otto to a
server you already have, skip to step 2. Otherwise, create one:

1. Open Discord (desktop app or <https://discord.com/app>).
2. Click the **+** icon at the bottom of the server sidebar → **Create My
   Own** → give it a name (e.g. "Otto Test").

### 2. Create a bot application

1. Go to <https://discord.com/developers/applications> and click **New
   Application**. Give it a name (this becomes the bot's display name)
   and accept the terms.
2. In the left sidebar, click **Bot**.
3. Click **Reset Token** and copy the token immediately — Discord only
   shows it once. This is your `DISCORD_BOT_TOKEN`.
4. On the same page, scroll to **Privileged Gateway Intents** and enable
   **Server Members Intent** if you want `list_guild_members` to work
   (without it, Discord returns an empty list or a 403).

### 3. Invite the bot into your server

1. In the left sidebar, click **OAuth2 → URL Generator**.
2. Under **Scopes**, check `bot`.
3. Under **Bot Permissions** (appears once `bot` is checked), select at
   minimum: `View Channels`, `Read Message History`, `Send Messages`,
   `Add Reactions`.
4. Copy the generated URL at the bottom, open it in a browser, pick your
   server from the dropdown, and click **Authorize**.
5. The bot now appears in your server's member list (it'll show offline
   until Otto actually connects to it).

### 4. Connect it in Otto

1. In Otto, go to the **Agents** page → **Tools** tab.
2. Find the **Discord** card and click **Credentials**.
3. Paste the bot token from step 2.3 into `DISCORD_BOT_TOKEN` and save.
4. Click **Start** (or toggle the server on) to connect it.

Once connected, the agent has `list_guilds`, `get_guild_info`,
`list_guild_members`, `list_channels`, `get_channel_messages`,
`send_message`, and `add_reaction` available, scoped to whatever
channels/permissions you granted in step 3.3.

## Known limitations

* `429` (rate limited) and other non-2xx responses are returned as a
  structured `{"error": ..., "status_code": ..., "retry_after_seconds":
  ...}` result instead of raising, so the agent can see what happened and
  back off deliberately.
* `list_guild_members` requires the Server Members privileged intent
  (step 2.4 above) — without it, Discord returns an empty list or a 403.

## Why is this a built-in MCP?

The canonical source lives at `backend/builtin_mcps/discord/server.py` so
the orchestrator gets Discord tools out of the box without the user
having to author them via `mcp_builder`. On every backend startup the
file is copied into `mcp_server/discord/`, the venv is provisioned with
`uv`, and the registered `MCPServerConfig` points at
`<dir>/.venv/bin/python <dir>/server.py`.
