# Slack (built-in MCP)

Read + write tools over the Slack Web API (`slack.com/api`):

| Tool | Purpose |
|------|---------|
| `list_channels` | List channels visible to the bot. |
| `get_channel_info` | Metadata for a single channel. |
| `join_channel` | Join a public channel (bot must be invited to private ones). |
| `get_channel_history` | Recent messages in a channel. |
| `get_thread_replies` | All replies in a message thread. |
| `send_message` | Post a message, or reply in a thread. |
| `add_reaction` | Add an emoji reaction to a message. |
| `list_users` | List workspace members. |
| `get_user_info` | Profile details for one user. |

## Required credentials

* `SLACK_BOT_TOKEN` — a Bot User OAuth Token (starts with `xoxb-`).

## Setup walkthrough

### 1. Get a Slack workspace

If you're connecting Otto to a workspace you already use, skip ahead to
step 2. Otherwise, create a free one:

1. Go to <https://slack.com/get-started#/createnew> and sign up.
2. Follow the prompts to name your workspace (e.g. "Otto Test") and create
   a first channel — you can skip inviting teammates.

You now have a workspace where you (and your bot) are the only members.

### 2. Create a Slack app

1. Go to <https://api.slack.com/apps> and click **Create New App** →
   **From scratch**.
2. Give it a name (this becomes the bot's display name, e.g. "Otto") and
   pick the workspace from step 1.

### 3. Add Bot Token Scopes

1. In the left sidebar, click **OAuth & Permissions**.
2. Scroll to **Scopes → Bot Token Scopes** and click **Add an OAuth
   Scope** for each of:
   * `channels:read` and `channels:history` — list/read public channels
   * `chat:write` — `send_message`
   * `users:read` — `list_users` / `get_user_info`
   * `groups:read` and `groups:history` — only if you want private-channel
     access too
   * `reactions:write` — only if you want `add_reaction`

### 4. Install the app and copy the token

1. Scroll back up on the same **OAuth & Permissions** page and click
   **Install to Workspace** (or **Install App** in the left sidebar).
   Review the requested scopes and click **Allow**.
2. Copy the **Bot User OAuth Token** shown at the top of the page — it
   starts with `xoxb-`. This is your `SLACK_BOT_TOKEN`.

### 5. Invite the bot into your channels

The bot can only see and post in channels it has actually joined:

* For public channels: either run `/invite @YourBotName` in the channel
  from Slack, or just let Otto call the `join_channel` tool itself once
  it's connected.
* For private channels: someone has to `/invite @YourBotName` manually —
  there's no API-only way to join a private channel.

### 6. Connect it in Otto

1. In Otto, go to the **Agents** page → **Tools** tab.
2. Find the **Slack** card and click **Credentials**.
3. Paste the bot token from step 4.2 into `SLACK_BOT_TOKEN` and save.
4. Click **Start** (or toggle the server on) to connect it.

Once connected, the agent has `list_channels`, `get_channel_info`,
`join_channel`, `get_channel_history`, `get_thread_replies`,
`send_message`, `add_reaction`, `list_users`, and `get_user_info`
available, scoped to whatever you granted in steps 3 and 5.

## Known limitations

* Slack's Web API returns HTTP 200 even for a failed call (`{"ok": false,
  "error": "..."}`); every tool surfaces that `error` slug directly rather
  than a generic HTTP error.
* The bot can only see/post in channels it has joined (or been invited to,
  for private channels) and within the scopes granted above.

## Why is this a built-in MCP?

The canonical source lives at `backend/builtin_mcps/slack/server.py` so the
orchestrator gets Slack tools out of the box without the user having to
author them via `mcp_builder`. On every backend startup the file is copied
into `mcp_server/slack/`, the venv is provisioned with `uv`, and the
registered `MCPServerConfig` points at `<dir>/.venv/bin/python <dir>/server.py`.
