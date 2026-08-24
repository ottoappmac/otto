# Blender (built-in MCP)

Live control of a running Blender session — not a headless
`blender --background` wrapper. The agent sees and drives the same
scene/viewport you have open.

This MCP is a client for **BlenderMCP**, the popular open-source
Blender addon by Siddharth Ahuja
(<https://github.com/ahujasid/blender-mcp>) — not a bespoke Otto
addon. If you've ever tried an AI/MCP workflow with Blender before,
there's a good chance you already have it installed.

| Tool | Purpose |
|------|---------|
| `ping` | Check the addon is reachable. |
| `get_addon_info` | Addon/protocol/Blender version handshake. |
| `get_scene_info` | Rich scene snapshot: objects, selection, camera, lights, animation. |
| `get_object_info` | Transform, materials, bounding box, mesh stats for one object. |
| `execute_blender_code` | Run arbitrary Python with full `bpy` access — use this to create/delete/transform objects, assign materials, etc. |
| `get_viewport_screenshot` | Grab the current viewport as a PNG image. |

## One-time setup

1. **Install the addon** if you don't already have it: download
   `addon.py` from <https://github.com/ahujasid/blender-mcp>, then in
   Blender go to **Edit → Preferences → Add-ons → Install...**, pick
   the file, and enable the **"Blender MCP"** checkbox.
2. **Start it.** In the 3D viewport press `N` to open the sidebar,
   select the **BlenderMCP** tab, and click **Connect to Claude**
   (that's the addon's own button label — it works with any MCP
   client, Otto included). Tick its "auto start" option if you want
   this automatic every time you open Blender.
3. **Enable the tool in Otto.** Turn on the "Blender" server on the
   Tools page (or add it to an agent's tool list) and start chatting —
   `ping` / `get_scene_info` are good first calls to confirm the
   connection.

## Connection details

* The addon binds `127.0.0.1:9876` by default; never exposed off-machine.
* If you changed the port in the addon's scene properties, set
  `BLENDER_MCP_PORT` to match via the Tools page (Blender →
  Credentials) or `request_credential('blender', 'BLENDER_MCP_PORT', ...)`
  mid-chat. `BLENDER_MCP_HOST` is there for symmetry but shouldn't
  normally need changing.
* Wire protocol is a single TCP connection per command with no
  length/newline delimiter — the addon (and this client) both buffer
  received bytes and retry `json.loads()` until it parses. This file
  is a clean-room client for that protocol; no addon code is vendored.
* `get_viewport_screenshot` asks the addon to write a PNG to a temp
  file, then reads it back — Otto and Blender must be on the same
  machine (true by construction here: this MCP subprocess and the
  addon both run on your local machine).

## Trust boundary

`execute_blender_code` runs arbitrary Python inside Blender with full
`bpy` (and therefore filesystem) access — the same trust level as any
other code-execution tool in this codebase. Only connect the addon
when you intend to let the agent drive that Blender session.

## Why is this a built-in MCP?

The canonical source lives at `backend/builtin_mcps/blender/server.py`
so the orchestrator gets Blender tools out of the box. On every
backend startup the file is copied into the per-MCP folder under
`mcp_server/blender/`, the venv is provisioned with `uv`, and the
registered `MCPServerConfig` points at
`<dir>/.venv/bin/python <dir>/server.py`. There's no addon bundled
here — see the setup steps above for installing BlenderMCP itself.
