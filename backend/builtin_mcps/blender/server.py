#!/usr/bin/env python3
"""Built-in MCP server: Blender.

Live control of a running Blender session, via the popular open-source
**BlenderMCP** addon by Siddharth Ahuja
(<https://github.com/ahujasid/blender-mcp>) rather than a bespoke Otto
addon. Reasons to piggyback on that project instead of shipping our
own:

* It's the addon most Blender users who've ever tried an AI/MCP
  workflow already have installed (View3D > Sidebar > BlenderMCP,
  default port 9876) — reusing it means zero extra setup for them and
  no confusing "which Blender addon do I install" choice.
* It's actively maintained and ships real capability (PolyHaven /
  Sketchfab / Hyper3D asset generation, manual-edit capture, etc.)
  that would take real effort to duplicate.

This file is a clean-room MCP-side client for that addon's plain-TCP
wire protocol — no addon code is vendored here, just interoperability
with its documented request/response shape.

Wire protocol (observed from the addon's ``BlenderMCPServer`` class):

* One TCP connection per command. Send ``json.dumps({"type": ...,
  "params": {...}}).encode()`` with **no length prefix or delimiter**.
* The addon buffers received bytes and repeatedly attempts
  ``json.loads(buffer)`` until it parses (so it can accept the request
  in multiple TCP reads); the client here mirrors that on the response
  side, since the addon's replies aren't newline- or length-delimited
  either.
* ``get_viewport_screenshot`` doesn't return image bytes over the
  socket — the addon writes a PNG to a ``filepath`` *we* supply and
  reports back once it's on disk. Since the addon and this MCP
  subprocess run on the same machine, we hand it a temp path and read
  it back ourselves.

Connection details:

* Host/port default to ``127.0.0.1:9876`` (the addon's default).
  Override via ``BLENDER_MCP_HOST`` / ``BLENDER_MCP_PORT`` if you
  changed the addon's port — set them from the Tools page (Blender →
  Credentials) or via ``request_credential``. They aren't really
  secret, just reusing the existing credential-vault plumbing for a
  user-settable override.

Trust boundaries: everything here talks to localhost only.
``execute_blender_code`` runs arbitrary Python inside Blender with
full ``bpy`` (and therefore filesystem) access — the same trust level
as any other code-execution tool in this codebase.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import socket
import tempfile
import uuid
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP, Image

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("otto.mcp.blender")


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9876
CONNECT_TIMEOUT = 5.0
DEFAULT_READ_TIMEOUT = 15.0
LONG_READ_TIMEOUT = 60.0

_NOT_RUNNING_HINT = (
    "Open Blender, go to the 3D viewport's sidebar (press N) → the "
    "'BlenderMCP' tab, and click 'Connect to Claude' (that's the "
    "addon's own label for 'start the socket server' — it works with "
    "any MCP client, including Otto). If the addon isn't installed "
    "yet, see this MCP's README."
)


mcp = FastMCP("Blender")


class BlenderConnectionError(RuntimeError):
    """Raised when the BlenderMCP addon isn't reachable."""


def _host() -> str:
    return (os.environ.get("BLENDER_MCP_HOST") or "").strip() or DEFAULT_HOST


def _port() -> int:
    raw = (os.environ.get("BLENDER_MCP_PORT") or "").strip()
    if not raw:
        return DEFAULT_PORT
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_PORT


def _send_command(
    command_type: str,
    params: Optional[dict[str, Any]] = None,
    *,
    timeout: float = DEFAULT_READ_TIMEOUT,
) -> dict[str, Any]:
    host, port = _host(), _port()
    payload = json.dumps({"type": command_type, "params": params or {}}).encode("utf-8")

    try:
        sock = socket.create_connection((host, port), timeout=CONNECT_TIMEOUT)
    except OSError as exc:
        raise BlenderConnectionError(
            f"Could not reach the BlenderMCP addon at {host}:{port} "
            f"({exc}). {_NOT_RUNNING_HINT}"
        ) from exc

    try:
        sock.settimeout(timeout)
        sock.sendall(payload)
        buf = b""
        response: Optional[dict[str, Any]] = None
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            buf += chunk
            try:
                response = json.loads(buf.decode("utf-8"))
                break
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue  # incomplete — the addon may still be writing
    except socket.timeout as exc:
        raise BlenderConnectionError(
            f"Timed out waiting for Blender to respond to {command_type!r} "
            f"after {timeout:.0f}s."
        ) from exc
    finally:
        try:
            sock.close()
        except OSError:
            pass

    if response is None:
        raise BlenderConnectionError(
            f"Blender closed the connection without a parseable response to "
            f"{command_type!r} (got {len(buf)} bytes)."
        )

    if response.get("status") != "success":
        raise RuntimeError(response.get("message") or "Blender reported an unknown error")

    result = response.get("result") or {}
    # Several of the addon's own handlers catch their internal exceptions
    # and return {"error": "..."} instead of raising, so the outer
    # envelope still says "success". Surface that the same way as a
    # protocol-level error so callers don't have to check both places.
    if isinstance(result, dict) and set(result) == {"error"}:
        raise RuntimeError(result["error"])
    return result


@mcp.tool()
def ping() -> dict[str, Any]:
    """Check whether the BlenderMCP addon is running and reachable.

    Call this first if other Blender tools are failing — it raises a
    clear error pointing at the addon setup steps when Blender isn't
    listening.
    """
    return _send_command("ping")


@mcp.tool()
def get_addon_info() -> dict[str, Any]:
    """Get the BlenderMCP addon's version, protocol version, and Blender version.

    Useful as a capability handshake before relying on a specific
    feature — the addon may run an older or newer protocol version.
    """
    return _send_command("get_addon_info")


@mcp.tool()
def get_scene_info() -> dict[str, Any]:
    """Get a snapshot of the currently open Blender scene.

    Returns scene name, frame range/fps, up to 50 objects (with
    transform, visibility, materials, bounding box, parent/child
    relations, and any active animation), the current selection, the
    active camera and its lens settings, and every light. Call this
    first to see what's in the file before making changes.
    """
    return _send_command("get_world_state_snapshot")


@mcp.tool()
def get_object_info(name: str) -> dict[str, Any]:
    """Get detailed info about one object in the Blender scene.

    Args:
        name: Exact object name (as shown in get_scene_info's object list).

    Returns transform (location/rotation/scale), visibility, assigned
    materials, world-space bounding box, and — for meshes —
    vertex/edge/polygon counts.
    """
    if not name or not name.strip():
        raise ValueError("name is required")
    return _send_command("get_object_info", {"name": name})


@mcp.tool()
def execute_blender_code(code: str) -> dict[str, Any]:
    """Run arbitrary Python code inside the running Blender session.

    The full ``bpy`` API is available. Use this for everything beyond
    scene inspection: creating/deleting/transforming objects, assigning
    materials, modifiers, geometry nodes, animation keyframes, UV
    unwrapping, importing/exporting, rendering to disk, etc.
    ``print()`` output is captured and returned as the result.

    Example — add a red cube at the origin::

        import bpy
        bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
        obj = bpy.context.active_object
        mat = bpy.data.materials.new(name="Red")
        mat.use_nodes = True
        mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (1, 0, 0, 1)
        obj.data.materials.append(mat)
        print(obj.name)

    Args:
        code: Python source to exec() inside Blender. ``bpy`` is
            already imported into the execution namespace.

    Raises with the Blender-side error message on failure.
    """
    if not code or not code.strip():
        raise ValueError("code is required")
    result = _send_command("execute_code", {"code": code}, timeout=LONG_READ_TIMEOUT)
    return {"stdout": result.get("result", "")}


@mcp.tool()
def get_viewport_screenshot(max_size: int = 800) -> Image:
    """Render the current viewport and return it as an image.

    Captures whatever the active 3D viewport is currently showing
    (camera view or not, shading mode, etc.) via an offscreen GPU
    render, independent of whether the Blender window is focused or
    even visible on screen.

    Args:
        max_size: Downscale so the longest edge is at most this many
            pixels, to keep payloads small. Default 800.
    """
    tmp_path = os.path.join(
        tempfile.gettempdir(), f"otto_blender_viewport_{uuid.uuid4().hex}.png",
    )
    try:
        _send_command(
            "get_viewport_screenshot",
            {"max_size": max_size, "filepath": tmp_path, "format": "png"},
            timeout=LONG_READ_TIMEOUT,
        )
        try:
            with open(tmp_path, "rb") as f:
                data = f.read()
        except OSError as exc:
            raise BlenderConnectionError(
                f"Blender reported success but the screenshot file wasn't "
                f"readable at {tmp_path!r} ({exc}). Otto and Blender must be "
                f"running on the same machine for this tool to work."
            ) from exc
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return Image(data=data, format="png")


if __name__ == "__main__":
    mcp.run()
