"""MCP JSON config import/export routes (separate prefix from /api/mcp-servers)."""

from __future__ import annotations

import re

from fastapi import APIRouter

from backend.config import AppConfig, MCPServerConfig
from backend.state import mcp_mgr

router = APIRouter(prefix="/api", tags=["mcp"])


@router.put("/mcp-config")
async def save_mcp_servers_json(payload: dict):
    """Replace all non-builtin MCP servers from a JSON config.

    Accepts: ``{ "mcpServers": { "Name": { ... }, ... } }``
    Validates, then replaces all non-builtin entries.
    """
    mcp_servers = payload.get("mcpServers") or payload.get("mcp_servers") or {}
    if not isinstance(mcp_servers, dict):
        return {"error": "Expected 'mcpServers' to be an object"}

    errors: list[str] = []
    new_entries: list[MCPServerConfig] = []
    seen_ids: set[str] = set()

    for name, spec in mcp_servers.items():
        if not isinstance(spec, dict):
            errors.append(f"'{name}': value must be an object")
            continue

        server_id = re.sub(r"[^a-z0-9-]", "-", name.lower().strip()).strip("-")
        if not server_id:
            errors.append(f"'{name}': could not derive a valid ID")
            continue
        if server_id in seen_ids:
            errors.append(f"'{name}': duplicate ID '{server_id}'")
            continue
        seen_ids.add(server_id)

        has_command = bool(spec.get("command"))
        transport = spec.get("transport", "stdio" if has_command else "streamable_http")

        if transport == "stdio" and not has_command:
            errors.append(f"'{name}': stdio transport requires 'command'")
            continue
        if transport != "stdio" and not spec.get("url"):
            errors.append(f"'{name}': {transport} transport requires 'url'")
            continue

        new_entries.append(MCPServerConfig(
            id=server_id,
            name=name,
            transport=transport,
            url=spec.get("url"),
            command=spec.get("command"),
            args=spec.get("args", []),
            env=spec.get("env", {}),
            enabled=True,
            auto_start=False,
            builtin=False,
        ))

    if errors:
        return {"error": "Validation failed", "details": errors}

    cfg = await AppConfig.aload()
    removed_ids = {s.id for s in cfg.mcp_servers if not s.builtin}
    cfg.mcp_servers = [s for s in cfg.mcp_servers if s.builtin] + new_entries
    await cfg.asave()

    for rid in removed_ids - {s.id for s in new_entries}:
        await mcp_mgr.disconnect(rid)

    return {"status": "saved", "count": len(new_entries)}
