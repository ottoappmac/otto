"""MCP server management API routes."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.config import AppConfig, MCPServerConfig
from backend.mcp_manager import reset_circuit_breaker
from backend.schemas import MCPAuthStatus, MCPServerAddRequest, MCPServerStatus
from backend.state import mcp_mgr, session_mgr
from backend.utils import slugify

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mcp-servers", tags=["mcp"])


def _current_os() -> str:
    from backend.utils import platform_label
    return platform_label()


def _missing_secrets(srv: MCPServerConfig) -> list[str]:
    """Subset of ``srv.required_secrets`` not yet stored in the vault.

    Vault lookup is name-only — never returns a value — so this is safe
    to compute eagerly on every list/status response.  When the vault
    backend is unavailable (e.g. CI without a keychain) we treat every
    required name as missing rather than as present, which keeps the
    "Start" button disabled instead of silently launching with no creds.
    """
    if not srv.required_secrets:
        return []
    try:
        from backend.credential_vault import vault
    except Exception:
        return list(srv.required_secrets)
    return [n for n in srv.required_secrets if not vault.has(srv.id, n)]


def _auth_status(srv: MCPServerConfig) -> MCPAuthStatus:
    """Snapshot the interactive-auth state for a server.

    Static MCPs short-circuit to a default ``MCPAuthStatus`` (kind
    ``"static"``, ``needs_login=False``).  For interactive flows we
    consult the vault for the bundle and the registered provider for
    its expiry verdict — there's no token contents in the response,
    only the booleans the frontend needs to pick the right affordance.

    Vault errors degrade gracefully: anything we can't read becomes
    ``needs_login=True`` so the user is prompted instead of silently
    leaving them in a broken state.
    """
    auth = srv.auth
    if auth is None or auth.kind == "static":
        return MCPAuthStatus(kind="static")

    try:
        from backend.auth import get_provider
        from backend.credential_vault import vault
    except Exception:
        return MCPAuthStatus(kind=auth.kind, needs_login=True)

    try:
        bundle = vault.get_bundle(srv.id)
    except Exception:
        bundle = None

    if not bundle:
        return MCPAuthStatus(kind=auth.kind, needs_login=True)

    try:
        provider = get_provider(auth.kind)
    except KeyError:
        # Manifest references a kind we don't ship — surface as needs-login
        # so the UI prompts instead of silently using a stale bundle.
        return MCPAuthStatus(
            kind=auth.kind, has_bundle=True, expired=True, needs_login=True,
        )

    expired = provider.is_expired(bundle)
    # ``needs_login`` is true only when we know we can't recover
    # silently — i.e. the bundle is expired AND there's no refresh
    # path (no refresh_token, or the provider can't refresh).  For
    # OAuth flows that means a missing refresh_token.  For
    # browser_capture it's always true once expired (no refresh path).
    refresh_possible = bool(bundle.get("refresh_token")) and auth.kind in (
        "oauth_device", "oauth_authcode",
    )
    needs_login = expired and not refresh_possible

    return MCPAuthStatus(
        kind=auth.kind,
        has_bundle=True,
        expired=expired,
        needs_login=needs_login,
        expiry_iso=bundle.get("expiry_iso"),
    )


def _status_for(srv: MCPServerConfig, *, os_supported: bool) -> dict:
    """Render a full ``MCPServerStatus`` payload for a registered server.

    Centralises the fan-in from MCPManager + vault so list / connect /
    start / stop responses all carry the same shape — the frontend can
    mutate its local state directly from any of them without re-fetching.
    """
    conn = mcp_mgr.connections.get(srv.id)
    connected = conn.connected if conn else False
    tools = [t.name for t in conn.tools] if conn else []
    return MCPServerStatus(
        id=srv.id,
        name=srv.name,
        connected=connected,
        tool_count=len(tools),
        tools=tools,
        excluded_tools=srv.excluded_tools,
        error=None if connected else (conn.error if conn else None),
        auto_start=srv.auto_start,
        process_running=mcp_mgr.is_process_running(srv.id),
        transport=srv.transport,
        url=srv.url,
        port=srv.port,
        command=srv.command,
        args=srv.args,
        builtin=srv.builtin,
        requires_os=srv.requires_os,
        os_supported=os_supported,
        server_type=conn.server_type if conn else "generic",
        context_cache_active=conn.context_cache_active if conn else False,
        generated=srv.generated,
        required_secrets=list(srv.required_secrets),
        missing_secrets=_missing_secrets(srv),
        optional_secrets=list(srv.optional_secrets),
        auth=_auth_status(srv),
    ).model_dump()


@router.get("")
async def list_mcp_servers():
    current_os = _current_os()
    cfg = await AppConfig.aload()
    statuses: list[dict] = []
    for srv in cfg.mcp_servers:
        os_ok = srv.requires_os is None or srv.requires_os == current_os
        statuses.append(_status_for(srv, os_supported=os_ok))
    statuses.sort(key=lambda s: (not s["builtin"], s["name"]))
    return statuses


@router.post("")
async def add_mcp_server(req: MCPServerAddRequest):
    cfg = await AppConfig.aload()
    server_id = slugify(req.name)

    if any(s.id == server_id for s in cfg.mcp_servers):
        return JSONResponse(status_code=409, content={"error": f"Server '{server_id}' already exists"})

    new_server = MCPServerConfig(
        id=server_id,
        name=req.name,
        transport=req.transport,
        url=req.url,
        port=req.port,
        command=req.command,
        args=req.args,
        env=req.env,
        enabled=True,
        auto_start=req.auto_start,
        builtin=False,
    )
    cfg.mcp_servers.append(new_server)
    await cfg.asave()
    return {"status": "added", "id": server_id}


@router.put("/{server_id}")
async def update_mcp_server(server_id: str, req: MCPServerAddRequest):
    cfg = await AppConfig.aload()
    for srv in cfg.mcp_servers:
        if srv.id == server_id:
            srv.name = req.name
            srv.transport = req.transport
            srv.url = req.url
            srv.port = req.port
            srv.command = req.command
            srv.args = req.args
            srv.env = req.env
            srv.auto_start = req.auto_start
            await cfg.asave()
            await mcp_mgr.disconnect(server_id)
            return {"status": "updated", "id": server_id}
    return JSONResponse(status_code=404, content={"error": "Server not found"})


@router.put("/{server_id}/excluded-tools")
async def update_excluded_tools(server_id: str, payload: dict):
    """Update the list of excluded tools for an MCP server."""
    cfg = await AppConfig.aload()
    srv = next((s for s in cfg.mcp_servers if s.id == server_id), None)
    if not srv:
        return JSONResponse(status_code=404, content={"error": "Server not found"})
    srv.excluded_tools = payload.get("excluded_tools", [])
    await cfg.asave()
    await session_mgr.refresh_tools(await AppConfig.aload())
    return {"status": "updated", "excluded_tools": srv.excluded_tools}


@router.delete("/{server_id}")
async def remove_mcp_server(server_id: str):
    from backend.agent_library import list_agents

    cfg = await AppConfig.aload()
    srv = next((s for s in cfg.mcp_servers if s.id == server_id), None)
    if srv and srv.builtin:
        return JSONResponse(status_code=409, content={"error": "Built-in servers cannot be removed"})
    agents = await asyncio.to_thread(list_agents)
    referencing_agents = [a.name for a in agents if server_id in a.tools]
    if referencing_agents:
        return JSONResponse(status_code=409, content={"error": f"Cannot delete: referenced by agent(s): {', '.join(referencing_agents)}"})

    await mcp_mgr.disconnect(server_id)

    if srv and srv.generated:
        # Agent-built servers have a source file, manifest, and vault
        # entries.  Route through the registry so they all get cleaned
        # up atomically rather than leaving orphans on disk.
        from backend.mcp_builder import delete_generated_server
        result = await delete_generated_server(server_id)
        credentials_removed = result.get("credentials_removed", 0)
        return {"status": "removed", "credentials_removed": credentials_removed}

    cfg.mcp_servers = [s for s in cfg.mcp_servers if s.id != server_id]
    await cfg.asave()

    # Externally-registered servers may still have keychain entries
    # (e.g. an npx GitHub MCP with GITHUB_PERSONAL_ACCESS_TOKEN).
    # Clean them up so a future re-register starts from a clean slate.
    credentials_removed = 0
    if srv is not None:
        from backend.credential_vault import vault
        try:
            credentials_removed = vault.delete_all(server_id)
        except Exception:
            pass
    return {"status": "removed", "credentials_removed": credentials_removed}


@router.get("/export")
async def export_mcp_servers():
    """Export non-builtin MCP servers as standard JSON config."""
    cfg = await AppConfig.aload()
    servers: dict[str, dict] = {}
    for srv in cfg.mcp_servers:
        if srv.builtin:
            continue
        entry: dict = {}
        if srv.transport == "stdio":
            entry["command"] = srv.command or ""
            if srv.args:
                entry["args"] = srv.args
            if srv.env:
                entry["env"] = srv.env
        else:
            entry["url"] = srv.url or ""
            if srv.transport != "streamable_http":
                entry["transport"] = srv.transport
        servers[srv.name] = entry
    return {"mcpServers": servers}


@router.post("/import")
async def import_mcp_servers(payload: dict):
    """Import MCP servers from standard JSON config format.

    Accepts: ``{ "mcpServers": { "Name": { ... }, ... } }``
    """
    mcp_servers = payload.get("mcpServers") or payload.get("mcp_servers") or {}
    if not isinstance(mcp_servers, dict):
        return JSONResponse(status_code=400, content={"error": "Expected 'mcpServers' to be an object"})

    cfg = await AppConfig.aload()
    existing_ids = {s.id for s in cfg.mcp_servers}
    added = []
    skipped = []

    for name, spec in mcp_servers.items():
        if not isinstance(spec, dict):
            skipped.append(name)
            continue

        server_id = slugify(name)
        if server_id in existing_ids:
            skipped.append(name)
            continue

        has_command = bool(spec.get("command"))
        transport = spec.get("transport", "stdio" if has_command else "streamable_http")

        new_srv = MCPServerConfig(
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
        )
        cfg.mcp_servers.append(new_srv)
        existing_ids.add(server_id)
        added.append(name)

    if added:
        await cfg.asave()

    return {"added": added, "skipped": skipped}


@router.post("/{server_id}/connect")
async def connect_mcp_server(server_id: str):
    cfg = await AppConfig.aload()
    srv = next((s for s in cfg.mcp_servers if s.id == server_id), None)
    if not srv:
        return JSONResponse(status_code=404, content={"error": "Server not found"})
    reset_circuit_breaker(server_id)
    await mcp_mgr.connect(srv)
    await session_mgr.refresh_tools(await AppConfig.aload())
    return _status_for(srv, os_supported=True)


@router.post("/{server_id}/start")
async def start_mcp_process(server_id: str):
    """Bring an MCP server online.

    Two transports converge on this single endpoint:

    * **HTTP MCPs** (Playwright, eval services) — spawn the long-running
      HTTP subprocess via :meth:`MCPManager.ensure_process`, then connect
      the MCP client.  Requires ``auto_start=True`` like before.
    * **stdio MCPs** (anything the agent generated, plus user-imported
      stdio configs) — call :meth:`MCPManager.connect` directly; the
      langchain-mcp-adapters client owns the subprocess lifecycle.  Does
      NOT require ``auto_start`` because there's no separate process to
      pre-warm — connection IS the process.

    Stdio MCPs additionally fail fast with a structured 400 when any
    ``required_secret`` is missing from the vault, so the frontend can
    pop a "fill credentials" affordance instead of letting the
    subprocess die at spawn with a cryptic ``KeyError``.
    """
    current_os = _current_os()
    cfg = await AppConfig.aload()
    srv = next((s for s in cfg.mcp_servers if s.id == server_id), None)
    if not srv:
        return JSONResponse(status_code=404, content={"error": "Server not found"})
    if not srv.enabled:
        from backend.mcp_manager import HOOK_DISABLED_LABELS
        msg = HOOK_DISABLED_LABELS.get(server_id, "Server is disabled. Enable it first.")
        return JSONResponse(status_code=400, content={"error": msg})
    if srv.requires_os and srv.requires_os != current_os:
        return JSONResponse(status_code=400, content={"error": f"This server requires {srv.requires_os}"})

    is_stdio = srv.transport == "stdio"

    if not is_stdio and not srv.auto_start:
        return JSONResponse(status_code=400, content={"error": "Server is not configured for auto-start"})

    if is_stdio:
        missing = _missing_secrets(srv)
        if missing:
            return JSONResponse(status_code=400, content={
                "error": "missing_credentials",
                "message": (
                    f"Cannot start {srv.name!r}: missing credential(s) "
                    f"{', '.join(missing)} in the keychain."
                ),
                "missing": missing,
                "server_id": server_id,
            })

        auth_status = _auth_status(srv)
        if auth_status.needs_login:
            return JSONResponse(status_code=400, content={
                "error": "needs_login",
                "message": (
                    f"Cannot start {srv.name!r}: interactive login required."
                ),
                "auth_kind": auth_status.kind,
                "server_id": server_id,
            })

    try:
        reset_circuit_breaker(server_id)
        if is_stdio:
            conn = await mcp_mgr.connect(srv)
        else:
            await mcp_mgr.ensure_process(srv)
            conn = await mcp_mgr.connect(srv, skip_process_start=True)
        await session_mgr.refresh_tools(await AppConfig.aload())
        return {
            "status": "running",
            "process_running": mcp_mgr.is_process_running(srv.id),
            "connected": conn.connected,
            "tool_count": len(conn.tools),
            "error": conn.error,
            "missing_secrets": _missing_secrets(srv),
        }
    except Exception as exc:
        logger.exception("Failed to start process for %s", server_id)
        return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})


@router.post("/{server_id}/stop")
async def stop_mcp_process(server_id: str):
    """Bring an MCP server offline.

    For HTTP MCPs this stops the managed subprocess and closes the
    client.  For stdio MCPs there is no managed subprocess, so we just
    disconnect — closing the client kills the child process owned by
    ``MultiServerMCPClient``.
    """
    await mcp_mgr.stop_process(server_id)
    if server_id in mcp_mgr.connections:
        await mcp_mgr.disconnect(server_id)
    await session_mgr.refresh_tools(await AppConfig.aload())
    return {"status": "stopped", "process_running": False}


@router.post("/{server_id}/test")
async def test_mcp_server(server_id: str):
    cfg = await AppConfig.aload()
    srv = next((s for s in cfg.mcp_servers if s.id == server_id), None)
    if not srv:
        return JSONResponse(status_code=404, content={"success": False, "message": "Server not found"})
    success, message = await mcp_mgr.test_connection(srv)
    return {"success": success, "message": message}


# ---------------------------------------------------------------------------
# Interactive auth flows (OAuth device + auth-code, browser bearer-token
# capture).  Backed by :mod:`backend.auth` providers — every flow shares
# this single ``/auth/{login,status,logout}`` triplet so the frontend
# only needs one code path regardless of vendor.
# ---------------------------------------------------------------------------


@router.get("/{server_id}/auth/status")
async def get_mcp_auth_status(server_id: str):
    """Return the current auth bundle state for *server_id*.

    Boolean / name-only — never includes token contents.  Idempotent
    and cheap; the frontend can poll this after a login click without
    re-fetching the whole server list.
    """
    cfg = await AppConfig.aload()
    srv = next((s for s in cfg.mcp_servers if s.id == server_id), None)
    if not srv:
        return JSONResponse(status_code=404, content={"error": "Server not found"})
    return _auth_status(srv).model_dump()


@router.post("/{server_id}/auth/login")
async def login_mcp_auth(server_id: str):
    """Trigger the interactive login flow for *server_id*.

    The flow runs in the parent backend process (browser launch, CDP
    capture, OAuth poll, etc.) and persists a token bundle in the OS
    keychain on success.  The MCP subprocess itself is untouched —
    this endpoint exists purely to populate the vault before a
    subsequent ``/start`` succeeds.

    Errors are returned as ``{"status": "error", "message": ...}``
    rather than HTTP 5xx so the frontend can render the message
    inline next to the Login button without parsing FastAPI exception
    payloads.
    """
    cfg = await AppConfig.aload()
    srv = next((s for s in cfg.mcp_servers if s.id == server_id), None)
    if not srv:
        return JSONResponse(status_code=404, content={"error": "Server not found"})
    if srv.auth is None or srv.auth.kind == "static":
        return JSONResponse(status_code=400, content={
            "error": "static_auth",
            "message": (
                f"{srv.name!r} uses paste-a-string credentials — set them "
                f"via the credentials dialog, not the login flow."
            ),
        })

    from backend.auth import NeedsLoginError, get_provider
    from backend.credential_vault import vault

    try:
        provider = get_provider(srv.auth.kind)
    except KeyError:
        return JSONResponse(status_code=400, content={
            "error": "unknown_auth_kind",
            "message": f"No provider registered for {srv.auth.kind!r}.",
        })

    try:
        bundle = await provider.acquire(srv.auth, server_id)
    except NeedsLoginError as exc:
        return {
            "status": "error",
            "auth_kind": exc.kind,
            "reason": exc.reason,
            "message": str(exc),
        }
    except Exception as exc:
        logger.exception("auth.acquire failed for %s", server_id)
        return {
            "status": "error",
            "auth_kind": srv.auth.kind,
            "message": f"{type(exc).__name__}: {exc}",
        }

    try:
        vault.set_bundle(server_id, bundle)
    except Exception as exc:
        logger.exception("vault.set_bundle failed for %s", server_id)
        return {
            "status": "error",
            "auth_kind": srv.auth.kind,
            "message": f"Login succeeded but persisting bundle failed: {exc}",
        }

    return {
        "status": "ok",
        "auth": _auth_status(srv).model_dump(),
    }


@router.post("/{server_id}/auth/logout")
async def logout_mcp_auth(server_id: str):
    """Wipe the persisted auth bundle for *server_id* (without disconnecting).

    Per-name static credentials are left alone — those are managed via
    the existing vault routes.  After this returns, ``/auth/status``
    reports ``needs_login=True`` and ``/start`` refuses with a 400.
    """
    cfg = await AppConfig.aload()
    srv = next((s for s in cfg.mcp_servers if s.id == server_id), None)
    if not srv:
        return JSONResponse(status_code=404, content={"error": "Server not found"})

    try:
        from backend.credential_vault import vault

        existed = vault.delete_bundle(server_id)
    except Exception as exc:
        return JSONResponse(status_code=500, content={
            "error": "vault_unavailable",
            "message": str(exc),
        })

    return {
        "status": "logged_out" if existed else "no_bundle",
        "auth": _auth_status(srv).model_dump(),
    }
