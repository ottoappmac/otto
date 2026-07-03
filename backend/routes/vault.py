"""REST API for the credential vault.

These are the **only** routes the frontend (or curl, for power users)
should call to manage stored credentials.  Notice what's missing:

* No ``GET /secrets/{server_id}/{name}/value`` — values cannot leave
  the backend through HTTP.  The only outbound path is the env-var
  injection in :mod:`backend.mcp_manager` at MCP subprocess spawn time.

* No ``GET /secrets/{server_id}/value`` either — same reason.

Operations exposed:

* ``POST   /api/vault/secrets/{server_id}/{name}``  — store a value
* ``DELETE /api/vault/secrets/{server_id}/{name}``  — delete one
* ``DELETE /api/vault/secrets/{server_id}``         — delete all for a server
* ``GET    /api/vault/secrets/{server_id}``         — list NAMES (no values)
* ``GET    /api/vault/secrets``                     — list servers with stored secrets
* ``GET    /api/vault/health``                      — keychain backend status
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.credential_vault import (
    CredentialVaultError,
    vault,
)

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/vault", tags=["vault"])


class SecretSetRequest(BaseModel):
    value: str = Field(min_length=1, max_length=8192)


@router.get("/health")
def vault_health() -> dict:
    """Report whether the OS keychain is reachable from this process.

    On macOS, the first call may show a "Otto wants to access
    Keychain" prompt — this endpoint is the cheapest way to trigger it
    deliberately.
    """
    if not vault.available():
        return {"available": False, "backend": None}
    try:
        kr = vault._kr()  # internal — fine here, this is the vault's own routes
        backend = type(kr.get_keyring()).__name__
    except Exception as exc:
        return {"available": False, "backend": None, "error": str(exc)}
    return {"available": True, "backend": backend}


@router.get("/secrets")
def list_servers_with_secrets() -> dict:
    """Return all server_ids that have at least one stored secret."""
    return {"servers": vault.list_servers()}


@router.get("/secrets/{server_id}")
def list_secret_names(server_id: str) -> dict:
    """Return the *names* of secrets stored for *server_id*.

    Names only — never values.  Safe to expose to the LLM via a tool.
    """
    return {"server_id": server_id, "names": vault.list_names(server_id)}


@router.post("/secrets/{server_id}/{name}")
def set_secret(server_id: str, name: str, req: SecretSetRequest) -> dict:
    """Store ``req.value`` under ``(server_id, name)`` in the keychain.

    The body MUST be sent over the local Tauri/HTTP loopback only.  This
    endpoint does no rate-limiting because the only legitimate caller
    is the desktop app's credential dialog, but it should NEVER be
    exposed beyond ``127.0.0.1``.
    """
    try:
        vault.set(server_id, name, req.value)
    except CredentialVaultError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "stored", "server_id": server_id, "name": name}


@router.delete("/secrets/{server_id}/{name}")
def delete_secret(server_id: str, name: str) -> dict:
    try:
        existed = vault.delete(server_id, name)
    except CredentialVaultError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "deleted" if existed else "not_found",
            "server_id": server_id, "name": name}


@router.delete("/secrets/{server_id}")
def delete_all_for_server(server_id: str) -> dict:
    try:
        n = vault.delete_all(server_id)
    except CredentialVaultError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "deleted_all", "server_id": server_id, "removed": n}
