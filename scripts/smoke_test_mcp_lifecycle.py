"""End-to-end test of the stdio MCP lifecycle through HTTP routes.

Exercises the new behaviour added on top of the agent-built MCP path:

* ``GET /api/mcp-servers`` exposes ``generated``, ``required_secrets``,
  and ``missing_secrets`` per server.
* ``POST /api/mcp-servers/{id}/start`` refuses with 400
  ``missing_credentials`` when the vault is empty.
* After populating the vault via ``POST /api/vault/secrets``, ``/start``
  succeeds and the listed status reports ``connected=True`` and
  ``process_running=True``.
* ``POST /api/mcp-servers/{id}/stop`` brings it back down with
  ``process_running=False``.
* ``DELETE /api/mcp-servers/{id}`` cleans up source file, manifest, and
  every vault entry — verified by the keychain index.

Run while the backend is up on :18081::

    python scripts/smoke_test_mcp_lifecycle.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))


BASE = "http://localhost:18081"


async def main() -> None:
    import httpx

    from backend.mcp_builder import (
        MCPSpec, ToolSpec, generate_mcp_server,
    )

    SERVER_ID = "smoke-lifecycle"
    SECRET_NAME = "SMOKE_LIFECYCLE_TOKEN"
    SECRET_VALUE = "live_value_with_42_chars__abcdefghijklmnop"

    print("=" * 60)
    print("MCP Lifecycle Smoke Test (HTTP)")
    print("=" * 60)

    # 1. Generate the MCP via the in-process builder so the test is
    #    self-contained.  In real life the agent would call this via
    #    create_mcp_server.
    print("\n[1] Generating MCP server (in-process)")
    spec = MCPSpec(
        id=SERVER_ID,
        name="Smoke Lifecycle",
        description="Lifecycle test MCP.",
        required_secrets=[SECRET_NAME],
        allowed_imports=[],
        tools=[
            ToolSpec(
                name="token_length",
                description="Returns length of the configured token.",
                params=[],
                body=f"return {{'length': len(os.environ['{SECRET_NAME}'])}}",
            ),
        ],
    )
    result = await generate_mcp_server(spec)
    assert result["registered"]
    print(f"  generated -> {result['path']}")

    async with httpx.AsyncClient(base_url=BASE, timeout=15.0) as client:
        try:
            # 2. List MCP servers — verify our server appears with the
            #    new fields.
            print("\n[2] GET /api/mcp-servers includes new fields")
            r = await client.get("/api/mcp-servers")
            r.raise_for_status()
            servers = r.json()
            srv = next((s for s in servers if s["id"] == SERVER_ID), None)
            assert srv is not None, "server missing in list"
            print(f"  generated         -> {srv['generated']}")
            print(f"  required_secrets  -> {srv['required_secrets']}")
            print(f"  missing_secrets   -> {srv['missing_secrets']}")
            print(f"  transport         -> {srv['transport']}")
            assert srv["generated"] is True
            assert srv["required_secrets"] == [SECRET_NAME]
            assert srv["missing_secrets"] == [SECRET_NAME]
            assert srv["transport"] == "stdio"

            # 3. Start should refuse with 400 missing_credentials.
            print("\n[3] POST /start fails fast when vault is empty")
            r = await client.post(f"/api/mcp-servers/{SERVER_ID}/start")
            print(f"  status -> {r.status_code}")
            print(f"  body   -> {r.json()}")
            assert r.status_code == 400
            body = r.json()
            assert body["error"] == "missing_credentials"
            assert SECRET_NAME in body["missing"]

            # 4. Set the credential through the vault REST API.
            print("\n[4] POST /api/vault/secrets stores the credential")
            r = await client.post(
                f"/api/vault/secrets/{SERVER_ID}/{SECRET_NAME}",
                json={"value": SECRET_VALUE},
            )
            r.raise_for_status()
            print(f"  body -> {r.json()}")

            # 5. Start now succeeds, status flips to running.
            print("\n[5] POST /start succeeds after credentials are set")
            r = await client.post(f"/api/mcp-servers/{SERVER_ID}/start")
            r.raise_for_status()
            print(f"  body -> {r.json()}")
            assert r.json()["connected"] is True
            assert r.json()["process_running"] is True
            assert r.json()["tool_count"] == 1

            # 6. List again, confirm process_running=true and missing_secrets is empty.
            print("\n[6] GET /api/mcp-servers reflects running state")
            r = await client.get("/api/mcp-servers")
            r.raise_for_status()
            srv = next(s for s in r.json() if s["id"] == SERVER_ID)
            print(f"  connected={srv['connected']} running={srv['process_running']} missing={srv['missing_secrets']}")
            assert srv["connected"] is True
            assert srv["process_running"] is True
            assert srv["missing_secrets"] == []

            # 7. Stop brings it back down.
            print("\n[7] POST /stop brings it offline")
            r = await client.post(f"/api/mcp-servers/{SERVER_ID}/stop")
            r.raise_for_status()
            print(f"  body -> {r.json()}")
            assert r.json()["process_running"] is False

        finally:
            # 8. Delete cleans up source file, manifest, and vault entries.
            print("\n[8] DELETE /api/mcp-servers cleans up everything")
            r = await client.delete(f"/api/mcp-servers/{SERVER_ID}")
            r.raise_for_status()
            print(f"  body -> {r.json()}")
            assert r.json()["status"] == "removed"
            assert r.json().get("credentials_removed") == 1

            # Verify on-disk state
            from backend.mcp_builder import server_path, manifest_path
            assert not server_path(SERVER_ID).exists(), "source file leaked"
            assert not manifest_path(SERVER_ID).exists(), "manifest leaked"

            # Verify vault state
            r = await client.get(f"/api/vault/secrets/{SERVER_ID}")
            r.raise_for_status()
            assert r.json()["names"] == []
            print("  OK — no leftover files or vault entries")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
