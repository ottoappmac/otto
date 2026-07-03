"""End-to-end smoke test: generate an MCP, spawn it, list its tools.

This goes one level deeper than ``smoke_test_mcp_builder.py``:
* Stores a credential in the vault.
* Connects via the real ``MCPManager`` (which is what the running
  backend uses), so we exercise the secret-injection path.
* Calls the generated tool over MCP and asserts the response.

Run from the venv::

    python scripts/smoke_test_mcp_integration.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))


async def main() -> None:
    from backend.config import AppConfig
    from backend.credential_vault import vault
    from backend.mcp_builder import (
        MCPSpec, ToolSpec,
        delete_generated_server, generate_mcp_server,
    )
    from backend.mcp_manager import MCPManager

    SERVER_ID = "smoke-integration"
    SECRET_NAME = "SMOKE_INT_TOKEN"
    SECRET_VALUE = "tok_smoketest_value_with_known_length_30"

    print("=" * 60)
    print("MCP Integration Smoke Test")
    print("=" * 60)

    # 1. Store credential in vault
    print("\n[1] Storing credential in vault")
    vault.set(SERVER_ID, SECRET_NAME, SECRET_VALUE)
    assert vault.has(SERVER_ID, SECRET_NAME)
    print(f"  OK — vault has {SERVER_ID}/{SECRET_NAME}")

    # 2. Generate a tiny MCP that echoes the token's length
    print("\n[2] Generating MCP server")
    spec = MCPSpec(
        id=SERVER_ID,
        name="Smoke Integration",
        description="Tiny MCP for end-to-end integration testing.",
        required_secrets=[SECRET_NAME],
        allowed_imports=[],
        tools=[
            ToolSpec(
                name="token_length",
                description="Return the length of the configured secret token.",
                params=[],
                body=f"token = os.environ['{SECRET_NAME}']\nreturn {{'length': len(token)}}",
            ),
            ToolSpec(
                name="echo",
                description="Echo a message back, with a fixed prefix.",
                params=[("message", "str", None)],
                body="return f'echo: {message}'",
            ),
        ],
    )
    result = await generate_mcp_server(spec)
    assert result["registered"]
    assert not result["missing_secrets"]
    print(f"  OK — generated at {result['path']}")

    # 3. Connect via MCPManager — this exercises subprocess spawn and
    #    secret hydration into env.
    print("\n[3] Connecting via MCPManager (real path)")
    cfg = await AppConfig.aload()
    srv = next(s for s in cfg.mcp_servers if s.id == SERVER_ID)
    mgr = MCPManager()
    try:
        conn = await mgr.connect(srv)
        if not conn.connected:
            raise RuntimeError(f"connect failed: {conn.error}")
        print(f"  OK — connected, {len(conn.tools)} tools loaded:")
        for t in conn.tools:
            print(f"     - {t.name}")
        tool_names = {t.name for t in conn.tools}
        assert "token_length" in tool_names, tool_names
        assert "echo" in tool_names, tool_names

        # 4. Call token_length — should report exactly len(SECRET_VALUE)
        print("\n[4] Calling token_length tool")
        token_tool = next(t for t in conn.tools if t.name == "token_length")
        out = await token_tool.ainvoke({})
        print(f"  result -> {out!r}")
        assert str(len(SECRET_VALUE)) in str(out), out
        print(f"  OK — tool saw token of length {len(SECRET_VALUE)}")

        # 5. Call echo
        print("\n[5] Calling echo tool")
        echo_tool = next(t for t in conn.tools if t.name == "echo")
        out = await echo_tool.ainvoke({"message": "hello world"})
        print(f"  result -> {out!r}")
        assert "hello world" in str(out)
        print("  OK — echo round-tripped through MCP")

    finally:
        # 6. Cleanup
        print("\n[6] Cleanup")
        try:
            await mgr.disconnect_all()
        except Exception as exc:
            print(f"  disconnect_all failed: {exc}")
        out = await delete_generated_server(SERVER_ID)
        print(f"  delete -> {out['status']} (creds_removed={out['credentials_removed']})")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
