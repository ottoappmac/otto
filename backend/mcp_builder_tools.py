"""LangChain tools that let the agent author and manage its own MCP servers.

These wrap :mod:`backend.mcp_builder` and :mod:`backend.credential_vault`
behind the LangChain ``@tool`` interface so they can be injected into a
DeepAgent's toolbox.

Important: **none of these tools return credential values.**  They can
only check whether a credential is set (``is_credential_set``) or
trigger the user to supply one (``request_credential``).  The actual
value lives in the OS keychain and only enters an MCP subprocess at
spawn time, never in the agent's chat history.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool
from langgraph.types import interrupt

from backend.config import MCPAuthConfig
from backend.credential_vault import vault
from backend.mcp_builder import (
    ALLOWED_THIRD_PARTY,
    MCPGenerationError,
    MCPSpec,
    ToolSpec,
    delete_generated_server,
    generate_mcp_server,
    list_generated_servers,
)
from backend.utils import run_coro_sync

logger = logging.getLogger(__name__)


def build_mcp_builder_tools() -> list:
    """Return the agent-facing MCP authoring tools."""

    @tool
    def list_allowed_mcp_imports() -> str:
        """List the third-party Python packages that generated MCP servers
        may import.  Use this BEFORE drafting an MCP spec — if the API
        you want to integrate isn't covered here, fall back to ``httpx``
        and call the API's REST endpoints directly.

        Returns a JSON list of allowed top-level import names.
        """
        return json.dumps(sorted(ALLOWED_THIRD_PARTY), indent=2)

    @tool
    def list_my_mcp_servers() -> str:
        """List MCP servers that were authored by the agent (i.e. via
        ``create_mcp_server``).  Built-in servers like ``playwright-mcp``
        are not included — use ``list_available_mcp_servers`` for those.

        Returns a JSON array of generated servers with their tool names
        and required credentials (no values, just names).
        """
        return json.dumps(list_generated_servers(), indent=2)

    @tool
    def is_credential_set(server_id: str, name: str) -> str:
        """Check whether a credential is already stored in the keychain
        for the given MCP server.  Returns 'yes' or 'no' — NEVER returns
        the value.

        Use this to decide whether to call ``request_credential`` before
        connecting an MCP that needs that credential.

        Args:
            server_id: The MCP server id (e.g. ``"stripe"``).
            name:      Credential name in SHOUTY_SNAKE_CASE
                       (e.g. ``"STRIPE_SECRET_KEY"``).
        """
        return "yes" if vault.has(server_id, name) else "no"

    @tool
    def list_credentials_for(server_id: str) -> str:
        """Return the NAMES of credentials stored for a given MCP
        server.  Names only — never values.

        Args:
            server_id: The MCP server id (e.g. ``"stripe"``).
        """
        return json.dumps(vault.list_names(server_id))

    @tool
    def request_credential(
        server_id: str,
        name: str,
        display_label: str,
        instructions: str,
        signup_url: str = "",
    ) -> str:
        """Ask the user to supply a credential for an MCP server.

        Pauses the agent (LangGraph ``interrupt``) and surfaces a UI
        prompt of type ``request_credential``.  The frontend is
        expected to render a secret-input dialog and POST the value to
        ``/api/vault/secrets/{server_id}/{name}`` BEFORE resuming.  We
        deliberately do NOT pass the value back through the
        interrupt's resume payload — the agent must never see it.

        After the user submits, this tool returns 'stored' if the
        credential is now in the vault, or 'cancelled' / 'missing' if
        the user aborted.

        Use this only after the credential ladder has been exhausted:
        ``is_credential_set`` first, then trigger the MCP's declared
        auth flow if non-static, then attempt auto-signup via
        ``web-voyager`` if the API has a self-serve dev console.

        Args:
            server_id:     The MCP server id (e.g. ``"stripe"``).
            name:          Credential name in SHOUTY_SNAKE_CASE
                           (e.g. ``"STRIPE_SECRET_KEY"``).
            display_label: Human-readable label for the UI dialog
                           (e.g. ``"Stripe secret API key"``).
            instructions:  Brief instructions for the user on where to
                           obtain the credential
                           (e.g. ``"Find this in dashboard.stripe.com → Developers → API keys"``).
            signup_url:    Optional URL where the user can register and
                           obtain the key.  Rendered as a one-click
                           button in the dialog.  Pass the same URL the
                           MCP's manifest declared in ``auth.signup_url``
                           if available; otherwise the documented dev
                           console URL.
        """
        payload: dict[str, Any] = {
            "type": "request_credential",
            "server_id": server_id,
            "name": name,
            "display_label": display_label,
            "instructions": instructions,
            "signup_url": signup_url,
        }
        # The agent loop suspends here until the frontend resumes
        # with a decision.  The frontend's flow:
        #   1. Show a secure dialog (masked input).
        #   2. On submit, POST /api/vault/secrets/{id}/{name}.
        #   3. Resume the agent with {"decisions": [{"answer": "stored"}]}.
        #   4. If user cancels, resume with "cancelled".
        result = interrupt(payload)

        # Whatever the frontend echoed back, re-check the vault as the
        # source of truth — the LLM should never trust a self-reported
        # "stored" without verification.
        if vault.has(server_id, name):
            return "stored"

        if isinstance(result, dict):
            decisions = result.get("decisions") or []
            if decisions and isinstance(decisions[0], dict):
                ans = str(decisions[0].get("answer") or "")
                if ans.lower() == "cancelled":
                    return "cancelled"
        return "missing"

    @tool
    def create_mcp_server(
        server_id: str,
        display_name: str,
        description: str,
        required_secrets: list[str],
        allowed_imports: list[str],
        tools_json: str,
        auth_json: str = "",
    ) -> str:
        """Generate, audit, and register a new MCP server.

        After this returns successfully, the new server appears in the
        config but is NOT yet connected — the user must first either
        set every credential in ``required_secrets`` (static auth) OR
        complete the configured login flow (interactive auth) via the
        Settings → Credentials → Login button.  Use ``request_credential``
        for missing static entries, then call ``connect_mcp_server`` to
        bring it online.

        The generated source file lives under the app data directory
        (``mcp_servers/<server_id>.py``) and is statically audited:
        forbidden imports, ``exec`` / ``eval`` calls, and string literals
        that look like API keys cause the call to fail.

        Args:
            server_id: kebab-case id, 2-63 chars, starts with a letter
                       (e.g. ``"stripe"``).
            display_name: Human-friendly name (e.g. ``"Stripe Payments"``).
            description: 1-3 sentences on what the MCP does.
            required_secrets: SHOUTY_SNAKE_CASE env-var names the
                              generated code references via
                              ``os.environ[name]`` (e.g.
                              ``["STRIPE_SECRET_KEY"]``). Each maps to
                              a vault entry the user must populate.
                              Leave empty when ``auth_json`` configures
                              an interactive flow that injects the
                              credential via ``env_mapping`` instead.
            allowed_imports: Subset of the names returned by
                             ``list_allowed_mcp_imports`` that this
                             server actually needs.  Anything outside
                             the allowlist is rejected by the auditor.
            tools_json: JSON array of tool specs.  Each entry is an
                        object with keys ``name`` (snake_case),
                        ``description`` (str), ``params`` (list of
                        ``[name, py_type_str, default_or_null]``), and
                        ``body`` (Python source for the tool body — no
                        ``def`` line, no decorator).
            auth_json: Optional JSON object describing an interactive
                       auth flow.  Keys mirror :class:`MCPAuthConfig` —
                       at minimum ``kind`` (``"oauth_device"``,
                       ``"oauth_authcode"``, or ``"browser_capture"``)
                       plus the per-kind URLs / client_id and
                       ``env_mapping`` mapping env-var names to bundle
                       fields (e.g. ``{"GITHUB_TOKEN": "access_token"}``).
                       Always include ``signup_url`` when the API has a
                       self-serve dev console — used by the credentials
                       dialog and by the auto-signup flow when manual
                       paste is the last resort.
                       Leave empty for the default static flow.

        Example tools_json:

        [
          {
            "name": "create_charge",
            "description": "Create a Stripe charge for the given amount.",
            "params": [
              ["amount", "int", null],
              ["currency", "str", "\\"usd\\""],
              ["description", "str", "\\"\\""]
            ],
            "body": "import stripe\\nstripe.api_key = os.environ['STRIPE_SECRET_KEY']\\ncharge = stripe.Charge.create(amount=amount, currency=currency, description=description)\\nreturn {'id': charge.id, 'status': charge.status}"
          }
        ]

        Example auth_json (GitHub OAuth device flow):

        {
          "kind": "oauth_device",
          "client_id": "Ov23liYourClientId",
          "device_url": "https://github.com/login/device/code",
          "token_url": "https://github.com/login/oauth/access_token",
          "scopes": ["repo", "read:user"],
          "env_mapping": {"GITHUB_TOKEN": "access_token"}
        }
        """
        try:
            tool_objs = json.loads(tools_json)
        except json.JSONDecodeError as exc:
            return f"Error: tools_json is not valid JSON: {exc}"

        if not isinstance(tool_objs, list):
            return "Error: tools_json must be a JSON array."

        tools: list[ToolSpec] = []
        for i, t in enumerate(tool_objs):
            if not isinstance(t, dict):
                return f"Error: tools_json[{i}] is not an object."
            try:
                params_raw = t.get("params") or []
                params: list[tuple[str, str, Any]] = []
                for p in params_raw:
                    if not isinstance(p, list) or len(p) != 3:
                        return (
                            f"Error: tools_json[{i}].params has bad shape — "
                            f"each entry must be [name, type, default_or_null]"
                        )
                    pname, ptype, pdefault = p
                    params.append((pname, ptype, pdefault))
                tools.append(ToolSpec(
                    name=t["name"],
                    description=t.get("description", ""),
                    params=params,
                    body=t.get("body", "raise NotImplementedError"),
                ))
            except KeyError as exc:
                return f"Error: tools_json[{i}] missing key {exc}"

        auth_cfg: MCPAuthConfig | None = None
        if auth_json.strip():
            try:
                auth_dict = json.loads(auth_json)
            except json.JSONDecodeError as exc:
                return f"Error: auth_json is not valid JSON: {exc}"
            if not isinstance(auth_dict, dict):
                return "Error: auth_json must be a JSON object."
            try:
                auth_cfg = MCPAuthConfig.model_validate(auth_dict)
            except Exception as exc:
                return f"Error: auth_json failed validation: {exc}"

        spec = MCPSpec(
            id=server_id,
            name=display_name,
            description=description,
            required_secrets=required_secrets,
            allowed_imports=allowed_imports,
            tools=tools,
            auth=auth_cfg,
        )

        try:
            result = run_coro_sync(generate_mcp_server(spec))
        except MCPGenerationError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            logger.exception("create_mcp_server failed")
            return f"Error: {exc}"

        return json.dumps(result, indent=2)

    @tool
    def delete_mcp_server(server_id: str) -> str:
        """Permanently delete an MCP server (generated or externally registered).

        Removes the config entry and every keychain credential under that
        server_id.  For agent-built servers it also removes the generated
        source file and manifest.  Built-in servers cannot be deleted
        through this tool.

        Args:
            server_id: The MCP server id (e.g. ``"stripe"``).
        """
        try:
            result = run_coro_sync(delete_generated_server(server_id))
        except Exception as exc:
            logger.exception("delete_mcp_server failed")
            return f"Error: {exc}"
        return json.dumps(result, indent=2)

    @tool
    def connect_mcp_server(server_id: str) -> str:
        """Bring an agent-built MCP server online.

        Spawns the subprocess (with credentials hydrated from the
        keychain into env vars), connects the MCP client, and refreshes
        the agent's tool list so subsequent calls can use the new
        tools.  Returns an error string if any required credential is
        still missing — call ``request_credential`` first in that case.

        Args:
            server_id: The MCP server id (e.g. ``"stripe"``).
        """
        async def _connect():
            from backend.config import AppConfig
            from backend.state import mcp_mgr, session_mgr

            cfg = await AppConfig.aload()
            srv = next((s for s in cfg.mcp_servers if s.id == server_id), None)
            if srv is None:
                return f"Error: MCP server {server_id!r} is not registered."
            missing = [n for n in srv.required_secrets if not vault.has(server_id, n)]
            if missing:
                return (
                    f"Error: missing credentials in vault for {server_id}: "
                    f"{missing}.  Call request_credential first."
                )
            conn = await mcp_mgr.connect(srv)
            try:
                await session_mgr.refresh_tools(cfg)
            except Exception:
                logger.debug("refresh_tools failed after connect", exc_info=True)
            return json.dumps({
                "id": server_id,
                "connected": conn.connected,
                "error": conn.error,
                "tools": [t.name for t in (conn.tools or [])],
            }, indent=2)

        return run_coro_sync(_connect())

    @tool
    def register_external_mcp_server(
        server_id: str,
        display_name: str,
        transport: str,
        command: str = "",
        args: list[str] | None = None,
        url: str = "",
        required_secrets: list[str] | None = None,
    ) -> str:
        """Register an existing third-party MCP server (stdio or HTTP).

        Use this for off-the-shelf servers you DON'T need to author —
        e.g. ``@modelcontextprotocol/server-github`` (stdio), Notion's
        hosted MCP (HTTP), or any vendor-provided server.

        For ``transport='stdio'`` the backend will run
        ``command`` ``args...`` as a subprocess at start time and pipe
        ``required_secrets`` into its environment from the keychain.
        For ``transport='streamable_http'`` or ``'sse'`` the backend will
        connect to ``url`` directly.

        After this returns, the user must populate every name in
        ``required_secrets`` via ``request_credential`` before
        ``connect_mcp_server`` will succeed.  If you don't know what
        secrets the server needs, leave ``required_secrets`` empty —
        the user can still set env vars manually via the Settings UI.

        Args:
            server_id: kebab-case id (must be unique across all MCPs).
            display_name: Human-friendly name (e.g. ``"GitHub MCP"``).
            transport: ``"stdio"``, ``"streamable_http"``, or ``"sse"``.
            command: For stdio, the executable (e.g. ``"npx"``).
            args: For stdio, the argv list (e.g.
                  ``["-y", "@modelcontextprotocol/server-github"]``).
            url: For HTTP/SSE, the endpoint URL.
            required_secrets: SHOUTY_SNAKE_CASE env-var names the server
                              expects in its environment (e.g.
                              ``["GITHUB_PERSONAL_ACCESS_TOKEN"]``).
        """
        async def _add():
            from backend.config import AppConfig, MCPServerConfig

            if transport not in ("stdio", "streamable_http", "sse"):
                return f"Error: transport must be stdio, streamable_http, or sse (got {transport!r})"
            if transport == "stdio" and not command:
                return "Error: stdio transport requires command"
            if transport != "stdio" and not url:
                return f"Error: {transport} transport requires url"

            cfg = await AppConfig.aload()
            if any(s.id == server_id for s in cfg.mcp_servers):
                return f"Error: a server with id {server_id!r} already exists. Use a different id or delete the existing server first."

            cfg.mcp_servers.append(MCPServerConfig(
                id=server_id,
                name=display_name,
                transport=transport,
                command=command if transport == "stdio" else None,
                args=list(args or []),
                url=url if transport != "stdio" else None,
                enabled=True,
                auto_start=False,
                builtin=False,
                generated=False,
                required_secrets=list(required_secrets or []),
            ))
            await cfg.asave()

            missing = [n for n in (required_secrets or []) if not vault.has(server_id, n)]
            return json.dumps({
                "id": server_id,
                "registered": True,
                "transport": transport,
                "required_secrets": list(required_secrets or []),
                "missing_secrets": missing,
            }, indent=2)

        return run_coro_sync(_add())

    return [
        list_allowed_mcp_imports,
        list_my_mcp_servers,
        is_credential_set,
        list_credentials_for,
        request_credential,
        create_mcp_server,
        register_external_mcp_server,
        delete_mcp_server,
        connect_mcp_server,
    ]
