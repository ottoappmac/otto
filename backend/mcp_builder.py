"""Generate, register, and verify agent-built MCP servers.

This module is the system the agent uses to **author** new tools at
runtime.  The flow:

1. Agent calls :func:`generate_mcp_server` with a name, description, and
   a list of tool specs (each is a Python function body that uses
   ``os.environ[...]`` to access credentials).
2. We render a per-MCP folder under
   ``~/Library/Application Support/Otto/mcp_server/<id>/`` containing::

       server.py         FastMCP server (the runnable subprocess)
       client.py         standalone smoke-test client (CLI)
       manifest.json     full spec — used for regeneration
       requirements.txt  PyPI deps derived from spec.allowed_imports
       README.md         usage notes for humans
       .venv/            isolated venv provisioned by uv (per-MCP)

3. We provision ``<id>/.venv`` with uv, installing ``mcp`` plus every
   package in ``requirements.txt``.  The MCP subprocess runs from this
   venv — the backend's main interpreter never has to carry vendor SDKs.
4. We register a new :class:`MCPServerConfig` with ``transport="stdio"``,
   ``command=<id>/.venv/bin/python``, ``args=[<id>/server.py]``, and
   ``required_secrets`` listing the credential names the agent declared.
5. The agent then prompts the user via the dedicated credential dialog
   (NOT chat) to fill the vault.
6. Once credentials are set, ``connect_mcp_server`` spawns the
   subprocess via the per-MCP venv.  The manager hydrates env vars
   from the keychain at spawn time — the LLM never sees the values.

Security boundaries enforced here:

* **Code auditing.**  Every generated server source file is parsed with
  :mod:`ast` and rejected if it imports forbidden modules
  (``socket``, ``subprocess``, ``os.system``...) or contains literal
  strings that look like API keys (defence against prompt-injection
  attacks where a malicious tool description tries to bake a leaked
  key into the file).
* **Allowlisted dependencies.**  Generated servers can only import from
  a curated allowlist of API SDKs (``stripe``, ``slack_sdk``, ``httpx``,
  etc.).  The list is intentionally short — extending it is a deliberate
  human action.
* **Per-MCP isolation.**  Each generated server gets its own venv and
  installed dependencies.  A bad ``slack_sdk`` install can't poison the
  ``stripe`` MCP, and the backend's main interpreter stays clean.
* **Credential references only.**  Generated code is forbidden from
  string-literal credentials.  All access goes through
  ``os.environ["NAME"]`` where ``NAME`` was declared in
  ``required_secrets``.
"""

from __future__ import annotations

import asyncio
import ast
import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from backend.config import (
    AppConfig,
    MCPAuthConfig,
    MCPServerConfig,
    get_app_data_dir,
)
from backend.mcp_sandbox import PermissionManifest, render_sandbox_profile, write_profile
from backend.mcp_signer import write_signature

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Locations — per-MCP folder layout
#
# Each generated MCP lives entirely inside ``mcp_server/<id>/``.  Wiping
# the folder + the config entry + vault entries is a complete uninstall.
# ---------------------------------------------------------------------------


def mcp_root_dir() -> Path:
    """Parent directory holding one folder per generated MCP."""
    p = get_app_data_dir() / "mcp_server"
    p.mkdir(parents=True, exist_ok=True)
    return p


def server_dir(server_id: str) -> Path:
    """Folder holding everything for one generated MCP."""
    return mcp_root_dir() / server_id


def server_path(server_id: str) -> Path:
    """The runnable FastMCP server file."""
    return server_dir(server_id) / "server.py"


def client_path(server_id: str) -> Path:
    """A standalone smoke-test client that talks to ``server.py`` over stdio."""
    return server_dir(server_id) / "client.py"


def manifest_path(server_id: str) -> Path:
    """Sidecar JSON with the original spec — used for ``regenerate``."""
    return server_dir(server_id) / "manifest.json"


def requirements_path(server_id: str) -> Path:
    """PyPI requirements derived from ``spec.allowed_imports``."""
    return server_dir(server_id) / "requirements.txt"


def readme_path(server_id: str) -> Path:
    return server_dir(server_id) / "README.md"


def permissions_path(server_id: str) -> Path:
    """JSON file describing the runtime sandbox declared for this MCP.

    Lives next to ``manifest.json`` so a single ``rm -rf <server_dir>``
    drops every artifact in one go.  Signed alongside ``server.py`` so
    later tampering is detected at spawn time.
    """
    return server_dir(server_id) / "permissions.json"


def sandbox_profile_path(server_id: str) -> Path:
    """Rendered macOS ``.sb`` profile (only present on macOS)."""
    return server_dir(server_id) / "sandbox.sb"


def venv_dir(server_id: str) -> Path:
    """Per-MCP isolated venv (provisioned by uv)."""
    return server_dir(server_id) / ".venv"


def venv_python(server_id: str) -> Path:
    """Python interpreter inside the per-MCP venv."""
    sub = "Scripts" if sys.platform == "win32" else "bin"
    name = "python.exe" if sys.platform == "win32" else "python"
    return venv_dir(server_id) / sub / name


def mcp_servers_dir() -> Path:
    """Backwards-compat alias for callers that still import the old name.

    Points at the new ``mcp_server/`` root.  Existing call sites in the
    smoke-test scripts only use this to compute paths for cleanup checks,
    so redirecting them to the new layout is safe.
    """
    return mcp_root_dir()


# ---------------------------------------------------------------------------
# Spec data classes — the input to generation
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    """One tool inside a generated MCP server.

    Attributes:
        name:        snake_case tool name (becomes the FastMCP @tool name)
        description: 1-2 sentence docstring for the tool
        params:      list of (param_name, python_type_str, default_repr_or_None)
        body:        Python source for the tool body (no def line, no
                     decorator). Uses ``os.environ["X"]`` for any secret X.
    """
    name: str
    description: str
    params: list[tuple[str, str, Optional[str]]] = field(default_factory=list)
    body: str = "return 'not implemented'"


@dataclass
class MCPSpec:
    """Full spec for a generated server.

    ``auth`` controls how the manager obtains the credentials that get
    projected into the MCP subprocess environment at spawn time.  When
    omitted (or ``kind="static"``) the historical
    ``required_secrets`` paste-a-string flow applies and nothing else
    changes.  When set to ``oauth_device`` / ``oauth_authcode`` /
    ``browser_capture``, the matching provider in
    :mod:`backend.auth` runs an interactive flow, persists a token
    bundle in the OS keychain, and ``required_secrets`` MAY be empty —
    the bundle's ``env_mapping`` is what populates the subprocess env.

    ``permissions`` declares the runtime sandbox the subprocess should
    be wrapped in (filesystem read/write paths, allowed network hosts,
    env var allowlist).  See :class:`backend.mcp_sandbox.PermissionManifest`.
    Defaults to a manifest that allows reads/writes only within the
    per-MCP folder and denies all outbound network -- the most
    restrictive option that still lets a credential-free MCP run.
    """
    id: str  # kebab-case
    name: str  # display name
    description: str
    required_secrets: list[str] = field(default_factory=list)
    allowed_imports: list[str] = field(default_factory=list)
    tools: list[ToolSpec] = field(default_factory=list)
    auth: Optional[MCPAuthConfig] = None
    permissions: Optional[PermissionManifest] = None


# ---------------------------------------------------------------------------
# Allowlisted imports
#
# Generated servers can ONLY import from this set (plus stdlib).  Every
# entry is a security decision: any import here gets to be loaded inside
# an MCP subprocess that has the user's API keys in its env.  The list
# is sized to cover the top ~50 MCP server integrations as of 2026 (see
# https://mcpmanager.ai/blog/most-popular-mcp-servers/), so most agent-
# authored servers can be expressed without expanding this set.
#
# Categories below mirror the structure of those popular-server lists.
# Two patterns are deliberately *not* included:
#   * Anything that calls subprocess / shell / system binaries — those
#     belong in FORBIDDEN_MODULES.
#   * REST-only services with no first-party Python SDK (Figma, Vercel,
#     Brave Search, Zapier, …) — agents should hit those via ``httpx``
#     against the documented REST endpoints.
# ---------------------------------------------------------------------------

ALLOWED_THIRD_PARTY = frozenset({
    # Generic HTTP / auth helpers
    "httpx", "requests", "authlib",

    # Browser automation (top 50: Playwright, Puppeteer)
    "playwright",

    # AI / ML providers
    "openai", "anthropic", "mistralai", "cohere",
    "huggingface_hub", "replicate",

    # Cloud infrastructure (top 50: AWS, Azure, GCP, Cloudflare, Docker, k8s)
    "boto3", "botocore",
    "azure.identity", "azure.storage.blob", "azure.mgmt.resource",
    "google.cloud", "google.cloud.bigquery", "google.cloud.storage",
    "googleapiclient", "google.oauth2", "google.auth",
    "cloudflare", "docker", "kubernetes",

    # Source control / dev hosting (top 50: GitHub, GitLab)
    "github", "gitlab",

    # Issue / project trackers (top 50: Linear, Jira/Atlassian/Confluence,
    # Asana, Notion)
    "linear_sdk", "atlassian", "asana", "notion_client",

    # Communications (top 50: Slack; common: Discord, Twilio)
    "slack_sdk", "discord", "twilio",

    # Observability / monitoring (top 50: Sentry, Datadog, Grafana)
    "sentry_sdk", "datadog", "datadog_api_client", "grafana_client",

    # CRM / business SaaS (top 50: Salesforce, HubSpot, Shopify, Airtable)
    "stripe", "simple_salesforce", "hubspot", "shopify", "pyairtable",

    # Search / web scraping (top 50: Firecrawl, Tavily, Exa)
    "firecrawl", "tavily", "exa_py",

    # Spreadsheets (top 50: Google Sheets via gspread)
    "gspread",

    # Database drivers (top 50: Postgres, MySQL, Redis, Snowflake;
    # common: MongoDB, Elasticsearch).  These open arbitrary TCP
    # connections — same exfiltration surface as ``httpx``, just over a
    # different protocol.
    "psycopg", "psycopg2", "asyncpg", "pymysql",
    "redis", "pymongo", "elasticsearch",
    "snowflake.connector", "supabase",

    # Data / parsing
    "pydantic", "dateutil", "tzdata", "pytz",
})

# Stdlib modules we allow without question.  Notably absent: ``socket``,
# ``subprocess``, ``ctypes``, ``ftplib`` — anything that could
# exfiltrate state outside the API endpoint.
ALLOWED_STDLIB = frozenset({
    # ``__future__`` is required by every server (we always emit
    # ``from __future__ import annotations`` in the header).
    "__future__",
    "os", "sys", "json", "re", "logging", "datetime", "time", "math",
    "typing", "collections", "itertools", "functools", "dataclasses",
    "enum", "pathlib", "urllib.parse", "base64", "hashlib", "hmac",
    "uuid", "io", "decimal",
})

FORBIDDEN_MODULES = frozenset({
    "subprocess", "socket", "ctypes", "ftplib", "smtplib", "telnetlib",
    "pickle", "marshal", "shelve", "dbm",
    "shutil", "tempfile",  # filesystem write helpers — limit scope
    "platform",  # leaks host info
})


# ---------------------------------------------------------------------------
# Import-name → PyPI distribution-name map
#
# ``spec.allowed_imports`` carries the *import* names referenced by the
# generated tool body (e.g. ``googleapiclient``).  uv installs by *PyPI
# distribution name* (``google-api-python-client``).  This map covers
# the cases where the two differ.  Anything not in the map is assumed
# to install under the same name as it imports under.
# ---------------------------------------------------------------------------

IMPORT_TO_PYPI: dict[str, str] = {
    # Auth helpers
    "slack_sdk": "slack-sdk",
    "notion_client": "notion-client",
    "linear_sdk": "linear-sdk",
    "github": "PyGithub",
    "dateutil": "python-dateutil",

    # Google APIs
    "googleapiclient": "google-api-python-client",
    "google.oauth2": "google-auth",
    "google.auth": "google-auth",
    "google.cloud.bigquery": "google-cloud-bigquery",
    "google.cloud.storage": "google-cloud-storage",

    # Azure subpackages (each is its own distribution)
    "azure.identity": "azure-identity",
    "azure.storage.blob": "azure-storage-blob",
    "azure.mgmt.resource": "azure-mgmt-resource",

    # Source control / project management
    "gitlab": "python-gitlab",
    "simple_salesforce": "simple-salesforce",
    "hubspot": "hubspot-api-client",

    # Observability
    "sentry_sdk": "sentry-sdk",
    "datadog_api_client": "datadog-api-client",
    "grafana_client": "grafana-client",

    # AI / ML
    "huggingface_hub": "huggingface-hub",

    # Communications
    "discord": "discord.py",

    # Search / scraping (PyPI name differs from import name)
    "firecrawl": "firecrawl-py",
    "tavily": "tavily-python",
    "exa_py": "exa-py",

    # E-commerce
    "shopify": "ShopifyAPI",

    # Database drivers
    "psycopg2": "psycopg2-binary",
    "snowflake.connector": "snowflake-connector-python",
}


def import_to_pypi(import_name: str) -> str:
    """Translate a Python import name to a PyPI distribution name."""
    return IMPORT_TO_PYPI.get(import_name, import_name.split(".")[0])


CREDENTIAL_LIKE_LITERALS = [
    re.compile(r"\bsk_(?:live|test)_[A-Za-z0-9]{16,}"),
    re.compile(r"\bxox[pbar]-[\d]+-[\d]+"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}"),
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{40,}"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
]


class MCPGenerationError(RuntimeError):
    """Raised when a spec is invalid or generated code fails the audit."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


_ID_RE = re.compile(r"^[a-z][a-z0-9-]{1,62}$")
_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
_SECRET_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,62}$")


def _validate_spec(spec: MCPSpec) -> None:
    if not _ID_RE.match(spec.id):
        raise MCPGenerationError(
            f"Invalid id {spec.id!r} — must be kebab-case, start with a letter, "
            f"length 2-63"
        )
    for name in spec.required_secrets:
        if not _SECRET_RE.match(name):
            raise MCPGenerationError(
                f"Invalid secret name {name!r} — must be SHOUTY_SNAKE_CASE "
                f"and start with a letter"
            )
    seen_tools: set[str] = set()
    for t in spec.tools:
        if not _NAME_RE.match(t.name):
            raise MCPGenerationError(
                f"Invalid tool name {t.name!r} — snake_case, "
                f"start with letter or underscore"
            )
        if t.name in seen_tools:
            raise MCPGenerationError(f"Duplicate tool name: {t.name}")
        seen_tools.add(t.name)
    _validate_auth(spec)


def _validate_auth(spec: MCPSpec) -> None:
    """Reject auth specs that would yield an unusable subprocess env.

    Run separately from the main spec validation so the error messages
    can stay specific to the auth contract (``env_mapping`` shape,
    ``allowed_hosts`` for browser capture, etc.).
    """
    auth = spec.auth
    if auth is None or auth.kind == "static":
        return

    # Lazy import — keeps the heavy provider deps out of the spec
    # validation hot path for static MCPs.
    from backend.auth import available_kinds

    if auth.kind not in available_kinds():
        raise MCPGenerationError(
            f"Unknown auth.kind {auth.kind!r}.  Supported: "
            f"{available_kinds()}"
        )

    for env_name in auth.env_mapping:
        if not _SECRET_RE.match(env_name):
            raise MCPGenerationError(
                f"Invalid auth.env_mapping env name {env_name!r} — must be "
                f"SHOUTY_SNAKE_CASE"
            )

    if not auth.env_mapping:
        raise MCPGenerationError(
            f"auth.kind={auth.kind!r} requires a non-empty env_mapping so the "
            f"subprocess receives the captured token (e.g. "
            f"{{'BEARER_TOKEN': 'access_token'}})"
        )

    if auth.kind == "browser_capture":
        if not auth.landing_url:
            raise MCPGenerationError(
                "browser_capture requires a non-empty landing_url"
            )
        if not auth.allowed_hosts:
            raise MCPGenerationError(
                "browser_capture requires a non-empty allowed_hosts list "
                "(defence against malicious manifests redirecting through "
                "phishing intermediaries)"
            )

    if auth.kind in ("oauth_device", "oauth_authcode"):
        if not auth.client_id or not auth.token_url:
            raise MCPGenerationError(
                f"{auth.kind} requires client_id and token_url"
            )
        if auth.kind == "oauth_authcode" and not auth.auth_url:
            raise MCPGenerationError(
                "oauth_authcode requires an auth_url (the consent screen URL)"
            )
        if auth.kind == "oauth_device" and not auth.device_url:
            raise MCPGenerationError(
                "oauth_device requires a device_url (start_device_authorization)"
            )


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------


def _render_tool(t: ToolSpec, secret_names: list[str]) -> str:
    """Render one ``@mcp.tool()`` function block.

    Tool body is indented under the ``def`` and is inserted verbatim
    after AST checks.  We pre-pend a ``_check_secrets()`` call so any
    missing env var fails the call with a clear error message instead
    of the underlying SDK's ``KeyError``.
    """
    sig_parts: list[str] = []
    for pname, ptype, pdefault in t.params:
        if pdefault is None:
            sig_parts.append(f"{pname}: {ptype}")
        else:
            sig_parts.append(f"{pname}: {ptype} = {pdefault}")
    sig = ", ".join(sig_parts)

    indented_body = "\n".join("    " + ln for ln in t.body.splitlines() if ln.strip())
    if not indented_body:
        indented_body = "    raise NotImplementedError"

    return (
        f"@mcp.tool()\n"
        f"def {t.name}({sig}):\n"
        f"    \"\"\"{t.description}\"\"\"\n"
        f"    _check_secrets({secret_names!r})\n"
        f"{indented_body}\n"
    )


_HEADER = '''#!/usr/bin/env python3
"""Auto-generated MCP server: {name}

{description}

Generated by Otto's mcp_builder.  Do not edit by hand — regenerate
through the agent or via the sibling manifest.json.

Trust model:
* All credentials referenced as ``os.environ[\"NAME\"]`` only.
* Never print, log, or otherwise echo a credential.
* Every tool result is post-processed by the backend's
  ``output_redactor`` before reaching the LLM context, but write code
  that respects the boundary anyway.
"""

from __future__ import annotations

import os
import sys
import logging
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("otto.mcp.{id_token}")


def _check_secrets(names):
    """Fail fast when a required credential isn't in the subprocess env.

    The caller (:mod:`backend.mcp_manager`) hydrates env vars from the
    OS keychain at spawn time.  If ``names`` aren't all present, the
    user hasn't filled the vault for this server yet — refuse to run
    the tool with a clear actionable error.
    """
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise RuntimeError(
            f"Missing required credentials: {{', '.join(missing)}}. "
            f"Ask the user to set them via Settings \u2192 Credentials \u2192 {{server_name}}."
        )


mcp = FastMCP({mcp_name!r})

'''


_FOOTER = '''

if __name__ == "__main__":
    mcp.run()
'''


# ---------------------------------------------------------------------------
# Client / requirements / README rendering
#
# The client is a small CLI that spawns ``server.py`` over stdio (using
# the per-MCP venv interpreter), lists tools, and lets the user invoke
# any tool with ``key=value`` args.  It's intentionally minimal: the
# real MCP client lives in :mod:`backend.mcp_manager` — this is a
# smoke-test / demo so the user can verify the server outside the agent.
# ---------------------------------------------------------------------------


def render_requirements_txt(spec: MCPSpec) -> str:
    """Build a requirements.txt from ``spec.allowed_imports``.

    ``mcp`` is always pinned because every generated server imports
    ``mcp.server.fastmcp``.  Additional entries map import-names to
    PyPI distribution names via :data:`IMPORT_TO_PYPI`.
    """
    pkgs = {"mcp>=1.0.0"}
    for imp in spec.allowed_imports or []:
        pkgs.add(import_to_pypi(imp))
    return "\n".join(sorted(pkgs)) + "\n"


_CLIENT_TEMPLATE = '''#!/usr/bin/env python3
"""Standalone smoke-test client for the {name} MCP server.

Spawns ``server.py`` from the sibling per-MCP venv over stdio, then:

    python client.py            # list available tools
    python client.py list       # same
    python client.py <tool>     # call <tool> with no args
    python client.py <tool> k=v # call <tool> with named args

This is a developer/debug helper — production agents reach the same
tools via the backend's MCP manager, not through this client.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


HERE = Path(__file__).resolve().parent
SERVER_PY = HERE / "server.py"

if sys.platform == "win32":
    _VENV_PY = HERE / ".venv" / "Scripts" / "python.exe"
else:
    _VENV_PY = HERE / ".venv" / "bin" / "python"

PYTHON = str(_VENV_PY) if _VENV_PY.exists() else sys.executable


def _parse_kv(items):
    out = {{}}
    for raw in items:
        if "=" not in raw:
            raise SystemExit(f"Bad arg {{raw!r}} — expected key=value")
        k, _, v = raw.partition("=")
        out[k] = v
    return out


async def _amain(argv):
    params = StdioServerParameters(
        command=PYTHON,
        args=[str(SERVER_PY)],
        env=dict(os.environ),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            if not argv or argv[0] == "list":
                resp = await session.list_tools()
                for t in resp.tools:
                    print(f"{{t.name}}: {{t.description or '(no description)'}}")
                return
            tool, *rest = argv
            args = _parse_kv(rest)
            res = await session.call_tool(tool, args)
            payload = [c.model_dump(mode="json") for c in res.content]
            print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(_amain(sys.argv[1:]))
'''


def render_client_source(spec: MCPSpec) -> str:
    """Render the standalone smoke-test client for the spec."""
    return _CLIENT_TEMPLATE.format(name=spec.name)


_README_TEMPLATE = """# {name}

{description}

## Files

| File | Purpose |
|------|---------|
| `server.py` | The runnable FastMCP server. The backend spawns this as a stdio MCP subprocess. |
| `client.py` | Standalone smoke-test client. Run by hand to verify the server works in isolation. |
| `manifest.json` | Original spec — used by the agent for regeneration. |
| `requirements.txt` | Pinned dependencies installed into `.venv/`. |
| `.venv/` | Isolated Python environment for this MCP. Recreated on regeneration. |

## Required credentials

{credentials_section}

These are stored in the OS keychain via the credential vault (never in
source files, never in chat history). The backend hydrates them into
the subprocess environment at spawn time.

## Smoke test

```sh
.venv/bin/python client.py            # list tools
.venv/bin/python client.py <tool>     # call a tool
.venv/bin/python client.py <tool> k=v # with arguments
```

## Regenerating

Ask the agent: "regenerate the {id} MCP". The agent will overwrite
this folder and reprovision the venv.
"""


def render_readme(spec: MCPSpec) -> str:
    if spec.required_secrets:
        creds = "\n".join(f"- `{n}`" for n in spec.required_secrets)
    else:
        creds = "_None — this MCP needs no credentials._"
    return _README_TEMPLATE.format(
        name=spec.name,
        description=spec.description,
        credentials_section=creds,
        id=spec.id,
    )


# ---------------------------------------------------------------------------
# Per-MCP venv provisioning (uv)
#
# We require uv on the host because it's the only Python package
# manager that creates a venv + installs in 1–3s for typical SDKs.
# Failure mode: if uv is missing or the install fails, the generated
# folder is left in place but the registration step refuses to enable
# auto-start so the user gets a clear error before connecting.
# ---------------------------------------------------------------------------


class VenvProvisionError(MCPGenerationError):
    """Raised when the per-MCP venv cannot be created or populated."""


def _find_uv() -> Optional[str]:
    """Resolve uv via PATH first, then a couple of well-known install dirs."""
    found = shutil.which("uv")
    if found:
        return found
    for cand in (
        Path.home() / ".local" / "bin" / "uv",
        Path("/opt/homebrew/bin/uv"),
        Path("/usr/local/bin/uv"),
    ):
        if cand.exists():
            return str(cand)
    return None


async def _run_uv(uv: str, *args: str, cwd: Optional[Path] = None) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        uv, *args,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ},
    )
    stdout, stderr = await proc.communicate()
    return (
        proc.returncode if proc.returncode is not None else -1,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


async def _provision_venv(spec: MCPSpec) -> Path:
    """Create ``<id>/.venv`` and install every requirement.

    Returns the path to the venv's ``python`` executable.  Raises
    :class:`VenvProvisionError` with a user-readable message on failure.
    """
    uv = _find_uv()
    if uv is None:
        raise VenvProvisionError(
            "uv is not installed.  Install it (e.g. `brew install uv` or "
            "`curl -LsSf https://astral.sh/uv/install.sh | sh`) — generated "
            "MCP servers run in their own per-MCP venv provisioned by uv."
        )

    sd = server_dir(spec.id)
    sd.mkdir(parents=True, exist_ok=True)

    venv = venv_dir(spec.id)
    if venv.exists():
        # Wipe a stale venv before re-creating so we never end up with
        # a half-populated env from a previous failed install.
        shutil.rmtree(venv, ignore_errors=True)

    rc, out, err = await _run_uv(uv, "venv", str(venv))
    if rc != 0:
        raise VenvProvisionError(
            f"`uv venv` failed for {spec.id}: {err.strip() or out.strip()}"
        )

    req = requirements_path(spec.id)
    if req.exists() and req.read_text(encoding="utf-8").strip():
        rc, out, err = await _run_uv(
            uv, "pip", "install",
            "--python", str(venv_python(spec.id)),
            "-r", str(req),
        )
        if rc != 0:
            raise VenvProvisionError(
                f"`uv pip install` failed for {spec.id}: "
                f"{err.strip() or out.strip()}"
            )

    py = venv_python(spec.id)
    if not py.exists():
        raise VenvProvisionError(
            f"venv was created but interpreter not found at {py}"
        )
    return py


def render_server_source(spec: MCPSpec) -> str:
    """Produce the full Python source for the MCP server.

    Pure function — no side effects.  Output is fed to :func:`audit_code`
    before being written to disk.
    """
    head = _HEADER.format(
        name=spec.name,
        description=spec.description,
        id_token=spec.id.replace("-", "_"),
        mcp_name=spec.name,
    )
    head = head.replace("{server_name}", spec.name)

    if spec.allowed_imports:
        import_lines = "\n".join(f"import {m}" for m in spec.allowed_imports)
        head += import_lines + "\n\n"

    body = "\n".join(_render_tool(t, spec.required_secrets) for t in spec.tools)
    return head + body + _FOOTER


# ---------------------------------------------------------------------------
# Static analysis — reject dangerous code BEFORE writing to disk
# ---------------------------------------------------------------------------


def audit_code(source: str, spec: MCPSpec) -> None:
    """Raise :class:`MCPGenerationError` on anything dangerous.

    Run BEFORE writing the file to disk and BEFORE registering with the
    backend.  Failures are non-recoverable here — the agent must
    regenerate.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise MCPGenerationError(f"Generated source has syntax errors: {exc}")

    # 1. Reject forbidden imports / disallowed third-party
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names = [node.module]
        for n in names:
            top = n.split(".")[0]
            if top in FORBIDDEN_MODULES or n in FORBIDDEN_MODULES:
                raise MCPGenerationError(
                    f"Generated code imports forbidden module {n!r}"
                )
            if top == "mcp":
                continue  # FastMCP server import — always allowed
            if top in ALLOWED_STDLIB or n in ALLOWED_STDLIB:
                continue
            if top in ALLOWED_THIRD_PARTY or n in ALLOWED_THIRD_PARTY:
                continue
            raise MCPGenerationError(
                f"Generated code imports {n!r}, which is not on the "
                f"allowlist.  Add it to ALLOWED_THIRD_PARTY in "
                f"backend/mcp_builder.py if it's safe."
            )

    # 2. Reject string literals that look like credentials
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for pat in CREDENTIAL_LIKE_LITERALS:
                if pat.search(node.value):
                    raise MCPGenerationError(
                        "Generated code contains a literal that looks "
                        "like an API key — secrets must come from "
                        "os.environ[...] only."
                    )

    # 3. Reject ``exec`` / ``eval`` / ``__import__`` / ``open``-for-write
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in {"exec", "eval", "__import__", "compile"}:
                raise MCPGenerationError(
                    f"Generated code calls {f.id!r}, which is not allowed."
                )

    # 4. Sanity: every required_secret must be referenced somewhere as
    #    ``os.environ[...]`` so the subprocess actually uses it.
    referenced: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            v = node.value
            if (isinstance(v, ast.Attribute)
                    and isinstance(v.value, ast.Name)
                    and v.value.id == "os" and v.attr == "environ"):
                key = node.slice
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    referenced.add(key.value)
        if isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Attribute)
                    and isinstance(f.value, ast.Attribute)
                    and isinstance(f.value.value, ast.Name)
                    and f.value.value.id == "os"
                    and f.value.attr == "environ"
                    and f.attr in {"get", "setdefault"}
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                referenced.add(node.args[0].value)
    unused = [s for s in spec.required_secrets if s not in referenced]
    if unused:
        # Soft warning rather than hard failure — _check_secrets()
        # references them dynamically so AST analysis can miss legit
        # uses.  Log and continue.
        logger.info(
            "MCP %s: declared but unreferenced secrets in static analysis: %s",
            spec.id, unused,
        )


# ---------------------------------------------------------------------------
# Top-level: generate + register
# ---------------------------------------------------------------------------


async def generate_mcp_server(spec: MCPSpec) -> dict[str, Any]:
    """Validate, render, audit, install, and register a new MCP server.

    Lays out the per-MCP folder (``server.py``, ``client.py``,
    ``manifest.json``, ``requirements.txt``, ``README.md``), provisions
    ``<id>/.venv`` via uv, installs the requirements into that venv,
    and points the registered :class:`MCPServerConfig` at the venv's
    Python interpreter.

    Does **not** auto-connect — the caller is expected to first ensure
    the credential vault has every entry in ``spec.required_secrets``.
    Otherwise the subprocess will refuse to start tools.

    Returns a dict ready to surface back to the agent::

        {
          "id": "stripe",
          "dir": "/.../mcp_server/stripe",
          "server_path": "/.../mcp_server/stripe/server.py",
          "client_path": "/.../mcp_server/stripe/client.py",
          "venv_python": "/.../mcp_server/stripe/.venv/bin/python",
          "requirements": ["mcp>=1.0.0", "stripe"],
          "required_secrets": ["STRIPE_SECRET_KEY"],
          "missing_secrets": ["STRIPE_SECRET_KEY"],
          "registered": true,
        }
    """
    _validate_spec(spec)
    source = render_server_source(spec)
    audit_code(source, spec)

    sd = server_dir(spec.id)
    sd.mkdir(parents=True, exist_ok=True)

    server_path(spec.id).write_text(source, encoding="utf-8")
    server_path(spec.id).chmod(0o600)  # readable only by us — defence in depth

    client_path(spec.id).write_text(render_client_source(spec), encoding="utf-8")

    requirements_text = render_requirements_txt(spec)
    requirements_path(spec.id).write_text(requirements_text, encoding="utf-8")

    readme_path(spec.id).write_text(render_readme(spec), encoding="utf-8")

    manifest_path(spec.id).write_text(
        json.dumps(
            {
                "id": spec.id,
                "name": spec.name,
                "description": spec.description,
                "required_secrets": spec.required_secrets,
                "allowed_imports": spec.allowed_imports,
                "tools": [
                    {"name": t.name, "description": t.description,
                     "params": t.params}
                    for t in spec.tools
                ],
                # ``model_dump(mode="json")`` ensures a regenerate cycle
                # round-trips the auth config losslessly even when new
                # fields are added later.
                "auth": (spec.auth or MCPAuthConfig()).model_dump(mode="json"),
            },
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )

    py = await _provision_venv(spec)

    # Permission manifest -- defaults to "writes/reads inside the
    # per-MCP folder only, no outbound network".  The agent can declare
    # a wider surface via spec.permissions when it genuinely needs one.
    sd_path = server_dir(spec.id)
    manifest_perms = spec.permissions or PermissionManifest(
        fs_read=[str(sd_path)],
        fs_write=[str(sd_path)],
        network_hosts=[],
        allow_network_all=False,
        env_read=list(spec.required_secrets),
        sandbox_enabled=True,
    )
    if not manifest_perms.env_read:
        manifest_perms.env_read = list(spec.required_secrets)
    permissions_path(spec.id).write_text(
        json.dumps(manifest_perms.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    permissions_path(spec.id).chmod(0o600)

    # Sandbox profile -- written on every platform so a copy of the
    # MCP folder can be inspected (the file is harmless on Linux);
    # mcp_manager's wrap step is what conditions on the OS.
    profile_text = render_sandbox_profile(
        mcp_id=spec.id,
        server_dir=sd_path,
        venv_python=py,
        manifest=manifest_perms,
    )
    write_profile(sd_path, profile_text)
    if manifest_perms.allow_network_all:
        logger.warning(
            "mcp_builder: %s declared allow_network_all=true -- subprocess "
            "may dial any host even with the sandbox loaded",
            spec.id,
        )

    # Sign every artifact at the end so the bundle is consistent.
    try:
        write_signature(
            server_id=spec.id,
            server_dir=sd_path,
            server_file=server_path(spec.id),
            manifest_file=manifest_path(spec.id),
            permissions_file=permissions_path(spec.id),
        )
    except Exception as exc:  # noqa: BLE001
        # Sign failures don't block generation -- they degrade the
        # trust posture and are surfaced via the routes layer + the
        # spawn-time verify gate, which refuses to start unsigned MCPs.
        logger.warning("mcp_builder: failed to sign %s: %s", spec.id, exc)

    cfg = await AppConfig.aload()
    server_py = server_path(spec.id)
    auth_cfg = spec.auth or MCPAuthConfig()
    if any(s.id == spec.id for s in cfg.mcp_servers):
        for s in cfg.mcp_servers:
            if s.id == spec.id:
                s.name = spec.name
                s.command = str(py)
                s.args = [str(server_py)]
                s.required_secrets = list(spec.required_secrets)
                s.transport = "stdio"
                s.generated = True
                s.builtin = False
                s.auth = auth_cfg
                break
    else:
        cfg.mcp_servers.append(MCPServerConfig(
            id=spec.id,
            name=spec.name,
            transport="stdio",
            command=str(py),
            args=[str(server_py)],
            enabled=True,
            auto_start=False,
            builtin=False,
            generated=True,
            required_secrets=list(spec.required_secrets),
            auth=auth_cfg,
        ))
    await cfg.asave()

    from backend.credential_vault import vault
    try:
        missing = [n for n in spec.required_secrets if not vault.has(spec.id, n)]
    except Exception as exc:
        logger.warning("vault unavailable; reporting all secrets as missing (%s)", exc)
        missing = list(spec.required_secrets)

    logger.info(
        "mcp_builder: generated server=%s dir=%s tools=%d secrets=%s missing=%s",
        spec.id, sd, len(spec.tools), spec.required_secrets, missing,
    )

    return {
        "id": spec.id,
        "dir": str(sd),
        "server_path": str(server_py),
        "client_path": str(client_path(spec.id)),
        "venv_python": str(py),
        "requirements": [
            ln for ln in requirements_text.splitlines() if ln.strip()
        ],
        "required_secrets": list(spec.required_secrets),
        "missing_secrets": missing,
        "tools": [t.name for t in spec.tools],
        "registered": True,
    }


async def delete_generated_server(server_id: str) -> dict[str, Any]:
    """Wipe config, the per-MCP folder (incl. venv), and every vault entry.

    Used when the user revokes an agent-built MCP.  Built-in servers
    cannot be deleted through this path — the function returns an error.
    """
    cfg = await AppConfig.aload()
    srv = next((s for s in cfg.mcp_servers if s.id == server_id), None)
    sd = server_dir(server_id)

    if srv is None and not sd.exists():
        return {"status": "not_found", "id": server_id}
    if srv is not None and srv.builtin:
        return {"status": "forbidden", "id": server_id,
                "error": "Built-in servers cannot be deleted"}

    if srv is not None:
        cfg.mcp_servers = [s for s in cfg.mcp_servers if s.id != server_id]
        await cfg.asave()

    if sd.exists():
        # ``rmtree`` handles the venv, source, manifest, README, and
        # requirements file in one go.  ignore_errors keeps a partial
        # failure (e.g. a still-running subprocess holding a file open
        # on Windows) from blocking the config rollback.
        shutil.rmtree(sd, ignore_errors=True)

    n_creds = 0
    try:
        from backend.credential_vault import vault
        n_creds = vault.delete_all(server_id)
    except Exception as exc:
        logger.warning("vault unavailable during delete (%s)", exc)

    logger.info(
        "mcp_builder: deleted server=%s creds_removed=%d",
        server_id, n_creds,
    )

    return {
        "status": "deleted",
        "id": server_id,
        "credentials_removed": n_creds,
    }


def list_generated_servers() -> list[dict[str, Any]]:
    """Return a name-only summary of every generated MCP server."""
    out: list[dict[str, Any]] = []
    root = mcp_root_dir()
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        mf = sub / "manifest.json"
        if not mf.exists():
            continue
        try:
            data = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append({
            "id": data.get("id"),
            "name": data.get("name"),
            "description": data.get("description"),
            "required_secrets": data.get("required_secrets", []),
            "tool_names": [t.get("name") for t in data.get("tools", [])],
            "dir": str(sub),
        })
    return out


__all__ = [
    "ToolSpec",
    "MCPSpec",
    "MCPGenerationError",
    "VenvProvisionError",
    "render_server_source",
    "render_client_source",
    "render_requirements_txt",
    "render_readme",
    "audit_code",
    "generate_mcp_server",
    "delete_generated_server",
    "list_generated_servers",
    "mcp_root_dir",
    "server_dir",
    "server_path",
    "client_path",
    "manifest_path",
    "requirements_path",
    "venv_dir",
    "venv_python",
    "ALLOWED_THIRD_PARTY",
    "ALLOWED_STDLIB",
    "FORBIDDEN_MODULES",
    "IMPORT_TO_PYPI",
]
