"""Registry + bootstrap for source-bundled built-in MCP servers.

Each :class:`BuiltinMCP` declares the metadata the backend needs to
deploy a repo-resident MCP into the user's app-data dir and register
it as a regular ``stdio`` :class:`MCPServerConfig`.

Two helpers split the lifecycle along sync/async lines so the
synchronous config-load path stays cheap:

* :func:`sync_builtin_mcp_files` is fast and synchronous — it just
  copies ``server.py`` / ``requirements.txt`` from the repo into
  ``<app_data>/mcp_server/<id>/`` when the bundled bytes don't match
  what's on disk.  Safe to call from :meth:`AppConfig._ensure_default_servers`.
* :func:`ensure_builtin_mcp_venvs` is async and provisions a per-MCP
  ``.venv`` via ``uv``.  Called once during backend lifespan startup,
  before MCP connections are attempted.

Why this layout (vs. running the script with ``sys.executable``):

The backend's main interpreter doesn't carry vendor SDKs (``httpx`` is
the only third-party dep here, but a future built-in MCP might need
``stripe``, ``slack_sdk``, etc.).  Per-MCP venvs keep the dependency
graphs isolated and let agent-authored MCPs and built-in MCPs share
the same spawn path in :mod:`backend.mcp_manager`.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from backend.config import MCPServerConfig
from backend.mcp_builder import (
    VenvProvisionError,
    _find_uv,
    _run_uv,
    server_dir,
    venv_dir,
    venv_python,
)

logger = logging.getLogger(__name__)


_BUILTIN_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class BuiltinMCP:
    """Metadata for one repo-bundled built-in MCP server.

    Attributes:
        id: Stable kebab-case identifier (matches the folder name under
            ``backend/builtin_mcps/`` after replacing ``-`` with ``_``).
        name: Human-readable name shown in the UI.
        description: One-line summary for tooltips and docs.
        source_dir_name: Folder name under ``backend/builtin_mcps/``
            holding the source files (``server.py``, ``requirements.txt``,
            optional ``README.md``).
        required_secrets: Env-var names hydrated from the credential
            vault at subprocess spawn time.  Tools should reference
            them via ``os.environ``.  Missing values block the Start
            button.
        optional_secrets: Like ``required_secrets`` but the Start button
            is *not* gated on them — used when the MCP has a sensible
            default and the credential is purely a personalisation
            (e.g. ``EDGAR_USER_AGENT``).  The slot still appears in the
            credentials dialog so the user can update the value.
        auto_start: Whether the MCP should attempt to connect on
            startup.  Always ``False`` for stdio MCPs in this codebase
            — the connection itself spawns the subprocess; the flag is
            kept for parity with the HTTP MCPs.
        enabled: Initial enabled state.
    """

    id: str
    name: str
    description: str
    source_dir_name: str
    required_secrets: tuple[str, ...] = ()
    optional_secrets: tuple[str, ...] = ()
    auto_start: bool = False
    enabled: bool = True
    excluded_tools: tuple[str, ...] = field(default_factory=tuple)
    # Limits the MCP to a single OS family.  Enforced by
    # ``backend/server.py:_startup_mcp`` which skips connection when the
    # current ``platform_label()`` doesn't match.  ``None`` means runs
    # everywhere.
    requires_os: Optional[str] = None


BUILTIN_MCPS: tuple[BuiltinMCP, ...] = (
    BuiltinMCP(
        id="edgar-sec",
        name="SEC EDGAR Filings",
        description=(
            "Read-only tools over SEC EDGAR: full-text filing search, "
            "company submissions, XBRL company facts and frames, ticker→CIK "
            "lookup, and per-filing document indexes."
        ),
        source_dir_name="edgar_sec",
        # SEC asks every API client to identify itself but doesn't
        # enforce a particular value — the MCP ships with a sensible
        # default so it works out of the box.  Users can personalise
        # via Tools → SEC EDGAR Filings → Credentials, or the orchestrator
        # can call ``request_credential('edgar-sec', 'EDGAR_USER_AGENT', …)``
        # mid-chat.
        optional_secrets=("EDGAR_USER_AGENT",),
    ),
    BuiltinMCP(
        id="macos-osascript",
        name="macOS osascript",
        description=(
            "Execute AppleScript or JXA snippets via the system "
            "osascript binary.  macOS-only — gates on TCC / Automation "
            "permissions the user has already granted to the host."
        ),
        source_dir_name="macos_osascript",
        requires_os="macos",
    ),
    BuiltinMCP(
        id="slack",
        name="Slack",
        description=(
            "Read + write tools over the Slack Web API: list/join "
            "channels, read channel and thread history, send messages, "
            "add reactions, and look up users."
        ),
        source_dir_name="slack",
        required_secrets=("SLACK_BOT_TOKEN",),
    ),
    BuiltinMCP(
        id="discord",
        name="Discord",
        description=(
            "Read + write tools over the Discord REST API: list "
            "servers/channels/members, read channel messages, send "
            "messages, and add reactions."
        ),
        source_dir_name="discord",
        required_secrets=("DISCORD_BOT_TOKEN",),
    ),
    BuiltinMCP(
        id="microsoft-teams",
        name="Microsoft Teams",
        description=(
            "Read-only tools over Microsoft Graph for Teams: list "
            "teams/channels/members/users and best-effort channel "
            "message reads.  App-only auth can't send messages — see "
            "the MCP's README for the Protected API approval needed to "
            "read channel messages."
        ),
        source_dir_name="microsoft_teams",
        required_secrets=("TEAMS_TENANT_ID", "TEAMS_CLIENT_ID", "TEAMS_CLIENT_SECRET"),
    ),
)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _source_dir(mcp: BuiltinMCP) -> Path:
    return _BUILTIN_ROOT / mcp.source_dir_name


def _source_files(mcp: BuiltinMCP) -> list[Path]:
    """Files we copy from the repo into the user's app-data folder.

    Anything matching the ignore globs (``__pycache__``, ``*.pyc``)
    is left behind so the deployed copy stays clean.
    """
    src = _source_dir(mcp)
    if not src.exists():
        return []
    return [
        p for p in src.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and not p.name.endswith(".pyc")
    ]


# ---------------------------------------------------------------------------
# File sync — synchronous, called from config load
# ---------------------------------------------------------------------------


def sync_builtin_mcp_files() -> dict[str, bool]:
    """Mirror every built-in MCP's source files into the app-data dir.

    Returns a map of ``id`` → ``True`` if any file changed (which is
    the signal that the venv may need to be rebuilt because
    ``requirements.txt`` could have changed).

    Idempotent and cheap: each file is compared by bytes before being
    rewritten, so a no-op restart costs one ``stat`` + one ``read`` per
    file.  Safe to call from synchronous config-load contexts.
    """
    results: dict[str, bool] = {}
    for mcp in BUILTIN_MCPS:
        try:
            results[mcp.id] = _sync_one(mcp)
        except Exception as exc:
            logger.warning(
                "builtin_mcps: failed to sync source files for %s: %s",
                mcp.id, exc,
            )
            results[mcp.id] = False
    return results


def _sync_one(mcp: BuiltinMCP) -> bool:
    src = _source_dir(mcp)
    if not src.exists():
        logger.warning(
            "builtin_mcps: source dir missing for %s — expected at %s",
            mcp.id, src,
        )
        return False

    dst = server_dir(mcp.id)
    dst.mkdir(parents=True, exist_ok=True)

    changed = False
    for src_file in _source_files(mcp):
        rel = src_file.relative_to(src)
        dst_file = dst / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        new_bytes = src_file.read_bytes()
        if dst_file.exists() and dst_file.read_bytes() == new_bytes:
            continue
        dst_file.write_bytes(new_bytes)
        if dst_file.name == "server.py":
            try:
                dst_file.chmod(0o600)
            except OSError:
                pass
        changed = True
    if changed:
        logger.info("builtin_mcps: refreshed source files for %s -> %s", mcp.id, dst)
    return changed


# ---------------------------------------------------------------------------
# Venv provisioning — async, called from backend lifespan startup
# ---------------------------------------------------------------------------


async def ensure_builtin_mcp_venvs(
    *, force_rebuild: Optional[set[str]] = None,
) -> dict[str, str]:
    """Make sure every built-in MCP has a working ``.venv``.

    Args:
        force_rebuild: ids whose venv must be wiped and re-installed
            (used when ``requirements.txt`` changed since last boot).

    Returns a map of ``id`` → status string (``"ready"``, ``"created"``,
    ``"rebuilt"``, ``"error: …"``).  Errors are logged at WARNING and
    do not raise — a misconfigured venv shouldn't crash the backend;
    the corresponding MCP just won't start until the user fixes their
    Python/uv environment.
    """
    forced = force_rebuild or set()
    statuses: dict[str, str] = {}

    uv = _find_uv()
    if uv is None:
        msg = (
            "uv is not installed.  Built-in MCP servers run in their own "
            "per-MCP venv provisioned by uv.  Install via "
            "`brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`."
        )
        logger.warning("builtin_mcps: %s", msg)
        for mcp in BUILTIN_MCPS:
            statuses[mcp.id] = f"error: {msg}"
        return statuses

    for mcp in BUILTIN_MCPS:
        try:
            statuses[mcp.id] = await _ensure_one_venv(
                mcp, uv=uv, force=mcp.id in forced,
            )
        except VenvProvisionError as exc:
            logger.warning("builtin_mcps: venv provisioning failed for %s: %s", mcp.id, exc)
            statuses[mcp.id] = f"error: {exc}"
        except Exception:
            logger.exception("builtin_mcps: unexpected error provisioning %s", mcp.id)
            statuses[mcp.id] = "error: unexpected"
    return statuses


async def _ensure_one_venv(mcp: BuiltinMCP, *, uv: str, force: bool) -> str:
    sd = server_dir(mcp.id)
    sd.mkdir(parents=True, exist_ok=True)
    req = sd / "requirements.txt"
    if not req.exists():
        raise VenvProvisionError(
            f"requirements.txt missing for {mcp.id} at {req} — was source sync skipped?"
        )

    venv = venv_dir(mcp.id)
    py = venv_python(mcp.id)
    marker = sd / ".requirements.sha256"
    digest = hashlib.sha256(req.read_bytes()).hexdigest()
    stored = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""
    needs_rebuild = force or not py.exists() or stored != digest

    if not needs_rebuild:
        return "ready"

    if venv.exists():
        await asyncio.to_thread(shutil.rmtree, venv, True)

    rc, out, err = await _run_uv(uv, "venv", str(venv))
    if rc != 0:
        raise VenvProvisionError(
            f"`uv venv` failed for {mcp.id}: {err.strip() or out.strip()}"
        )

    rc, out, err = await _run_uv(
        uv, "pip", "install",
        "--python", str(py),
        "-r", str(req),
    )
    if rc != 0:
        raise VenvProvisionError(
            f"`uv pip install` failed for {mcp.id}: {err.strip() or out.strip()}"
        )

    if not py.exists():
        raise VenvProvisionError(
            f"venv created but interpreter not found at {py}"
        )

    marker.write_text(digest, encoding="utf-8")

    logger.info(
        "builtin_mcps: provisioned venv for %s (force=%s) -> %s",
        mcp.id, force, py,
    )
    return "rebuilt" if (force or stored) else "created"


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


def builtin_mcp_config(mcp: BuiltinMCP) -> MCPServerConfig:
    """Build the ``MCPServerConfig`` entry for one built-in MCP.

    The command/args paths are deterministic — ``server_dir(id)`` and
    ``venv_python(id)`` resolve the same way regardless of whether the
    files exist yet, which lets us emit the config entry at first-run
    and let the lifespan startup populate the disk asynchronously.
    """
    py = venv_python(mcp.id)
    server_py = server_dir(mcp.id) / "server.py"
    return MCPServerConfig(
        id=mcp.id,
        name=mcp.name,
        transport="stdio",
        command=str(py),
        args=[str(server_py)],
        enabled=mcp.enabled,
        auto_start=mcp.auto_start,
        builtin=True,
        generated=False,
        required_secrets=list(mcp.required_secrets),
        optional_secrets=list(mcp.optional_secrets),
        excluded_tools=list(mcp.excluded_tools),
        requires_os=mcp.requires_os,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _self_check() -> int:
    """``python -m backend.builtin_mcps.registry`` — list bundled MCPs."""
    for mcp in BUILTIN_MCPS:
        files = [str(p.relative_to(_BUILTIN_ROOT)) for p in _source_files(mcp)]
        print(f"{mcp.id}: {mcp.name}")
        print(f"  description: {mcp.description}")
        print(f"  required_secrets: {list(mcp.required_secrets)}")
        print(f"  source files: {files}")
    return 0


if __name__ == "__main__":
    sys.exit(_self_check())
