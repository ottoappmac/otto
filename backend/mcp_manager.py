"""Generic MCP server connection manager.

Replaces the hardcoded per-server loaders in tool_factory with a config-driven
approach that can connect to *any* MCP server the user adds.

Includes a ProcessManager that can spawn and supervise child processes for
MCP servers that have ``auto_start=True`` (e.g. Playwright MCP).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable
from urllib.parse import urlparse

import httpx

_FATAL = (KeyboardInterrupt, SystemExit)

# Pinned Playwright MCP version and the exact ``playwright`` build it bundles.
# These MUST stay in lock-step: the browser auto-provisioner
# (``backend.server._ensure_playwright_browser``) installs Chromium using
# ``PLAYWRIGHT_VERSION`` so the revision matches what
# ``@playwright/mcp@PLAYWRIGHT_MCP_VERSION`` launches.  Bump both together
# (check ``npm view @playwright/mcp@<ver> dependencies``).
PLAYWRIGHT_MCP_VERSION = "0.0.70"
PLAYWRIGHT_VERSION = "1.60.0-alpha-1774999321000"

from langchain_core.tools import BaseTool

from tools.loop_guard import ToolLoopGuard, wrap_with_loop_guard

# ── Per-server recovery hints fed into ToolLoopGuard ─────────────────────────
#
# When a tool gets stuck in an identical-args failure loop, ToolLoopGuard
# raises a ToolLoopDetected whose message is converted to a tool-result
# message by LangChain's ``handle_tool_error=True`` and shown to the
# model on the next turn.  These per-server hints make that recovery
# message actionable for the specific tool family the agent is using.

_MACOS_NATIVE_RECOVERY_HINT = (
    "macOS desktop tools require an `app_name` argument (e.g. "
    "`app_name=\"Microsoft Word\"`).  Repeating the same call won't help. "
    "Call `get_screen_controls(app_name=...)` (or `wait_for_controls(app_name=..., "
    "timeout=20)` for Electron apps) to inspect, then act on a specific control "
    "index with `press_control` / `type_into_control`. If an app's content stays "
    "unreadable (common for Slack/Electron), report that you could not read it — "
    "do NOT loop or fabricate content."
)

# Read-only observation tools whose successful repeated calls are normal
# (the agent re-scans between actions).  Excluded from success-loop and
# no-progress detection so that legitimate re-reads don't trip the guard.
# A macOS-specific subset of the shared ``loop_guard.OBSERVATION_TOOLS``
# exemption set; kept as its own name because it ALSO drives the desktop-lock
# wrapping decision below (focus-stealing + observation tools hold the lock).
_MACOS_OBSERVATION_TOOLS: frozenset[str] = frozenset({
    "get_screen_controls",
    "wait_for_controls",
    "list_apps",
    "take_screenshot",
    "capture_app_screenshot",
    "get_control_value",
    "read_clipboard",
})

# Read-only browser tools whose repeated identical results are normal (a
# snapshot of an unchanged page, console/network dumps).  Exempt from the
# no-progress guard for the Playwright connection so legitimate re-snapshots
# between actions don't trip it.
_PLAYWRIGHT_OBSERVATION_TOOLS: frozenset[str] = frozenset({
    "browser_snapshot",
    "browser_take_screenshot",
    "browser_console_messages",
    "browser_network_requests",
})

# macos-native tools that steal focus or synthesize mouse/keyboard input —
# i.e. that actually drive the single foreground GUI.  Each must hold the
# host-global desktop lock so two concurrent agents can't fight over the
# screen (the exact failure the lock exists to prevent).  Everything else
# (the read-only observation tools above) skips the lock and stays parallel.
_MACOS_FOCUS_STEALING_TOOLS: frozenset[str] = frozenset({
    "press_control",
    "type_into_control",
    "batch_actions",
    "launch_app",
    "open_app",
    "activate_app",
    "click",
    "double_click",
    "right_click",
    "scroll",
    "type_text",
    "hotkey",
    "spotlight_search",
})

_PLAYWRIGHT_RECOVERY_HINT = (
    "For Playwright browser tools, the `ref` argument MUST be a literal "
    "snapshot id like \"e23\" (NOT a CSS selector, NOT an XPath).  Call "
    "`browser_snapshot()` first, pick a different element id, or use a "
    "different tool."
)

# Playwright MCP reports most *action* failures (stale element ref, action
# timeout, navigation error) as a SUCCESSFUL tool result (``isError=False``)
# with the failure written into the text body — so neither an exception nor the
# ``isError`` flag fires, and the loop guard records them as successes.  These
# sentinels mark such a result as a real failure so the guard's failure-loop
# detector counts it.  Matched against the ``### Error`` section header (not the
# bare word "error", which also appears in the healthy "Console: N errors"
# summary line of successful snapshots) plus a few specific failure phrases.
_PLAYWRIGHT_ERROR_SENTINELS: tuple[str, ...] = (
    "### Error",
    "not found in the current page snapshot",
    "net::ERR_",
)

# Appended to the `browser_type` tool description so the model presses Enter
# for search boxes in the same call.  The tool schema is weighted more heavily
# than system-prompt text on most ReAct stacks, so this is the highest-signal
# place to put the guidance.
_BROWSER_TYPE_SUBMIT_HINT = (
    "\n\n"
    "IMPORTANT: pass `submit=true` to press Enter right after typing whenever the "
    "field commits on Enter — search boxes, URL/address bars, chat inputs, and "
    "single-line forms. Only omit `submit=true` when the user said not to press "
    "Enter, or for a multi-field form with a separate submit button "
    "(use `browser_fill_form` + `browser_click`)."
)


def _patch_browser_type_submit_hint(tool: BaseTool) -> None:
    """Append the submit=true hint to the `browser_type` tool description.

    Idempotent: checks for a marker substring before appending so repeated
    tool loads don't stack hints.
    """
    if tool.name != "browser_type":
        return
    current = tool.description or ""
    if "pass `submit=true` to press Enter" in current:
        return
    tool.description = current.rstrip() + _BROWSER_TYPE_SUBMIT_HINT

_GENERIC_MCP_RECOVERY_HINT = (
    "Read the error message returned by the previous call and change "
    "your arguments accordingly, or pick a different tool."
)


def _loop_recovery_kwargs() -> dict:
    """Common ToolLoopGuard recovery knobs sourced from the environment.

    Spread into every ``ToolLoopGuard(...)`` so a tripped guard also requests
    a one-shot sampling temperature bump on local MLX models (a no-op for API
    providers), which helps the model escape a deterministic identical-call
    loop instead of re-deriving the same action.  Also enables no-progress
    detection ("different args, same result") on every MCP guard.

    ``max_escalations`` + ``on_escalate`` arm the cooperative hard stop: a model
    that keeps looping past the escalation limit despite corrective messages
    triggers ``request_loop_abort_current``, which flags the *current* run (the
    subagent that owns this asyncio context) so it unwinds gracefully at the
    next step boundary instead of burning the whole recursion budget.  These
    per-connection guards are shared across sessions, so the abort target is
    resolved from the run-scoped contextvar rather than bound here."""
    from utilities.environment import Environment
    from backend.streaming_subagent import request_loop_abort_current

    return {
        "recovery_temperature": Environment.get_loop_recovery_temperature(),
        "recovery_temperature_turns": (
            Environment.get_loop_recovery_temperature_turns()
        ),
        "max_no_progress": Environment.get_loop_guard_max_no_progress(),
        "max_escalations": Environment.get_loop_guard_max_escalations(),
        "on_escalate": request_loop_abort_current,
    }

HOOK_DISABLED_LABELS: dict[str, str] = {
    "claude-eval-hook": (
        "Claude Evaluator Hook is disabled. "
        "Enable it in Settings → Integrations → Claude Hook."
    ),
    "openclaw-eval-hook": (
        "OpenClaw integration is disabled. "
        "Enable it in Settings → Integrations → OpenClaw."
    ),
}

from backend.config import MCPServerConfig

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Tool-name disambiguation across servers
# -----------------------------------------------------------------------


def dedupe_tool_names(
    server_tools: Iterable[tuple[str, list[BaseTool]]],
) -> list[BaseTool]:
    """Flatten ``(server_id, tools)`` pairs into one list, renaming collisions.

    A flat, bare-name tool list is what every tool-dispatch path in this
    codebase eventually builds — LangGraph's ``ToolNode``
    (``tools_by_name[tool.name] = tool``), our react-mode dispatch shim
    (:mod:`middleware.react_middleware`), and the system prompt's "available
    tools" section (:func:`deep_agent.prompt._build_direct_tools_section`)
    all key purely on ``tool.name``. Built-in MCP servers are not required
    to have globally-unique tool names (Slack, Discord, and Microsoft Teams
    all expose generic verbs like ``list_channels`` / ``send_message`` /
    ``list_users``), so whenever two or more of the *server_tools* passed in
    here share a name, a bare flat list would let one silently shadow the
    other — last one inserted wins, with no error and no way for the LLM to
    address the shadowed tool.

    Only colliding names are renamed, to ``"<server_id>__<tool_name>"``
    (e.g. ``"slack__list_users"`` / ``"microsoft-teams__list_users"``).
    Tool names that are unique across *server_tools* are returned as-is —
    this keeps the common case (one server, or no overlapping verbs)
    identical to before, and avoids touching anything that compares tool
    names against the original bare strings *before* this function runs:
    ``excluded_tools`` filtering in ``MCPManager._load_generic``, the Tools
    page's per-server tool lists, and ``discover_available_tools``.
    """
    by_name: dict[str, list[tuple[str, BaseTool]]] = {}
    for server_id, tools in server_tools:
        for tool in tools:
            by_name.setdefault(tool.name, []).append((server_id, tool))

    result: list[BaseTool] = []
    for name, entries in by_name.items():
        if len(entries) == 1:
            result.append(entries[0][1])
            continue
        logger.warning(
            "mcp_manager: tool name %r is exposed by multiple connected "
            "servers (%s) — disambiguating with a server-id prefix so "
            "tool dispatch can't silently collide.",
            name, ", ".join(server_id for server_id, _ in entries),
        )
        for server_id, tool in entries:
            result.append(tool.model_copy(update={"name": f"{server_id}__{name}"}))
    return result


# -----------------------------------------------------------------------
# Managed child-process wrapper
# -----------------------------------------------------------------------

@dataclass
class ManagedProcess:
    """Wraps a child process so we can start / health-check / stop it.

    Uses asyncio.create_subprocess_exec when available, falling back to
    subprocess.Popen on Windows where the event loop may not support
    async subprocesses (e.g. uvicorn --reload with SelectorEventLoop).
    """

    server_id: str
    command: list[str]
    env: dict[str, str] | None = None
    health_url: str | None = None
    _proc: asyncio.subprocess.Process | None = field(default=None, repr=False)
    _popen: Any = field(default=None, repr=False)

    @property
    def pid(self) -> int | None:
        if self._proc is not None:
            return self._proc.pid
        if self._popen is not None:
            return self._popen.pid
        return None

    @property
    def running(self) -> bool:
        if self._proc is not None:
            return self._proc.returncode is None
        if self._popen is not None:
            return self._popen.poll() is None
        return False

    async def start(self, ready_timeout: float = 20.0) -> None:
        if self.running:
            return

        merged_env = {**os.environ, **(self.env or {})}
        logger.info(
            "Starting managed process [%s]: %s (PATH includes: %s)",
            self.server_id,
            " ".join(self.command),
            merged_env.get("PATH", "")[:200],
        )

        if sys.platform == "win32":
            await self._start_win32(merged_env)
        else:
            await self._start_posix(merged_env)

        if self.health_url:
            await self._wait_ready(ready_timeout)

    async def _start_posix(self, env: dict[str, str]) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            *self.command,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

    async def _start_win32(self, env: dict[str, str]) -> None:
        import subprocess as _sp

        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self.command,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=_sp.CREATE_NEW_PROCESS_GROUP,
            )
        except NotImplementedError:
            logger.info(
                "asyncio subprocess not supported on this event loop, "
                "falling back to subprocess.Popen for [%s]",
                self.server_id,
            )
            self._popen = _sp.Popen(
                self.command,
                env=env,
                stdout=_sp.DEVNULL,
                stderr=_sp.PIPE,
                creationflags=_sp.CREATE_NEW_PROCESS_GROUP,
            )

    async def _wait_ready(self, timeout: float) -> None:
        """Poll the health URL until the server responds or we time out."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        async with httpx.AsyncClient(timeout=3) as client:
            while loop.time() < deadline:
                if not self.running:
                    output = await self._read_process_output()
                    rc = self._returncode()
                    raise RuntimeError(
                        f"Managed process [{self.server_id}] exited early "
                        f"(code {rc}): {output[:1000]}"
                    )
                try:
                    resp = await client.get(self.health_url)
                    logger.info("Managed process [%s] ready (status=%d)", self.server_id, resp.status_code)
                    return
                except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, OSError):
                    pass
                try:
                    resp = await client.post(self.health_url)
                    logger.info("Managed process [%s] ready (status=%d)", self.server_id, resp.status_code)
                    return
                except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, OSError):
                    pass
                await asyncio.sleep(0.5)
        raise TimeoutError(f"Managed process [{self.server_id}] did not become ready within {timeout}s")

    async def _read_process_output(self) -> str:
        """Read combined stdout + stderr from the exited process."""
        parts: list[str] = []

        if self._proc is not None:
            for stream, label in [(self._proc.stdout, "stdout"), (self._proc.stderr, "stderr")]:
                if stream is None:
                    continue
                try:
                    raw = await asyncio.wait_for(stream.read(8192), timeout=2.0)
                    if raw:
                        parts.append(f"[{label}] {raw.decode(errors='replace')}")
                except Exception:
                    pass

        if self._popen is not None:
            if self._popen.stderr:
                try:
                    raw = await asyncio.to_thread(self._popen.stderr.read, 8192)
                    if raw:
                        parts.append(f"[stderr] {raw.decode(errors='replace')}")
                except Exception:
                    pass

        return "\n".join(parts) if parts else "(no output captured)"

    def _returncode(self) -> str:
        if self._proc is not None:
            return str(self._proc.returncode) if self._proc.returncode is not None else "?"
        if self._popen is not None:
            return str(self._popen.returncode) if self._popen.returncode is not None else "?"
        return "?"

    async def stop(self) -> None:
        if not self.running:
            return
        cur_pid = self.pid
        logger.info("Stopping managed process [%s] (pid=%s)", self.server_id, cur_pid)

        if self._proc is not None:
            await self._stop_async_proc()
        elif self._popen is not None:
            await asyncio.to_thread(self._stop_popen)

        if self.health_url:
            parsed = urlparse(self.health_url)
            if parsed.port:
                await _kill_stale_port_holder(parsed.port)

    async def _stop_async_proc(self) -> None:
        try:
            if sys.platform == "win32":
                self._proc.terminate()
            else:
                pgid = os.getpgid(self._proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                if sys.platform == "win32":
                    self._proc.kill()
                else:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                await self._proc.wait()
        except (ProcessLookupError, PermissionError, OSError):
            try:
                self._proc.kill()
                await self._proc.wait()
            except ProcessLookupError:
                pass
        self._proc = None

    def _stop_popen(self) -> None:
        try:
            self._popen.terminate()
            try:
                self._popen.wait(timeout=5)
            except Exception:
                self._popen.kill()
                self._popen.wait()
        except (ProcessLookupError, PermissionError, OSError):
            try:
                self._popen.kill()
            except ProcessLookupError:
                pass
        self._popen = None


async def _kill_stale_port_holder(port: int) -> None:
    """Best-effort: find and kill any process listening on *port*.

    This handles the case where a previous Playwright MCP child process
    survived its parent and is still holding the port.
    """

    def _port_in_use() -> bool:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(0.5)
            return sock.connect_ex(("127.0.0.1", port)) == 0
        except OSError:
            return False
        finally:
            sock.close()

    if not await asyncio.to_thread(_port_in_use):
        return

    logger.warning("Port %d is in use — attempting to free it", port)
    try:
        if sys.platform == "win32":
            cmd = ["netstat", "-ano"]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await proc.communicate()
            if stdout:
                for line in stdout.decode(errors="replace").splitlines():
                    if f":{port}" in line and "LISTENING" in line:
                        parts = line.split()
                        try:
                            pid = int(parts[-1])
                            logger.info("Killing stale process on port %d (pid=%d)", port, pid)
                            os.kill(pid, signal.SIGTERM)
                        except (ValueError, ProcessLookupError, PermissionError):
                            pass
        else:
            proc = await asyncio.create_subprocess_exec(
                "lsof", "-ti", f":{port}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await proc.communicate()
            if stdout:
                for pid_str in stdout.decode().strip().split("\n"):
                    pid = int(pid_str.strip())
                    logger.info("Killing stale process on port %d (pid=%d)", port, pid)
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except (ProcessLookupError, PermissionError):
                        pass
        await asyncio.sleep(1)
    except Exception as exc:
        logger.debug("Could not free port %d: %s", port, exc)


class MCPSignatureError(RuntimeError):
    """Raised when a generated MCP's signature does not validate."""


def _harden_generated_command(
    *,
    config: MCPServerConfig,
    command: str,
    args: list[str],
) -> tuple[str, list[str]]:
    """Verify the signature and apply the sandbox profile for *config*.

    Returns the (possibly-rewritten) ``(command, args)`` pair.  Raises
    :class:`MCPSignatureError` if signature verification fails -- the
    caller catches RuntimeError in the connect path, so this propagates
    cleanly to the connection's ``error`` field without a backtrace.
    """
    from pathlib import Path
    from backend.mcp_builder import (
        manifest_path,
        permissions_path,
        sandbox_profile_path,
        server_dir,
        server_path,
    )
    from backend.mcp_sandbox import PermissionManifest, wrap_command
    from backend.mcp_signer import verify_directory

    sd = server_dir(config.id)
    server_file = server_path(config.id)
    perms_file = permissions_path(config.id)
    perms_file_arg = perms_file if perms_file.is_file() else None

    ok, reason, envelope = verify_directory(
        server_id=config.id,
        server_dir=sd,
        server_file=server_file,
        manifest_file=manifest_path(config.id),
        permissions_file=perms_file_arg,
    )
    if not ok:
        fp = (envelope or {}).get("key_fingerprint", "?")
        raise MCPSignatureError(
            f"refusing to spawn generated MCP {config.id!r}: {reason} "
            f"(signing key fingerprint={fp}).  Regenerate the server or "
            f"contact the user before re-enabling it."
        )

    # Load the permission manifest from disk -- the on-disk copy is
    # what was signed, so we always trust it over any in-memory state.
    manifest = PermissionManifest()
    if perms_file.is_file():
        try:
            import json as _json
            manifest = PermissionManifest.from_dict(
                _json.loads(perms_file.read_text(encoding="utf-8"))
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "mcp_manager: could not parse permissions.json for %s (%s) -- "
                "running without a sandbox",
                config.id, exc,
            )
            manifest = PermissionManifest(sandbox_enabled=False)

    # Make sure a profile exists on disk for the sandbox wrap step.
    # If signing happened on a non-macOS host the file may still be
    # absent; in that case wrap_command no-ops and we just rely on the
    # signature + AST audit.
    if not sandbox_profile_path(config.id).is_file():
        logger.debug(
            "mcp_manager: sandbox profile missing for %s -- spawning unwrapped",
            config.id,
        )

    full_command = [command, *args]
    wrapped = wrap_command(
        command=full_command,
        server_dir=Path(sd),
        manifest=manifest,
    )
    return wrapped[0], list(wrapped[1:])


def _hydrate_secrets(config: MCPServerConfig) -> dict[str, str]:
    """Pull credentials from the vault and return a subprocess env dict.

    Two paths converge here, both producing the same flat env shape so
    the rest of the spawn pipeline stays unchanged:

    * **Static auth** (``config.auth.kind == "static"``, the default).
      Hydrates ``required_secrets`` (must be present — missing names
      log a WARNING because the route layer should have already
      blocked Start) and ``optional_secrets`` (best-effort — missing
      is fine, the MCP has its own default).

    * **Interactive auth** (any other ``config.auth.kind``).  Reads the
      structured token bundle persisted by an :mod:`backend.auth`
      provider, refreshes it silently if the access token is stale,
      and projects the bundle into env vars via the provider's
      ``env_for``.  Static ``required_secrets`` / ``optional_secrets``
      still apply on top — useful for MCPs that need both an OAuth
      token AND a static config knob (e.g. tenant id).

    Raises :class:`backend.auth.NeedsLoginError` when the bundle is
    missing or stale and silent refresh failed.  The route layer turns
    that into a 400 so the frontend swaps its "Set credentials"
    affordance for a "Login" button.
    """
    out = _hydrate_static_secrets(config)
    out.update(_hydrate_auth_bundle(config))
    return out


def _hydrate_static_secrets(config: MCPServerConfig) -> dict[str, str]:
    if not config.required_secrets and not config.optional_secrets:
        return {}
    try:
        from backend.credential_vault import vault
    except Exception as exc:
        logger.warning("vault unavailable (%s) — skipping secret hydration", exc)
        return {}
    out: dict[str, str] = {}
    for name in config.required_secrets:
        val = vault.get(config.id, name)
        if val is not None:
            out[name] = val
        else:
            logger.warning(
                "MCP %s: required secret %r not in vault — subprocess will fail",
                config.id, name,
            )
    for name in config.optional_secrets:
        val = vault.get(config.id, name)
        if val is not None:
            out[name] = val
        # No log line on missing — optional means missing is the
        # expected case until the user customises it.
    return out


def _hydrate_auth_bundle(config: MCPServerConfig) -> dict[str, str]:
    """Project the persisted OAuth / browser-capture bundle into env vars.

    Lazy-imports both ``backend.auth`` and ``backend.credential_vault``
    so the static-only path stays cheap.  Synchronous wrapper around
    the provider's async ``refresh`` because :func:`_hydrate_secrets`
    is called from sync subprocess-spawn contexts.
    """
    auth = config.auth
    if auth is None or auth.kind == "static":
        return {}

    from backend.auth import NeedsLoginError, get_provider
    from backend.credential_vault import vault

    provider = get_provider(auth.kind)

    try:
        bundle = vault.get_bundle(config.id)
    except Exception as exc:
        logger.warning(
            "vault.get_bundle failed for %s (%s) — treating as missing",
            config.id, exc,
        )
        bundle = None

    if not bundle:
        raise NeedsLoginError(config.id, kind=auth.kind, reason="no_bundle")

    if provider.is_expired(bundle):
        refreshed = _run_provider_coro(provider.refresh(auth, config.id, bundle))
        if refreshed is None:
            raise NeedsLoginError(
                config.id, kind=auth.kind, reason="refresh_unavailable",
            )
        try:
            vault.set_bundle(config.id, refreshed)
        except Exception as exc:
            logger.warning(
                "vault.set_bundle failed for %s after refresh (%s); "
                "using in-memory bundle for this spawn",
                config.id, exc,
            )
        bundle = refreshed

    return provider.env_for(auth, bundle)


def _run_provider_coro(coro: Any) -> Any:
    """Run an async provider call from sync context, even on a hot loop.

    The MCP spawn pipeline is sync end-to-end (uvloop ``Process`` /
    ``Popen`` start), but :class:`AuthProvider.refresh` is async so it
    can share an ``httpx.AsyncClient``.  ``asyncio.run`` would fail
    inside a running loop; falling back to a thread + ``asyncio.run``
    keeps this safe regardless of which context the spawn is in.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


def _wrap_with_output_redactor(tool: BaseTool) -> None:
    """Run every tool result through :func:`backend.output_redactor.redact_value`.

    This is the last line of defence — even if a generated MCP
    accidentally returns a credential in a tool response, the regex
    pass strips it before LangChain hands it back to the LLM.  Always
    safe to apply: redaction is a no-op on payloads that don't contain
    credential-shaped strings.
    """
    from functools import wraps
    from langchain_core.tools import StructuredTool

    if not isinstance(tool, StructuredTool) or tool.coroutine is None:
        return
    if getattr(tool, "_otto_redactor_wrapped", False):
        return

    original = tool.coroutine

    @wraps(original)
    async def _redacted(*args: Any, **kwargs: Any) -> Any:
        from backend.output_redactor import redact_value

        result = await original(*args, **kwargs)
        return redact_value(result)

    tool.coroutine = _redacted
    tool._otto_redactor_wrapped = True  # type: ignore[attr-defined]


def _build_mcp_command(
    config: MCPServerConfig,
    *,
    server_name: str,
    script_path: str,
    label: str,
    default_url: str,
    default_port: int,
) -> tuple[list[str], dict[str, str]]:
    """Build command + env for a Python-based MCP server subprocess.

    When running as a PyInstaller frozen binary, spawns the binary itself
    with ``--mcp-server <name>`` so the child reuses the bundled Python
    runtime and all packages.  No external Python interpreter needed.

    In development, spawns ``sys.executable`` (the venv Python) with the
    script path directly.
    """
    from pathlib import Path

    parsed = urlparse(config.url or default_url)
    port = str(parsed.port or default_port)

    if getattr(sys, "frozen", False):
        cmd = [
            sys.executable, "--mcp-server", server_name,
            "--port", port, "--host", "127.0.0.1",
        ]
    else:
        server_script = Path(__file__).resolve().parent.parent / script_path
        if not server_script.exists():
            raise FileNotFoundError(
                f"{label} script not found at {server_script}"
            )
        cmd = [sys.executable, str(server_script), "--port", port, "--host", "127.0.0.1"]

    return cmd, {}


def _build_eval_service_command(config: MCPServerConfig) -> tuple[list[str], dict[str, str]]:
    """Build command and env for the Agent Evaluator Service MCP server.

    Returns ``(command, extra_env)`` where *extra_env* is merged into the
    subprocess environment.  The backend's ANTHROPIC_API_KEY (or
    OPENAI_API_KEY) is forwarded so the evaluator LLM can authenticate.
    """
    cmd, extra_env = _build_mcp_command(
        config,
        server_name="agent-eval-service",
        script_path="src/tools/evaluation/mcp_server.py",
        label="Agent Evaluator Service",
        default_url="http://localhost:8941/mcp",
        default_port=8941,
    )

    extra_env.update({
        "DEEPEVAL_TELEMETRY_OPT_OUT": "1",
        "DEEPEVAL_DISABLE_DOTENV": "1",
        "CONFIDENT_API_KEY": "",
    })
    from backend.config import AppConfig
    try:
        cfg = AppConfig.load()
        for key, val in cfg.to_env_dict().items():
            if val:
                extra_env[key] = val
        for aws_key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
            val = os.environ.get(aws_key, "")
            if val:
                extra_env[aws_key] = val
    except Exception:
        logger.debug("Could not load AppConfig for DeepEval env — using os.environ fallback")
        for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL_PROVIDER", "ANTHROPIC_MODEL_NAME",
                     "ANTHROPIC_BEDROCK_REGION", "LLM_PROVIDER",
                     "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
            val = os.environ.get(key, "")
            if val:
                extra_env[key] = val

    return cmd, extra_env


def _resolve_npx_path() -> str | None:
    """Resolve an absolute path to ``npx``, or ``None`` if not found.

    Bare ``"npx"`` isn't guaranteed to resolve via ``PATH`` inside a
    packaged app, so this checks the system ``PATH`` first, then falls
    back to the portable Node install managed by
    :mod:`backend.node_provisioner` (its ``bin/`` may not be on ``PATH``
    yet if Node was installed after this process started).
    """
    npx = shutil.which("npx")
    if npx:
        return npx
    from backend.node_provisioner import node_bin_dir

    candidate = node_bin_dir() / "npx"
    if candidate.exists():
        return str(candidate)
    return None


def _build_playwright_command(
    config: MCPServerConfig,
    *,
    instance_id: str | None = None,
) -> list[str]:
    """Build the npx command to start Playwright MCP from the server config.

    Args:
        config: Server config with URL/port.
        instance_id: When provided, creates instance-specific profile and
            output directories so concurrent browser instances don't share
            the same Chromium user-data-dir.
    """
    from backend.config import get_app_data_dir

    npx = _resolve_npx_path()
    if not npx:
        raise FileNotFoundError(
            "npx not found — install Node.js to auto-start Playwright MCP "
            "(POST /api/node/install)"
        )

    parsed = urlparse(config.url or "http://localhost:8931/mcp")
    port = str(parsed.port or 8931)

    # Pinning to a specific version avoids npx fetching the registry on every
    # cold start, which can add 30-60 s on a slow connection.
    cmd = [npx, "--yes", f"@playwright/mcp@{PLAYWRIGHT_MCP_VERSION}", "--port", port]

    suffix = f"-{instance_id}" if instance_id else ""

    user_data_dir = get_app_data_dir() / f"playwright-profile{suffix}"
    user_data_dir.mkdir(parents=True, exist_ok=True)
    cmd.extend(["--user-data-dir", str(user_data_dir)])

    output_dir = get_app_data_dir() / f"playwright-output{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd.extend(["--output-dir", str(output_dir)])

    pw_env = {k: v for k, v in os.environ.items() if k.startswith("PLAYWRIGHT_MCP_")}

    if pw_env.get("PLAYWRIGHT_MCP_HEADLESS", "").lower() in ("true", "1"):
        cmd.append("--headless")

    # Default to Playwright's bundled Chromium rather than the "chrome"
    # channel (real Google Chrome).  Chrome isn't present on a clean
    # install / VM and installing it can require admin rights, whereas
    # the matching Chromium is auto-provisioned sudo-free at startup by
    # ``backend.server._ensure_playwright_browser``.  Users who want
    # their system Chrome can still set ``PLAYWRIGHT_MCP_BROWSER=chrome``.
    browser = pw_env.get("PLAYWRIGHT_MCP_BROWSER") or "chromium"
    cmd.extend(["--browser", browser])

    viewport = pw_env.get("PLAYWRIGHT_MCP_VIEWPORT_SIZE")
    if viewport:
        cmd.extend(["--viewport-size", viewport])

    caps = pw_env.get("PLAYWRIGHT_MCP_CAPS")
    if caps:
        cmd.extend(["--caps", caps])

    device = pw_env.get("PLAYWRIGHT_MCP_DEVICE")
    if device:
        cmd.extend(["--device", device])

    return cmd


def _build_claude_eval_hook_command(config: MCPServerConfig) -> tuple[list[str], dict[str, str]]:
    """Build command and env for the Claude Evaluator Hook MCP server."""
    return _build_mcp_command(
        config,
        server_name="claude-eval-hook",
        script_path="src/tools/transcripts/claude_mcp_server.py",
        label="Claude Evaluator Hook",
        default_url="http://localhost:8942/mcp",
        default_port=8942,
    )


def _build_openclaw_eval_hook_command(config: MCPServerConfig) -> tuple[list[str], dict[str, str]]:
    """Build command and env for the OpenClaw Evaluator Hook MCP server."""
    return _build_mcp_command(
        config,
        server_name="openclaw-eval-hook",
        script_path="src/tools/transcripts/openclaw_mcp_server.py",
        label="OpenClaw Evaluator Hook",
        default_url="http://localhost:8943/mcp",
        default_port=8943,
    )


# -----------------------------------------------------------------------
# Circuit breaker – module-level so it survives MCPManager re-creation.
# All functions below must only be called from the async event-loop
# thread; there is no locking because the dict is never accessed from
# worker threads.
# -----------------------------------------------------------------------

_MAX_CONSECUTIVE_FAILURES = 3
_CIRCUIT_BREAKER_COOLDOWN_SECS = 300  # 5 min


@dataclass
class _FailureRecord:
    consecutive_failures: int = 0
    last_failure_ts: float = 0.0
    last_error: str = ""


_failure_registry: dict[str, _FailureRecord] = {}


def reset_circuit_breaker(server_id: str) -> None:
    """Clear the failure record for *server_id* so the next connect attempt proceeds."""
    _failure_registry.pop(server_id, None)


def _circuit_open(server_id: str) -> str | None:
    """Return an error message if the circuit breaker is open, else ``None``."""
    rec = _failure_registry.get(server_id)
    if rec is None or rec.consecutive_failures < _MAX_CONSECUTIVE_FAILURES:
        return None
    elapsed = time.monotonic() - rec.last_failure_ts
    if elapsed >= _CIRCUIT_BREAKER_COOLDOWN_SECS:
        _failure_registry.pop(server_id, None)
        return None
    remaining = int(_CIRCUIT_BREAKER_COOLDOWN_SECS - elapsed)
    return (
        f"Skipped — {rec.consecutive_failures} consecutive connection failures "
        f"(last: {rec.last_error}). Retrying in {remaining}s or reconnect manually."
    )


def _record_failure(server_id: str, error: str) -> None:
    rec = _failure_registry.setdefault(server_id, _FailureRecord())
    rec.consecutive_failures += 1
    rec.last_failure_ts = time.monotonic()
    rec.last_error = error


def _record_success(server_id: str) -> None:
    _failure_registry.pop(server_id, None)


# -----------------------------------------------------------------------
# Connection tracking
# -----------------------------------------------------------------------

@dataclass
class MCPConnection:
    config: MCPServerConfig
    tools: list[BaseTool] = field(default_factory=list)
    connected: bool = False
    error: str | None = None
    server_type: str = "generic"
    context_cache_active: bool = False
    _helper: Any = field(default=None, repr=False)
    _context_cache: Any = field(default=None, repr=False)
    # Per-connection ToolLoopGuard.  All tools loaded for this MCP
    # server share one guard so the failure-history window is
    # meaningful (a model that loops on tool A then tool B then tool A
    # is still looping, just with extra steps).  Scope is the
    # connection lifetime, which matches the MCPManager's
    # session-or-process-scoped reuse model.
    _loop_guard: Any = field(default=None, repr=False)
    # macos-native only: pre-wrapped vision-combo variants
    # ({"read_screen": <tool>, "extra": [<capture tool>, ...]}) so a
    # vision-capable subagent can swap them in without re-wrapping.
    macos_native_vision: Any = field(default=None, repr=False)

    async def close(self) -> None:
        if self._helper is not None:
            try:
                await self._helper.close()
            except RuntimeError as exc:
                # Known anyio quirk: ``stdio_client`` (used for stdio MCPs
                # via ``MultiServerMCPClient``) opens an anyio cancel
                # scope inside the task that called ``connect``.  When
                # that connect happened in one FastAPI request task and
                # ``close`` is invoked from a different task (e.g. a
                # later /stop request), anyio raises::
                #
                #   RuntimeError: Attempted to exit cancel scope in a
                #   different task than it was entered in
                #
                # The child process gets killed regardless — the only
                # casualty is the clean shutdown of the anyio scope —
                # so we log and move on rather than 500-ing the route.
                # See https://github.com/agronholm/anyio/issues/374
                if "cancel scope" in str(exc).lower():
                    logger.warning(
                        "MCP %s: ignoring cross-task cancel-scope error on close; "
                        "child process is still terminated. (%s)",
                        getattr(self.config, "id", "?"), exc,
                    )
                else:
                    raise
            except Exception as exc:
                logger.warning(
                    "MCP %s: error during close (%s); continuing",
                    getattr(self.config, "id", "?"), exc,
                )
            finally:
                self._helper = None
        self.connected = False


class MCPManager:
    """Manages connections to multiple MCP servers based on config."""

    def __init__(self) -> None:
        self._connections: dict[str, MCPConnection] = {}
        self._processes: dict[str, ManagedProcess] = {}
        self._pw_pool: Any = None  # Optional[PlaywrightPool]

    @property
    def pw_pool(self) -> Any:
        """The :class:`PlaywrightPool` for concurrent browser isolation, or ``None``."""
        return self._pw_pool

    @pw_pool.setter
    def pw_pool(self, pool: Any) -> None:
        self._pw_pool = pool

    @property
    def connections(self) -> dict[str, MCPConnection]:
        return dict(self._connections)

    @property
    def processes(self) -> dict[str, ManagedProcess]:
        return dict(self._processes)

    def get_all_tools(self) -> list[BaseTool]:
        """Flatten every connected server's tools into one list for the agent graph.

        See :func:`dedupe_tool_names` for why this isn't a plain
        ``extend`` loop — it never was a plain loop in the original code
        either, but a flat list with possible bare-name collisions across
        servers (e.g. Slack's and Microsoft Teams' ``list_users``) used to
        be exactly what got built here.
        """
        return dedupe_tool_names(
            (conn.config.id, conn.tools)
            for conn in self._connections.values()
            if conn.connected
        )

    # ------------------------------------------------------------------
    # Process lifecycle
    # ------------------------------------------------------------------

    async def ensure_process(self, config: MCPServerConfig) -> None:
        """Start the managed process for a server if auto_start is set."""
        if not config.auto_start:
            return
        if not config.enabled:
            return
        existing = self._processes.get(config.id)
        if existing and existing.running:
            return

        extra_env: dict[str, str] | None = None

        if config.id == "playwright-mcp":
            cmd = _build_playwright_command(config)
            health = config.url or "http://localhost:8931/mcp"
            parsed = urlparse(health)
            port = parsed.port or 8931
            await _kill_stale_port_holder(port)
        elif config.id == "agent-eval-service":
            cmd, extra_env = _build_eval_service_command(config)
            health = config.url or "http://localhost:8941/mcp"
            parsed = urlparse(health)
            port = parsed.port or 8941
            await _kill_stale_port_holder(port)
        elif config.id == "claude-eval-hook":
            cmd, extra_env = _build_claude_eval_hook_command(config)
            health = config.url or "http://localhost:8942/mcp"
            parsed = urlparse(health)
            port = parsed.port or 8942
            await _kill_stale_port_holder(port)
        elif config.id == "openclaw-eval-hook":
            cmd, extra_env = _build_openclaw_eval_hook_command(config)
            health = config.url or "http://localhost:8943/mcp"
            parsed = urlparse(health)
            port = parsed.port or 8943
            await _kill_stale_port_holder(port)
        else:
            return

        if existing and not existing.running:
            await existing.stop()
            self._processes.pop(config.id, None)

        proc = ManagedProcess(
            server_id=config.id,
            command=cmd,
            env=extra_env,
            health_url=health,
        )

        await proc.start()
        self._processes[config.id] = proc

    async def stop_process(self, server_id: str) -> None:
        proc = self._processes.pop(server_id, None)
        if proc:
            await proc.stop()

    async def stop_all_processes(self) -> None:
        for proc in self._processes.values():
            await proc.stop()
        self._processes.clear()

    def is_process_running(self, server_id: str) -> bool:
        """Whether the server has a live child process or stdio session.

        For HTTP MCPs (Playwright, eval services) we own a long-running
        subprocess and track it in ``self._processes`` — the answer is
        whatever ``ManagedProcess.running`` says.

        For stdio MCPs (notably the ``mcp_builder``-generated ones) the
        subprocess is owned by ``MultiServerMCPClient`` inside
        ``langchain-mcp-adapters``.  There is no separate ``ManagedProcess``,
        so a successful connection IS the running process.  Reporting
        the connection state here lets the same Start/Stop UI work for
        both transports without per-id special-casing on the frontend.
        """
        proc = self._processes.get(server_id)
        if proc is not None:
            return proc.running
        conn = self._connections.get(server_id)
        if conn is not None and conn.connected:
            cfg = conn.config
            if cfg is not None and cfg.transport == "stdio":
                return True
        return False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(
        self, config: MCPServerConfig, *, skip_process_start: bool = False,
    ) -> MCPConnection:
        if config.id in self._connections:
            try:
                await self._connections[config.id].close()
            except Exception:
                logger.debug("Ignoring error closing stale connection for %s", config.id)
            del self._connections[config.id]

        conn = MCPConnection(config=config)

        if not config.enabled:
            conn.error = HOOK_DISABLED_LABELS.get(config.id, "Server is disabled")
            self._connections[config.id] = conn
            return conn

        cb_msg = _circuit_open(config.id)
        if cb_msg is not None:
            conn.error = cb_msg
            logger.debug("MCP %s: circuit breaker open — skipping", config.name)
            self._connections[config.id] = conn
            return conn

        if not skip_process_start:
            try:
                if config.auto_start:
                    await self.ensure_process(config)
            except _FATAL:
                raise
            except BaseException as exc:
                conn.error = f"Failed to start process: {exc}"
                _record_failure(config.id, conn.error)
                self._connections[config.id] = conn
                return conn

        try:
            if config.builtin or config.id in ("playwright-mcp",):
                tools = await self._load_builtin(config, conn)
            else:
                tools = await self._load_generic(config, conn)

            conn.tools = tools
            conn.connected = True
            _record_success(config.id)
            logger.info("MCP %s: %d tools loaded — %s", config.name, len(tools), [t.name for t in tools])
        except _FATAL:
            raise
        except BaseException as exc:
            # ``NeedsLoginError`` from the auth-bundle hydration path
            # gets a friendlier surface so the UI can render a "Login"
            # button.  We don't bump the circuit breaker for it — a
            # login is something the user has to do, not a transient
            # network failure to back off from.
            from backend.auth import NeedsLoginError as _NLE

            if isinstance(exc, _NLE):
                conn.error = (
                    f"needs_login ({exc.kind}) — open Settings → "
                    f"Credentials → Login for this server"
                )
                logger.info(
                    "MCP %s: needs interactive login (kind=%s, reason=%s)",
                    config.name, exc.kind, exc.reason,
                )
            else:
                conn.error = str(exc)
                _record_failure(config.id, conn.error)
                logger.warning("MCP %s: connection failed — %s", config.name, exc)

        self._connections[config.id] = conn
        return conn

    async def connect_all(
        self,
        configs: list[MCPServerConfig],
        *,
        skip_process_start: bool = False,
        timeout: float = 10.0,
    ) -> None:
        """Connect to all enabled servers in parallel.

        Each server gets its own *timeout* (default 10s).  Under uvloop,
        anyio cancel scopes can stall the event loop while waiting for a TCP
        connection to an unreachable host to time out at the OS level.
        ``asyncio.wait_for`` cancels at the asyncio layer before that happens,
        keeping the event loop responsive even when a server is offline.
        """
        async def _connect_with_timeout(cfg: MCPServerConfig) -> None:
            try:
                await asyncio.wait_for(
                    self.connect(cfg, skip_process_start=skip_process_start),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                conn = MCPConnection(config=cfg)
                conn.error = f"Connection timed out after {timeout:.0f}s"
                _record_failure(cfg.id, conn.error)
                self._connections[cfg.id] = conn
                logger.warning("MCP %s: connection timed out after %.0fs", cfg.name, timeout)

        await asyncio.gather(*(
            _connect_with_timeout(cfg)
            for cfg in configs
            if cfg.enabled
        ))

    async def disconnect(self, server_id: str) -> None:
        if server_id in self._connections:
            await self._connections[server_id].close()
            del self._connections[server_id]

    async def disconnect_all(self) -> None:
        if self._pw_pool is not None:
            try:
                await self._pw_pool.shutdown()
            except Exception:
                logger.debug("Error shutting down PlaywrightPool", exc_info=True)
            self._pw_pool = None
        for conn in self._connections.values():
            try:
                await conn.close()
            except Exception:
                pass
        self._connections.clear()
        await self.stop_all_processes()

    async def close(self) -> None:
        """Alias for disconnect_all — matches the interface Session.close() expects."""
        await self.disconnect_all()

    async def test_connection(self, config: MCPServerConfig) -> tuple[bool, str]:
        try:
            helper = await self._create_helper(config)
            await helper.connect_all()
            tools = helper.get_tools()
            await helper.close()
            return True, f"Connected — {len(tools)} tools discovered"
        except _FATAL:
            raise
        except BaseException as exc:
            return False, str(exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _create_helper(self, config: MCPServerConfig) -> Any:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from tools.anthropic.mcps import MCPHelper

        effective_url = config.url
        connection: dict[str, Any] = {}

        if config.transport == "streamable_http" and effective_url:
            connection = {"transport": "streamable_http", "url": effective_url}
        elif config.transport == "stdio" and config.command:
            # Hydrate any required_secrets from the credential vault into
            # the subprocess env at spawn time.  Values come from macOS
            # Keychain (or the OS-equivalent) and are never persisted to
            # ``config.env`` so they cannot leak via config export, log
            # files, or the LLM context.  See ``backend.credential_vault``
            # for the trust-model rationale.
            #
            # Layered on top of the MCP SDK's own safe-subset baseline
            # (PATH/HOME/etc.) rather than starting from an empty dict —
            # any server declaring secrets or static env otherwise loses
            # those defaults entirely, which is invisible for our
            # venv-python built-ins (launched via an absolute interpreter
            # path) but breaks anything that shells out via PATH, like an
            # ``npx``-run third-party package.
            from mcp.client.stdio import get_default_environment

            env_for_child = get_default_environment()
            env_for_child.update(config.env or {})
            secret_env = _hydrate_secrets(config)
            env_for_child.update(secret_env)

            spawn_command = config.command
            spawn_args = list(config.args or [])
            if spawn_command == "npx":
                resolved = _resolve_npx_path()
                if not resolved:
                    raise FileNotFoundError(
                        f"npx not found — install Node.js to run {config.name} "
                        "(POST /api/node/install)"
                    )
                spawn_command = resolved

            # Generated MCPs (those authored at runtime by the agent)
            # carry a signed manifest + a sandbox profile.  Verify the
            # signature here -- refusing to spawn on mismatch -- and
            # wrap the command in ``sandbox-exec`` on macOS.  Static
            # built-in MCPs skip both: they're shipped with the
            # backend, so their authenticity is the backend binary's
            # signature, not a per-server HMAC.
            if config.generated:
                spawn_command, spawn_args = _harden_generated_command(
                    config=config,
                    command=spawn_command,
                    args=spawn_args,
                )

            connection = {
                "transport": "stdio",
                "command": spawn_command,
                "args": spawn_args,
                "env": env_for_child or None,
            }
        elif config.transport == "sse" and config.url:
            connection = {"transport": "sse", "url": config.url}
        else:
            raise ValueError(f"Invalid MCP config for {config.name}: transport={config.transport}")

        client = MultiServerMCPClient(connections={config.id: connection})
        return MCPHelper(client)

    async def _load_generic(self, config: MCPServerConfig, conn: MCPConnection) -> list[BaseTool]:
        from deep_agent.tool_factory import (
            _strip_null_image_content,
        )

        helper = await self._create_helper(config)
        await helper.connect_all()

        tools = helper.get_tools()

        excluded = set(config.excluded_tools)
        tools = [t for t in tools if t.name not in excluded]

        if conn._loop_guard is None:
            conn._loop_guard = ToolLoopGuard(
                recovery_hint=_GENERIC_MCP_RECOVERY_HINT,
                **_loop_recovery_kwargs(),
            )
        for t in tools:
            t.handle_tool_error = True
            _strip_none_args(t)
            _strip_null_image_content(t)
            wrap_with_loop_guard(t, conn._loop_guard)
            # Always wrap generic MCP outputs with the credential
            # redactor — regardless of whether the server is built-in
            # or agent-generated, defense-in-depth wins.
            _wrap_with_output_redactor(t)

        conn._helper = helper
        return tools

    async def _load_builtin(self, config: MCPServerConfig, conn: MCPConnection) -> list[BaseTool]:
        """Load a built-in MCP server using the original specialized loaders."""
        if config.id == "playwright-mcp":
            return await self._load_builtin_playwright(config, conn)
        if config.id == "macos-native":
            return await self._load_builtin_macos_native(config, conn)
        if config.id in ("agent-eval-service", "claude-eval-hook", "openclaw-eval-hook"):
            return await self._load_builtin_simple(config, conn, server_type=config.id)
        return await self._load_generic(config, conn)

    async def _load_builtin_macos_native(
        self, config: MCPServerConfig, conn: MCPConnection,
    ) -> list[BaseTool]:
        """In-process macOS Accessibility + pyautogui tools (no MCP subprocess)."""
        if sys.platform != "darwin":
            raise RuntimeError("macOS native desktop automation requires macOS")

        from utilities.environment import Environment
        from tools.navigation.computer import MacOSNavigator, MacOSToolkit

        toolkit = MacOSToolkit(
            ax_ipc_timeout=Environment.get_ax_ipc_timeout(),
            scan_max_depth=Environment.get_scan_depth(),
            scan_max_elements=Environment.get_scan_max_elements(),
            scan_max_workers=Environment.get_scan_max_workers(),
        )
        navigator = MacOSNavigator(toolkit=toolkit)
        tools = navigator.get_tools(vision=False)
        excluded = set(config.excluded_tools)
        tools = [t for t in tools if t.name not in excluded]

        if conn._loop_guard is None:
            conn._loop_guard = ToolLoopGuard(
                recovery_hint=_MACOS_NATIVE_RECOVERY_HINT,
                max_identical_success=3,
                success_exempt_tools=_MACOS_OBSERVATION_TOOLS,
                **_loop_recovery_kwargs(),
            )
        # One sticky-lease owner per session/connection.  Every desktop tool
        # of THIS agent — focus-stealing AND read-only observation — shares
        # the owner, so the agent holds the screen across its whole
        # activate→scan→act burst and a second agent can't steal focus
        # between the calls.  The lease frees on idle (see _desktop_lock).
        lease_owner = f"macos-native-{uuid.uuid4().hex[:12]}"

        def _prep(t: BaseTool) -> None:
            t.handle_tool_error = True
            _strip_none_args(t)
            # Applied inside the loop guard (guard wraps last, runs first) so
            # a tripped guard never makes the agent wait for the screen first.
            if (
                t.name in _MACOS_FOCUS_STEALING_TOOLS
                or t.name in _MACOS_OBSERVATION_TOOLS
            ):
                _wrap_with_desktop_lock(t, lease_owner)
            wrap_with_loop_guard(t, conn._loop_guard)

        for t in tools:
            _prep(t)

        # Pre-wrap the vision-combo variants (text+image read_screen and the
        # capture_app_screenshot tool). They are distinct tool objects from the
        # text toolset above, so wrapping them here does not double-wrap. A
        # vision-capable subagent swaps these in (see session_manager); text
        # models keep the text-only read_screen and never see this stash.
        vision_read = getattr(toolkit, "read_screen_vision", None)
        if vision_read is not None and "read_screen" not in excluded:
            vision_extra = [t for t in toolkit.vision_tools if t.name not in excluded]
            _prep(vision_read)
            for t in vision_extra:
                _prep(t)
            conn.macos_native_vision = {
                "read_screen": vision_read,
                "extra": vision_extra,
                "toolkit": toolkit,
            }

        conn.server_type = "macos-native"
        logger.info("macos-native: loaded %d tools", len(tools))
        return tools

    async def _load_builtin_playwright(self, config: MCPServerConfig, conn: MCPConnection) -> list[BaseTool]:
        from tools.navigation.web.playwright_mcp import create_playwright_mcp_client
        from tools.anthropic.mcps import MCPHelper

        mcps = create_playwright_mcp_client()
        helper = MCPHelper(mcps)
        await helper.connect_all()

        tools = helper.get_tools()
        excluded = set(config.excluded_tools)
        tools = [t for t in tools if t.name not in excluded]

        if conn._loop_guard is None:
            from utilities.environment import Environment
            from tools.loop_guard import (
                DEFAULT_HIGH_COST_TOOLS,
                DEFAULT_MAX_HIGH_COST_REPEATS,
            )

            conn._loop_guard = ToolLoopGuard(
                recovery_hint=_PLAYWRIGHT_RECOVERY_HINT,
                # Playwright reports most action failures as *successful* results
                # (the failure is only in the text body), so the success-loop
                # detector is what catches a model clicking the same element
                # forever.  Observation tools (snapshots, console dumps) are
                # exempt so legitimate re-snapshots between actions don't trip.
                max_identical_success=Environment.get_loop_guard_max_success(),
                success_exempt_tools=_PLAYWRIGHT_OBSERVATION_TOOLS,
                # The exempt snapshot tools above are otherwise unbounded; a
                # cumulative per-run ceiling curbs dozens of redundant same-URL
                # navigations / re-snapshots without tripping on a few.
                high_cost_tools=DEFAULT_HIGH_COST_TOOLS,
                max_high_cost_repeats=DEFAULT_MAX_HIGH_COST_REPEATS,
                **_loop_recovery_kwargs(),
            )
        for t in tools:
            t.handle_tool_error = True
            _strip_none_args(t)
            _patch_browser_type_submit_hint(t)
            # Detect text-level Playwright failures BEFORE the loop guard wraps
            # the coroutine, so the guard's try/except sees the raised
            # ToolException and counts it as a failure.
            _raise_on_pw_error(t)
            wrap_with_loop_guard(t, conn._loop_guard)

        conn._helper = helper
        return tools

    async def _load_builtin_simple(
        self, config: MCPServerConfig, conn: MCPConnection, server_type: str,
    ) -> list[BaseTool]:
        """Load a built-in MCP server that needs no special post-processing."""
        helper = await self._create_helper(config)
        await helper.connect_all()

        tools = helper.get_tools()
        excluded = set(config.excluded_tools)
        tools = [t for t in tools if t.name not in excluded]

        if conn._loop_guard is None:
            conn._loop_guard = ToolLoopGuard(
                recovery_hint=_GENERIC_MCP_RECOVERY_HINT,
                **_loop_recovery_kwargs(),
            )
        for t in tools:
            t.handle_tool_error = True
            _strip_none_args(t)
            wrap_with_loop_guard(t, conn._loop_guard)

        conn._helper = helper
        conn.server_type = server_type
        return tools


def _strip_none_args(tool: BaseTool) -> None:
    """Wrap an MCP tool's coroutine to drop None-valued arguments."""
    from functools import wraps
    from langchain_core.tools import StructuredTool

    if not isinstance(tool, StructuredTool) or tool.coroutine is None:
        return

    original = tool.coroutine

    @wraps(original)
    async def _cleaned(*args: Any, **kwargs: Any) -> Any:
        cleaned = {k: v for k, v in kwargs.items() if v is not None}
        return await original(*args, **cleaned)

    tool.coroutine = _cleaned


def _tool_result_text(result: Any) -> str:
    """Best-effort plain-text view of a tool result for sentinel matching.

    Handles the common LangChain shapes: a plain string, a ``(content, artifact)``
    response_format=content_and_artifact tuple, or a list of content blocks
    (strings or ``{"type": "text", "text": ...}`` dicts)."""
    if isinstance(result, str):
        return result
    if isinstance(result, tuple) and result:
        return _tool_result_text(result[0])
    if isinstance(result, list):
        parts: list[str] = []
        for block in result:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
        return "\n".join(parts)
    return str(result)


def _raise_on_pw_error(tool: BaseTool) -> None:
    """Turn Playwright text-level failures into real ``ToolException``s.

    Playwright MCP returns action failures (stale ``ref``, timeout, navigation
    error) as ``isError=False`` results with the failure only in the text body,
    so the loop guard records them as *successes* and its failure-loop detector
    never trips (the exact gap that let a ``browser_click`` retry ~300 times).
    Detect the sentinels and raise ``ToolException`` so that (a) the loop guard
    records ``ok=False`` and trips after ``max_identical`` repeats, and (b)
    ``handle_tool_error=True`` still surfaces the text to the model as a
    corrective message rather than crashing the run.

    Wrapped *inside* :func:`tools.loop_guard.wrap_with_loop_guard` (i.e. applied
    before it) so the guard's ``try/except`` observes the raise.
    """
    from functools import wraps
    from langchain_core.tools import StructuredTool, ToolException

    if not isinstance(tool, StructuredTool) or tool.coroutine is None:
        return

    original = tool.coroutine

    @wraps(original)
    async def _checked(*args: Any, **kwargs: Any) -> Any:
        result = await original(*args, **kwargs)
        text = _tool_result_text(result)
        if any(sentinel in text for sentinel in _PLAYWRIGHT_ERROR_SENTINELS):
            raise ToolException(text)
        return result

    tool.coroutine = _checked


def _wrap_with_desktop_lock(tool: BaseTool, owner: str) -> None:
    """Hold the host-global *sticky* desktop lease across a macos-native call.

    Every desktop tool of one session shares an ``owner`` id, so the agent
    keeps the screen across its whole activate→scan→act burst — including the
    read-only observation tools — rather than handing it back between calls.
    That's what stops a second concurrent desktop agent from stealing focus
    mid-sequence (the exact bug the lease exists to prevent).  The lease is
    the same cross-process ``fcntl.flock`` the ``macos-osascript`` MCP uses,
    so AppleScript- and Accessibility-driven actions serialize against each
    other across every agent run on the machine, and it frees automatically
    once the owner goes idle (see :mod:`_desktop_lock`).

    Re-entrant per owner, so composite tools that invoke other tools
    internally (``launch_app`` → ``activate_app``) don't deadlock.  If the
    screen never frees up within the wait cap, returns a clear "desktop busy"
    message (rather than raising) so the agent can back off and retry instead
    of tripping the failure-loop guard.
    """
    from functools import wraps
    from langchain_core.tools import StructuredTool

    from backend.builtin_mcps.macos_osascript._desktop_lock import (
        DesktopBusy,
        acquire_desktop,
        desktop_busy_message,
        end_desktop_call,
        resolve_owner,
    )

    if not isinstance(tool, StructuredTool) or tool.coroutine is None:
        return

    if getattr(tool.coroutine, "__desktop_lock_wrapped__", False):
        return

    original = tool.coroutine

    @wraps(original)
    async def _locked(*args: Any, **kwargs: Any) -> Any:
        # The connection (and this wrapper) is shared across sessions, so the
        # baked-in ``owner`` is the same for every concurrent desktop agent.
        # Resolve the per-run owner from the contextvar set by the subagent
        # runner so two concurrent runs serialize instead of sharing one
        # re-entrant lease. Falls back to ``owner`` for non-subagent callers.
        eff_owner = resolve_owner(owner)
        try:
            await acquire_desktop(eff_owner)
        except DesktopBusy as exc:
            return desktop_busy_message(exc.waited_ms)
        try:
            return await original(*args, **kwargs)
        finally:
            await end_desktop_call(eff_owner)

    setattr(_locked, "__desktop_lock_wrapped__", True)
    tool.coroutine = _locked
