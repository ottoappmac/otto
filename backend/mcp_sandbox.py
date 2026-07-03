"""Render and apply macOS ``sandbox-exec`` profiles for generated MCPs.

This module is the second half of OTTO's defence-in-depth strategy
around runtime-generated MCP servers (the first half being the AST
audit in :mod:`backend.mcp_builder`).

The agent-authored Python file has already passed:

1. Forbidden-imports rejection (``subprocess``, ``socket``, ...).
2. Credential-literal rejection (no API keys baked into source).
3. HMAC signature verification via :mod:`backend.mcp_signer`.

But once the subprocess starts, the *process* has full POSIX
permissions of the user.  A vulnerable dependency or a buggy LLM
prompt could still:

* Read arbitrary files outside the MCP's working directory.
* Open TCP sockets to arbitrary hosts (data exfiltration).
* Fork helper processes that aren't covered by the static audit.

``sandbox-exec`` -- the Apple-internal but stable-enough tool that
backs the App Sandbox -- lets us wrap the subprocess in a kernel-level
policy declared by a ``.sb`` file.  The profile rendered here:

* Allows everything Python needs to start (interpreter, stdlib, venv).
* Restricts file writes to the per-MCP folder + ``/private/tmp``.
* Restricts outbound network to the hosts the MCP declared in its
  permission manifest.
* Blocks ptrace / debug API access.

On Linux / Windows we currently no-op (returning ``False`` from
:func:`is_supported`); the same MCPs still run, just without the
extra layer.  Tightening that is a follow-up.

References
~~~~~~~~~~

The ``.sb`` language is undocumented but widely reverse-engineered; the
profile shape used here follows the conventions in
``/System/Library/Sandbox/Profiles/`` and Apple's own
``Profile.scm`` examples.
"""

from __future__ import annotations

import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Permission manifest
# ---------------------------------------------------------------------------


@dataclass
class PermissionManifest:
    """Declared permissions for one generated MCP subprocess.

    Empty lists mean "deny" -- the renderer never falls back to
    "allow all" silently.  ``allow_network_all`` is a single explicit
    knob that overrides the host allowlist for cases like an MCP that
    talks to a service with rotating IPs we can't enumerate.
    """

    # Absolute filesystem paths the MCP is allowed to read.  Anything
    # not on this list (plus the always-allowed system / venv paths
    # baked into the template) is blocked.
    fs_read: list[str] = field(default_factory=list)
    # Absolute filesystem paths the MCP is allowed to write.  Reads
    # are NOT implicit -- include a path in both lists if needed.
    fs_write: list[str] = field(default_factory=list)
    # ``host[:port]`` entries the MCP may dial.  ``port`` is optional;
    # when present the rule restricts to that exact port.
    network_hosts: list[str] = field(default_factory=list)
    # Escape hatch for MCPs whose target endpoints can't be enumerated.
    # Logged at WARNING and surfaced in the trust panel.
    allow_network_all: bool = False
    # Environment variables the MCP is allowed to read.  Hydration in
    # :mod:`backend.mcp_manager` already filters this; the field is
    # carried here so the manifest is a complete description of the
    # subprocess's permissioned surface.
    env_read: list[str] = field(default_factory=list)

    # Whether sandbox-exec should be applied at all.  ``False`` means
    # "trust the audit but skip the kernel layer" -- useful when
    # iterating on a new MCP and the sandbox is the suspect during
    # debugging.
    sandbox_enabled: bool = True

    def to_dict(self) -> dict:
        return {
            "fs_read": list(self.fs_read),
            "fs_write": list(self.fs_write),
            "network_hosts": list(self.network_hosts),
            "allow_network_all": bool(self.allow_network_all),
            "env_read": list(self.env_read),
            "sandbox_enabled": bool(self.sandbox_enabled),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PermissionManifest":
        if not isinstance(data, dict):
            return cls()
        return cls(
            fs_read=[str(s) for s in data.get("fs_read") or []],
            fs_write=[str(s) for s in data.get("fs_write") or []],
            network_hosts=[str(s) for s in data.get("network_hosts") or []],
            allow_network_all=bool(data.get("allow_network_all")),
            env_read=[str(s) for s in data.get("env_read") or []],
            sandbox_enabled=bool(data.get("sandbox_enabled", True)),
        )


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------


def is_supported() -> bool:
    """Return True when the host can apply sandbox-exec wrapping."""
    if sys.platform != "darwin":
        return False
    return resolve_sandbox_exec() is not None


def resolve_sandbox_exec() -> str | None:
    """Return the path to ``sandbox-exec`` or ``None`` when missing."""
    found = shutil.which("sandbox-exec")
    if found:
        return found
    # sandbox-exec ships in /usr/bin on every modern macOS release.
    cand = Path("/usr/bin/sandbox-exec")
    if cand.is_file():
        return str(cand)
    return None


# ---------------------------------------------------------------------------
# Profile rendering
# ---------------------------------------------------------------------------


_PROFILE_HEADER = """(version 1)
(deny default)
(import "system.sb")
"""


# Common allow rules every Python subprocess needs to start at all.
# Errors here cause cryptic "Killed: 9" exits with no helpful logs, so
# we err on the side of allowing well-known interpreter machinery.
_COMMON_ALLOWS = """
; --- Common interpreter / runtime requirements ------------------------------
(allow process-fork)
(allow signal (target self))
(allow sysctl-read)
(allow mach-lookup)
(allow ipc-posix-shm)
(allow file-ioctl)

; Python needs to read its own interpreter, stdlib, and the venv's
; site-packages.  The venv path is filled in per-MCP below.
(allow file-read* (subpath "/usr/lib"))
(allow file-read* (subpath "/usr/local"))
(allow file-read* (subpath "/System/Library"))
(allow file-read* (subpath "/Library/Developer/CommandLineTools"))
(allow file-read* (subpath "/opt/homebrew"))
(allow file-read* (subpath "/private/etc"))

; SSL trust roots so HTTPS clients can validate certs even with
; tight network rules.
(allow file-read* (literal "/etc/ssl/cert.pem"))
(allow file-read* (subpath "/private/etc/ssl"))

; stdin / stdout / stderr -- without these the MCP can't talk to us.
(allow file-read-data (literal "/dev/null"))
(allow file-write-data (literal "/dev/null"))
(allow file-read-data (literal "/dev/random"))
(allow file-read-data (literal "/dev/urandom"))

; tmpfile() / NamedTemporaryFile -- standard library plumbing.
(allow file* (subpath "/private/tmp"))
(allow file* (subpath "/private/var/tmp"))
(allow file* (subpath "/private/var/folders"))
"""


def _escape_sb_literal(path: str) -> str:
    """Escape a string for inclusion as an ``.sb`` literal.

    Apple's .sb files use a Scheme-like syntax; double quotes need to
    be doubled and backslashes need escaping.  We don't try to fix
    malformed paths -- the caller upstream validates them.
    """
    return path.replace("\\", "\\\\").replace('"', '\\"')


def render_sandbox_profile(
    *,
    mcp_id: str,
    server_dir: Path,
    venv_python: Path,
    manifest: PermissionManifest,
) -> str:
    """Render the full ``.sb`` profile for one MCP subprocess.

    The caller writes this to ``<server_dir>/sandbox.sb`` and points
    ``sandbox-exec -f <path>`` at it.  The profile pins:

    * Reads to: system paths + venv + per-MCP server dir + the
      ``fs_read`` allowlist.
    * Writes to: per-MCP server dir + the ``fs_write`` allowlist +
      always-temp paths in /private/tmp.
    * Network: explicit host:port pairs unless ``allow_network_all``.
    """
    parts: list[str] = [_PROFILE_HEADER, f"; mcp_id={mcp_id}", _COMMON_ALLOWS]

    parts.append("; --- Per-MCP isolated venv + working directory ----------------------------")
    parts.append(f'(allow file-read* (subpath "{_escape_sb_literal(str(venv_python.parent.parent))}"))')
    parts.append(f'(allow process-exec (literal "{_escape_sb_literal(str(venv_python))}"))')
    parts.append(f'(allow file-read* (subpath "{_escape_sb_literal(str(server_dir))}"))')
    # Writing to the MCP's own folder is required for FastMCP's logging
    # + any small state file the user's tool may want to drop.
    parts.append(f'(allow file* (subpath "{_escape_sb_literal(str(server_dir))}"))')

    if manifest.fs_read:
        parts.append("")
        parts.append("; --- Manifest fs_read ------------------------------------------------------")
        for raw in manifest.fs_read:
            path = _escape_sb_literal(raw)
            parts.append(f'(allow file-read* (subpath "{path}"))')

    if manifest.fs_write:
        parts.append("")
        parts.append("; --- Manifest fs_write -----------------------------------------------------")
        for raw in manifest.fs_write:
            path = _escape_sb_literal(raw)
            parts.append(f'(allow file* (subpath "{path}"))')

    parts.append("")
    parts.append("; --- Network -------------------------------------------------------------")
    if manifest.allow_network_all:
        # Logged at write time so the trust panel can flag the MCP.
        parts.append("(allow network-outbound)")
        parts.append("(allow network*)")
    elif manifest.network_hosts:
        parts.append("(allow network-bind (local ip))")
        for raw in manifest.network_hosts:
            host, port = _split_host_port(raw)
            host_esc = _escape_sb_literal(host)
            if port is not None:
                parts.append(
                    f'(allow network-outbound (remote ip "{host_esc}:{port}"))'
                )
            else:
                parts.append(
                    f'(allow network-outbound (remote ip "{host_esc}:*"))'
                )
        # DNS resolution is required for any host string; allowing
        # ``udp 53`` to any IP is the standard pattern.
        parts.append('(allow network-outbound (remote udp "*:53"))')
        parts.append('(allow network-outbound (remote tcp "*:53"))')
    else:
        # No hosts and not all-allow -> fully deny network.  Useful for
        # MCPs that only manipulate local files.
        parts.append("; (network outbound denied by default)")

    return "\n".join(parts) + "\n"


def _split_host_port(spec: str) -> tuple[str, int | None]:
    """Split ``host[:port]`` -- IPv6 brackets are not used in .sb."""
    spec = spec.strip()
    if ":" in spec and not spec.startswith("["):
        host, _, port = spec.rpartition(":")
        try:
            return host, int(port)
        except ValueError:
            return spec, None
    return spec, None


def write_profile(server_dir: Path, profile: str) -> Path:
    """Persist a profile to ``<server_dir>/sandbox.sb`` and return the path."""
    out = server_dir / "sandbox.sb"
    out.write_text(profile, encoding="utf-8")
    out.chmod(0o600)
    return out


def wrap_command(
    *,
    command: list[str],
    server_dir: Path,
    manifest: PermissionManifest,
) -> list[str]:
    """Return *command* possibly wrapped in ``sandbox-exec``.

    No wrap is applied when:

    * The host isn't macOS (Linux / Windows -- no support yet).
    * ``manifest.sandbox_enabled`` is False (debug mode).
    * The profile file is missing on disk.

    Otherwise the returned list invokes ``sandbox-exec -f <profile> --
    <command>``.
    """
    if not manifest.sandbox_enabled:
        return list(command)
    if not is_supported():
        return list(command)
    profile = server_dir / "sandbox.sb"
    if not profile.is_file():
        logger.warning(
            "mcp_sandbox: profile missing at %s -- subprocess will run without sandbox",
            profile,
        )
        return list(command)
    sbe = resolve_sandbox_exec()
    if not sbe:  # pragma: no cover - covered by is_supported()
        return list(command)
    return [sbe, "-f", str(profile), *command]


def default_manifest_for_paths(
    *,
    server_dir: Path,
    extra_read: Iterable[str] = (),
    extra_write: Iterable[str] = (),
    network_hosts: Iterable[str] = (),
    allow_network_all: bool = False,
) -> PermissionManifest:
    """Convenience builder used by the smoke tests and the route layer.

    Most generated MCPs have ``server_dir`` itself as the only writable
    path; this helper pre-populates that without forcing the caller to
    repeat it.
    """
    return PermissionManifest(
        fs_read=[str(server_dir), *[str(p) for p in extra_read]],
        fs_write=[str(server_dir), *[str(p) for p in extra_write]],
        network_hosts=list(network_hosts),
        allow_network_all=bool(allow_network_all),
    )


__all__ = [
    "PermissionManifest",
    "is_supported",
    "resolve_sandbox_exec",
    "render_sandbox_profile",
    "write_profile",
    "wrap_command",
    "default_manifest_for_paths",
]
