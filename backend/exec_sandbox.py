"""Run agent shell commands under a macOS Seatbelt profile.

``execute`` used to be ``/bin/sh -c`` as the user, with the full
environment. This module is the replacement. The profile was checked
on macOS 26: deny-after-allow wins, ``osascript`` cannot send Apple
Events, ``~/.ssh`` is unreadable, and python and git still start when
``/private/var/folders`` is writable.

When the sandbox is enabled and cannot be applied, the command is not
run. Turning the sandbox off is an explicit setting, not a fallback.
"""

from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Names a shell needs. Everything else from the parent environment is
# dropped, and anything that looks like a credential is dropped even if
# the user listed it in env_passthrough.
_KEEP_ENV = (
    "PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE",
    "TMPDIR", "SHELL", "TERM", "TZ", "TMP", "TEMP",
)
_SECRET_ENV = re.compile(
    r"(API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AWS_|ANTHROPIC|OPENAI|OTTO_API)",
    re.I,
)

_HINT = (
    "\n[sandbox] That was blocked by the execute sandbox. "
    "Network is off unless you enable it, and writes are limited to the "
    "session folder and the mapped project. "
    "Settings → Privacy & Security → Action policy."
)


@dataclass
class ShellResult:
    stdout: str = ""
    stderr: str = ""
    returncode: int = 1
    error: str = ""


def scrub_env(
    source: dict[str, str] | None,
    extra: dict[str, str] | None = None,
    passthrough: list[str] | None = None,
) -> dict[str, str]:
    """An environment safe to hand to a model-authored command."""
    source = source or {}
    out: dict[str, str] = {}
    for key in _KEEP_ENV:
        value = source.get(key)
        if value:
            out[key] = value
    for key in passthrough or []:
        if key in source and not _SECRET_ENV.search(key):
            out[key] = source[key]
    for key, value in (extra or {}).items():
        if value is not None and not _SECRET_ENV.search(key):
            out[key] = str(value)
    return out


def _escape(path: str) -> str:
    return path.replace("\\", "\\\\").replace('"', '\\"')


def _regex_literal(path: str) -> str:
    """Escape *path* for a Seatbelt ``#"..."`` regular expression."""
    return re.sub(r"([.^$|?*+()\[\]{}\\])", r"\\\1", path).replace('"', '\\"')


def _ci_token(name: str) -> str:
    """A Seatbelt character-class spelling of *name*, case-insensitive."""
    parts: list[str] = []
    for char in name:
        if char == ".":
            parts.append("[.]")
        elif char.isalpha():
            parts.append(f"[{char.upper()}{char.lower()}]")
        else:
            parts.append(char)
    return "".join(parts)


def protected_write_denies(paths: list[Path]) -> list[str]:
    """Deny writes to protected project paths under each write root.

    The project is a write path, so without these rules a sandboxed command
    can plant a hook or a clean filter. Matching is case-insensitive on the
    filename because the volume may be too. Deny rules are emitted after the
    write allows, and on macOS the later rule wins.
    """
    lines: list[str] = []
    seen: set[str] = set()
    for path in paths:
        real = _sb_path(path).rstrip("/")
        if not real or real in seen:
            continue
        seen.add(real)
        root = _regex_literal(real)
        for name in (".git", ".vscode", ".idea"):
            token = _ci_token(name)
            lines.append(f'(deny file-write* (regex #"^{root}(/.*)?/{token}(/|$)"))')
        for name in (".gitattributes", ".envrc", "AGENTS.md", "CLAUDE.md"):
            token = _ci_token(name)
            lines.append(f'(deny file-write* (regex #"^{root}(/.*)?/{token}$"))')
    return lines


def _sb_path(path: Path) -> str:
    """The path Seatbelt will see. ``/var`` is a symlink to ``/private/var``."""
    try:
        return str(path.expanduser().resolve())
    except OSError:
        return str(path)


def sensitive_read_denies(home: Path | None = None) -> list[Path]:
    home = home or Path.home()
    library = home / "Library"
    return [
        home / ".ssh",
        home / ".aws",
        home / ".gnupg",
        home / ".config" / "Otto",
        library / "Keychains",
        library / "Messages",
        library / "Mail",
        library / "Cookies",
        library / "Application Support" / "Otto",
        library / "Application Support" / "Google" / "Chrome",
        library / "Application Support" / "Firefox",
        library / "Application Support" / "Discord",
        library / "Safari",
    ]


def network_rules(*, allow_all: bool = False, hosts: list[str] | None = None) -> list[str]:
    """Seatbelt lines for shell network. Hosts beat an all-allow."""
    from backend.mcp_sandbox import _escape_sb_literal, _split_host_port

    cleaned = [item.strip() for item in (hosts or []) if item and item.strip()]
    if cleaned:
        # ``system.sb`` can allow network. Deny first, then allow the hosts,
        # because the later rule wins.
        lines = ["(deny network*)", "(allow network-bind (local ip))"]
        for raw in cleaned:
            host, port = _split_host_port(raw)
            host_esc = _escape_sb_literal(host)
            if port is not None:
                lines.append(f'(allow network-outbound (remote ip "{host_esc}:{port}"))')
            else:
                lines.append(f'(allow network-outbound (remote ip "{host_esc}:*"))')
        lines.append('(allow network-outbound (remote udp "*:53"))')
        lines.append('(allow network-outbound (remote tcp "*:53"))')
        return lines
    if allow_all:
        return ["(allow network*)"]
    return ["(deny network*)"]


def render_profile(
    *,
    write_paths: list[Path],
    read_paths: list[Path] | None = None,
    network: bool = False,
    network_hosts: list[str] | None = None,
    sensitive_paths: list[Path] | None = None,
    session_paths: list[Path] | None = None,
    session_write: bool = True,
) -> str:
    """A deny-default profile. Later rules beat earlier ones (checked on macOS 26).

    Sensitive-path denies are emitted *after* the write-path allows. A
    write grant on ``$HOME`` would otherwise re-allow ``~/.ssh``, because
    the later ``allow file*`` wins.

    ``session_paths`` are emitted after those denies. The session files
    directory lives under ``Application Support/Otto``, so the Otto deny
    would otherwise hide the uploads the user just dropped. Only that
    directory is re-opened. The rest of Otto stays denied.
    """
    lines = [
        "(version 1)",
        "(deny default)",
        '(import "system.sb")',
        "(allow process-fork)",
        "(allow process-exec)",
        "(allow signal (target self))",
        "(allow sysctl-read)",
        "(allow file-read*)",
        "(allow file-ioctl)",
        "(allow ipc-posix-shm)",
        "(allow mach-lookup)",
        # Apple Events stay closed even though mach-lookup is open.
        "(deny appleevent-send)",
        '(deny mach-lookup (global-name "com.apple.coreservices.appleevents"))',
        '(deny mach-lookup (global-name "com.apple.axserver"))',
        '(deny process-exec (literal "/usr/bin/osascript"))',
        '(deny process-exec (literal "/usr/bin/shortcuts"))',
        '(deny process-exec (literal "/usr/sbin/screencapture"))',
        '(allow file-write* (subpath "/private/tmp"))',
        '(allow file-write* (subpath "/private/var/folders"))',
        '(allow file-write* (subpath "/tmp"))',
    ]
    for path in write_paths:
        lines.append(f'(allow file* (subpath "{_escape(_sb_path(path))}"))')
    for path in read_paths or []:
        lines.append(f'(allow file-read* (subpath "{_escape(_sb_path(path))}"))')
    sensitive = list(sensitive_paths if sensitive_paths is not None else sensitive_read_denies())
    for path in sensitive:
        lines.append(f'(deny file-read* (subpath "{_escape(_sb_path(path))}"))')
        lines.append(f'(deny file-write* (subpath "{_escape(_sb_path(path))}"))')
    # ``file*`` does not override an earlier ``deny file-read*``. Spell the
    # read and write filters out, or getcwd and cat of an upload still fail.
    for path in session_paths or []:
        escaped = _escape(_sb_path(path))
        lines.append(f'(allow file-read* (subpath "{escaped}"))')
        if session_write:
            lines.append(f'(allow file-write* (subpath "{escaped}"))')
    # After every file allow, including the session-directory reopen. A later
    # allow would put .git back in reach.
    covered = list(write_paths)
    if session_write:
        covered.extend(session_paths or [])
    lines.extend(protected_write_denies(covered))
    lines.extend(network_rules(allow_all=network, hosts=network_hosts))
    return "\n".join(lines) + "\n"


def _sandbox_exec() -> str | None:
    from backend.mcp_sandbox import resolve_sandbox_exec
    return resolve_sandbox_exec()


def run_shell(
    command: str,
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
    sandbox: bool,
    network: bool = False,
    network_hosts: list[str] | None = None,
    write_paths: list[Path] | None = None,
    read_paths: list[Path] | None = None,
    session_paths: list[Path] | None = None,
    session_write: bool = True,
    profile_dir: Path | None = None,
) -> ShellResult:
    """Run ``command`` via ``/bin/sh -c``. Sandboxed when ``sandbox`` is set."""
    if not command or not str(command).strip():
        return ShellResult(error="Command must be a non-empty string.", returncode=1)

    argv = ["/bin/sh", "-c", command]
    if sandbox:
        if sys.platform != "darwin":
            return ShellResult(
                error=(
                    "The execute sandbox is on, but this system has no Seatbelt. "
                    "The command was not run. Turn the sandbox off in "
                    "Settings → Privacy & Security to run it unsandboxed."
                ),
                returncode=1,
            )
        binary = _sandbox_exec()
        if not binary:
            return ShellResult(
                error=(
                    "The execute sandbox is on, but sandbox-exec was not found. "
                    "The command was not run."
                ),
                returncode=1,
            )
        folder = profile_dir or cwd
        try:
            folder.mkdir(parents=True, exist_ok=True)
            profile_path = folder / ".execute.sb"
            profile_path.write_text(
                render_profile(
                    write_paths=list(write_paths or [cwd]),
                    read_paths=list(read_paths or []),
                    network=network,
                    network_hosts=list(network_hosts or []),
                    session_paths=list(session_paths or []),
                    session_write=session_write,
                ),
                encoding="utf-8",
            )
            profile_path.chmod(0o600)
        except OSError as exc:
            return ShellResult(
                error=f"Could not write the sandbox profile ({exc}). The command was not run.",
                returncode=1,
            )
        argv = [binary, "-f", str(profile_path), *argv]

    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        return ShellResult(error=f"Failed to start the command ({exc}).", returncode=1)

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(proc.pid)
        try:
            stdout, stderr = proc.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
        return ShellResult(
            stdout=stdout or "",
            stderr=(stderr or "") + f"\nCommand timed out after {timeout} seconds.",
            returncode=124,
        )
    stderr = stderr or ""
    if sandbox and proc.returncode not in (0, None) and "Operation not permitted" in stderr:
        stderr += _HINT
    return ShellResult(stdout=stdout or "", stderr=stderr, returncode=proc.returncode or 0)


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def sandbox_settings() -> tuple[bool, bool, list[str], list[str]]:
    """``(enabled, network, extra_read, extra_write)`` from saved settings.

    A settings read that fails enables the sandbox: failing open would
    run the command on the host.
    """
    try:
        from backend.config import AppConfig
        box = AppConfig.load().action_policy.execute_sandbox
        return (
            bool(box.enabled),
            bool(box.network),
            list(box.extra_read_paths),
            list(box.extra_write_paths),
        )
    except Exception:
        logger.exception("execute sandbox: settings unreadable; sandbox stays on")
        return True, False, [], []


def passthrough_names() -> list[str]:
    try:
        from backend.config import AppConfig
        return list(AppConfig.load().action_policy.execute_sandbox.env_passthrough)
    except Exception:
        return []
