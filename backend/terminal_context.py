"""What a terminal in a Code session is allowed to do.

The same rules as ``execute``: the mapped folder is the working directory,
network follows the folder's grant, and a read-only folder or an Ask or Plan
session gets no write path. Typing in the terminal does not lift any of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TerminalContext:
    cwd: Path
    sandbox: bool
    network: bool
    network_hosts: list[str] = field(default_factory=list)
    write_paths: list[Path] = field(default_factory=list)
    read_paths: list[Path] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    readonly: bool = False


def session_mode(session_id: str) -> str:
    try:
        from backend.state import session_mgr
        return session_mgr._mode_for(session_id)
    except Exception:
        return "agent"


def build_context(session_id: str) -> tuple[TerminalContext | None, str | None]:
    """``(context, None)`` or ``(None, reason)`` when a terminal must not start."""
    from backend.exec_sandbox import sandbox_settings
    from backend.session_manager import _read_workspace_meta, _sandbox_always_writable
    from backend.shell_policy import shell_network
    from backend.workspace_grants import get_grant

    workspace = _read_workspace_meta(session_id)
    if workspace is None or not workspace.grant_id:
        return None, "No project folder is mapped."
    grant = get_grant(workspace.grant_id)
    if grant is None:
        return None, "The project folder is no longer allowed."
    root = Path(workspace.host_path)
    enabled, global_network, extra_read, extra_write = sandbox_settings()
    mode = session_mode(session_id)
    readonly = mode in ("ask", "plan") or workspace.mode == "read"
    if readonly and not enabled:
        return None, (
            "This folder is read-only here, and the shell sandbox is turned off in Settings, "
            "so a terminal cannot be held to that. Turn the sandbox on to open one."
        )
    if readonly and _sandbox_always_writable(root, extra_write):
        return None, (
            "This folder is read-only, but it sits where the shell sandbox always allows writes, "
            "so a terminal cannot be held to that."
        )
    network, hosts = shell_network(global_network, bool(grant.network), list(grant.network_hosts))
    write_paths = [] if readonly else [root]
    for raw in extra_write:
        if raw:
            write_paths.append(Path(raw).expanduser())
    read_paths = [Path(raw).expanduser() for raw in extra_read if raw]
    return TerminalContext(
        cwd=root,
        sandbox=enabled,
        network=network,
        network_hosts=hosts,
        write_paths=write_paths,
        read_paths=read_paths,
        env={"PROJECT_ROOT": str(root)},
        readonly=readonly,
    ), None


def open_terminal(session_id: str, *, kind: str = "shell", command: str | None = None, rows: int = 24, cols: int = 100) -> dict:
    """Start a terminal in the session's project. ``{ok, terminal}`` or ``{ok: False, error}``."""
    from backend import terminal

    ctx, reason = build_context(session_id)
    if ctx is None:
        return {"ok": False, "error": reason}
    return terminal.create(
        session_id,
        ctx.cwd,
        kind=kind,
        command=command,
        sandbox=ctx.sandbox,
        network=ctx.network,
        network_hosts=ctx.network_hosts,
        write_paths=ctx.write_paths,
        read_paths=ctx.read_paths,
        env=ctx.env,
        rows=rows,
        cols=cols,
    )
