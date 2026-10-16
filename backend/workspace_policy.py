"""Decide whether a host folder may be granted, and whether a path is inside it.

Validation follows symlinks and shows the real target. Grants are rejected
for the filesystem root, the home directory itself, other system roots,
Otto's own app-data directory, and any folder that is or contains a path
the shell sandbox treats as sensitive (``~/.ssh`` and the like).
"""

from __future__ import annotations

from pathlib import Path

from backend.exec_sandbox import sensitive_read_denies
from backend.schemas import SessionWorkspace

_BLOCKED = (
    "/",
    "/usr",
    "/bin",
    "/sbin",
    "/etc",
    "/var",
    "/private",
    "/System",
    "/Library",
    "/Applications",
    "/opt",
    "/tmp",
    "/Users",
    "/home",
    "/private/tmp",
    "/private/var",
    "/Volumes",
    "/private/var/root",
)


class GrantRejected(ValueError):
    """The path exists but must not be granted."""


def _posix(path: Path) -> str:
    text = path.as_posix().rstrip("/")
    return text or "/"


def _blocked_roots() -> set[str]:
    found: set[str] = set()
    for raw in _BLOCKED:
        try:
            found.add(_posix(Path(raw).resolve()))
        except OSError:
            found.add(raw.rstrip("/") or "/")
    try:
        found.add(_posix(Path.home().resolve()))
    except OSError:
        pass
    return found


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def _sensitive_paths() -> list[Path]:
    paths = list(sensitive_read_denies())
    try:
        from backend.config import get_app_data_dir
        paths.append(get_app_data_dir())
    except Exception:
        pass
    return paths


def rejection_reason(path: Path) -> str | None:
    """Why *path* cannot be granted, or None when it is a normal directory."""
    try:
        src = path.expanduser().resolve()
    except OSError as exc:
        return f"Cannot resolve path: {exc}"
    if not src.exists():
        return f"Folder not found: {path}"
    if not src.is_dir():
        return "Choose a folder, not a file."
    if _posix(src) in _blocked_roots():
        return (
            f"{src} is too broad to allow. Choose a project folder inside it, "
            "not your home directory or a system folder."
        )
    for sens in _sensitive_paths():
        try:
            target = sens.expanduser().resolve()
        except OSError:
            target = sens
        if src == target or _is_relative_to(src, target):
            return f"{src} is inside a protected location ({target})."
        if _is_relative_to(target, src):
            return (
                f"{src} contains a protected location ({target}). "
                "Choose a project folder that does not include it."
            )
    return None


def validate_grantable(path: str | Path) -> Path:
    """Resolve *path* or raise FileNotFoundError, NotADirectoryError, or GrantRejected."""
    src = Path(path).expanduser()
    try:
        resolved = src.resolve()
    except OSError as exc:
        raise FileNotFoundError(f"Cannot resolve path: {path}") from exc
    if not resolved.exists():
        raise FileNotFoundError(f"Folder not found: {path}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"workspace must be a directory: {path}")
    reason = rejection_reason(resolved)
    if reason:
        raise GrantRejected(reason)
    return resolved


def path_inside(child: Path, root: Path) -> bool:
    try:
        child.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def allowance_for_session(session_id: str) -> tuple[list[Path], list[Path]]:
    """Grant roots and explicitly dropped files for *session_id*."""
    from backend.session_manager import _read_linked_files, _read_workspace_meta

    roots: list[Path] = []
    workspace = _read_workspace_meta(session_id)
    if workspace is not None and workspace.grant_id:
        from backend.workspace_grants import get_grant
        grant = get_grant(workspace.grant_id)
        if grant is not None:
            root = Path(grant.realpath)
            if root.is_dir():
                roots.append(root)
    linked = []
    for raw in _read_linked_files(session_id):
        path = Path(raw)
        if path.is_file():
            linked.append(path)
    return roots, linked


def active_workspace(session_id: str) -> SessionWorkspace | None:
    """The mapped folder, only when it is backed by a live grant."""
    from backend.session_manager import _read_workspace_meta
    workspace = _read_workspace_meta(session_id)
    if workspace is None or not workspace.grant_id:
        return None
    from backend.workspace_grants import get_grant
    if get_grant(workspace.grant_id) is None:
        return None
    return workspace
