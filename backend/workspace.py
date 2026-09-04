"""Session workspace — a mapped host folder the coding agent works in.

A workspace is a directory symlink under ``files/links/<name>/`` plus
metadata on the session.  File tools keep virtual paths
(``/links/<name>/…``); ``execute`` can run with cwd at the host folder.

The tree/file APIs list **one directory level** and never recurse through
``node_modules`` / ``.git`` / similar, so mapping a real repo is safe.
"""

from __future__ import annotations

import fnmatch
import logging
from pathlib import Path
from typing import Any, Optional

from backend.schemas import SessionWorkspace
from backend.utils import is_resolved_path_allowed

logger = logging.getLogger(__name__)

# Directories we never list (even if not in .gitignore).
_SKIP_DIR_NAMES = frozenset({
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".next",
    ".turbo",
    ".nuxt",
    "target",
    "Pods",
    ".gradle",
    ".idea",
    ".cache",
    "coverage",
    ".parcel-cache",
    "vendor",
})

# Dot-directories that *are* useful to show.
_KEEP_DOT_DIRS = frozenset({".github", ".cursor", ".vscode", ".otto"})

_MAX_TREE_ENTRIES = 400
_MAX_FILE_BYTES = 1_000_000


def skip_dir_name(name: str) -> bool:
    """Return True when *name* should be omitted from a workspace tree listing."""
    if name in _SKIP_DIR_NAMES:
        return True
    if name.startswith(".") and name not in _KEEP_DOT_DIRS:
        return True
    return False


def _gitignore_patterns(root: Path) -> list[str]:
    """Return simple gitignore patterns from the workspace root (best-effort)."""
    gi = root / ".gitignore"
    if not gi.is_file():
        return []
    try:
        lines = gi.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    patterns: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        patterns.append(line.rstrip("/"))
    return patterns


def _ignored_by_gitignore(name: str, rel_posix: str, patterns: list[str]) -> bool:
    for pat in patterns:
        base = pat.split("/")[-1]
        if fnmatch.fnmatch(name, base) or fnmatch.fnmatch(rel_posix, pat.lstrip("/")):
            return True
    return False


def resolve_workspace_child(workspace: SessionWorkspace, rel: str) -> Path:
    """Resolve *rel* (workspace-relative) to a host path, rejecting traversal."""
    root = Path(workspace.host_path).expanduser().resolve()
    rel_norm = (rel or "").replace("\\", "/").lstrip("/")
    if rel_norm in ("", "."):
        return root
    if ".." in Path(rel_norm).parts or rel_norm.startswith("~"):
        raise ValueError("Path traversal not allowed")
    full = (root / rel_norm).resolve()
    if not full.is_relative_to(root):
        raise ValueError(f"Path '{rel}' is outside the mapped folder")
    return full


def list_tree(
    workspace: SessionWorkspace,
    rel: str = "",
) -> dict[str, Any]:
    """List one directory under the mapped folder.

    Returns ``{path, entries: [{name, path, is_dir, size}], truncated, skipped}``.
    """
    root = Path(workspace.host_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Mapped folder not found: {workspace.host_path}")

    target = resolve_workspace_child(workspace, rel)
    if not target.is_dir():
        raise NotADirectoryError(f"Not a directory: {rel or '/'}")

    patterns = _gitignore_patterns(root)
    rel_prefix = (rel or "").replace("\\", "/").strip("/")

    entries: list[dict[str, Any]] = []
    skipped = 0
    truncated = False

    try:
        children = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as exc:
        raise OSError(f"Cannot list {target}: {exc}") from exc

    for child in children:
        name = child.name
        child_rel = f"{rel_prefix}/{name}" if rel_prefix else name
        is_dir = child.is_dir()
        if is_dir and skip_dir_name(name):
            skipped += 1
            continue
        if _ignored_by_gitignore(name, child_rel, patterns):
            skipped += 1
            continue
        if len(entries) >= _MAX_TREE_ENTRIES:
            truncated = True
            skipped += 1
            continue
        size: int | None = None
        if child.is_file() and not child.is_symlink():
            try:
                size = child.stat().st_size
            except OSError:
                size = 0
        entries.append({
            "name": name,
            "path": child_rel,
            "is_dir": is_dir,
            "size": size,
        })

    return {
        "path": rel_prefix,
        "entries": entries,
        "truncated": truncated,
        "skipped": skipped,
    }


def read_file(workspace: SessionWorkspace, rel: str) -> dict[str, Any]:
    """Read a text file under the mapped folder (size-capped)."""
    target = resolve_workspace_child(workspace, rel)
    if not target.is_file():
        raise FileNotFoundError(f"File not found: {rel}")
    try:
        data = target.read_bytes()
    except OSError as exc:
        raise OSError(f"Cannot read {rel}: {exc}") from exc
    truncated = False
    if len(data) > _MAX_FILE_BYTES:
        data = data[:_MAX_FILE_BYTES]
        truncated = True
    text = data.decode("utf-8", errors="replace")
    return {
        "path": rel.replace("\\", "/").lstrip("/"),
        "content": text,
        "truncated": truncated,
        "size": target.stat().st_size if target.exists() else len(data),
    }


def ensure_workspace_link(files_dir: Path, source: Path) -> SessionWorkspace:
    """Create (or reuse) a directory symlink under ``files/links/<name>``.

    Unlike the generic session-links API this uses a stable name (the
    folder basename) so the virtual path does not churn if the user remaps.
    """
    src = source.expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"source not found: {source}")
    if not src.is_dir():
        raise NotADirectoryError(f"workspace must be a directory: {source}")

    links_dir = files_dir / "links"
    links_dir.mkdir(parents=True, exist_ok=True)

    name = src.name or "project"
    target = links_dir / name
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == src:
            pass  # already mapped to this folder
        else:
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.is_dir() and not target.is_symlink():
                # Don't clobber a real directory the agent created.
                n = 1
                while True:
                    candidate = links_dir / f"{name}-{n}"
                    if not candidate.exists():
                        target = candidate
                        name = candidate.name
                        break
                    n += 1
            else:
                target.unlink()
            target.symlink_to(src, target_is_directory=True)
    else:
        target.symlink_to(src, target_is_directory=True)

    return SessionWorkspace(
        host_path=str(src),
        virtual_path=f"/links/{name}",
        name=name,
    )


def remove_workspace_link(files_dir: Path, workspace: SessionWorkspace) -> None:
    """Remove the workspace symlink (does not touch the host folder)."""
    name = workspace.name
    if not name or "/" in name or name in (".", ".."):
        return
    target = files_dir / "links" / name
    if target.is_symlink() or (target.exists() and target.is_file()):
        target.unlink()


def virtual_to_rel(workspace: SessionWorkspace, virtual_path: str) -> Optional[str]:
    """Strip ``/links/<name>/`` from a virtual path; None if it isn't under the workspace."""
    prefix = workspace.virtual_path.rstrip("/")
    v = virtual_path if virtual_path.startswith("/") else "/" + virtual_path
    if v == prefix:
        return ""
    if v.startswith(prefix + "/"):
        return v[len(prefix) + 1:]
    return None


def is_workspace_vpath_allowed(files_dir: Path, vpath: str, full: Path) -> bool:
    """Containment check for serving a session-virtual path that may live under /links/."""
    return is_resolved_path_allowed(full, files_dir, vpath)
