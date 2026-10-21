"""Shared utility functions for the backend."""

from __future__ import annotations

import asyncio
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_coro_sync(coro):
    """Run an async coroutine from inside a synchronous LangChain ``@tool`` body.

    LangChain ``@tool``-wrapped functions execute inside the agent's
    asyncio loop already, but the underlying tool callable is sync.
    Resolve the surrounding loop and dispatch the coroutine onto it via
    ``asyncio.run_coroutine_threadsafe`` so we don't block the loop or
    spawn a competing one; fall back to ``asyncio.run`` only when no
    loop is running (e.g. unit tests calling the tool directly).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result()


def slugify(name: str) -> str:
    """Convert a human name to a URL/filesystem-safe slug."""
    return re.sub(r"[^a-z0-9-]", "-", name.lower().strip()).strip("-")


def is_safe_path_segment(segment: str) -> bool:
    """Return True when *segment* is safe to join as a single child path component.

    Rejects empty values, ``.``/``..`` traversal, path separators, and null
    bytes so a user-supplied ID can never escape its parent directory.
    """
    if not segment or segment in (".", ".."):
        return False
    return not any(ch in segment for ch in ("/", "\\", "\x00"))


def remap_to_virtual_path(key: str, root: Path) -> str:
    """Normalize an agent-supplied path into a session-virtual path.

    The agent is handed the *real* session files directory (via the system
    prompt and the ``SESSION_FILES`` env var), so it naturally builds absolute
    paths like ``{root}/output/report.md``.  The virtual file tools only accept
    virtual paths (``/output/report.md``) or bare relative paths, so an absolute
    path that genuinely lives inside *root* must be remapped rather than rejected.

    Returns the virtual equivalent (``/output/report.md``) when *key* is an
    absolute path inside *root*; otherwise returns *key* unchanged so the caller's
    existing handling (relative join, traversal guard, real-looking-path guard)
    still applies.  This only normalizes the *spelling* of in-root paths — it
    never widens the set of reachable locations.
    """
    if not key.startswith("/"):
        return key
    try:
        rel = Path(key).relative_to(root)
    except ValueError:
        return key
    posix = rel.as_posix()
    if posix == ".":
        return "/"
    return "/" + posix


def is_resolved_path_allowed(
    full: Path,
    root: Path,
    vpath: str,
    *,
    grant_roots: list[Path] | None = None,
    linked_files: list[Path] | None = None,
) -> bool:
    """Return True if *full* is allowed once symlinks are resolved.

    Defense-in-depth for the virtual filesystem: after the lexical containment
    check, re-resolve symlinks so a link planted inside the session root cannot
    be used to read or write a real path outside it.

    An escape is allowed only when the virtual path is under ``/links/`` and
    the resolved path sits inside a folder grant for this session, or is a
    file the user explicitly dropped (``linked_files``). A symlink the agent
    creates under ``/links/`` does not widen the boundary.

    *vpath* is the session-virtual path (always leading-slash form, e.g.
    ``/links/foo/bar.txt``).  Both sides are resolved to normalize platform
    quirks (e.g. macOS ``/var`` -> ``/private/var``).
    """
    resolved = full.resolve()
    if resolved.is_relative_to(root.resolve()):
        return True
    if vpath.lstrip("/").split("/", 1)[0] != "links":
        return False
    for grant_root in grant_roots or []:
        try:
            if resolved.is_relative_to(Path(grant_root).expanduser().resolve()):
                return True
        except (OSError, ValueError):
            continue
    for linked in linked_files or []:
        try:
            if resolved == Path(linked).expanduser().resolve():
                return True
        except (OSError, ValueError):
            continue
    return False


def extract_text_content(content: Any) -> str:
    """Return the plain-text portion of an LLM message content block.

    Handles both the simple string form and the list-of-dicts form used by
    models that interleave text with tool-use blocks.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
    return ""


def platform_label() -> str:
    """Return a normalized OS label: 'windows', 'macos', or 'linux'."""
    return {"win32": "windows", "darwin": "macos", "linux": "linux"}.get(
        sys.platform, sys.platform
    )


def open_in_file_manager(target: Path, *, reveal_file: bool = False) -> None:
    """Open a directory (or reveal a file) in the OS file manager."""
    system = platform.system()
    if system == "Darwin":
        if reveal_file and target.is_file():
            subprocess.Popen(["open", "-R", str(target)])
        else:
            subprocess.Popen(["open", str(target)])
    elif system == "Windows":
        if reveal_file and target.is_file():
            subprocess.Popen(["explorer", "/select,", str(target)])
        else:
            subprocess.Popen(["explorer", str(target)])
    else:
        subprocess.Popen(["xdg-open", str(target)])
