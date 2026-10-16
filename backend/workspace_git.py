"""Checkpoints and diffs for a granted project folder.

Git repos get a snapshot commit under ``refs/otto/checkpoints/<session>/<n>``
before each turn. The snapshot uses a temporary index, so the user's
index, stash, and branches are left alone. Repo config can run programs
during diff and status, so those commands disable external diff drivers,
textconv, fsmonitor, and hooks.

Folders that are not a git repo keep a byte copy of each file the agent
writes, taken before the write. Revert refuses to clobber a file whose
current hash does not match the hash recorded after the agent's write.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from backend.schemas import SessionWorkspace

logger = logging.getLogger(__name__)

_TIMEOUT = 20
_NO_HOOKS = Path(tempfile.gettempdir()) / "otto-no-hooks"
_NO_HOOKS.mkdir(exist_ok=True)
_MAX_FILES = 400
_MAX_PREIMAGE = 1_000_000
_MAX_DIFF_BYTES = 200_000


def _git_env(index: Path | None = None) -> dict[str, str]:
    env = {
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", ""),
    }
    if index is not None:
        env["GIT_INDEX_FILE"] = str(index)
    return env


def _git(repo: Path, *args: str, index: Path | None = None, timeout: int = _TIMEOUT) -> subprocess.CompletedProcess[str]:
    proc = _git_raw(repo, *args, index=index, timeout=timeout)
    return subprocess.CompletedProcess(
        proc.args,
        proc.returncode,
        (proc.stdout or b"").decode("utf-8", errors="replace"),
        (proc.stderr or b"").decode("utf-8", errors="replace"),
    )


def _git_raw(repo: Path, *args: str, index: Path | None = None, timeout: int = _TIMEOUT) -> subprocess.CompletedProcess[bytes]:
    cmd = [
        "git",
        "-C", str(repo),
        "-c", "core.fsmonitor=",
        "-c", f"core.hooksPath={_NO_HOOKS}",
        "--no-optional-locks",
        *args,
    ]
    return subprocess.run(
        cmd,
        cwd=str(repo),
        env=_git_env(index),
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def repo_root(host: Path) -> Path | None:
    try:
        proc = _git(host, "rev-parse", "--show-toplevel")
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.info("git rev-parse failed: %s", exc)
        return None
    if proc.returncode != 0:
        return None
    text = (proc.stdout or "").strip()
    if not text:
        return None
    root = Path(text)
    return root if root.is_dir() else None


def _state_path(session_id: str) -> Path:
    from backend.session_manager import _sessions_dir
    return _sessions_dir() / f"{session_id}.workspace.json"


def _load_state(session_id: str) -> dict[str, Any]:
    path = _state_path(session_id)
    if not path.is_file():
        return {"checkpoints": [], "writes": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"checkpoints": [], "writes": {}}
    if not isinstance(data, dict):
        return {"checkpoints": [], "writes": {}}
    data.setdefault("checkpoints", [])
    data.setdefault("writes", {})
    return data


def _save_state(session_id: str, data: dict[str, Any]) -> None:
    path = _state_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def snapshot_before_turn(workspace: SessionWorkspace, session_id: str) -> dict[str, Any]:
    """Capture the tree before a turn. Never raises; failures are ``ok: False``."""
    if workspace.mode == "read":
        return {"ok": True, "skipped": True, "reason": "read-only folder"}
    host = Path(workspace.host_path)
    if not host.is_dir():
        return {"ok": False, "reason": "mapped folder is missing"}
    root = repo_root(host)
    if root is None:
        return {
            "ok": False,
            "reason": "not a git repository",
            "coverage": "shell changes are not checkpointed; file-tool edits keep a private copy",
        }
    try:
        return _git_snapshot(root, session_id)
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "checkpoint timed out"}
    except OSError as exc:
        return {"ok": False, "reason": f"checkpoint failed: {exc}"}


def _git_snapshot(root: Path, session_id: str) -> dict[str, Any]:
    state = _load_state(session_id)
    turn = len(state["checkpoints"]) + 1
    ref = f"refs/otto/checkpoints/{session_id}/{turn}"
    with tempfile.TemporaryDirectory(prefix="otto-index-") as tmp:
        index = Path(tmp) / "index"
        added = _git(root, "add", "-A", "--", ".", index=index)
        if added.returncode != 0:
            return {"ok": False, "reason": (added.stderr or "git add failed").strip()[:400]}
        tree = _git(root, "write-tree", index=index)
        if tree.returncode != 0:
            return {"ok": False, "reason": (tree.stderr or "git write-tree failed").strip()[:400]}
        tree_id = (tree.stdout or "").strip()
        if not tree_id:
            return {"ok": False, "reason": "git write-tree returned no tree"}
        commit = _git(
            root, "commit-tree", tree_id, "-m", f"otto checkpoint {session_id} turn {turn}",
        )
        if commit.returncode != 0:
            return {"ok": False, "reason": (commit.stderr or "git commit-tree failed").strip()[:400]}
        commit_id = (commit.stdout or "").strip()
    updated = _git(root, "update-ref", ref, commit_id)
    if updated.returncode != 0:
        return {"ok": False, "reason": (updated.stderr or "git update-ref failed").strip()[:400]}
    record = {
        "id": f"{session_id}:{turn}",
        "turn": turn,
        "ref": ref,
        "commit": commit_id,
        "kind": "git",
    }
    state["checkpoints"].append(record)
    _save_state(session_id, state)
    return {"ok": True, "checkpoint": record}


def record_write(session_id: str, rel: str, before: bytes | None, after: bytes | None) -> None:
    """Remember hashes for a file-tool write so revert can detect a conflict."""
    rel = rel.replace("\\", "/").lstrip("/")
    if not rel or not session_id:
        return
    state = _load_state(session_id)
    writes: dict[str, Any] = state["writes"]
    prev = writes.get(rel) if isinstance(writes.get(rel), dict) else {}
    before_hash = prev.get("before") if before is None else _sha(before)
    if before is None and "before" not in prev and after is not None:
        before_hash = None
    writes[rel] = {
        "before": before_hash,
        "after": _sha(after) if after is not None else None,
    }
    state["writes"] = writes
    _save_state(session_id, state)


def capture_preimage(session_id: str, root: Path, target: Path) -> None:
    """Copy a non-git file before the agent overwrites it."""
    if repo_root(root) is not None:
        return
    if not target.is_file():
        return
    try:
        rel = target.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return
    try:
        data = target.read_bytes()
    except OSError:
        return
    if len(data) > _MAX_PREIMAGE:
        return
    dest_dir = _state_path(session_id).parent / session_id / "preimages"
    dest = dest_dir / rel
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


def git_status(workspace: SessionWorkspace) -> dict[str, Any]:
    host = Path(workspace.host_path)
    root = repo_root(host)
    if root is None:
        return {"git": False, "branch": None, "dirty": 0}
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git(root, "status", "--porcelain", "-uall", "--no-ext-diff", "--no-textconv")
    lines = [ln for ln in (status.stdout or "").splitlines() if ln.strip()]
    return {
        "git": True,
        "branch": (branch.stdout or "").strip() or None,
        "dirty": len(lines),
    }


_MAX_MARKER_BYTES = 2_000_000


def _marker_count(base: Path, rel: str) -> int:
    """Lines in the working copy of *rel* that carry the debug marker."""
    from backend.session_mode import count_debug_markers

    try:
        path = (base / rel).resolve()
        if not path.is_file() or path.stat().st_size > _MAX_MARKER_BYTES:
            return 0
        return count_debug_markers(path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, ValueError):
        return 0


def list_changes(workspace: SessionWorkspace, session_id: str, turn: int | None = None) -> dict[str, Any]:
    """Changed files since the checkpoint, plus leftover ``otto-debug`` lines.

    Debug mode asks the agent to mark temporary logging. Counting the marker
    here means cleanup is checked by code, not taken on trust.
    """
    result = _list_changes(workspace, session_id, turn)
    host = Path(workspace.host_path)
    base = repo_root(host) or host
    total = 0
    for item in result.get("files") or []:
        count = _marker_count(base, item.get("path") or "")
        item["debug_markers"] = count
        total += count
    result["debug_markers"] = total
    return result


def _list_changes(workspace: SessionWorkspace, session_id: str, turn: int | None = None) -> dict[str, Any]:
    host = Path(workspace.host_path)
    root = repo_root(host)
    state = _load_state(session_id)
    checkpoints = state.get("checkpoints") or []
    chosen = None
    if turn is not None:
        for item in checkpoints:
            if item.get("turn") == turn:
                chosen = item
                break
    elif checkpoints:
        chosen = checkpoints[-1]
    if root is None:
        return {
            "git": False,
            "checkpoint": chosen,
            "files": _nongit_files(session_id, host),
            "message": "This folder is not a git repository. Only file-tool edits can be reverted.",
        }
    if chosen is None or not chosen.get("commit"):
        return {"git": True, "checkpoint": None, "files": [], "message": "No checkpoint for this chat yet."}
    proc = _git(
        root,
        "diff", "--numstat", "--no-ext-diff", "--no-textconv",
        chosen["commit"],
    )
    files = []
    for line in (proc.stdout or "").splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        added, deleted, path = parts[0], parts[1], parts[-1]
        files.append({
            "path": path,
            "added": None if added == "-" else int(added or 0),
            "deleted": None if deleted == "-" else int(deleted or 0),
            "status": "modified",
        })
        if len(files) >= _MAX_FILES:
            break
    status = _git(root, "status", "--porcelain", "-uall", "--no-ext-diff", "--no-textconv")
    known = {item["path"] for item in files}
    for line in (status.stdout or "").splitlines():
        if len(line) < 4:
            continue
        code = line[:2]
        path = line[3:].strip()
        if path not in known and code.strip().endswith("?"):
            files.append({"path": path, "added": None, "deleted": None, "status": "untracked"})
    return {"git": True, "checkpoint": chosen, "checkpoints": checkpoints, "files": files}


def _nongit_files(session_id: str, host: Path) -> list[dict[str, Any]]:
    state = _load_state(session_id)
    files = []
    for rel in (state.get("writes") or {}):
        files.append({"path": rel, "added": None, "deleted": None, "status": "modified"})
    return files


def file_diff(workspace: SessionWorkspace, session_id: str, rel: str, turn: int | None = None) -> dict[str, Any]:
    host = Path(workspace.host_path)
    root = repo_root(host)
    rel = rel.replace("\\", "/").lstrip("/")
    if root is None:
        return {"path": rel, "diff": "", "message": "No git diff for a folder that is not a repository."}
    changes = list_changes(workspace, session_id, turn=turn)
    commit = (changes.get("checkpoint") or {}).get("commit")
    if not commit:
        return {"path": rel, "diff": "", "message": "No checkpoint yet."}
    proc = _git(
        root, "diff", "--no-ext-diff", "--no-textconv", commit, "--", rel,
    )
    text = proc.stdout or ""
    truncated = False
    if len(text.encode("utf-8", errors="replace")) > _MAX_DIFF_BYTES:
        text = text[:_MAX_DIFF_BYTES]
        truncated = True
    return {"path": rel, "diff": text, "truncated": truncated}


def _current_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return _sha(path.read_bytes())
    except OSError:
        return None


def revert(
    workspace: SessionWorkspace,
    session_id: str,
    *,
    paths: list[str] | None = None,
    turn: int | None = None,
) -> dict[str, Any]:
    """Restore files from the pre-turn checkpoint. Conflicts are not overwritten."""
    host = Path(workspace.host_path).resolve()
    state = _load_state(session_id)
    checkpoints = [c for c in state.get("checkpoints") or [] if c.get("kind") == "git"]
    chosen = None
    if turn is not None:
        chosen = next((c for c in checkpoints if c.get("turn") == turn), None)
    elif checkpoints:
        chosen = checkpoints[-1]
    writes: dict[str, Any] = state.get("writes") or {}
    root = repo_root(host)
    targets = paths or []
    if not targets and chosen is None and root is None:
        targets = list(writes.keys())
    if not targets and chosen is not None:
        listed = list_changes(workspace, session_id, turn=chosen.get("turn"))
        targets = [item["path"] for item in listed.get("files") or []]
    restored: list[str] = []
    conflicts: list[dict[str, str]] = []
    for rel in targets:
        rel_norm = rel.replace("\\", "/").lstrip("/")
        if not rel_norm or ".." in Path(rel_norm).parts:
            conflicts.append({"path": rel, "reason": "path is not inside the folder"})
            continue
        target = (host / rel_norm).resolve()
        if not target.is_relative_to(host):
            conflicts.append({"path": rel, "reason": "path is outside the folder"})
            continue
        recorded = writes.get(rel_norm) if isinstance(writes.get(rel_norm), dict) else None
        current = _current_hash(target)
        if recorded and recorded.get("after") and current and current != recorded.get("after"):
            conflicts.append({
                "path": rel_norm,
                "reason": "the file changed after Otto's last edit",
            })
            continue
        if root is not None and chosen and chosen.get("commit"):
            shown = _git_raw(root, "show", f"{chosen['commit']}:{rel_norm}")
            if shown.returncode != 0:
                if target.is_file():
                    target.unlink()
                    restored.append(rel_norm)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(shown.stdout or b"")
            restored.append(rel_norm)
            continue
        pre = _state_path(session_id).parent / session_id / "preimages" / rel_norm
        if pre.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(pre.read_bytes())
            restored.append(rel_norm)
            continue
        conflicts.append({"path": rel_norm, "reason": "no checkpoint for this file"})
    return {"restored": restored, "conflicts": conflicts, "checkpoint": chosen}
