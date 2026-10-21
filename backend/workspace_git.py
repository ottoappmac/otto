"""Checkpoints and diffs for a granted project folder.

Git repos get a snapshot commit under ``refs/otto/checkpoints/<session>/<n>``
before each turn. The snapshot uses a temporary index, so the user's
index, stash, and branches are left alone. The working tree is hashed with
``git hash-object --no-filters`` rather than ``git add``, so a clean filter,
fsmonitor hook, or ``.git/hooks`` script planted in the project cannot run.
Hooks are pointed at ``/dev/null``. Global and system git config are not read.

Revert restores only paths Otto's file tools or shell commands changed in
that turn. Other edits are reported as changed outside Otto and left alone.
Paths are relative to the granted folder, including when the grant is a
subdirectory of a larger repository.

Folders that are not a git repo keep a byte copy of each file the agent
writes, taken before the write. Revert refuses to clobber a file whose
current hash does not match the hash recorded after the agent's write.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from backend.schemas import SessionWorkspace

logger = logging.getLogger(__name__)

_TIMEOUT = 20
_MAX_FILES = 400
_MAX_PREIMAGE = 1_000_000
_MAX_DIFF_BYTES = 200_000

# Command-line config wins over anything in .git/config, including values
# pulled in by include.path. These are the keys that execute a program.
_GIT_HARDENING = (
    "-c", "core.fsmonitor=",
    "-c", "core.hooksPath=/dev/null",
    "-c", "core.sshCommand=",
    "-c", "core.pager=cat",
    "-c", "diff.external=",
    "-c", "protocol.ext.allow=never",
    "-c", "protocol.file.allow=never",
)


def _git_env(index: Path | None = None) -> dict[str, str]:
    env = {
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_PAGER": "cat",
        "GIT_EDITOR": "true",
        "GIT_AUTHOR_NAME": "Otto",
        "GIT_AUTHOR_EMAIL": "otto@localhost",
        "GIT_COMMITTER_NAME": "Otto",
        "GIT_COMMITTER_EMAIL": "otto@localhost",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", ""),
    }
    if index is not None:
        env["GIT_INDEX_FILE"] = str(index)
    return env


def _git(
    repo: Path,
    *args: str,
    index: Path | None = None,
    timeout: int = _TIMEOUT,
    stdin: bytes | None = None,
) -> subprocess.CompletedProcess[str]:
    proc = _git_raw(repo, *args, index=index, timeout=timeout, stdin=stdin)
    return subprocess.CompletedProcess(
        proc.args,
        proc.returncode,
        (proc.stdout or b"").decode("utf-8", errors="replace"),
        (proc.stderr or b"").decode("utf-8", errors="replace"),
    )


def _git_raw(
    repo: Path,
    *args: str,
    index: Path | None = None,
    timeout: int = _TIMEOUT,
    stdin: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    cmd = ["git", "-C", str(repo), *_GIT_HARDENING, "--no-optional-locks", *args]
    return subprocess.run(
        cmd,
        cwd=str(repo),
        env=_git_env(index),
        input=stdin,
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


def _latest_turn(state: dict[str, Any]) -> int:
    checkpoints = state.get("checkpoints") or []
    if not checkpoints:
        return 0
    try:
        return int(checkpoints[-1].get("turn") or 0)
    except (TypeError, ValueError):
        return 0


def _user_index(session_id: str) -> int | None:
    """How many user messages are already saved. The next turn belongs to that index."""
    try:
        from backend.session_manager import load_messages
        rows = load_messages(session_id)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return sum(1 for row in rows if isinstance(row, dict) and row.get("type") == "user")


def _split_z(data: bytes) -> list[str]:
    return [part.decode("utf-8", "surrogateescape") for part in data.split(b"\0") if part]


def _parse_stage_z(data: bytes) -> list[tuple[str, str, str]]:
    """``ls-files --stage -z`` rows as ``(mode, sha, path)``."""
    rows: list[tuple[str, str, str]] = []
    for rec in data.split(b"\0"):
        if not rec or b"\t" not in rec:
            continue
        meta, path_b = rec.split(b"\t", 1)
        parts = meta.decode("ascii", "replace").split()
        if len(parts) < 3 or not path_b:
            continue
        rows.append((parts[0], parts[1], path_b.decode("utf-8", "surrogateescape")))
    return rows


def _under_prefix(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + "/")


def _hash_bytes(root: Path, data: bytes) -> str | None:
    proc = _git(root, "hash-object", "-w", "--stdin", stdin=data)
    if proc.returncode != 0:
        return None
    text = (proc.stdout or "").strip()
    return text or None


def _hash_files(root: Path, paths: list[str]) -> dict[str, str] | str:
    """Blob ids for regular files. ``--no-filters`` skips clean filters.

    Returns an error string when a file that is still there cannot be hashed.
    A file that disappears mid-walk is omitted.
    """
    if not paths:
        return {}
    found: dict[str, str] = {}
    normal = [path for path in paths if "\n" not in path and "\r" not in path]
    weird = [path for path in paths if path not in normal]
    if normal:
        payload = ("\n".join(normal) + "\n").encode("utf-8", "surrogateescape")
        proc = _git_raw(root, "hash-object", "--no-filters", "-w", "--stdin-paths", stdin=payload)
        shas = [line.strip() for line in (proc.stdout or b"").decode("ascii", "replace").splitlines() if line.strip()]
        if proc.returncode == 0 and len(shas) == len(normal):
            found.update(zip(normal, shas))
        else:
            weird = normal + weird
            found = {}
    for path in weird:
        full = root / path
        if not full.is_file() or full.is_symlink():
            continue
        one = _git(root, "hash-object", "--no-filters", "-w", "--", path)
        if one.returncode != 0:
            if full.is_file():
                return (one.stderr or f"could not hash {path}").strip()[:400]
            continue
        sha = (one.stdout or "").strip()
        if not sha:
            return f"could not hash {path}"
        found[path] = sha
    return found


def _index_tree(root: Path) -> tuple[str | None, str]:
    """Hash the working tree, including untracked files, without trusting repo config.

    ``git add`` would run clean filters from ``.gitattributes`` and ``.git/config``.
    Blobs are written with ``hash-object --no-filters`` into a temporary index,
    so the user's index is not touched and no filter or hook runs.
    """
    staged = _git_raw(root, "ls-files", "-z", "--stage")
    if staged.returncode != 0:
        return None, (staged.stderr or b"git ls-files failed").decode("utf-8", "replace").strip()[:400]
    others = _git_raw(root, "ls-files", "-z", "--others", "--exclude-standard")
    if others.returncode != 0:
        return None, (others.stderr or b"git ls-files failed").decode("utf-8", "replace").strip()[:400]

    stage_rows = _parse_stage_z(staged.stdout or b"")
    gitlinks: dict[str, str] = {}
    tracked: set[str] = set()
    for mode, sha, path in stage_rows:
        tracked.add(path)
        if mode == "160000":
            gitlinks[path] = sha

    def nested(path: str) -> bool:
        return any(_under_prefix(path, link) for link in gitlinks)

    regular: list[str] = []
    symlinks: list[tuple[str, str]] = []
    modes: dict[str, str] = {}
    seen: set[str] = set()
    candidates = [path for _mode, _sha, path in stage_rows if _mode != "160000"]
    candidates.extend(path for path in _split_z(others.stdout or b"") if path not in tracked)
    for path in candidates:
        if not path or path in seen or nested(path):
            continue
        seen.add(path)
        full = root / path
        try:
            if full.is_symlink():
                symlinks.append((path, os.readlink(full)))
                modes[path] = "120000"
            elif full.is_file():
                regular.append(path)
                modes[path] = "100755" if full.stat().st_mode & 0o111 else "100644"
        except OSError:
            continue

    hashed = _hash_files(root, regular)
    if isinstance(hashed, str):
        return None, hashed
    blobs: dict[str, str] = dict(hashed)
    for path, target in symlinks:
        # hash-object on a symlink path follows the link. Hash the target text.
        sha = _hash_bytes(root, target.encode("utf-8", "surrogateescape"))
        if sha is None:
            return None, f"could not hash symlink {path}"
        blobs[path] = sha

    info: list[bytes] = []
    for path, sha in gitlinks.items():
        link = root / path
        if not link.is_dir():
            continue
        current = _git(link, "rev-parse", "HEAD")
        head = (current.stdout or "").strip() if current.returncode == 0 else ""
        info.append(f"160000 {head or sha} 0\t{path}\n".encode("utf-8", "surrogateescape"))
    for path, sha in blobs.items():
        if "\n" in path or "\t" in path:
            continue
        info.append(f"{modes[path]} {sha} 0\t{path}\n".encode("utf-8", "surrogateescape"))

    with tempfile.TemporaryDirectory(prefix="otto-index-") as tmp:
        index = Path(tmp) / "index"
        updated = _git(root, "update-index", "--index-info", index=index, stdin=b"".join(info))
        if updated.returncode != 0:
            return None, (updated.stderr or "git update-index failed").strip()[:400]
        for path, sha in blobs.items():
            if "\n" not in path and "\t" not in path:
                continue
            one = _git(
                root, "update-index", "--add", "--cacheinfo", modes[path], sha, path, index=index,
            )
            if one.returncode != 0:
                return None, (one.stderr or f"could not index {path}").strip()[:400]
        tree = _git(root, "write-tree", index=index)
        tree_id = (tree.stdout or "").strip()
        if tree.returncode != 0 or not tree_id:
            return None, (tree.stderr or "git write-tree failed").strip()[:400]
        return tree_id, ""


def _scope_prefix(host: Path, root: Path) -> str:
    """Repo-relative path of the granted folder, or ``\"\"`` when it is the repo root."""
    try:
        rel = host.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return ""
    return "" if rel in ("", ".") else rel


def _strip_scope(path: str, prefix: str) -> str | None:
    """Grant-relative path, or None when *path* is outside the grant."""
    if not prefix:
        return path
    head = prefix + "/"
    if path.startswith(head):
        return path[len(head):]
    return None


def _repo_rel(host: Path, root: Path, rel: str) -> str | None:
    """Map a grant-relative path to a repo-relative one, rejecting escapes."""
    rel = rel.replace("\\", "/").lstrip("/")
    if not rel or ".." in Path(rel).parts:
        return None
    prefix = _scope_prefix(host, root)
    return f"{prefix}/{rel}" if prefix else rel


def snapshot_before_turn(workspace: SessionWorkspace, session_id: str) -> dict[str, Any]:
    """Capture the tree before a turn. Never raises; failures are ``ok: False``."""
    if workspace.mode == "read":
        return {"ok": True, "skipped": True, "reason": "read-only folder"}
    host = Path(workspace.host_path)
    if not host.is_dir():
        return {"ok": False, "reason": "mapped folder is missing"}
    root = repo_root(host)
    if root is None:
        state = _load_state(session_id)
        turn = _latest_turn(state) + 1
        record = {
            "id": f"{session_id}:{turn}",
            "turn": turn,
            "kind": "copy",
            "user_index": _user_index(session_id),
        }
        state["checkpoints"].append(record)
        _save_state(session_id, state)
        return {
            "ok": False,
            "reason": "not a git repository",
            "coverage": "shell changes are not checkpointed; file-tool edits keep a private copy",
            "checkpoint": record,
        }
    try:
        return _git_snapshot(root, session_id)
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "checkpoint timed out"}
    except OSError as exc:
        return {"ok": False, "reason": f"checkpoint failed: {exc}"}


def _git_snapshot(root: Path, session_id: str) -> dict[str, Any]:
    state = _load_state(session_id)
    turn = _latest_turn(state) + 1
    ref = f"refs/otto/checkpoints/{session_id}/{turn}"
    tree_id, err = _index_tree(root)
    if not tree_id:
        return {"ok": False, "reason": err or "git write-tree returned no tree"}
    commit = _git(
        root, "commit-tree", tree_id, "-m", f"otto checkpoint {session_id} turn {turn}",
    )
    if commit.returncode != 0:
        return {"ok": False, "reason": (commit.stderr or "git commit-tree failed").strip()[:400]}
    commit_id = (commit.stdout or "").strip()
    if not commit_id:
        return {"ok": False, "reason": "git commit-tree returned no commit"}
    updated = _git(root, "update-ref", ref, commit_id)
    if updated.returncode != 0:
        return {"ok": False, "reason": (updated.stderr or "git update-ref failed").strip()[:400]}
    record = {
        "id": f"{session_id}:{turn}",
        "turn": turn,
        "ref": ref,
        "commit": commit_id,
        "kind": "git",
        "user_index": _user_index(session_id),
    }
    state["checkpoints"].append(record)
    _save_state(session_id, state)
    return {"ok": True, "checkpoint": record}


def _is_file_record(value: Any) -> bool:
    return isinstance(value, dict) and any(key in value for key in ("before", "after", "created"))


def _writes_by_turn(writes: Any) -> dict[str, dict[str, Any]]:
    """Normalize legacy flat ``{path: record}`` maps into ``{turn: {path: record}}``."""
    if not isinstance(writes, dict) or not writes:
        return {}
    if all(_is_file_record(value) for value in writes.values()):
        return {"0": {path: rec for path, rec in writes.items() if isinstance(rec, dict)}}
    out: dict[str, dict[str, Any]] = {}
    for key, value in writes.items():
        if isinstance(value, dict) and not _is_file_record(value):
            out[str(key)] = {path: rec for path, rec in value.items() if isinstance(rec, dict)}
    return out


def record_write(session_id: str, rel: str, before: bytes | None, after: bytes | None) -> None:
    """Remember hashes for a file-tool write so revert can detect a conflict.

    Hashes are stored under the current checkpoint turn. A shell edit in a
    later turn no longer collides with an earlier file-tool hash.
    """
    rel = rel.replace("\\", "/").lstrip("/")
    if not rel or not session_id:
        return
    state = _load_state(session_id)
    turn = str(_latest_turn(state))
    by_turn = _writes_by_turn(state.get("writes"))
    bucket = by_turn.setdefault(turn, {})
    prev = bucket.get(rel) if isinstance(bucket.get(rel), dict) else {}
    if before is None:
        before_hash = prev.get("before") if "before" in prev else None
    else:
        before_hash = _sha(before)
    created = bool(prev.get("created")) or (before is None and "before" not in prev)
    bucket[rel] = {
        "before": before_hash,
        "after": _sha(after) if after is not None else None,
        "created": created,
    }
    state["writes"] = by_turn
    _save_state(session_id, state)


def record_agent_change(
    session_id: str,
    rel: str,
    *,
    after_hash: str | None,
    created: bool = False,
) -> None:
    """Mark a path Otto changed, without inventing a preimage hash.

    Used for shell edits, where the bytes from before the command are not
    kept. A previous file-tool record on this turn keeps its ``created`` flag.
    """
    rel = rel.replace("\\", "/").lstrip("/")
    if not rel or not session_id or ".." in Path(rel).parts:
        return
    state = _load_state(session_id)
    turn = str(_latest_turn(state))
    by_turn = _writes_by_turn(state.get("writes"))
    bucket = by_turn.setdefault(turn, {})
    prev = bucket.get(rel) if isinstance(bucket.get(rel), dict) else {}
    bucket[rel] = {
        "before": prev.get("before"),
        "after": after_hash,
        "created": bool(prev.get("created")) or created,
    }
    state["writes"] = by_turn
    _save_state(session_id, state)


def _agent_paths(state: dict[str, Any], turn: int | None) -> set[str]:
    key = str(turn if turn is not None else _latest_turn(state))
    bucket = _writes_by_turn(state.get("writes")).get(key, {})
    return {path for path in bucket if isinstance(path, str) and path}


def shell_baseline(host: Path) -> str | None:
    """Tree id taken just before a shell command. None when this is not a git repo."""
    root = repo_root(host)
    if root is None:
        return None
    tree_id, err = _index_tree(root)
    if not tree_id:
        logger.info("shell baseline skipped: %s", err or "no tree")
        return None
    return tree_id


def record_shell_changes(session_id: str, host: Path, before_tree: str) -> None:
    """Attribute files a shell command changed inside the granted folder.

    The diff is the tree before the command against the tree after it, so an
    edit the user made earlier in the turn is not treated as the command's
    unless the command touched that file too.
    """
    if not session_id or not before_tree:
        return
    root = repo_root(host)
    if root is None:
        return
    after, err = _index_tree(root)
    if not after:
        logger.info("shell attribution skipped: %s", err or "no tree")
        return
    if after == before_tree:
        return
    files, diff_err = _diff_trees(root, before_tree, after, host)
    if diff_err:
        logger.info("shell attribution diff failed: %s", diff_err)
        return
    for item in files:
        rel = item.get("path") or ""
        if not rel:
            continue
        target = host / rel
        created = item.get("status") == "added"
        if item.get("status") == "deleted" or not (target.is_symlink() or target.exists()):
            record_agent_change(session_id, rel, after_hash=None, created=created)
            continue
        record_agent_change(
            session_id,
            rel,
            after_hash=_current_hash(target),
            created=created,
        )


def _preimage_path(session_id: str, turn: int, rel: str) -> Path:
    return _state_path(session_id).parent / session_id / "preimages" / str(turn) / rel


def capture_preimage(session_id: str, root: Path, target: Path) -> None:
    """Copy a non-git file before the agent overwrites it this turn."""
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
    turn = _latest_turn(_load_state(session_id))
    dest = _preimage_path(session_id, turn, rel)
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
    total = 0
    for item in result.get("files") or []:
        count = _marker_count(host, item.get("path") or "")
        item["debug_markers"] = count
        total += count
    result["debug_markers"] = total
    return result


def _checkpoint_summary(checkpoints: list[Any]) -> list[dict[str, Any]]:
    summary = []
    for item in checkpoints:
        if not isinstance(item, dict):
            continue
        summary.append({
            "id": item.get("id"),
            "turn": item.get("turn"),
            "kind": item.get("kind"),
            "user_index": item.get("user_index"),
            "commit": item.get("commit"),
        })
    return summary


def _chosen_checkpoint(checkpoints: list[Any], turn: int | None) -> dict[str, Any] | None:
    rows = [item for item in checkpoints if isinstance(item, dict)]
    if turn is not None:
        return next((item for item in rows if item.get("turn") == turn), None)
    return rows[-1] if rows else None


_STATUS_NAME = {"A": "added", "M": "modified", "D": "deleted", "T": "modified"}


def _parse_name_status_z(data: bytes) -> dict[str, str]:
    parts = data.split(b"\0")
    if parts and parts[-1] == b"":
        parts.pop()
    statuses: dict[str, str] = {}
    index = 0
    while index + 1 < len(parts):
        letter = parts[index].decode("ascii", errors="replace")[:1]
        path = parts[index + 1].decode("utf-8", "surrogateescape")
        if path:
            statuses[path] = _STATUS_NAME.get(letter, "modified")
        index += 2
    return statuses


def _parse_numstat_z(data: bytes) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for rec in data.split(b"\0"):
        if not rec:
            continue
        text = rec.decode("utf-8", "surrogateescape")
        added, sep, rest = text.partition("\t")
        deleted, sep2, path = rest.partition("\t")
        if not sep or not sep2 or not path:
            continue
        rows.append((added, deleted, path))
    return rows


def _diff_trees(
    root: Path,
    old: str,
    new: str,
    host: Path | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Files that differ between two trees.

    Paths are relative to *host* when it is set, and files outside that folder
    are omitted. ``diff-tree`` itself reports repo-root-relative paths.
    """
    prefix = _scope_prefix(host, root) if host is not None else ""
    status = _git_raw(
        root, "diff-tree", "-z", "-r", "--no-renames", "--name-status", old, new,
    )
    numbers = _git_raw(
        root, "diff-tree", "-z", "-r", "--no-renames", "--numstat", old, new,
    )
    if status.returncode != 0 or numbers.returncode != 0:
        err = (status.stderr or numbers.stderr or b"git diff-tree failed").decode("utf-8", "replace")
        return [], err.strip()[:400]
    statuses = _parse_name_status_z(status.stdout or b"")
    files: list[dict[str, Any]] = []
    for added, deleted, path in _parse_numstat_z(numbers.stdout or b""):
        shown = _strip_scope(path, prefix)
        if shown is None:
            continue
        files.append({
            "path": shown,
            "added": None if added == "-" else int(added or 0),
            "deleted": None if deleted == "-" else int(deleted or 0),
            "status": statuses.get(path, "modified"),
        })
        if len(files) >= _MAX_FILES:
            break
    return files, ""


def _diff_against_checkpoint(root: Path, commit: str, host: Path) -> tuple[list[dict[str, Any]], str]:
    """Files that differ between *commit* and the working tree, including untracked files.

    Both sides are trees of raw blob contents, so a file that was already
    untracked at the checkpoint does not show up unless its bytes changed.
    """
    tree_id, err = _index_tree(root)
    if not tree_id:
        return [], err or "Could not read the working tree."
    return _diff_trees(root, commit, tree_id, host)


def _list_changes(workspace: SessionWorkspace, session_id: str, turn: int | None = None) -> dict[str, Any]:
    host = Path(workspace.host_path)
    root = repo_root(host)
    state = _load_state(session_id)
    checkpoints = state.get("checkpoints") or []
    chosen = _chosen_checkpoint(checkpoints, turn)
    summary = _checkpoint_summary(checkpoints)
    if root is None:
        return {
            "git": False,
            "checkpoint": chosen,
            "checkpoints": summary,
            "files": _nongit_files(state, chosen.get("turn") if chosen else None),
            "message": "This folder is not a git repository. Only file-tool edits can be reverted.",
        }
    if chosen is None or not chosen.get("commit"):
        return {
            "git": True,
            "checkpoint": chosen,
            "checkpoints": summary,
            "files": [],
            "message": "No checkpoint for this chat yet.",
        }
    files, err = _diff_against_checkpoint(root, chosen["commit"], host)
    agent = _agent_paths(state, chosen.get("turn"))
    for item in files:
        item["origin"] = "otto" if item.get("path") in agent else "outside"
    result: dict[str, Any] = {
        "git": True,
        "checkpoint": chosen,
        "checkpoints": summary,
        "files": files,
    }
    if err:
        result["message"] = err
    return result


def _nongit_files(state: dict[str, Any], turn: int | None) -> list[dict[str, Any]]:
    by_turn = _writes_by_turn(state.get("writes"))
    key = str(turn if turn is not None else _latest_turn(state))
    files = []
    for rel, rec in (by_turn.get(key) or {}).items():
        created = bool(isinstance(rec, dict) and rec.get("created"))
        files.append({
            "path": rel,
            "added": None,
            "deleted": None,
            "status": "added" if created else "modified",
            "origin": "otto",
        })
    return files


def file_diff(workspace: SessionWorkspace, session_id: str, rel: str, turn: int | None = None) -> dict[str, Any]:
    host = Path(workspace.host_path)
    root = repo_root(host)
    rel = rel.replace("\\", "/").lstrip("/")
    state = _load_state(session_id)
    chosen = _chosen_checkpoint(state.get("checkpoints") or [], turn)
    if root is None:
        return _nongit_diff(workspace, session_id, rel, chosen)
    commit = (chosen or {}).get("commit")
    if not commit:
        return {"path": rel, "diff": "", "message": "No checkpoint yet."}
    git_rel = _repo_rel(host, root, rel)
    if git_rel is None:
        return {"path": rel, "diff": "", "message": "path is outside the folder"}
    proc = _git(
        root, "diff", "--no-ext-diff", "--no-textconv", "--no-renames", commit, "--", git_rel,
    )
    text = proc.stdout or ""
    truncated = False
    if len(text.encode("utf-8", errors="replace")) > _MAX_DIFF_BYTES:
        text = text[:_MAX_DIFF_BYTES]
        truncated = True
    return {"path": rel, "diff": text, "truncated": truncated, "hunks": 0 if truncated else _hunk_count(text)}


def _hunk_count(diff: str) -> int:
    return sum(1 for line in diff.splitlines() if line.startswith("@@"))


def _nongit_diff(
    workspace: SessionWorkspace,
    session_id: str,
    rel: str,
    chosen: dict[str, Any] | None,
) -> dict[str, Any]:
    host = Path(workspace.host_path).resolve()
    turn = int(chosen.get("turn")) if chosen and chosen.get("turn") is not None else _latest_turn(_load_state(session_id))
    target = host / rel
    pre = _preimage_path(session_id, turn, rel)
    if not pre.is_file():
        legacy = _state_path(session_id).parent / session_id / "preimages" / rel
        pre = legacy if legacy.is_file() else pre
    try:
        old = pre.read_text(encoding="utf-8", errors="replace") if pre.is_file() else ""
        new = target.read_text(encoding="utf-8", errors="replace") if target.is_file() else ""
    except OSError as exc:
        return {"path": rel, "diff": "", "message": str(exc)}
    if not pre.is_file() and not target.is_file():
        return {"path": rel, "diff": "", "message": "No copy of this file was kept."}
    text = "".join(difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{rel}",
        tofile=f"b/{rel}",
    ))
    return {"path": rel, "diff": text, "truncated": False, "hunks": _hunk_count(text)}


def _current_hash(path: Path) -> str | None:
    try:
        if path.is_symlink():
            return _sha(os.readlink(path).encode("utf-8", "surrogateescape"))
        if not path.is_file():
            return None
        return _sha(path.read_bytes())
    except OSError:
        return None


def _guard_record(state: dict[str, Any], rel: str, turn: int | None) -> dict[str, Any] | None:
    """Hash to compare against, only for the checkpoint currently being reverted.

    Older turns keep their own records. A shell edit in a later turn must not
    look like "the user changed this after Otto's edit" when reverting that
    later turn, and reverting an earlier turn is an explicit rollback.
    """
    latest = _latest_turn(state)
    if turn is not None and turn != latest:
        return None
    bucket = _writes_by_turn(state.get("writes")).get(str(latest if turn is None else turn), {})
    rec = bucket.get(rel)
    return rec if isinstance(rec, dict) else None


def _unlink_file(target: Path) -> bool:
    """Remove a file or symlink. Leave directories alone."""
    if target.is_symlink() or target.is_file():
        target.unlink()
        return True
    return False


def _restore_from_commit(root: Path, commit: str, git_path: str, target: Path) -> str:
    """Put one worktree path back to the blob in *commit*.

    Returns ``restored``, ``absent`` when the path is not in the tree, or a
    short reason when the path cannot be put back. ``git cat-file`` reads the
    stored blob, so smudge filters do not run. Symlinks and the executable bit
    come from the tree mode.
    """
    listed = _git(root, "ls-tree", "-z", commit, "--", git_path)
    if listed.returncode != 0:
        return "absent"
    chosen: list[str] | None = None
    for rec in (listed.stdout or "").split("\0"):
        if not rec or "\t" not in rec:
            continue
        meta, _, path = rec.partition("\t")
        if path == git_path:
            chosen = meta.split()
            break
    if not chosen or len(chosen) < 3:
        return "absent"
    mode, kind, sha = chosen[0], chosen[1], chosen[2]
    if mode == "160000" or kind == "commit":
        return "submodule changes are not restored"
    if kind != "blob":
        return "absent"
    blob = _git_raw(root, "cat-file", "blob", sha)
    if blob.returncode != 0:
        return "absent"
    data = blob.stdout or b""
    if target.exists() and target.is_dir() and not target.is_symlink():
        return "a directory is in the way"
    try:
        if target.is_symlink() or target.is_file():
            target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        if mode == "120000":
            os.symlink(data.decode("utf-8", "surrogateescape"), target)
        else:
            target.write_bytes(data)
            os.chmod(target, 0o755 if mode == "100755" else 0o644)
    except OSError as exc:
        return str(exc)
    return "restored"


def revert(
    workspace: SessionWorkspace,
    session_id: str,
    *,
    paths: list[str] | None = None,
    turn: int | None = None,
) -> dict[str, Any]:
    """Restore files from a pre-turn checkpoint. Conflicts are not overwritten.

    *turn* selects the checkpoint. Restoring an earlier turn puts those files
    back to that snapshot, which undoes later turns for the same paths.
    """
    host = Path(workspace.host_path).resolve()
    state = _load_state(session_id)
    checkpoints = [c for c in state.get("checkpoints") or [] if isinstance(c, dict)]
    chosen = _chosen_checkpoint(checkpoints, turn)
    root = repo_root(host)
    targets = list(paths or [])
    if not targets and chosen is not None:
        listed = list_changes(workspace, session_id, turn=chosen.get("turn"))
        targets = [
            item["path"] for item in listed.get("files") or []
            if item.get("origin") != "outside"
        ]
    elif not targets and root is None:
        targets = [item["path"] for item in _nongit_files(state, None)]
    restored: list[str] = []
    conflicts: list[dict[str, str]] = []
    turn_id = chosen.get("turn") if chosen else turn
    for rel in targets:
        rel_norm = rel.replace("\\", "/").lstrip("/")
        if not rel_norm or ".." in Path(rel_norm).parts:
            conflicts.append({"path": rel, "reason": "path is not inside the folder"})
            continue
        # Do not resolve the last component. A symlink has to be replaced,
        # not followed, and resolve() would write through it.
        target = host.joinpath(*Path(rel_norm).parts)
        try:
            parent = target.parent.resolve()
            inside = parent == host or parent.is_relative_to(host)
        except OSError:
            inside = False
        if not inside:
            conflicts.append({"path": rel, "reason": "path is outside the folder"})
            continue
        if root is not None and rel_norm not in _agent_paths(state, turn_id):
            conflicts.append({"path": rel_norm, "reason": "changed outside Otto"})
            continue
        recorded = _guard_record(state, rel_norm, turn_id)
        current = _current_hash(target)
        if recorded is not None and current != recorded.get("after"):
            conflicts.append({
                "path": rel_norm,
                "reason": "the file changed after Otto's last edit",
            })
            continue
        if root is not None and chosen and chosen.get("commit"):
            git_rel = _repo_rel(host, root, rel_norm)
            if git_rel is None:
                conflicts.append({"path": rel_norm, "reason": "path is outside the folder"})
                continue
            outcome = _restore_from_commit(root, chosen["commit"], git_rel, target)
            if outcome == "restored":
                restored.append(rel_norm)
            elif outcome == "absent":
                if _unlink_file(target):
                    restored.append(rel_norm)
            else:
                conflicts.append({"path": rel_norm, "reason": outcome})
            continue
        turn_num = int(chosen.get("turn")) if chosen and chosen.get("turn") is not None else _latest_turn(state)
        pre = _preimage_path(session_id, turn_num, rel_norm)
        if not pre.is_file() and turn_num == 0:
            legacy = _state_path(session_id).parent / session_id / "preimages" / rel_norm
            if legacy.is_file():
                pre = legacy
        if pre.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(pre.read_bytes())
            restored.append(rel_norm)
            continue
        if recorded and recorded.get("created"):
            if target.is_file():
                target.unlink()
            restored.append(rel_norm)
            continue
        conflicts.append({"path": rel_norm, "reason": "no checkpoint for this file"})
    return {"restored": restored, "conflicts": conflicts, "checkpoint": chosen}


def revert_hunk(
    workspace: SessionWorkspace,
    session_id: str,
    *,
    path: str,
    hunk: int,
    turn: int | None = None,
) -> dict[str, Any]:
    """Reverse one hunk of the checkpoint diff. Refuses a hunk that does not apply."""
    host = Path(workspace.host_path).resolve()
    root = repo_root(host)
    if root is None:
        return {"restored": [], "conflicts": [{"path": path, "reason": "hunk undo needs a git repository"}]}
    rel = path.replace("\\", "/").lstrip("/")
    state = _load_state(session_id)
    chosen = _chosen_checkpoint(state.get("checkpoints") or [], turn)
    turn_id = chosen.get("turn") if chosen else turn
    if rel not in _agent_paths(state, turn_id):
        return {"restored": [], "conflicts": [{"path": path, "reason": "changed outside Otto"}]}
    info = file_diff(workspace, session_id, path, turn=turn)
    if info.get("truncated"):
        return {"restored": [], "conflicts": [{"path": path, "reason": "the diff is truncated"}]}
    diff = info.get("diff") or ""
    patch = _hunk_patch(diff, hunk)
    if patch is None:
        return {"restored": [], "conflicts": [{"path": path, "reason": "that hunk is not in the diff"}]}
    with tempfile.NamedTemporaryFile(prefix="otto-hunk-", suffix=".patch", delete=False) as handle:
        handle.write(patch.encode("utf-8"))
        patch_path = handle.name
    try:
        check = _git(root, "apply", "--reverse", "--check", "--whitespace=nowarn", patch_path)
        if check.returncode != 0:
            reason = (check.stderr or "the hunk does not apply").strip()[:300]
            return {"restored": [], "conflicts": [{"path": path, "reason": reason or "the hunk does not apply"}]}
        applied = _git(root, "apply", "--reverse", "--whitespace=nowarn", patch_path)
        if applied.returncode != 0:
            reason = (applied.stderr or "the hunk does not apply").strip()[:300]
            return {"restored": [], "conflicts": [{"path": path, "reason": reason or "the hunk does not apply"}]}
    finally:
        Path(patch_path).unlink(missing_ok=True)
    rel = path.replace("\\", "/").lstrip("/")
    _refresh_after_hash(session_id, rel, host / rel)
    return {"restored": [rel], "conflicts": [], "hunk": hunk}


def _hunk_patch(diff: str, hunk: int) -> str | None:
    """A one-hunk patch, keeping the file header ``git apply`` needs."""
    lines = diff.splitlines(keepends=True)
    header: list[str] = []
    hunks: list[list[str]] = []
    current: list[str] | None = None
    for line in lines:
        if line.startswith("@@"):
            if current is not None:
                hunks.append(current)
            current = [line]
        elif current is not None:
            current.append(line)
        else:
            header.append(line)
    if current is not None:
        hunks.append(current)
    if hunk < 0 or hunk >= len(hunks):
        return None
    body = header + hunks[hunk]
    text = "".join(body)
    if not text.endswith("\n"):
        text += "\n"
    return text


def _refresh_after_hash(session_id: str, rel: str, target: Path) -> None:
    """A partial undo means the recorded post-edit hash is no longer current."""
    state = _load_state(session_id)
    by_turn = _writes_by_turn(state.get("writes"))
    bucket = by_turn.get(str(_latest_turn(state)))
    if not bucket or rel not in bucket or not isinstance(bucket[rel], dict):
        return
    bucket[rel]["after"] = _current_hash(target)
    state["writes"] = by_turn
    _save_state(session_id, state)


def forget_session(session_id: str, host: Path | None) -> None:
    """Drop checkpoint refs in the user's repo and the private checkpoint state.

    Refs left behind keep every snapshotted blob reachable, so ``git gc``
    cannot collect them.
    """
    if host is not None:
        try:
            root = repo_root(host)
        except (OSError, subprocess.TimeoutExpired):
            root = None
        if root is not None:
            prefix = f"refs/otto/checkpoints/{session_id}"
            try:
                listed = _git(root, "for-each-ref", "--format=%(refname)", prefix)
            except (OSError, subprocess.TimeoutExpired):
                listed = None
            if listed is not None and listed.returncode == 0:
                for ref in (listed.stdout or "").splitlines():
                    ref = ref.strip()
                    if ref.startswith(prefix):
                        try:
                            _git(root, "update-ref", "-d", ref)
                        except (OSError, subprocess.TimeoutExpired):
                            logger.debug("could not delete %s", ref, exc_info=True)
    try:
        _state_path(session_id).unlink(missing_ok=True)
    except OSError:
        logger.debug("could not delete workspace state for %s", session_id, exc_info=True)
    pre = _state_path(session_id).parent / session_id / "preimages"
    if pre.is_dir():
        shutil.rmtree(pre, ignore_errors=True)
