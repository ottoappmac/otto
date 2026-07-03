"""Custom triggers — polling-based "fire an agent when X happens" rules.

Two trigger types are supported today:

* ``fileos``    — watch a path on the filesystem.  Sub-modes:
                  ``mtime`` / ``size`` / ``exists`` / ``new_files``.
* ``macostool`` — periodically run an osascript snippet and react when
                  its stdout changes (optionally gated by a regex match).

This module is a sibling of :mod:`backend.scheduler` and follows the
same patterns:

* JSON files under ``<app_data>/triggers/<id>/`` are the source of truth.
* The shared :class:`AsyncIOScheduler` instance owned by ``scheduler.py``
  hosts both cron schedule jobs and interval trigger jobs — one
  scheduler instance, two job-id namespaces (``trigger:<id>``).
* When a trigger fires, a fresh :class:`Session` is spawned for the
  configured agent with ``trigger_source="trigger"`` and ``trigger_id``
  set so History can link sessions back to their cause.

Per-trigger watermark state (last mtime / last seen file list / sha256
of last stdout) lives on ``TriggerSpec.state_json`` and is rewritten
on every poll, so triggers survive backend restarts without re-firing
on already-seen events.
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import logging
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from apscheduler.jobstores.base import JobLookupError
from apscheduler.triggers.interval import IntervalTrigger

from backend.config import AppConfig, get_app_data_dir
from backend.schemas import TriggerRun, TriggerSpec
from backend.utils import is_safe_path_segment

logger = logging.getLogger(__name__)

MAX_TRIGGERS = 5
MIN_POLL_SECONDS = 5
MAX_POLL_SECONDS = 24 * 60 * 60
_MISFIRE_GRACE_SECS = 60

# Maps a TriggerRun terminal status onto the corresponding Session status so
# the Runs page reflects the run's outcome instead of a ghost "running" entry.
_SESSION_STATUS_FOR_RUN = {
    "success": "completed",
    "error": "error",
    "cancelled": "stopped",
}
_VALID_ID_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,62}[A-Za-z0-9]$|^[A-Za-z0-9]$",
)


def validate_trigger_id(trigger_id: str) -> str | None:
    """Return an error message when *trigger_id* isn't a safe filename."""
    if not trigger_id:
        return "Trigger ID must not be empty."
    if not _VALID_ID_RE.match(trigger_id):
        return (
            "Trigger ID must be 1-64 chars using letters, digits, spaces, "
            "hyphens, or underscores, and must start and end with a letter "
            "or digit."
        )
    return None


def validate_poll_seconds(poll_seconds: int) -> str | None:
    if not isinstance(poll_seconds, int):
        return "poll_seconds must be an integer."
    if poll_seconds < MIN_POLL_SECONDS:
        return f"poll_seconds must be >= {MIN_POLL_SECONDS}."
    if poll_seconds > MAX_POLL_SECONDS:
        return f"poll_seconds must be <= {MAX_POLL_SECONDS} (24h)."
    return None


def validate_spec(spec: TriggerSpec) -> str | None:
    """Type-specific validation that the request schema can't express."""
    err = validate_poll_seconds(spec.poll_seconds)
    if err:
        return err
    if not spec.prompt or not spec.prompt.strip():
        return "prompt must be a non-empty string."

    if spec.type == "fileos":
        if not spec.path or not spec.path.strip():
            return "path is required for fileos triggers."
        if spec.watch == "new_files" and not spec.glob:
            return "glob is required when watch=new_files."
    elif spec.type == "macostool":
        if not spec.script or not spec.script.strip():
            return "script is required for macostool triggers."
        if spec.match:
            try:
                re.compile(spec.match)
            except re.error as exc:
                return f"match regex is invalid: {exc}"
    elif spec.type == "http":
        if not spec.url or not spec.url.strip():
            return "url is required for http triggers."
        if spec.http_mode == "json_value" and not spec.json_path:
            return "json_path is required when http_mode=json_value."
        if spec.http_mode == "regex" and not spec.match:
            return "match (regex) is required when http_mode=regex."
        if spec.match:
            try:
                re.compile(spec.match)
            except re.error as exc:
                return f"match regex is invalid: {exc}"
    elif spec.type == "git":
        if not spec.repo_path or not spec.repo_path.strip():
            return "repo_path is required for git triggers."
        if spec.author_filter:
            try:
                re.compile(spec.author_filter)
            except re.error as exc:
                return f"author_filter regex is invalid: {exc}"
    elif spec.type == "shell":
        if not spec.command or not spec.command.strip():
            return "command is required for shell triggers."
        if spec.shell_mode == "regex" and not spec.match:
            return "match (regex) is required when shell_mode=regex."
        if spec.match:
            try:
                re.compile(spec.match)
            except re.error as exc:
                return f"match regex is invalid: {exc}"
    return None


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------


def _triggers_dir() -> Path:
    d = get_app_data_dir() / "triggers"
    d.mkdir(parents=True, exist_ok=True)
    return d


def trigger_dir(trigger_id: str) -> Path:
    # Fail closed on traversal-capable IDs: this helper is the single join
    # point for every trigger filesystem path, including routes that receive
    # the ID straight from the URL.
    if not is_safe_path_segment(trigger_id):
        raise ValueError(f"Unsafe trigger ID: {trigger_id!r}")
    return _triggers_dir() / trigger_id


def runs_dir(trigger_id: str) -> Path:
    return trigger_dir(trigger_id) / "runs"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def load_trigger(trigger_id: str) -> Optional[TriggerSpec]:
    if not is_safe_path_segment(trigger_id):
        return None  # no trigger can exist under an unsafe ID
    path = trigger_dir(trigger_id) / "trigger.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return TriggerSpec.model_validate(data)


def load_all_triggers() -> list[TriggerSpec]:
    base = _triggers_dir()
    triggers: list[TriggerSpec] = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        spec_path = d / "trigger.json"
        if spec_path.is_file():
            try:
                data = json.loads(spec_path.read_text(encoding="utf-8"))
                triggers.append(TriggerSpec.model_validate(data))
            except Exception:
                logger.debug("Skipping corrupt trigger: %s", d.name, exc_info=True)
    return triggers


def save_trigger(spec: TriggerSpec) -> TriggerSpec:
    spec.updated_at = datetime.now(timezone.utc)
    d = trigger_dir(spec.id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "trigger.json").write_text(
        spec.model_dump_json(indent=2), encoding="utf-8",
    )
    return spec


def delete_trigger_files(trigger_id: str) -> bool:
    d = trigger_dir(trigger_id)
    if not d.exists():
        return False
    shutil.rmtree(d, ignore_errors=True)
    return True


# ---------------------------------------------------------------------------
# Run history
# ---------------------------------------------------------------------------


def _save_run(run: TriggerRun) -> None:
    run_dir = runs_dir(run.trigger_id) / run.id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(
        run.model_dump_json(indent=2), encoding="utf-8",
    )


def load_runs(trigger_id: str, limit: int = 20) -> list[TriggerRun]:
    base = runs_dir(trigger_id)
    if not base.exists():
        return []
    runs: list[TriggerRun] = []
    for d in sorted(base.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        run_path = d / "run.json"
        if run_path.is_file():
            try:
                data = json.loads(run_path.read_text(encoding="utf-8"))
                runs.append(TriggerRun.model_validate(data))
            except Exception:
                logger.debug("Skipping corrupt run: %s", d.name, exc_info=True)
        if len(runs) >= limit:
            break
    return runs


def load_runs_paginated(
    trigger_id: str,
    limit: int = 20,
    offset: int = 0,
    after: Optional[datetime] = None,
    before: Optional[datetime] = None,
    status: Optional[str] = None,
) -> tuple[list[TriggerRun], int]:
    """Return a page of runs plus the total count matching the filters."""
    base = runs_dir(trigger_id)
    if not base.exists():
        return [], 0
    all_runs: list[TriggerRun] = []
    for d in sorted(base.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        run_path = d / "run.json"
        if not run_path.is_file():
            continue
        try:
            data = json.loads(run_path.read_text(encoding="utf-8"))
            run = TriggerRun.model_validate(data)
            started = run.started_at
            if after and started < after:
                continue
            if before and started > before:
                continue
            if status and run.status != status:
                continue
            all_runs.append(run)
        except Exception:
            logger.debug("Skipping corrupt run: %s", d.name, exc_info=True)
    total = len(all_runs)
    return all_runs[offset:offset + limit], total


def _prune_old_runs(trigger_id: str, keep_last_n: int) -> None:
    base = runs_dir(trigger_id)
    if not base.exists():
        return
    run_dirs = sorted(d for d in base.iterdir() if d.is_dir())
    to_delete = run_dirs[:-keep_last_n] if len(run_dirs) > keep_last_n else []
    for old_run in to_delete:
        shutil.rmtree(old_run, ignore_errors=True)


def _fix_orphaned_runs(trigger_id: str) -> None:
    """Mark any 'running' runs as cancelled when their task is gone."""
    for run in load_runs(trigger_id, limit=5):
        if run.status == "running":
            run.status = "cancelled"
            run.error = "Interrupted — server restarted while running"
            run.finished_at = datetime.now(timezone.utc)
            _save_run(run)
            logger.info("Reset orphaned run %s for trigger %s", run.id, trigger_id)


# ---------------------------------------------------------------------------
# Condition checks
# ---------------------------------------------------------------------------


def _check_fileos(spec: TriggerSpec) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    """Evaluate a fileos trigger.

    Returns ``(fired, new_state, event_payload)`` so callers can persist
    ``new_state`` (the watermark for next poll) regardless of whether
    the trigger fired this round, and forward ``event_payload`` to the
    spawned session as context.
    """
    raw_path = (spec.path or "").strip()
    expanded = Path(raw_path).expanduser()
    state = dict(spec.state_json or {})
    event: dict[str, Any] = {"path": str(expanded), "watch": spec.watch}

    if spec.watch == "exists":
        existed_now = expanded.exists()
        prev = state.get("existed")
        new_state = {"existed": existed_now}
        if prev is None:
            return False, new_state, event
        fired = existed_now != prev
        event["existed_before"] = prev
        event["existed_now"] = existed_now
        return fired, new_state, event

    if spec.watch == "mtime":
        if not expanded.exists():
            return False, {"mtime": None}, event
        st = expanded.stat()
        mt = st.st_mtime
        prev = state.get("mtime")
        new_state = {"mtime": mt}
        if prev is None:
            return False, new_state, event
        fired = mt > float(prev)
        event["mtime_before"] = prev
        event["mtime_now"] = mt
        return fired, new_state, event

    if spec.watch == "size":
        if not expanded.exists():
            return False, {"size": None}, event
        sz = expanded.stat().st_size
        prev = state.get("size")
        new_state = {"size": sz}
        if prev is None:
            return False, new_state, event
        fired = sz != int(prev)
        event["size_before"] = prev
        event["size_now"] = sz
        return fired, new_state, event

    if spec.watch == "new_files":
        glob = spec.glob or "*"
        if not expanded.exists() or not expanded.is_dir():
            return False, {"seen": []}, event
        try:
            current = sorted(
                str(p) for p in expanded.iterdir()
                if p.is_file() and fnmatch.fnmatch(p.name, glob)
            )
        except OSError:
            return False, dict(state), event
        prev_seen = list(state.get("seen") or [])
        new_paths = [p for p in current if p not in prev_seen]
        new_state = {"seen": current}
        if not prev_seen and new_paths:
            return False, new_state, event
        if not new_paths:
            return False, new_state, event
        event["new_paths"] = new_paths
        event["glob"] = glob
        return True, new_state, event

    return False, dict(state), event


async def _check_macostool(
    spec: TriggerSpec,
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    """Run osascript and decide whether the result implies a fire."""
    script = spec.script or ""
    state = dict(spec.state_json or {})
    event: dict[str, Any] = {"language": spec.language}

    try:
        proc = await asyncio.create_subprocess_exec(
            "osascript", "-l", spec.language, "-e", script,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        event["error"] = "osascript not found (non-macOS host?)"
        return False, dict(state), event

    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(), timeout=min(spec.poll_seconds, 60),
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        event["error"] = "osascript timed out"
        return False, dict(state), event

    stdout = (stdout_b or b"").decode("utf-8", errors="replace")
    stderr = (stderr_b or b"").decode("utf-8", errors="replace")
    exit_code = proc.returncode if proc.returncode is not None else -1

    h = hashlib.sha256(stdout.encode("utf-8")).hexdigest()
    prev = state.get("last_stdout_hash")
    new_state = {"last_stdout_hash": h}
    event["stdout"] = stdout[:4096]
    event["exit_code"] = exit_code
    if stderr:
        event["stderr"] = stderr[:1024]

    if exit_code != 0:
        return False, new_state, event

    if spec.match:
        try:
            if not re.search(spec.match, stdout):
                return False, new_state, event
            event["match_regex"] = spec.match
        except re.error:
            return False, new_state, event

    if prev is None:
        return False, new_state, event
    return (h != prev), new_state, event


def _json_path_lookup(payload: Any, dotted: str) -> Any:
    """Walk ``dotted`` (e.g. ``data.items.0.id``) into ``payload``.

    Returns ``None`` if any key/index is missing or types don't line up.
    Designed to be defensive — never raises, never bubbles AttributeError.
    """
    cur = payload
    for part in dotted.split("."):
        if cur is None:
            return None
        if isinstance(cur, list):
            try:
                idx = int(part)
            except ValueError:
                return None
            if idx < 0 or idx >= len(cur):
                return None
            cur = cur[idx]
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


async def _check_http(
    spec: TriggerSpec,
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    """Poll an HTTP endpoint and decide whether to fire.

    Four sub-modes — all share the same request/response cycle and only
    differ in *what* is compared against the previous watermark:

    * ``status_change`` — fires when HTTP status code changes
    * ``body_hash``     — fires when the response body sha256 changes
    * ``json_value``    — fires when ``json_path`` extracts a different value
    * ``regex``         — fires when ``match`` regex appears (and didn't last poll)
    """
    import httpx

    url = (spec.url or "").strip()
    state = dict(spec.state_json or {})
    event: dict[str, Any] = {
        "url": url,
        "method": spec.method,
        "mode": spec.http_mode,
    }

    if not url:
        event["error"] = "url is required"
        return False, dict(state), event

    timeout = float(min(spec.poll_seconds, 60))
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.request(
                spec.method,
                url,
                headers=spec.headers or None,
                content=spec.body if spec.body else None,
            )
    except httpx.HTTPError as exc:
        event["error"] = f"request failed: {exc!s}"
        return False, dict(state), event
    except Exception as exc:  # network errors, DNS, etc.
        event["error"] = f"unexpected error: {exc!s}"
        return False, dict(state), event

    body = resp.text or ""
    status = int(resp.status_code)
    event["status_code"] = status
    event["body_preview"] = body[:4096]

    if spec.http_mode == "status_change":
        prev = state.get("last_status")
        new_state = {**state, "last_status": status}
        if prev is None:
            return False, new_state, event
        event["old_status"] = int(prev)
        return (status != int(prev)), new_state, event

    if spec.http_mode == "body_hash":
        h = hashlib.sha256(body.encode("utf-8")).hexdigest()
        prev = state.get("last_hash")
        new_state = {**state, "last_hash": h, "last_status": status}
        if prev is None:
            return False, new_state, event
        return (h != prev), new_state, event

    if spec.http_mode == "json_value":
        if not spec.json_path:
            event["error"] = "json_path is required for json_value mode"
            return False, dict(state), event
        try:
            parsed = json.loads(body) if body else None
        except json.JSONDecodeError as exc:
            event["error"] = f"response is not valid JSON: {exc!s}"
            return False, dict(state), event
        value = _json_path_lookup(parsed, spec.json_path)
        prev = state.get("last_value")
        new_state = {**state, "last_value": value, "last_status": status}
        event["json_path"] = spec.json_path
        event["matched_value"] = value
        if "last_value" not in state:
            return False, new_state, event
        event["old_value"] = prev
        return (value != prev), new_state, event

    if spec.http_mode == "regex":
        if not spec.match:
            event["error"] = "match (regex) is required for regex mode"
            return False, dict(state), event
        try:
            present_now = bool(re.search(spec.match, body))
        except re.error as exc:
            event["error"] = f"invalid regex: {exc!s}"
            return False, dict(state), event
        prev = bool(state.get("last_match", False))
        new_state = {**state, "last_match": present_now, "last_status": status}
        event["match_regex"] = spec.match
        event["match_present"] = present_now
        if "last_match" not in state:
            return False, new_state, event
        # Fire on rising edge (became present this poll).
        return (present_now and not prev), new_state, event

    return False, dict(state), event


async def _check_git(
    spec: TriggerSpec,
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    """Poll a local git repo for new commits on a branch.

    Uses ``git log <last_sha>..<branch>`` to enumerate new commits since
    the last watermark.  On first run only the current tip is recorded
    (no fire) so the trigger doesn't replay history.
    """
    repo = (spec.repo_path or "").strip()
    branch = spec.branch or "HEAD"
    state = dict(spec.state_json or {})
    expanded = Path(repo).expanduser()
    event: dict[str, Any] = {"repo_path": str(expanded), "branch": branch}

    if not repo:
        event["error"] = "repo_path is required"
        return False, dict(state), event
    if not expanded.exists() or not (expanded / ".git").exists():
        event["error"] = f"not a git repository: {expanded}"
        return False, dict(state), event

    async def _run_git(*args: str) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            "git", "-C", str(expanded), *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out_b, err_b = await asyncio.wait_for(
                proc.communicate(), timeout=min(spec.poll_seconds, 30),
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return -1, "", "git timed out"
        return (
            proc.returncode if proc.returncode is not None else -1,
            (out_b or b"").decode("utf-8", errors="replace"),
            (err_b or b"").decode("utf-8", errors="replace"),
        )

    rc, tip, err = await _run_git("rev-parse", branch)
    if rc != 0:
        event["error"] = f"git rev-parse failed: {err.strip() or 'unknown'}"
        return False, dict(state), event
    tip_sha = tip.strip()
    if not tip_sha:
        event["error"] = "empty rev-parse output"
        return False, dict(state), event

    last_sha = state.get("last_sha")
    new_state = {**state, "last_sha": tip_sha}

    if last_sha is None:
        return False, new_state, event
    if last_sha == tip_sha:
        return False, new_state, event

    log_args = [
        "log",
        f"{last_sha}..{tip_sha}",
        "--pretty=format:%H%x1f%h%x1f%an%x1f%ae%x1f%aI%x1f%s",
        "--no-merges",
    ]
    if spec.author_filter:
        log_args.extend(["--author", spec.author_filter])
    if spec.path_filter:
        log_args.extend(["--", spec.path_filter])

    rc, out, err = await _run_git(*log_args)
    if rc != 0:
        event["error"] = f"git log failed: {err.strip() or 'unknown'}"
        # Still update watermark so we don't keep retrying the same range.
        return False, new_state, event

    new_commits: list[dict[str, Any]] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("\x1f")
        if len(parts) < 6:
            continue
        sha, short, author_name, author_email, iso_date, subject = parts[:6]
        new_commits.append({
            "sha": sha,
            "short_sha": short,
            "author": f"{author_name} <{author_email}>",
            "date": iso_date,
            "message": subject,
        })

    if not new_commits:
        # Filter ate everything — treat as no-fire but keep new tip.
        return False, new_state, event

    event["new_commits"] = new_commits
    event["count"] = len(new_commits)
    return True, new_state, event


async def _check_shell(
    spec: TriggerSpec,
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    """Run an arbitrary shell command and decide whether to fire.

    Three sub-modes:

    * ``stdout_change``     — sha256 of stdout changed since last poll
    * ``regex``             — ``match`` regex now matches stdout (rising edge)
    * ``exit_code_change``  — exit code differs from last poll
    """
    cmd = (spec.command or "").strip()
    state = dict(spec.state_json or {})
    event: dict[str, Any] = {
        "command": cmd,
        "cwd": spec.cwd,
        "mode": spec.shell_mode,
    }

    if not cmd:
        event["error"] = "command is required"
        return False, dict(state), event

    cwd_expanded: Optional[str] = None
    if spec.cwd:
        cwd_expanded = str(Path(spec.cwd).expanduser())

    proc_env = None
    if spec.env:
        import os as _os
        proc_env = {**_os.environ, **spec.env}

    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            cwd=cwd_expanded,
            env=proc_env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as exc:
        event["error"] = f"failed to launch shell: {exc!s}"
        return False, dict(state), event

    try:
        out_b, err_b = await asyncio.wait_for(
            proc.communicate(), timeout=min(spec.poll_seconds, 60),
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        event["error"] = "shell command timed out"
        return False, dict(state), event

    stdout = (out_b or b"").decode("utf-8", errors="replace")
    stderr = (err_b or b"").decode("utf-8", errors="replace")
    exit_code = proc.returncode if proc.returncode is not None else -1
    event["stdout"] = stdout[:4096]
    event["exit_code"] = exit_code
    if stderr:
        event["stderr"] = stderr[:1024]

    if spec.shell_mode == "stdout_change":
        h = hashlib.sha256(stdout.encode("utf-8")).hexdigest()
        prev = state.get("last_stdout_hash")
        new_state = {**state, "last_stdout_hash": h, "last_exit_code": exit_code}
        if prev is None:
            return False, new_state, event
        return (h != prev), new_state, event

    if spec.shell_mode == "exit_code_change":
        prev = state.get("last_exit_code")
        new_state = {**state, "last_exit_code": exit_code}
        if prev is None:
            return False, new_state, event
        event["old_exit_code"] = int(prev)
        return (exit_code != int(prev)), new_state, event

    if spec.shell_mode == "regex":
        if not spec.match:
            event["error"] = "match (regex) is required for regex mode"
            return False, dict(state), event
        try:
            present_now = bool(re.search(spec.match, stdout))
        except re.error as exc:
            event["error"] = f"invalid regex: {exc!s}"
            return False, dict(state), event
        prev = bool(state.get("last_match", False))
        new_state = {**state, "last_match": present_now, "last_exit_code": exit_code}
        event["match_regex"] = spec.match
        event["match_present"] = present_now
        if "last_match" not in state:
            return False, new_state, event
        return (present_now and not prev), new_state, event

    return False, dict(state), event


# ---------------------------------------------------------------------------
# Fire path — spawn an agent session with the event payload
# ---------------------------------------------------------------------------


_running_trigger_tasks: dict[str, asyncio.Task] = {}


def _build_prompt(spec: TriggerSpec, event_payload: dict[str, Any]) -> str:
    """Append the event payload to the user-authored prompt as JSON.

    Keeping the payload as a JSON-fenced block (rather than rewriting the
    prompt) lets the worker agent reliably parse it with ``json.loads``
    and lets the user write the prompt template once without worrying
    about variable interpolation syntax.
    """
    payload_json = json.dumps(
        {"trigger_id": spec.id, "type": spec.type, "event": event_payload},
        indent=2,
    )
    return (
        f"{spec.prompt.rstrip()}\n\n"
        f"Trigger event payload:\n```json\n{payload_json}\n```"
    )


async def _execute_trigger_run(
    spec: TriggerSpec,
    run: TriggerRun,
    timestamp: str,
    event_payload: dict[str, Any],
) -> None:
    """Spawn a session, stream the prompt, record the run."""
    from backend.session_manager import _session_files_dir
    from backend.state import message_queues, running_tasks, session_mgr

    trigger_id = spec.id
    try:
        cfg = await AppConfig.aload()
        session = await session_mgr.create_session(
            config=cfg,
            agent_name=spec.agent_name,
            is_scheduled_run=True,
            schedule_id=None,
            trigger_source="trigger",
            trigger_id=spec.id,
        )
        run.session_id = session.id
        await asyncio.to_thread(_save_run, run)

        current_task = asyncio.current_task()
        if current_task:
            running_tasks[session.id] = current_task

        prompt = _build_prompt(spec, event_payload)

        async def _stream() -> int:
            from backend.routes.sessions import _LazyPersistingSubagentQueue
            from backend.streaming_subagent import (
                reset_subagent_queue,
                set_subagent_queue,
            )

            _NO_USER_ANSWER = (
                "No user is available to respond — this is an automated trigger run. "
                "Handle the situation by making reasonable assumptions, using defaults, "
                "or clearly reporting what is missing and stopping cleanly."
            )

            count = 0
            sent_done = False
            auto_resumes = 0
            # Tracks the decisions to send on the next resume; set inside _drain.
            _next_decisions: list[dict] = []
            token = set_subagent_queue(_LazyPersistingSubagentQueue(session.id))

            async def _drain(stream_iter) -> bool:
                """Drain *stream_iter* into the session queue.

                Returns True when a terminal event (done) was sent, False when
                the stream ended on an interrupt that was auto-handled (caller
                should resume with _next_decisions).
                """
                nonlocal count, auto_resumes, _next_decisions
                async for resp in stream_iter:
                    count += 1
                    rtype = resp.get("type")
                    q = message_queues.get(session.id)
                    if q is not None:
                        await q.put(resp)
                    if rtype == "done":
                        return True
                    if rtype == "ask_user":
                        # No human present: always auto-answer and continue.
                        auto_resumes += 1
                        logger.info(
                            "Trigger run %s: auto-resuming ask_user (attempt %d)",
                            trigger_id, auto_resumes,
                        )
                        _next_decisions = [{"answer": _NO_USER_ANSWER}]
                        return False
                    if rtype == "hitl_request":
                        # No human present: auto-approve execute commands.
                        auto_resumes += 1
                        logger.info(
                            "Trigger run %s: auto-approving hitl_request (attempt %d)",
                            trigger_id, auto_resumes,
                        )
                        _next_decisions = [{"type": "approve"}]
                        return False
                return True

            try:
                terminal = await _drain(session_mgr.stream_message(session.id, prompt))
                while not terminal:
                    terminal = await _drain(
                        session_mgr.stream_resume(
                            session.id,
                            decisions=_next_decisions,
                        )
                    )
                sent_done = True
            finally:
                reset_subagent_queue(token)
                if not sent_done:
                    q = message_queues.get(session.id)
                    if q is not None:
                        await q.put({"type": "done", "content": ""})
            return count

        message_count = await asyncio.wait_for(_stream(), timeout=spec.timeout_seconds)

        run.status = "success"
        run.message_count = message_count

        def _copy_session_output() -> None:
            session_files = _session_files_dir(session.id)
            run_dir = runs_dir(trigger_id) / timestamp
            run_dir.mkdir(parents=True, exist_ok=True)
            if session_files.exists() and any(session_files.iterdir()):
                shutil.copytree(session_files, run_dir / "files", dirs_exist_ok=True)

        await asyncio.to_thread(_copy_session_output)

    except asyncio.CancelledError:
        logger.info("Trigger run %s/%s cancelled by user", trigger_id, timestamp)
        run.status = "cancelled"
        run.error = "Cancelled by user"
    except asyncio.TimeoutError:
        logger.error(
            "Trigger run %s/%s timed out after %ds",
            trigger_id, timestamp, spec.timeout_seconds,
        )
        run.status = "error"
        run.error = f"Timed out after {spec.timeout_seconds}s"
    except Exception as exc:
        logger.error(
            "Trigger run %s/%s failed: %s",
            trigger_id, timestamp, exc, exc_info=True,
        )
        run.status = "error"
        run.error = str(exc)
    finally:
        if run.session_id:
            running_tasks.pop(run.session_id, None)
            # Mirror the run's terminal outcome onto the session record before
            # closing it. The streaming error/timeout/cancel paths never reach
            # the "completed" stamp in _check_interrupts_or_done, so without
            # this the session meta is persisted with its start-of-run
            # "running" status — leaving a ghost "running" entry on the Runs
            # page even though the trigger itself shows an error.
            live = session_mgr.get_session(run.session_id)
            if live is not None and live.status in ("running", "idle"):
                now = datetime.now(timezone.utc)
                live.status = _SESSION_STATUS_FOR_RUN.get(run.status, "error")
                live.error = run.error
                if live.status == "error":
                    from backend.eval_runner import classify_error_code
                    live.error_code = classify_error_code(run.error or "")
                live.finished_at = now
                live.duration_ms = int((now - live.created_at).total_seconds() * 1000)
            try:
                await session_mgr.close_session(run.session_id)
            except Exception:
                logger.debug(
                    "Failed to close trigger session %s",
                    run.session_id, exc_info=True,
                )
            # Best-effort: analyze a failed run for a prompt fix (gated by
            # evaluation.analyze_errors). Runs after close_session persists meta.
            if run.status == "error":
                await session_mgr.maybe_analyze_error(run.session_id)

        run.finished_at = datetime.now(timezone.utc)
        await asyncio.to_thread(_save_run, run)

        latest = await asyncio.to_thread(load_trigger, trigger_id) or spec
        latest.last_status = run.status
        latest.last_error = run.error
        latest.last_run = run.started_at
        latest.last_event = event_payload
        await asyncio.to_thread(save_trigger, latest)

        await asyncio.to_thread(_prune_old_runs, trigger_id, latest.keep_last_n_runs)
        _running_trigger_tasks.pop(trigger_id, None)


async def _fire(spec: TriggerSpec, event_payload: dict[str, Any]) -> None:
    """Create + record a new TriggerRun, then spawn the executor task."""
    if spec.id in _running_trigger_tasks:
        logger.info(
            "Trigger %s already has an active run — skipping fire", spec.id,
        )
        return

    timestamp = (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        + f"-{uuid.uuid4().hex[:6]}"
    )
    run = TriggerRun(
        id=timestamp,
        trigger_id=spec.id,
        status="running",
        started_at=datetime.now(timezone.utc),
        event_payload=event_payload,
    )
    await asyncio.to_thread(_save_run, run)

    spec.last_run = run.started_at
    spec.last_status = "running"
    spec.last_error = None
    spec.last_event = event_payload
    await asyncio.to_thread(save_trigger, spec)

    task = asyncio.create_task(
        _execute_trigger_run(spec, run, timestamp, event_payload),
        name=f"trigger-{spec.id}",
    )
    _running_trigger_tasks[spec.id] = task


# ---------------------------------------------------------------------------
# Poll entry point — registered with APScheduler
# ---------------------------------------------------------------------------


def _job_id(trigger_id: str) -> str:
    """Namespaced APScheduler job id so triggers don't collide with schedules."""
    return f"trigger:{trigger_id}"


async def _poll_trigger(trigger_id: str, *, force: bool = False) -> None:
    """One poll tick — load spec, evaluate, persist watermark, maybe fire."""
    spec = await asyncio.to_thread(load_trigger, trigger_id)
    if spec is None:
        return
    if not spec.enabled and not force:
        return

    try:
        if spec.type == "fileos":
            fired, new_state, event = await asyncio.to_thread(_check_fileos, spec)
        elif spec.type == "macostool":
            fired, new_state, event = await _check_macostool(spec)
        elif spec.type == "http":
            fired, new_state, event = await _check_http(spec)
        elif spec.type == "git":
            fired, new_state, event = await _check_git(spec)
        elif spec.type == "shell":
            fired, new_state, event = await _check_shell(spec)
        else:
            logger.warning("Unknown trigger type %r for %s", spec.type, trigger_id)
            return
    except Exception:
        logger.warning("Trigger %s evaluation failed", trigger_id, exc_info=True)
        return

    if force:
        fired = True
        # When forced (manual "Run Now"), the diff-based check may have returned
        # an empty event because all files were already seen or because it was a
        # first-time seed.  Backfill the relevant payload field so the triggered
        # agent actually knows what to process.
        if spec.type == "fileos" and spec.watch == "new_files" and "new_paths" not in event:
            _glob = spec.glob or "*"
            _expanded = Path(spec.path or "").expanduser()
            if _expanded.is_dir():
                try:
                    event["new_paths"] = sorted(
                        str(p)
                        for p in _expanded.iterdir()
                        if p.is_file() and fnmatch.fnmatch(p.name, _glob)
                    )
                    event["glob"] = _glob
                except OSError:
                    event["new_paths"] = []
        elif spec.type == "git" and "new_commits" not in event and "error" not in event:
            # First-time force on a git trigger seeds last_sha but doesn't list
            # commits.  Pull the most recent few so the agent has context.
            _expanded = Path(spec.repo_path or "").expanduser()
            if _expanded.exists() and (_expanded / ".git").exists():
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "git", "-C", str(_expanded), "log",
                        "-5",
                        "--pretty=format:%H%x1f%h%x1f%an%x1f%ae%x1f%aI%x1f%s",
                        "--no-merges", spec.branch or "HEAD",
                        stdin=asyncio.subprocess.DEVNULL,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    out_b, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
                    out = (out_b or b"").decode("utf-8", errors="replace")
                    commits: list[dict[str, Any]] = []
                    for line in out.splitlines():
                        parts = line.split("\x1f")
                        if len(parts) >= 6:
                            commits.append({
                                "sha": parts[0],
                                "short_sha": parts[1],
                                "author": f"{parts[2]} <{parts[3]}>",
                                "date": parts[4],
                                "message": parts[5],
                            })
                    if commits:
                        event["new_commits"] = commits
                        event["count"] = len(commits)
                        event["note"] = "forced run — showing last 5 commits"
                except (asyncio.TimeoutError, FileNotFoundError, OSError):
                    pass

    spec.state_json = new_state
    await asyncio.to_thread(save_trigger, spec)

    if fired:
        await _fire(spec, event)


# ---------------------------------------------------------------------------
# APScheduler integration — reuses the scheduler.py singleton
# ---------------------------------------------------------------------------


def register_job(trigger_id: str, poll_seconds: int) -> None:
    """Add or replace the APScheduler interval job for *trigger_id*."""
    from backend.scheduler import get_scheduler

    scheduler = get_scheduler()
    scheduler.add_job(
        _poll_trigger,
        trigger=IntervalTrigger(seconds=poll_seconds),
        args=[trigger_id],
        id=_job_id(trigger_id),
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=_MISFIRE_GRACE_SECS,
    )


def remove_job(trigger_id: str) -> None:
    from backend.scheduler import get_scheduler

    scheduler = get_scheduler()
    try:
        scheduler.remove_job(_job_id(trigger_id))
    except JobLookupError:
        pass


def run_trigger_immediately(trigger_id: str) -> None:
    """Queue a one-off poll that bypasses the ``enabled`` check and the
    "did the watermark change" gate — fires the agent unconditionally."""
    from backend.scheduler import get_scheduler

    scheduler = get_scheduler()
    scheduler.add_job(
        _poll_trigger,
        args=[trigger_id],
        kwargs={"force": True},
        id=f"{_job_id(trigger_id)}-manual",
        replace_existing=True,
        max_instances=1,
    )


def stop_trigger_run(trigger_id: str) -> bool:
    task = _running_trigger_tasks.get(trigger_id)
    if task and not task.done():
        task.cancel()
        return True
    return False


def is_trigger_running(trigger_id: str) -> bool:
    task = _running_trigger_tasks.get(trigger_id)
    return task is not None and not task.done()


def init_trigger_manager() -> None:
    """Register every enabled trigger as an APScheduler job.

    Must be called AFTER :func:`backend.scheduler.init_scheduler` has
    started the shared scheduler instance.
    """
    from backend.managed_triggers import seed_managed_triggers
    seed_managed_triggers()

    for spec in load_all_triggers():
        if spec.last_status == "running":
            logger.info(
                "Resetting stale 'running' status for trigger %s", spec.id,
            )
            spec.last_status = "error"
            spec.last_error = "Interrupted — server restarted while running"
            save_trigger(spec)
            _fix_orphaned_runs(spec.id)
        if spec.enabled:
            try:
                register_job(spec.id, spec.poll_seconds)
                logger.info(
                    "Registered trigger: %s (every %ds, type=%s)",
                    spec.id, spec.poll_seconds, spec.type,
                )
            except Exception:
                logger.warning(
                    "Failed to register trigger %s", spec.id, exc_info=True,
                )

    logger.info("Trigger manager initialised")
