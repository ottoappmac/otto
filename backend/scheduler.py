"""Scheduled agent runs — APScheduler integration with persistent storage."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from apscheduler.jobstores.base import JobLookupError
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from backend.config import AppConfig, get_app_data_dir
from backend.schemas import ScheduleRun, ScheduleSpec
from backend.utils import is_safe_path_segment

logger = logging.getLogger(__name__)

MAX_SCHEDULES = 10
_MISFIRE_GRACE_SECS = 3600  # allow jobs to run up to 1 hour late

# Maps a ScheduleRun terminal status onto the corresponding Session status so
# the Runs page reflects the run's outcome instead of a ghost "running" entry.
_SESSION_STATUS_FOR_RUN = {
    "success": "completed",
    "error": "error",
    "cancelled": "stopped",
}
_VALID_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,62}[A-Za-z0-9]$|^[A-Za-z0-9]$")


def validate_schedule_id(schedule_id: str) -> str | None:
    """Return an error message if the ID is invalid, or None if OK."""
    if not schedule_id:
        return "Schedule ID must not be empty."
    if not _VALID_ID_RE.match(schedule_id):
        return (
            "Schedule ID must be 1-64 chars using letters, digits, spaces, hyphens, "
            "or underscores, and must start and end with a letter or digit."
        )


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def _schedules_dir() -> Path:
    d = get_app_data_dir() / "schedules"
    d.mkdir(parents=True, exist_ok=True)
    return d


def schedule_dir(schedule_id: str) -> Path:
    # Fail closed on traversal-capable IDs: this helper is the single join
    # point for every schedule filesystem path, including routes and agent
    # tools that receive the ID from user/LLM input.
    if not is_safe_path_segment(schedule_id):
        raise ValueError(f"Unsafe schedule ID: {schedule_id!r}")
    return _schedules_dir() / schedule_id


def runs_dir(schedule_id: str) -> Path:
    return schedule_dir(schedule_id) / "runs"


def attachments_dir(schedule_id: str) -> Path:
    # schedule_dir() applies the is_safe_path_segment guard, so this is the
    # single join point for every attachment filesystem path.
    return schedule_dir(schedule_id) / "attachments"


def list_attachments(schedule_id: str) -> list[dict]:
    """Return ``[{path, size}]`` for every attachment file of a schedule."""
    base = attachments_dir(schedule_id)
    if not base.exists():
        return []
    return [
        {"path": fp.relative_to(base).as_posix(), "size": fp.stat().st_size}
        for fp in sorted(base.rglob("*"))
        if fp.is_file()
    ]


def _link_attachments_into_session(schedule_id: str, files_dir: Path) -> list[str]:
    """Symlink each schedule attachment into the session files dir.

    Each attachment is linked individually under ``files_dir/attachments`` so
    the agent reads the single schedule-owned copy via the virtual path
    ``/attachments/...`` while any *new* file it writes there lands in the
    session dir (never the schedule dir). Returns the list of virtual paths
    that were linked.
    """
    base = attachments_dir(schedule_id)
    if not base.exists():
        return []
    virtual_paths: list[str] = []
    for src in sorted(base.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(base)
        dest = files_dir / "attachments" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            try:
                dest.symlink_to(src)
            except OSError:
                logger.warning(
                    "Failed to symlink attachment %s for schedule %s",
                    rel, schedule_id, exc_info=True,
                )
                continue
        virtual_paths.append("/attachments/" + rel.as_posix())
    return virtual_paths


# ---------------------------------------------------------------------------
# Schedule persistence (JSON files — no extra DB required)
# ---------------------------------------------------------------------------

def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically.

    Concurrent readers must never observe a half-written file: a refresh that
    races a ``save_schedule`` would otherwise read truncated JSON, fail to
    parse, and silently drop the schedule from the list. Writing to a temp
    file in the same directory and renaming makes the swap atomic.
    """
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_schedule(schedule_id: str) -> Optional[ScheduleSpec]:
    if not is_safe_path_segment(schedule_id):
        return None  # no schedule can exist under an unsafe ID
    path = schedule_dir(schedule_id) / "schedule.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return ScheduleSpec.model_validate(data)


def load_all_schedules() -> list[ScheduleSpec]:
    base = _schedules_dir()
    schedules: list[ScheduleSpec] = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        spec_path = d / "schedule.json"
        if spec_path.is_file():
            try:
                data = json.loads(spec_path.read_text(encoding="utf-8"))
                schedules.append(ScheduleSpec.model_validate(data))
            except Exception:
                logger.debug("Skipping corrupt schedule: %s", d.name, exc_info=True)
    return schedules


def save_schedule(spec: ScheduleSpec) -> ScheduleSpec:
    spec.updated_at = datetime.now(timezone.utc)
    d = schedule_dir(spec.id)
    d.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(d / "schedule.json", spec.model_dump_json(indent=2))
    return spec


def delete_schedule_files(schedule_id: str) -> bool:
    d = schedule_dir(schedule_id)
    if not d.exists():
        return False
    shutil.rmtree(d, ignore_errors=True)
    return True


# ---------------------------------------------------------------------------
# Run history persistence
# ---------------------------------------------------------------------------

def _save_run(run: ScheduleRun) -> None:
    run_dir = runs_dir(run.schedule_id) / run.id
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(run_dir / "run.json", run.model_dump_json(indent=2))


def load_runs(schedule_id: str, limit: int = 20) -> list[ScheduleRun]:
    base = runs_dir(schedule_id)
    if not base.exists():
        return []
    runs: list[ScheduleRun] = []
    for d in sorted(base.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        run_path = d / "run.json"
        if run_path.is_file():
            try:
                data = json.loads(run_path.read_text(encoding="utf-8"))
                runs.append(ScheduleRun.model_validate(data))
            except Exception:
                logger.debug("Skipping corrupt run: %s", d.name, exc_info=True)
        if len(runs) >= limit:
            break
    return runs


def load_runs_paginated(
    schedule_id: str,
    limit: int = 20,
    offset: int = 0,
    after: Optional[datetime] = None,
    before: Optional[datetime] = None,
    status: Optional[str] = None,
    search: Optional[str] = None,
) -> tuple[list[ScheduleRun], int]:
    """Return a page of runs plus the total count matching the filters."""
    base = runs_dir(schedule_id)
    if not base.exists():
        return [], 0
    q = search.strip().lower() if search else None
    all_runs: list[ScheduleRun] = []
    for d in sorted(base.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        run_path = d / "run.json"
        if not run_path.is_file():
            continue
        try:
            data = json.loads(run_path.read_text(encoding="utf-8"))
            run = ScheduleRun.model_validate(data)
            started = run.started_at
            if after and started < after:
                continue
            if before and started > before:
                continue
            if status and run.status != status:
                continue
            if q and q not in " ".join(
                filter(None, [run.id, run.session_id, run.status, run.error])
            ).lower():
                continue
            all_runs.append(run)
        except Exception:
            logger.debug("Skipping corrupt run: %s", d.name, exc_info=True)
    total = len(all_runs)
    return all_runs[offset:offset + limit], total


def compute_run_stats(
    schedule_id: str,
    after: Optional[datetime] = None,
    before: Optional[datetime] = None,
    status: Optional[str] = None,
    search: Optional[str] = None,
) -> dict:
    """Aggregate stats over a schedule's runs matching the given filters.

    ``running_now`` reflects all currently-running runs for the schedule
    (ignoring filters) so it stays a live indicator, mirroring the Runs page.
    """
    base = runs_dir(schedule_id)
    empty = {
        "total": 0,
        "success_rate": 0.0,
        "avg_duration_ms": None,
        "total_steps": 0,
        "avg_steps_per_run": None,
        "running_now": 0,
    }
    if not base.exists():
        return empty

    q = search.strip().lower() if search else None
    matched: list[ScheduleRun] = []
    running_now = 0
    for d in sorted(base.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        run_path = d / "run.json"
        if not run_path.is_file():
            continue
        try:
            run = ScheduleRun.model_validate(json.loads(run_path.read_text(encoding="utf-8")))
        except Exception:
            logger.debug("Skipping corrupt run: %s", d.name, exc_info=True)
            continue
        if run.status == "running":
            running_now += 1
        started = run.started_at
        if after and started < after:
            continue
        if before and started > before:
            continue
        if status and run.status != status:
            continue
        if q and q not in " ".join(
            filter(None, [run.id, run.session_id, run.status, run.error])
        ).lower():
            continue
        matched.append(run)

    total = len(matched)
    success = sum(1 for r in matched if r.status == "success")
    success_rate = round(success / total * 100, 1) if total else 0.0

    durations = [
        (r.finished_at - r.started_at).total_seconds() * 1000
        for r in matched
        if r.finished_at
    ]
    avg_duration_ms = int(sum(durations) / len(durations)) if durations else None

    total_steps = sum(r.message_count or 0 for r in matched)
    avg_steps_per_run = round(total_steps / total, 1) if total else None

    return {
        "total": total,
        "success_rate": success_rate,
        "avg_duration_ms": avg_duration_ms,
        "total_steps": total_steps,
        "avg_steps_per_run": avg_steps_per_run,
        "running_now": running_now,
    }


def _fix_orphaned_runs(schedule_id: str) -> None:
    """Mark any 'running' runs as cancelled when their task no longer exists."""
    for run in load_runs(schedule_id, limit=5):
        if run.status == "running":
            run.status = "cancelled"
            run.error = "Interrupted — server restarted while running"
            run.finished_at = datetime.now(timezone.utc)
            _save_run(run)
            logger.info("Reset orphaned run %s for schedule %s", run.id, schedule_id)


def _prune_old_runs(schedule_id: str, keep_last_n: int) -> None:
    base = runs_dir(schedule_id)
    if not base.exists():
        return
    run_dirs = sorted(d for d in base.iterdir() if d.is_dir())
    to_delete = run_dirs[:-keep_last_n] if len(run_dirs) > keep_last_n else []
    for old_run in to_delete:
        shutil.rmtree(old_run, ignore_errors=True)


def _sync_latest(schedule_id: str, run_dir: Path) -> None:
    """Copy run output files to the schedule's ``latest/`` directory."""
    latest = schedule_dir(schedule_id) / "latest"
    if latest.exists():
        shutil.rmtree(latest, ignore_errors=True)
    files_dir = run_dir / "files"
    if files_dir.exists() and any(files_dir.iterdir()):
        shutil.copytree(files_dir, latest)


# ---------------------------------------------------------------------------
# Core execution — runs the agent for a scheduled job
# ---------------------------------------------------------------------------

_running_schedule_tasks: dict[str, asyncio.Task] = {}


async def _execute_scheduled_run(
    spec: ScheduleSpec,
    run: ScheduleRun,
    timestamp: str,
) -> None:
    """Do the actual work of a scheduled run as a background task.

    This is spawned via ``asyncio.create_task`` so it never blocks the
    event loop — the HTTP server stays responsive while the agent runs.
    """
    from backend.session_manager import _session_files_dir
    from backend.state import message_queues, running_tasks, session_mgr

    schedule_id = spec.id
    try:
        cfg = await AppConfig.aload()
        session = await session_mgr.create_session(
            config=cfg,
            agent_name=spec.agent_name,
            is_scheduled_run=True,
            schedule_id=spec.id,
            trigger_source="schedule",
        )
        run.session_id = session.id
        await asyncio.to_thread(_save_run, run)

        # Make schedule attachments readable by the run via per-file symlinks.
        files_dir = _session_files_dir(session.id)
        attachment_paths = await asyncio.to_thread(
            _link_attachments_into_session, schedule_id, files_dir
        )
        if attachment_paths:
            prompt = (
                f"[Uploaded files: {', '.join(attachment_paths)}]\n\n"
                f"These read-only files are attached to this schedule. Write any "
                f"output only under `/output/`, never into `/attachments/`.\n\n"
                f"{spec.prompt}"
            )
        else:
            prompt = spec.prompt

        current_task = asyncio.current_task()
        if current_task:
            running_tasks[session.id] = current_task

        async def _stream() -> int:
            from backend.routes.sessions import _LazyPersistingSubagentQueue
            from backend.streaming_subagent import reset_subagent_queue, set_subagent_queue

            _NO_USER_ANSWER = (
                "No user is available to respond — this is an automated scheduled run. "
                "Handle the situation by making reasonable assumptions, using defaults, "
                "or clearly reporting what is missing and stopping cleanly."
            )

            count = 0
            sent_done = False
            auto_resumes = 0
            # Tracks the decisions to send on the next resume; set inside _drain.
            _next_decisions: list[dict] = []

            token = set_subagent_queue(
                _LazyPersistingSubagentQueue(session.id),
            )

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
                            "Scheduled run %s: auto-resuming ask_user (attempt %d)",
                            schedule_id, auto_resumes,
                        )
                        _next_decisions = [{"answer": _NO_USER_ANSWER}]
                        return False
                    if rtype == "hitl_request":
                        # No human present: auto-approve execute commands.
                        # The model may batch multiple approval-requiring tool
                        # calls into a single interrupt; the middleware requires
                        # exactly one decision per hanging tool call.
                        auto_resumes += 1
                        n_actions = len(
                            (resp.get("metadata") or {}).get("action_requests") or []
                        ) or 1
                        logger.info(
                            "Scheduled run %s: auto-approving hitl_request "
                            "(attempt %d, %d action(s))",
                            schedule_id, auto_resumes, n_actions,
                        )
                        _next_decisions = [{"type": "approve"} for _ in range(n_actions)]
                        return False
                return True

            try:
                terminal = await _drain(
                    session_mgr.stream_message(session.id, prompt)
                )
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
            run_dir = runs_dir(schedule_id) / timestamp
            run_dir.mkdir(parents=True, exist_ok=True)
            if session_files.exists() and any(session_files.iterdir()):
                # Exclude the symlinked attachments so run snapshots don't
                # duplicate the schedule's single copy of the input files.
                shutil.copytree(
                    session_files,
                    run_dir / "files",
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("attachments"),
                )
            _sync_latest(schedule_id, run_dir)

        await asyncio.to_thread(_copy_session_output)

    except asyncio.CancelledError:
        logger.info("Scheduled run %s/%s cancelled by user", schedule_id, timestamp)
        run.status = "cancelled"
        run.error = "Cancelled by user"
    except asyncio.TimeoutError:
        logger.error("Scheduled run %s/%s timed out after %ds", schedule_id, timestamp, spec.timeout_seconds)
        run.status = "error"
        run.error = f"Timed out after {spec.timeout_seconds}s"
    except Exception as exc:
        logger.error("Scheduled run %s/%s failed: %s", schedule_id, timestamp, exc, exc_info=True)
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
            # page even though the schedule itself shows an error.
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
                logger.debug("Failed to close scheduled session %s", run.session_id, exc_info=True)
            # Best-effort: analyze a failed run for a prompt fix (gated by
            # evaluation.analyze_errors). Runs after close_session persists meta.
            if run.status == "error":
                await session_mgr.maybe_analyze_error(run.session_id)

        run.finished_at = datetime.now(timezone.utc)
        await asyncio.to_thread(_save_run, run)

        spec = await asyncio.to_thread(load_schedule, schedule_id) or spec
        spec.last_status = run.status
        spec.last_error = run.error
        spec.last_run = run.started_at
        await asyncio.to_thread(save_schedule, spec)

        await asyncio.to_thread(_prune_old_runs, schedule_id, spec.keep_last_n_runs)
        _running_schedule_tasks.pop(schedule_id, None)


async def _run_scheduled_agent(schedule_id: str, *, force: bool = False) -> None:
    """APScheduler entry point — validates the schedule then spawns a
    background task so the event loop is never blocked.

    When *force* is True the ``enabled`` check is skipped (used by run-now).
    """
    if schedule_id in _running_schedule_tasks:
        logger.info("Schedule %s is already running — skipping", schedule_id)
        return

    spec = await asyncio.to_thread(load_schedule, schedule_id)
    if not spec:
        return
    if not spec.enabled and not force:
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S") + f"-{uuid.uuid4().hex[:6]}"
    run = ScheduleRun(
        id=timestamp,
        schedule_id=schedule_id,
        status="running",
        started_at=datetime.now(timezone.utc),
    )
    await asyncio.to_thread(_save_run, run)

    spec.last_run = run.started_at
    spec.last_status = "running"
    spec.last_error = None
    await asyncio.to_thread(save_schedule, spec)

    task = asyncio.create_task(
        _execute_scheduled_run(spec, run, timestamp),
        name=f"schedule-{schedule_id}",
    )
    _running_schedule_tasks[schedule_id] = task


# ---------------------------------------------------------------------------
# Scheduler lifecycle
# ---------------------------------------------------------------------------

_scheduler: Optional[AsyncIOScheduler] = None


def get_scheduler() -> AsyncIOScheduler:
    if _scheduler is None:
        raise RuntimeError("Scheduler not initialized — call init_scheduler() first")
    return _scheduler


def register_job(scheduler: AsyncIOScheduler, schedule_id: str, cron_expression: str) -> None:
    """Add or replace an APScheduler job for the given schedule."""
    trigger = CronTrigger.from_crontab(cron_expression)
    scheduler.add_job(
        _run_scheduled_agent,
        trigger=trigger,
        args=[schedule_id],
        id=schedule_id,
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=_MISFIRE_GRACE_SECS,
    )


def remove_job(scheduler: AsyncIOScheduler, schedule_id: str) -> None:
    try:
        scheduler.remove_job(schedule_id)
    except JobLookupError:
        pass


def run_schedule_immediately(schedule_id: str) -> None:
    """Queue a one-off manual run that bypasses the ``enabled`` check."""
    scheduler = get_scheduler()
    scheduler.add_job(
        _run_scheduled_agent,
        args=[schedule_id],
        kwargs={"force": True},
        id=f"{schedule_id}-manual",
        replace_existing=True,
        max_instances=1,
    )


def stop_schedule_run(schedule_id: str) -> bool:
    """Cancel a running schedule task.  Returns True if a task was cancelled."""
    task = _running_schedule_tasks.get(schedule_id)
    if task and not task.done():
        task.cancel()
        return True
    return False


def is_schedule_running(schedule_id: str) -> bool:
    """Check whether a schedule currently has an active background task."""
    task = _running_schedule_tasks.get(schedule_id)
    return task is not None and not task.done()


def init_scheduler() -> AsyncIOScheduler:
    """Create, populate, and start the scheduler.

    Reads all saved schedules from disk and registers APScheduler jobs for
    each enabled schedule.  Called once during FastAPI lifespan startup.
    """
    global _scheduler

    scheduler = AsyncIOScheduler(
        job_defaults={
            "coalesce": True,
            "max_instances": 1,
            "misfire_grace_time": _MISFIRE_GRACE_SECS,
        },
    )

    for spec in load_all_schedules():
        if spec.last_status == "running":
            logger.info("Resetting stale 'running' status for schedule %s", spec.id)
            spec.last_status = "error"
            spec.last_error = "Interrupted — server restarted while running"
            save_schedule(spec)
            _fix_orphaned_runs(spec.id)
        if spec.enabled:
            try:
                register_job(scheduler, spec.id, spec.cron_expression)
                logger.info("Registered schedule: %s (%s)", spec.id, spec.cron_expression)
            except Exception:
                logger.warning("Failed to register schedule %s", spec.id, exc_info=True)

    scheduler.start()
    _scheduler = scheduler
    logger.info("Scheduler started with %d job(s)", len(scheduler.get_jobs()))
    return scheduler


def shutdown_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Scheduler shut down")
