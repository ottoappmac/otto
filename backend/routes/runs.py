"""Unified /api/runs endpoints.

Merges sessions, schedule runs, and trigger runs into a single paginated list
with aggregate stats for the Overview dashboard.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Body, Query

from backend.config import get_app_data_dir
from backend.schemas import SessionInfo
from backend.session_manager import _sessions_dir
from backend.state import session_mgr, running_tasks

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/runs", tags=["runs"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sessions_dir_path() -> Path:
    return _sessions_dir()


# Message types that represent a discrete agent "step". ``agent`` covers each
# model turn and ``tool_call`` each tool invocation, for both the orchestrator
# and any subagents (subagent events are mirrored into the same
# ``.messages.json`` file, so counting here is inherently inclusive of them).
_STEP_TYPES = {"agent", "tool_call"}


def _count_session_steps(session_id: str) -> int:
    """Total steps (incl. subagents) for a session, from its messages file.

    Reads ``{session_id}.messages.json`` line-by-line (JSONL) and counts step
    events. Falls back to parsing a legacy JSON-array file if present. Returns
    0 when the file is missing or unreadable.
    """
    from backend.session_manager import _messages_path

    try:
        path = _messages_path(session_id)
    except Exception:
        return 0
    if not path.exists():
        return 0

    count = 0
    try:
        with path.open(encoding="utf-8") as fh:
            first = fh.read(1)
            fh.seek(0)
            if first == "[":
                # Legacy single-array format — parse the whole document.
                try:
                    for obj in json.load(fh):
                        if isinstance(obj, dict) and obj.get("type") in _STEP_TYPES:
                            count += 1
                except Exception:
                    return 0
                return count
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("type") in _STEP_TYPES:
                    count += 1
    except Exception:
        logger.debug("Failed counting steps for %s", session_id, exc_info=True)
    return count


def _load_all_session_infos() -> list[SessionInfo]:
    """Load all persisted session metadata files."""
    sessions: list[SessionInfo] = []
    for p in sorted(_sessions_dir_path().glob("*.json"), reverse=True):
        if p.name.endswith((".messages.json", ".eval.json")):
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            sessions.append(SessionInfo.model_validate(data))
        except Exception:
            logger.debug("Skipping corrupt session meta: %s", p.name, exc_info=True)
    return sessions


def _schedules_root() -> Path:
    return get_app_data_dir() / "schedules"


def _triggers_root() -> Path:
    return get_app_data_dir() / "triggers"


def _load_schedule_runs() -> list[dict[str, Any]]:
    """Load all ScheduleRun records from disk."""
    runs: list[dict[str, Any]] = []
    root = _schedules_root()
    if not root.exists():
        return runs
    for sched_dir in root.iterdir():
        runs_dir = sched_dir / "runs"
        if not runs_dir.is_dir():
            continue
        for run_dir in runs_dir.iterdir():
            run_file = run_dir / "run.json"
            if not run_file.exists():
                continue
            try:
                data = json.loads(run_file.read_text(encoding="utf-8"))
                runs.append(data)
            except Exception:
                logger.debug("Skipping corrupt schedule run: %s", run_file, exc_info=True)
    return runs


def _load_trigger_runs() -> list[dict[str, Any]]:
    """Load all TriggerRun records from disk."""
    runs: list[dict[str, Any]] = []
    root = _triggers_root()
    if not root.exists():
        return runs
    for trig_dir in root.iterdir():
        runs_dir = trig_dir / "runs"
        if not runs_dir.is_dir():
            continue
        for run_dir in runs_dir.iterdir():
            run_file = run_dir / "run.json"
            if not run_file.exists():
                continue
            try:
                data = json.loads(run_file.read_text(encoding="utf-8"))
                runs.append(data)
            except Exception:
                logger.debug("Skipping corrupt trigger run: %s", run_file, exc_info=True)
    return runs


def _session_to_run_dict(info: SessionInfo) -> dict[str, Any]:
    """Normalise a SessionInfo into the unified run shape."""
    # Resolve live status from in-memory session if available
    live = session_mgr.get_session(info.id)
    status = (live.status if live else info.status) or "idle"
    # Sessions persisted before status-tracking was added show as "idle".
    # If they are not currently active and have messages they are effectively done.
    if status == "idle" and not live and info.message_count > 0:
        status = "completed"
    # If the stored status is "running" but there is no live in-memory session and
    # no active asyncio task, the session is orphaned (e.g. server crashed mid-run).
    # Downgrade to "error" so the UI stops showing a ghost spinner.
    if status == "running" and not live:
        task = running_tasks.get(info.id)
        if task is None or task.done():
            status = "error"
    return {
        "id": info.id,
        "kind": "session",
        "title": info.title,
        "agent_name": info.agent_name,
        "trigger_source": info.trigger_source,
        "schedule_id": info.schedule_id,
        "trigger_id": info.trigger_id,
        "parent_session_id": info.parent_session_id,
        "chain_depth": info.chain_depth,
        "status": status,
        "started_at": info.created_at.isoformat(),
        "finished_at": info.finished_at.isoformat() if info.finished_at else None,
        "duration_ms": info.duration_ms,
        "message_count": info.message_count,
        "tools_used": info.tools_used,
        "llm_provider": info.llm_provider,
        "model": info.model,
        "input_tokens": info.input_tokens,
        "output_tokens": info.output_tokens,
        "estimated_cost_usd": info.estimated_cost_usd,
        "avg_prefill_tps": info.avg_prefill_tps,
        "avg_generation_tps": info.avg_generation_tps,
        "cache_hit_ratio": info.cache_hit_ratio,
        "peak_memory_gb": info.peak_memory_gb,
        "error": info.error,
        "session_id": info.id,
        "eval_status": info.eval_status,
        "eval_overall_score": info.eval_overall_score,
        "eval_pass_count": info.eval_pass_count,
        "eval_total": info.eval_total,
    }


def _schedule_run_to_run_dict(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": run.get("id", ""),
        "kind": "schedule_run",
        "title": f"Schedule run",
        "agent_name": None,
        "trigger_source": "schedule",
        "schedule_id": run.get("schedule_id"),
        "trigger_id": None,
        "parent_session_id": None,
        "chain_depth": 0,
        "status": _normalise_status(run.get("status", "unknown")),
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
        "duration_ms": _duration_ms(run.get("started_at"), run.get("finished_at")),
        "message_count": run.get("message_count", 0),
        "tools_used": [],
        "llm_provider": None,
        "model": None,
        "input_tokens": None,
        "output_tokens": None,
        "estimated_cost_usd": None,
        "avg_prefill_tps": None,
        "avg_generation_tps": None,
        "cache_hit_ratio": None,
        "peak_memory_gb": None,
        "error": run.get("error"),
        "session_id": run.get("session_id"),
    }


def _trigger_run_to_run_dict(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": run.get("id", ""),
        "kind": "trigger_run",
        "title": "Trigger run",
        "agent_name": None,
        "trigger_source": "trigger",
        "schedule_id": None,
        "trigger_id": run.get("trigger_id"),
        "parent_session_id": None,
        "chain_depth": 0,
        "status": _normalise_status(run.get("status", "unknown")),
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
        "duration_ms": _duration_ms(run.get("started_at"), run.get("finished_at")),
        "message_count": run.get("message_count", 0),
        "tools_used": [],
        "llm_provider": None,
        "model": None,
        "input_tokens": None,
        "output_tokens": None,
        "estimated_cost_usd": None,
        "avg_prefill_tps": None,
        "avg_generation_tps": None,
        "cache_hit_ratio": None,
        "peak_memory_gb": None,
        "error": run.get("error"),
        "session_id": run.get("session_id"),
    }


def _normalise_status(s: str) -> str:
    mapping = {"success": "completed", "done": "completed"}
    return mapping.get(s, s)


def _duration_ms(started: Any, finished: Any) -> Optional[int]:
    if not started or not finished:
        return None
    try:
        s = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
        f = datetime.fromisoformat(str(finished).replace("Z", "+00:00"))
        return int((f - s).total_seconds() * 1000)
    except Exception:
        return None


def _parse_dt(val: Any) -> Optional[datetime]:
    if val is None:
        return None
    try:
        if isinstance(val, datetime):
            return val
        return datetime.fromisoformat(str(val).replace("Z", "+00:00"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("")
async def api_list_runs(
    status: Optional[str] = Query(None, description="Filter by status"),
    agent: Optional[str] = Query(None, description="Filter by agent name"),
    source: Optional[str] = Query(None, description="Filter by trigger source"),
    search: Optional[str] = Query(None, description="Search title/agent"),
    schedule_id: Optional[str] = Query(None, description="Filter by schedule ID"),
    trigger_id: Optional[str] = Query(None, description="Filter by trigger ID"),
    date_from: Optional[str] = Query(None, description="ISO start date filter"),
    date_to: Optional[str] = Query(None, description="ISO end date filter"),
    order_by: Optional[str] = Query(
        None,
        description="Sort field: started_at | duration_ms | tokens | eval. Default = running first, then newest.",
    ),
    order: str = Query("desc", description="Sort direction: asc | desc"),
    limit: int = Query(100, le=500),
    offset: int = Query(0),
):
    """Unified, paginated list of all runs (sessions + schedule/trigger runs)."""
    # Load everything in parallel
    session_infos, sched_runs, trig_runs = await asyncio.gather(
        asyncio.to_thread(_load_all_session_infos),
        asyncio.to_thread(_load_schedule_runs),
        asyncio.to_thread(_load_trigger_runs),
    )

    # Merge active in-memory sessions first so live status is reflected
    active_ids = {s.id for s in session_mgr.list_active()}
    history_infos = [s for s in session_infos if s.id not in active_ids]
    all_infos = session_mgr.list_active() + history_infos

    # Build unified run dicts
    runs: list[dict[str, Any]] = [_session_to_run_dict(s) for s in all_infos]

    # Avoid double-listing schedule/trigger runs that already have a session
    session_ids_covered = {r["session_id"] for r in runs if r.get("session_id")}

    for sr in sched_runs:
        if sr.get("session_id") not in session_ids_covered:
            runs.append(_schedule_run_to_run_dict(sr))

    for tr in trig_runs:
        if tr.get("session_id") not in session_ids_covered:
            runs.append(_trigger_run_to_run_dict(tr))

    # --- Filtering ---
    dt_from = _parse_dt(date_from)
    dt_to = _parse_dt(date_to)

    def _matches(r: dict[str, Any]) -> bool:
        if status and r["status"] != status:
            return False
        if agent and r.get("agent_name", "") != agent:
            return False
        if source and r.get("trigger_source") != source:
            return False
        if schedule_id and r.get("schedule_id") != schedule_id:
            return False
        if trigger_id and r.get("trigger_id") != trigger_id:
            return False
        if search:
            q = search.lower()
            if q not in (r.get("title") or "").lower() and q not in (r.get("agent_name") or "").lower():
                return False
        started = _parse_dt(r.get("started_at"))
        if dt_from and started and started < dt_from:
            return False
        if dt_to and started and started > dt_to:
            return False
        return True

    filtered = [r for r in runs if _matches(r)]

    # --- Sorting ---
    # Only fields available before pagination can drive a server-side sort.
    # (step_count is computed lazily per-page below, so it is intentionally
    # not sortable here.)
    _SORTABLE = {"started_at", "duration_ms", "tokens", "eval"}

    def _sort_value(r: dict[str, Any], field: str):
        if field == "started_at":
            dt = _parse_dt(r.get("started_at"))
            return dt.timestamp() if dt else None
        if field == "duration_ms":
            return r.get("duration_ms")
        if field == "tokens":
            return (r.get("input_tokens") or 0) + (r.get("output_tokens") or 0)
        if field == "eval":
            return r.get("eval_overall_score")
        return None

    if order_by in _SORTABLE:
        reverse = order != "asc"

        def _explicit_sort_key(r: dict[str, Any]):
            v = _sort_value(r, order_by)
            # Missing values always sort last, regardless of direction.
            if v is None:
                return (1, 0.0)
            return (0, -float(v) if reverse else float(v))

        filtered.sort(key=_explicit_sort_key)
    else:
        # Default: running first, then by started_at descending.
        def _sort_key(r: dict[str, Any]):
            started = _parse_dt(r.get("started_at"))
            return (
                0 if r["status"] == "running" else 1,
                -(started.timestamp() if started else 0),
            )

        filtered.sort(key=_sort_key)
    total = len(filtered)
    page = filtered[offset: offset + limit]

    # Compute total steps (incl. subagents) lazily for just the returned page so
    # we don't read every session's message file on each list/poll request.
    async def _attach_steps(run: dict[str, Any]) -> None:
        sid = run.get("session_id")
        if sid:
            run["step_count"] = await asyncio.to_thread(_count_session_steps, sid)
        else:
            run["step_count"] = run.get("message_count", 0)

    await asyncio.gather(*[_attach_steps(r) for r in page])

    return {"total": total, "offset": offset, "limit": limit, "runs": page}


@router.get("/{session_id}/evaluation")
async def api_get_run_evaluation(session_id: str):
    """Return the persisted end-of-run evaluation for a session."""
    from backend.eval_runner import load_evaluation

    data = await asyncio.to_thread(load_evaluation, session_id)
    if data is None:
        return {"session_id": session_id, "status": "none", "results": []}
    return data


@router.post("/{session_id}/run-again")
async def api_run_again(
    session_id: str,
    prompt: Optional[str] = Body(None, embed=True),
):
    """Create a new session with the same agent and first user prompt as *session_id*.

    Loads the original session metadata (for ``agent_name``) and the first user
    message from the messages file, then creates a fresh session and kicks it off
    via ``kick_off_message`` — identical to how ambient/schedule runs are started.

    When *prompt* is supplied it overrides the original first-user-message, allowing
    callers (e.g. the Evaluation tab "Run with improved prompt" button) to re-run
    with an updated prompt without modifying the original session.

    Returns ``{"session_id": "<new-id>"}`` on success or a 404/422 on failure.
    """
    from backend.config import AppConfig
    from backend.session_dispatch import kick_off_message
    from backend.session_manager import _load_messages_async, _sessions_dir

    # --- resolve metadata ---
    # Try live session first, then fall back to the on-disk meta file.
    live = session_mgr.get_session(session_id)
    agent_name: str | None = None
    if live is not None:
        agent_name = live.to_info().agent_name
    else:
        meta_path = _sessions_dir() / f"{session_id}.json"
        if await asyncio.to_thread(meta_path.exists):
            try:
                raw = json.loads(await asyncio.to_thread(meta_path.read_text, "utf-8"))
                agent_name = raw.get("agent_name")
            except Exception:
                pass

    # --- find first user message ---
    try:
        messages = await _load_messages_async(session_id)
    except Exception:
        messages = []

    first_prompt: str | None = None
    for msg in messages:
        if msg.get("type") == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                first_prompt = content.strip()
                break
            # content can be a list of blocks (e.g. with images); extract text
            if isinstance(content, list):
                parts = [
                    block.get("text", "") for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                text = " ".join(p for p in parts if p).strip()
                if text:
                    first_prompt = text
                    break

    # Caller may supply an override prompt (e.g. an eval-suggested improvement).
    effective_prompt = (prompt.strip() if prompt and prompt.strip() else None) or first_prompt

    if not effective_prompt:
        from fastapi.responses import JSONResponse as _JSONResponse
        return _JSONResponse(
            status_code=422,
            content={"error": "No user prompt found in the original run — cannot re-run."},
        )

    # --- create new session and kick it off ---
    cfg = await AppConfig.aload()
    new_session = await session_mgr.create_session(
        config=cfg,
        agent_name=agent_name,
        trigger_source="manual",
    )
    asyncio.create_task(kick_off_message(new_session.id, effective_prompt))

    logger.info(
        "run-again: cloned session %s → %s (agent=%s, prompt=%d chars%s)",
        session_id, new_session.id, agent_name, len(effective_prompt),
        ", overridden" if prompt else "",
    )
    return {"session_id": new_session.id}


@router.post("/{session_id}/evaluation")
async def api_run_evaluation(session_id: str):
    """Kick off a manual evaluation and return the initial running state.

    Runs in the background so the client can poll the GET endpoint and
    reveal the evaluator's step-by-step trace live. Failed runs are routed to
    the errored-run analyzer (diagnosis + prompt fix) instead of metric scoring,
    since they have no completed output to score.
    """
    from backend.eval_runner import (
        launch_error_analysis,
        launch_evaluation,
        load_evaluation,
    )
    from backend.session_manager import _sessions_dir

    # Resolve the run's terminal status (live session first, then on-disk meta).
    live = session_mgr.get_session(session_id)
    status = live.to_info().status if live is not None else None
    if status is None:
        meta_path = _sessions_dir() / f"{session_id}.json"
        if await asyncio.to_thread(meta_path.exists):
            try:
                raw = json.loads(await asyncio.to_thread(meta_path.read_text, "utf-8"))
                status = raw.get("status")
            except Exception:
                status = None

    if status == "error":
        launch_error_analysis(session_id, manual=True)
    else:
        launch_evaluation(session_id, manual=True)
    # Give the task a moment to write its initial running state.
    await asyncio.sleep(0.1)
    data = await asyncio.to_thread(load_evaluation, session_id)
    return data or {"session_id": session_id, "status": "running", "steps": [], "results": []}


@router.get("/stats")
async def api_run_stats(
    period: str = Query("7d", description="Period: 24h | 7d | 30d | all | custom"),
    date_from: Optional[str] = Query(None, description="ISO date string for custom range start (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="ISO date string for custom range end (YYYY-MM-DD)"),
    search: Optional[str] = Query(None, description="Filter by title or agent name"),
    status: Optional[str] = Query(None, description="Filter by run status"),
    source: Optional[str] = Query(None, description="Filter by trigger source"),
):
    """Aggregate stats for the Overview dashboard."""
    from collections import Counter, defaultdict

    now = datetime.now(timezone.utc)

    # Load all runs (reuse list endpoint logic)
    session_infos, sched_runs, trig_runs = await asyncio.gather(
        asyncio.to_thread(_load_all_session_infos),
        asyncio.to_thread(_load_schedule_runs),
        asyncio.to_thread(_load_trigger_runs),
    )
    active_ids = {s.id for s in session_mgr.list_active()}
    history_infos = [s for s in session_infos if s.id not in active_ids]
    all_infos = session_mgr.list_active() + history_infos

    runs: list[dict[str, Any]] = [_session_to_run_dict(s) for s in all_infos]
    session_ids_covered = {r["session_id"] for r in runs if r.get("session_id")}
    for sr in sched_runs:
        if sr.get("session_id") not in session_ids_covered:
            runs.append(_schedule_run_to_run_dict(sr))
    for tr in trig_runs:
        if tr.get("session_id") not in session_ids_covered:
            runs.append(_trigger_run_to_run_dict(tr))

    # Apply search/status/source filters (mirror logic from the list endpoint)
    if search or status or source:
        q = search.lower() if search else None

        def _stats_matches(r: dict[str, Any]) -> bool:
            if status and r.get("status") != status:
                return False
            if source and r.get("trigger_source") != source:
                return False
            if q and q not in (r.get("title") or "").lower() and q not in (r.get("agent_name") or "").lower():
                return False
            return True

        runs = [r for r in runs if _stats_matches(r)]

    # Determine time window boundaries
    if period == "custom" and date_from:
        try:
            start_ts = datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            start_ts = now.timestamp() - 7 * 24 * 3600
        if date_to:
            try:
                end_dt = datetime.fromisoformat(date_to).replace(tzinfo=timezone.utc)
                end_ts = end_dt.replace(hour=23, minute=59, second=59).timestamp()
            except ValueError:
                end_ts = now.timestamp()
        else:
            end_ts = now.timestamp()
    elif period == "all":
        all_starts = [
            (_parse_dt(r.get("started_at")) or now).timestamp()
            for r in runs
        ]
        # Use a start slightly before the earliest run so it's always included.
        # When there are no filtered runs, set a window that yields an empty period_runs.
        start_ts = (min(all_starts) - 1) if all_starts else now.timestamp()
        end_ts = now.timestamp()
    else:
        period_hours = {"24h": 24, "7d": 168, "30d": 720}.get(period, 168)
        start_ts = now.timestamp() - period_hours * 3600
        end_ts = now.timestamp()

    period_runs = [
        r for r in runs
        if start_ts <= (_parse_dt(r.get("started_at")) or now).timestamp() <= end_ts
    ]

    # Determine bucket size from span
    span_hours = max(1.0, (end_ts - start_ts) / 3600)
    if span_hours <= 24:
        bucket_hours = 1
    elif span_hours <= 24 * 14:
        bucket_hours = 24
    elif span_hours <= 24 * 90:
        bucket_hours = 24
    else:
        bucket_hours = 24 * 7  # weekly buckets for spans > 90 days

    bucket_secs = bucket_hours * 3600
    num_buckets = max(1, int(span_hours / bucket_hours))
    # Cap to avoid excessive data points
    if num_buckets > 200:
        bucket_hours = max(bucket_hours, int(span_hours / 200) + 1)
        bucket_secs = bucket_hours * 3600
        num_buckets = max(1, int(span_hours / bucket_hours))

    # Status counts — computed from period_runs so they match the filtered set.
    # running_now also uses period_runs so it reflects the exact same records
    # visible in the table (respecting search, status, source, and date filters).
    status_counts: Counter[str] = Counter(r["status"] for r in period_runs)
    running_now = sum(1 for r in period_runs if r["status"] == "running")

    # Evaluation aggregates (only runs scored by the end-of-run evaluator).
    evaluated = [
        r for r in period_runs
        if r.get("eval_status") == "done" and isinstance(r.get("eval_overall_score"), (int, float))
    ]
    eval_count = len(evaluated)
    eval_avg_score = (
        round(sum(r["eval_overall_score"] for r in evaluated) / eval_count, 4)
        if eval_count else None
    )
    eval_metric_total = sum(r.get("eval_total") or 0 for r in evaluated)
    eval_metric_passed = sum(r.get("eval_pass_count") or 0 for r in evaluated)
    eval_pass_rate = (
        round(eval_metric_passed / eval_metric_total * 100, 1)
        if eval_metric_total else None
    )

    total_period = len(period_runs)
    completed = status_counts.get("completed", 0)
    success_rate = round(completed / total_period * 100, 1) if total_period > 0 else 0.0

    # Durations
    durations = [r["duration_ms"] for r in period_runs if r.get("duration_ms")]
    avg_duration_ms = int(sum(durations) / len(durations)) if durations else None

    # Steps (agent model turns + tool calls; uses message_count as a fast proxy
    # since per-run step_count requires reading every messages file at runtime).
    total_steps = sum(r.get("message_count") or 0 for r in period_runs)
    avg_steps_per_run = round(total_steps / total_period, 1) if total_period > 0 else None

    # Tokens / cost
    total_input_tokens = sum(r.get("input_tokens") or 0 for r in period_runs)
    total_output_tokens = sum(r.get("output_tokens") or 0 for r in period_runs)
    total_cost_usd = sum(r.get("estimated_cost_usd") or 0.0 for r in period_runs)

    # MLX throughput (token-weighted so larger runs dominate the average).
    # Reconstruct elapsed time from per-run avg TPS and token counts, then
    # divide the summed tokens by the summed time.  Runs without MLX stats
    # (omlx/cloud) simply contribute nothing.
    prefill_tok_sum = prefill_time_sum = 0.0
    gen_tok_sum = gen_time_sum = 0.0
    cache_ratios: list[float] = []
    peak_memory_gb = 0.0
    for r in period_runs:
        ptps = r.get("avg_prefill_tps")
        gtps = r.get("avg_generation_tps")
        in_tok = r.get("input_tokens") or 0
        out_tok = r.get("output_tokens") or 0
        if ptps:
            # Use actual token count for proper weighting; fall back to 1
            # for older MLX sessions whose input_tokens was not yet recorded.
            w_in = in_tok if in_tok > 0 else 1
            prefill_tok_sum += w_in
            prefill_time_sum += w_in / ptps
        if gtps:
            w_out = out_tok if out_tok > 0 else 1
            gen_tok_sum += w_out
            gen_time_sum += w_out / gtps
        if r.get("cache_hit_ratio") is not None:
            cache_ratios.append(float(r["cache_hit_ratio"]))
        if r.get("peak_memory_gb"):
            peak_memory_gb = max(peak_memory_gb, float(r["peak_memory_gb"]))

    avg_prefill_tps = round(prefill_tok_sum / prefill_time_sum, 1) if prefill_time_sum > 0 else None
    avg_generation_tps = round(gen_tok_sum / gen_time_sum, 1) if gen_time_sum > 0 else None
    cache_hit_ratio = round(sum(cache_ratios) / len(cache_ratios), 3) if cache_ratios else None
    peak_memory_gb = round(peak_memory_gb, 3) if peak_memory_gb else None

    # Top agents
    agent_counts: Counter[str] = Counter(
        r.get("agent_name") or "general-purpose" for r in period_runs
    )
    top_agents = [{"agent": a, "count": c} for a, c in agent_counts.most_common(10)]

    # Top tools
    tool_counts: Counter[str] = Counter()
    for r in period_runs:
        for t in r.get("tools_used") or []:
            tool_counts[t] += 1
    top_tools = [{"tool": t, "count": c} for t, c in tool_counts.most_common(10)]

    # Source breakdown
    source_counts: Counter[str] = Counter(
        r.get("trigger_source") or "manual" for r in period_runs
    )
    source_breakdown = [{"source": s, "count": c} for s, c in source_counts.most_common()]

    # Provider/model breakdown
    model_counts: Counter[str] = Counter(
        (r.get("llm_provider") or r.get("model") or "unknown") for r in period_runs
    )
    model_breakdown = [{"model": m, "count": c} for m, c in model_counts.most_common()]

    # Model cost breakdown
    model_cost: dict[str, float] = defaultdict(float)
    for r in period_runs:
        key = r.get("llm_provider") or r.get("model") or "unknown"
        model_cost[key] += r.get("estimated_cost_usd") or 0.0
    model_cost_breakdown = [
        {"model": m, "cost_usd": round(c, 6)}
        for m, c in sorted(model_cost.items(), key=lambda x: -x[1])
    ]

    # Time-series buckets
    buckets: list[dict[str, Any]] = []
    for i in range(num_buckets):
        b_start = start_ts + i * bucket_secs
        b_end = b_start + bucket_secs
        # Integer bucket counts can leave a tail gap between the last bucket's
        # end and end_ts; let the final bucket absorb it so the most recent runs
        # (often the only evaluated ones) aren't dropped from the series.
        if i == num_buckets - 1:
            b_end = max(b_end, end_ts + 1)
        b_runs = [
            r for r in period_runs
            if b_start <= (_parse_dt(r.get("started_at")) or now).timestamp() < b_end
        ]
        b_status: Counter[str] = Counter(r["status"] for r in b_runs)
        b_eval_scores = [
            r["eval_overall_score"] for r in b_runs
            if r.get("eval_status") == "done"
            and isinstance(r.get("eval_overall_score"), (int, float))
        ]
        buckets.append({
            "ts": datetime.fromtimestamp(b_start, tz=timezone.utc).isoformat(),
            "total": len(b_runs),
            "completed": b_status.get("completed", 0),
            "error": b_status.get("error", 0),
            "running": b_status.get("running", 0),
            # Mean evaluation score (0..1) for runs scored in this bucket; null
            # when no runs were evaluated so the chart can skip the gap.
            "eval_avg_score": (
                round(sum(b_eval_scores) / len(b_eval_scores), 4)
                if b_eval_scores else None
            ),
            "eval_count": len(b_eval_scores),
        })

    return {
        "period": period,
        "running_now": running_now,
        "total_period": total_period,
        "status_counts": dict(status_counts),
        "success_rate": success_rate,
        "eval_count": eval_count,
        "eval_avg_score": eval_avg_score,
        "eval_pass_rate": eval_pass_rate,
        "avg_duration_ms": avg_duration_ms,
        "total_steps": total_steps,
        "avg_steps_per_run": avg_steps_per_run,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cost_usd": round(total_cost_usd, 6),
        "avg_prefill_tps": avg_prefill_tps,
        "avg_generation_tps": avg_generation_tps,
        "cache_hit_ratio": cache_hit_ratio,
        "peak_memory_gb": peak_memory_gb,
        "top_agents": top_agents,
        "top_tools": top_tools,
        "source_breakdown": source_breakdown,
        "model_breakdown": model_breakdown,
        "model_cost_breakdown": model_cost_breakdown,
        "time_series": buckets,
    }
