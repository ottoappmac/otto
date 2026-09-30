"""Summarize distillation training data for the Distill dataset viewer."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config import AppConfig, get_app_data_dir
from backend.distillation.collector import (
    _load_eval_sidecar,
    _load_session_meta,
    _read_jsonl,
    extract_trajectory,
)
from backend.distillation.paths import sft_path, trajectories_path
from backend.distillation.quality import eval_passed, reject_reasons

_PREVIEW_CHARS = 160
_DETAIL_CONTENT_CHARS = 4_000


def _config() -> AppConfig:
    try:
        return AppConfig.load()
    except Exception:
        return AppConfig()


def _session_dirs(app_data: Path | None) -> tuple[Path, Path]:
    root = app_data or get_app_data_dir()
    return root / "transcripts", root / "sessions"


def _median(vals: list[int]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0


def _preview(messages: list[dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") != "user":
            continue
        text = " ".join(str(msg.get("content") or "").split())
        if not text:
            continue
        if len(text) <= _PREVIEW_CHARS:
            return text
        return text[: _PREVIEW_CHARS - 1] + "…"
    return ""


def _unanswered_tool_ids(messages: list[dict[str, Any]]) -> list[str]:
    called: list[str] = []
    answered: set[str] = set()
    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                tid = str(tc.get("id") or "")
                if tid:
                    called.append(tid)
        elif role == "tool":
            tid = str(msg.get("tool_call_id") or "")
            if tid:
                answered.add(tid)
    return [tid for tid in called if tid not in answered]


def _eval_summary(sidecar: dict[str, Any] | None) -> dict[str, Any]:
    if not sidecar:
        return {
            "eval_status": "",
            "eval_overall_score": None,
            "eval_pass_count": None,
            "eval_total": None,
            "eval_verdict": None,
        }
    verdict = eval_passed(sidecar)
    score = sidecar.get("overall_score")
    passed = sidecar.get("pass_count")
    total = sidecar.get("total")
    return {
        "eval_status": str(sidecar.get("status") or ""),
        "eval_overall_score": float(score) if isinstance(score, (int, float)) else None,
        "eval_pass_count": int(passed) if isinstance(passed, int) else None,
        "eval_total": int(total) if isinstance(total, int) else None,
        "eval_verdict": verdict,
    }


def _role_counts(messages: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(m.get("role") or "unknown") for m in messages)
    n_tool_calls = 0
    for msg in messages:
        n_tool_calls += len(msg.get("tool_calls") or [])
    return {
        "n_messages": len(messages),
        "n_user": counts.get("user", 0),
        "n_assistant": counts.get("assistant", 0),
        "n_tool": counts.get("tool", 0),
        "n_tool_calls": n_tool_calls,
    }


def _file_stats(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "bytes": 0, "rows": 0, "updated_at": None}
    stat = path.stat()
    rows = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows += 1
    updated = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    return {
        "path": str(path),
        "exists": True,
        "bytes": stat.st_size,
        "rows": rows,
        "updated_at": updated,
    }


def _session_row(
    traj: dict[str, Any],
    meta: dict[str, Any],
    *,
    min_tool_calls: int,
    teacher_model_id: str,
    filter_sessions_by_teacher: bool,
) -> dict[str, Any]:
    messages = list(traj.get("messages") or [])
    unanswered = _unanswered_tool_ids(messages)
    reasons = reject_reasons(
        traj,
        min_tool_calls=min_tool_calls,
        teacher_model_id=teacher_model_id,
        filter_sessions_by_teacher=filter_sessions_by_teacher,
    )
    ev = _eval_summary(traj.get("eval") if isinstance(traj.get("eval"), dict) else None)
    title = str(meta.get("title") or "").strip()
    preview = _preview(messages)
    model = str(traj.get("model") or meta.get("model") or "")
    return {
        "session_id": traj.get("session_id") or "",
        "title": title or preview or (traj.get("session_id") or ""),
        "preview": preview,
        "timestamp": str(traj.get("timestamp") or meta.get("created_at") or ""),
        "status": str(traj.get("status") or ""),
        "error": str(traj.get("error") or ""),
        "model": model,
        "llm_provider": str(meta.get("llm_provider") or ""),
        "tools_used": list(traj.get("tools_used") or []),
        "steps": int(traj.get("steps") or 0),
        "n_unanswered": len(unanswered),
        "eligible": not reasons,
        "reject_reasons": reasons,
        **_role_counts(messages),
        **ev,
    }


def scan_dataset(
    *,
    app_data: Path | None = None,
    teacher_model_id: str | None = None,
    student_model_id: str | None = None,
    filter_sessions_by_teacher: bool | None = None,
    min_tool_calls: int | None = None,
) -> dict[str, Any]:
    """Scan transcripts and summarize them for the Distill data viewer."""
    cfg = _config()
    dist = cfg.distillation
    teacher = (teacher_model_id if teacher_model_id is not None else dist.teacher_model_id).strip()
    student = (student_model_id if student_model_id is not None else dist.student_model_id).strip()
    filter_by_teacher = (
        dist.filter_sessions_by_teacher
        if filter_sessions_by_teacher is None
        else filter_sessions_by_teacher
    )
    min_calls = dist.min_tool_calls if min_tool_calls is None else min_tool_calls
    transcripts, sessions = _session_dirs(app_data)

    rows: list[dict[str, Any]] = []
    if transcripts.is_dir():
        for path in sorted(transcripts.glob("*.jsonl")):
            records = _read_jsonl(path)
            if not records:
                continue
            meta = _load_session_meta(sessions, path.stem)
            sidecar = _load_eval_sidecar(sessions, path.stem)
            traj = extract_trajectory(path.stem, records, meta, sidecar)
            rows.append(
                _session_row(
                    traj,
                    meta,
                    min_tool_calls=min_calls,
                    teacher_model_id=teacher,
                    filter_sessions_by_teacher=filter_by_teacher,
                )
            )

    rows.sort(key=lambda r: str(r.get("timestamp") or ""), reverse=True)

    tools = Counter()
    providers = Counter()
    models = Counter()
    reject = Counter()
    n_messages = 0
    n_missing_model = 0
    n_eval_pass = 0
    n_eval_fail = 0
    n_eval_none = 0
    n_unanswered = 0
    msg_lens: list[int] = []
    tool_lens: list[int] = []
    for row in rows:
        tools.update(row.get("tools_used") or [])
        provider = str(row.get("llm_provider") or "")
        if provider:
            providers[provider] += 1
        model = str(row.get("model") or "")
        models[model or "(none)"] += 1
        if not model:
            n_missing_model += 1
        for reason in row.get("reject_reasons") or []:
            reject[reason] += 1
        n_messages += int(row.get("n_messages") or 0)
        msg_lens.append(int(row.get("n_messages") or 0))
        tool_lens.append(int(row.get("n_tool_calls") or 0))
        if int(row.get("n_unanswered") or 0) > 0:
            n_unanswered += 1
        verdict = row.get("eval_verdict")
        if verdict is True:
            n_eval_pass += 1
        elif verdict is False:
            n_eval_fail += 1
        else:
            n_eval_none += 1

    n_eligible = sum(1 for r in rows if r.get("eligible"))
    # Tests pass ``app_data``; don't leak the live DistillationConfig.data_dir.
    data_dir = "" if app_data is not None else dist.data_dir
    traj_file = trajectories_path(app_data=app_data, data_dir=data_dir)
    sft_file = sft_path(app_data=app_data, data_dir=data_dir)

    return {
        "teacher_model_id": teacher,
        "student_model_id": student,
        "filter_sessions_by_teacher": filter_by_teacher,
        "min_tool_calls": min_calls,
        "stats": {
            "n_sessions": len(rows),
            "n_eligible": n_eligible,
            "n_rejected": len(rows) - n_eligible,
            "n_messages": n_messages,
            "median_messages": _median(msg_lens),
            "median_tool_calls": _median(tool_lens),
            "n_missing_model": n_missing_model,
            "n_eval_pass": n_eval_pass,
            "n_eval_fail": n_eval_fail,
            "n_eval_none": n_eval_none,
            "n_unanswered": n_unanswered,
            "tool_histogram": dict(tools.most_common()),
            "provider_histogram": dict(providers.most_common()),
            "model_histogram": dict(models.most_common()),
            "reject_histogram": dict(reject.most_common()),
        },
        "persisted": {
            "trajectories": _file_stats(traj_file),
            "sft": _file_stats(sft_file),
        },
        "sessions": rows,
    }


def session_detail(
    session_id: str,
    *,
    app_data: Path | None = None,
    teacher_model_id: str | None = None,
    student_model_id: str | None = None,
    filter_sessions_by_teacher: bool | None = None,
    min_tool_calls: int | None = None,
) -> dict[str, Any] | None:
    """Return one session's trajectory plus truncated messages."""
    sid = (session_id or "").strip()
    if not sid or "/" in sid or "\\" in sid or sid in {".", ".."}:
        return None
    cfg = _config()
    dist = cfg.distillation
    teacher = (teacher_model_id if teacher_model_id is not None else dist.teacher_model_id).strip()
    filter_by_teacher = (
        dist.filter_sessions_by_teacher
        if filter_sessions_by_teacher is None
        else filter_sessions_by_teacher
    )
    min_calls = dist.min_tool_calls if min_tool_calls is None else min_tool_calls
    transcripts, sessions = _session_dirs(app_data)
    path = transcripts / f"{sid}.jsonl"
    if not path.is_file():
        return None
    records = _read_jsonl(path)
    if not records:
        return None
    meta = _load_session_meta(sessions, sid)
    sidecar = _load_eval_sidecar(sessions, sid)
    traj = extract_trajectory(sid, records, meta, sidecar)
    row = _session_row(
        traj,
        meta,
        min_tool_calls=min_calls,
        teacher_model_id=teacher,
        filter_sessions_by_teacher=filter_by_teacher,
    )
    row["messages"] = [_public_message(m) for m in (traj.get("messages") or [])]
    row["student_model_id"] = (
        student_model_id if student_model_id is not None else dist.student_model_id
    ).strip()
    return row


def _public_message(msg: dict[str, Any]) -> dict[str, Any]:
    content = msg.get("content")
    if content is None:
        text = ""
    elif isinstance(content, str):
        text = content
    else:
        text = json.dumps(content, default=str)
    truncated = len(text) > _DETAIL_CONTENT_CHARS
    if truncated:
        text = text[:_DETAIL_CONTENT_CHARS] + "…"
    out: dict[str, Any] = {
        "role": str(msg.get("role") or ""),
        "content": text,
        "truncated": truncated,
    }
    if msg.get("name"):
        out["name"] = str(msg["name"])
    if msg.get("tool_call_id"):
        out["tool_call_id"] = str(msg["tool_call_id"])
    calls = []
    for tc in msg.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        calls.append({
            "id": str(tc.get("id") or ""),
            "name": str(fn.get("name") or ""),
        })
    if calls:
        out["tool_calls"] = calls
    return out
