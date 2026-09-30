"""Quality gate for Path A trajectories."""

from __future__ import annotations

from typing import Any

# SessionInfo.status values that must never enter the training set.
_REJECT_STATUSES = frozenset({"error", "stopped", "running", "idle", "awaiting_input"})


def eval_passed(sidecar: dict[str, Any] | None) -> bool | None:
    """Return True/False when an eval sidecar exists, else None.

    A missing sidecar is not a rejection — eval is optional.  A completed
    eval with zero passing metrics is a rejection *only* when the caller
    sets ``require_eval_pass``.
    """
    if not sidecar:
        return None
    if sidecar.get("status") != "done":
        return None
    total = sidecar.get("total")
    passed = sidecar.get("pass_count")
    if isinstance(total, int) and total > 0 and isinstance(passed, int):
        return passed > 0
    overall = sidecar.get("overall_score")
    if isinstance(overall, (int, float)):
        return overall > 0
    return None


def reject_reasons(
    trajectory: dict[str, Any],
    *,
    min_tool_calls: int = 2,
    teacher_model_id: str = "",
    filter_sessions_by_teacher: bool = False,
    require_eval_pass: bool = False,
) -> list[str]:
    """Return why *trajectory* is excluded from SFT, or ``[]`` if eligible."""
    reasons: list[str] = []
    status = str(trajectory.get("status") or "")
    if status != "completed" or status in _REJECT_STATUSES:
        reasons.append("not_completed")
    if trajectory.get("error"):
        reasons.append("has_error")

    tools = list(trajectory.get("tools_used") or [])
    if len(tools) < min_tool_calls:
        reasons.append("too_few_tools")

    if filter_sessions_by_teacher:
        expected = teacher_model_id.strip()
        actual = str(trajectory.get("model") or "").strip()
        if not expected or actual != expected:
            reasons.append("teacher_mismatch")

    if require_eval_pass:
        verdict = eval_passed(trajectory.get("eval"))
        if verdict is False:
            reasons.append("eval_failed")

    return reasons


def is_high_quality(
    trajectory: dict[str, Any],
    *,
    min_tool_calls: int = 2,
    teacher_model_id: str = "",
    filter_sessions_by_teacher: bool = False,
    require_eval_pass: bool = False,
) -> bool:
    """Return True if *trajectory* may be written to the SFT dataset."""
    return not reject_reasons(
        trajectory,
        min_tool_calls=min_tool_calls,
        teacher_model_id=teacher_model_id,
        filter_sessions_by_teacher=filter_sessions_by_teacher,
        require_eval_pass=require_eval_pass,
    )


def deduplicate(
    trajectories: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop trajectories that share the same user text + tool-name sequence."""
    seen: set[tuple[str, tuple[str, ...]]] = set()
    out: list[dict[str, Any]] = []
    for traj in trajectories:
        key = _dedup_key(traj)
        if key in seen:
            continue
        seen.add(key)
        out.append(traj)
    return out


def _dedup_key(trajectory: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    user = ""
    for msg in trajectory.get("messages") or []:
        if msg.get("role") == "user":
            user = str(msg.get("content") or "")
            break
    tools = tuple(str(t) for t in (trajectory.get("tools_used") or []))
    return (user, tools)
