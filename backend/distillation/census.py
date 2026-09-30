"""Path A census / collect CLI.

Usage:
    python -m backend.distillation.census
    python -m backend.distillation.census --persist
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from backend.config import AppConfig
from backend.distillation.collector import collect_trajectories, persist_trajectories
from backend.session_manager import _sessions_dir
from backend.session_transcript import _transcripts_dir


def _config() -> AppConfig:
    try:
        return AppConfig.load()
    except Exception:
        return AppConfig()


def run_census(
    *,
    persist: bool = False,
    app_data: Path | None = None,
    teacher_model_id: str | None = None,
    student_model_id: str | None = None,
    filter_sessions_by_teacher: bool | None = None,
    min_tool_calls: int | None = None,
) -> dict:
    cfg = _config()
    dist = cfg.distillation
    teacher = (teacher_model_id or dist.teacher_model_id).strip()
    student = (student_model_id or dist.student_model_id).strip()
    filter_by_teacher = (
        dist.filter_sessions_by_teacher
        if filter_sessions_by_teacher is None
        else filter_sessions_by_teacher
    )
    min_calls = dist.min_tool_calls if min_tool_calls is None else min_tool_calls
    transcripts = (
        (app_data / "transcripts") if app_data else _transcripts_dir()
    )
    sessions = (app_data / "sessions") if app_data else _sessions_dir()
    trajectories = collect_trajectories(
        transcripts_dir=transcripts,
        sessions_dir=sessions,
        min_tool_calls=min_calls,
        teacher_model_id=teacher,
        filter_sessions_by_teacher=filter_by_teacher,
    )
    tools = Counter()
    for traj in trajectories:
        tools.update(traj.get("tools_used") or [])
    report = {
        "n_trajectories": len(trajectories),
        "tool_histogram": dict(tools.most_common()),
        "median_steps": _median([t.get("steps") or 0 for t in trajectories]),
        "with_eval": sum(1 for t in trajectories if t.get("eval")),
        "teacher_model_id": teacher,
        "student_model_id": student,
        "filter_sessions_by_teacher": filter_by_teacher,
    }
    if persist and trajectories:
        dest = persist_trajectories(
            trajectories, app_data=app_data, data_dir=dist.data_dir,
        )
        report["wrote"] = str(dest)
    return report


def _median(vals: list[int]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Count Path A distillation trajectories")
    parser.add_argument("--persist", action="store_true", help="Write accepted traces to distillation/activity/")
    parser.add_argument("--app-data", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run_census(persist=args.persist, app_data=args.app_data)
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
