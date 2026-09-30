"""Dataset viewer: scan all sessions, stats, and session detail."""

from __future__ import annotations

import json
from pathlib import Path

from backend.distillation.collector import persist_trajectories
from backend.distillation.dataset import scan_dataset, session_detail
from backend.distillation.quality import reject_reasons


def _records(extra=None) -> list[dict]:
    recs = [
        {"ts": "2026-09-21T00:00:00Z", "type": "user", "content": "Search for AI jobs"},
        {"ts": "2026-09-21T00:00:01Z", "type": "assistant", "content": "I'll search."},
        {
            "ts": "2026-09-21T00:00:02Z",
            "type": "tool_call",
            "tool": "search_messages",
            "tool_call_id": "c1",
            "content": {"query": "AI jobs"},
        },
        {
            "ts": "2026-09-21T00:00:03Z",
            "type": "tool_result",
            "tool": "search_messages",
            "tool_call_id": "c1",
            "content": "3 hits",
        },
        {
            "ts": "2026-09-21T00:00:04Z",
            "type": "tool_call",
            "tool": "send_message",
            "tool_call_id": "c2",
            "content": {"to": "a@b.c", "body": "jobs"},
        },
        {
            "ts": "2026-09-21T00:00:05Z",
            "type": "tool_result",
            "tool": "send_message",
            "tool_call_id": "c2",
            "content": "sent",
        },
        {"ts": "2026-09-21T00:00:06Z", "type": "assistant", "content": "Done."},
    ]
    if extra:
        recs.extend(extra)
    return recs


def _write_session(root: Path, sid: str, status: str, recs, **meta):
    (root / "transcripts").mkdir(exist_ok=True)
    (root / "sessions").mkdir(exist_ok=True)
    (root / "transcripts" / f"{sid}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8",
    )
    payload = {"id": sid, "status": status, "title": meta.pop("title", sid), **meta}
    (root / "sessions" / f"{sid}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_reject_reasons_cover_status_and_tools():
    thin = {
        "status": "completed",
        "error": "",
        "tools_used": ["one"],
        "messages": [{"role": "user", "content": "hi"}],
    }
    assert "too_few_tools" in reject_reasons(thin)
    stopped = {**thin, "status": "stopped", "tools_used": ["a", "b"]}
    assert "not_completed" in reject_reasons(stopped)


def test_scan_includes_rejected_and_stats(tmp_path: Path):
    _write_session(
        tmp_path, "ok", "completed", _records(),
        model="mlx-community/Qwen3-8B-4bit", llm_provider="omlx",
        title="Find jobs",
    )
    _write_session(
        tmp_path, "thin", "completed",
        [
            {"ts": "2026-09-21T01:00:00Z", "type": "user", "content": "hi"},
            {"ts": "2026-09-21T01:00:01Z", "type": "assistant", "content": "hello"},
        ],
        model="", llm_provider="omlx",
    )
    unanswered = _records()[:-1]  # drop final assistant; leave c2 answered
    unanswered.append({
        "ts": "2026-09-21T00:00:07Z",
        "type": "tool_call",
        "tool": "execute",
        "tool_call_id": "c3",
        "content": {"cmd": "ls"},
    })
    _write_session(
        tmp_path, "gap", "completed", unanswered,
        model="mlx-community/Qwen3-8B-4bit", llm_provider="mlx",
    )
    _write_session(
        tmp_path, "dead", "error", _records(),
        model="x", error="boom",
    )

    report = scan_dataset(
        app_data=tmp_path,
        teacher_model_id="",
        student_model_id="mlx-community/Qwen3-8B-4bit",
        filter_sessions_by_teacher=False,
        min_tool_calls=2,
    )
    by_id = {r["session_id"]: r for r in report["sessions"]}
    assert set(by_id) == {"ok", "thin", "gap", "dead"}
    assert by_id["ok"]["eligible"] is True
    assert by_id["thin"]["eligible"] is False
    assert "too_few_tools" in by_id["thin"]["reject_reasons"]
    assert by_id["thin"]["model"] == ""
    assert by_id["gap"]["n_unanswered"] == 1
    assert by_id["dead"]["eligible"] is False
    assert "has_error" in by_id["dead"]["reject_reasons"]
    assert by_id["ok"]["preview"].startswith("Search for AI jobs")
    assert report["stats"]["n_sessions"] == 4
    assert report["stats"]["n_eligible"] == 2  # ok + gap (gap still has 3 tools)
    assert report["stats"]["n_missing_model"] == 1
    assert report["stats"]["n_unanswered"] == 1
    assert report["stats"]["tool_histogram"]["search_messages"] == 3
    assert report["persisted"]["trajectories"]["exists"] is False


def test_persisted_file_stats(tmp_path: Path):
    _write_session(tmp_path, "ok", "completed", _records(), model="m")
    from backend.distillation.collector import extract_trajectory

    traj = extract_trajectory(
        "ok", _records(), {"status": "completed", "model": "m"},
    )
    persist_trajectories([traj], tmp_path / "distillation" / "activity" / "trajectories.jsonl")
    report = scan_dataset(app_data=tmp_path, min_tool_calls=2)
    assert report["persisted"]["trajectories"]["exists"] is True
    assert report["persisted"]["trajectories"]["rows"] == 1
    assert report["persisted"]["trajectories"]["bytes"] > 0


def test_session_detail_truncates_and_lists_tools(tmp_path: Path):
    recs = _records()
    recs[3]["content"] = "x" * 5000
    _write_session(tmp_path, "ok", "completed", recs, model="m", title="Jobs")
    detail = session_detail("ok", app_data=tmp_path, min_tool_calls=2)
    assert detail is not None
    assert detail["session_id"] == "ok"
    assert detail["eligible"] is True
    roles = [m["role"] for m in detail["messages"]]
    assert roles[0] == "user"
    tool_msg = next(m for m in detail["messages"] if m["role"] == "tool" and m.get("tool_call_id") == "c1")
    assert tool_msg["truncated"] is True
    assert tool_msg["content"].endswith("…")
    first_ai = next(m for m in detail["messages"] if m.get("tool_calls"))
    assert first_ai["tool_calls"][0]["name"] == "search_messages"


def test_session_detail_missing(tmp_path: Path):
    assert session_detail("nope", app_data=tmp_path) is None
    assert session_detail("../etc/passwd", app_data=tmp_path) is None
