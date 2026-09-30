"""Path A collector, quality filter, and SFT formatter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.distillation.collector import (
    collect_trajectories,
    extract_trajectory,
    persist_trajectories,
    records_to_messages,
)
from backend.distillation.formatter import (
    assert_round_trip,
    qwen_probe_tokenizer,
    render_with_tokenizer,
    to_sft_jsonl,
    trajectory_to_sft_example,
)
from backend.distillation.quality import deduplicate, is_high_quality


def _records() -> list[dict]:
    return [
        {"ts": "2026-09-21T00:00:00Z", "type": "user", "content": "Search for AI jobs"},
        {"ts": "2026-09-21T00:00:01Z", "type": "assistant", "role": "assistant", "content": "I'll search."},
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
        {"ts": "2026-09-21T00:00:06Z", "type": "assistant", "role": "assistant", "content": "Done."},
    ]


def test_records_to_messages_groups_tool_calls():
    msgs = records_to_messages(_records())
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    first_ai = msgs[1]
    assert first_ai["content"] == "I'll search."
    assert first_ai["tool_calls"][0]["function"]["name"] == "search_messages"
    assert first_ai["tool_calls"][0]["function"]["arguments"]["query"] == "AI jobs"
    assert msgs[2]["role"] == "tool"
    assert msgs[2]["tool_call_id"] == "c1"


def test_extract_trajectory_lists_tools():
    traj = extract_trajectory("sid", _records(), {"status": "completed", "model": "mlx-community/Qwen3-32B-4bit"})
    assert traj["tools_used"] == ["search_messages", "send_message"]
    assert traj["steps"] == 2
    assert traj["status"] == "completed"


def test_stopped_and_error_sessions_are_excluded(tmp_path: Path):
    transcripts = tmp_path / "transcripts"
    sessions = tmp_path / "sessions"
    transcripts.mkdir()
    sessions.mkdir()

    def _write(sid: str, status: str, error: str = "", extra_records=None):
        recs = extra_records or _records()
        (transcripts / f"{sid}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8",
        )
        (sessions / f"{sid}.json").write_text(
            json.dumps({"id": sid, "status": status, "error": error, "model": "m"}),
            encoding="utf-8",
        )

    _write("ok", "completed")
    _write("stopped", "stopped")
    _write("errored", "error", error="boom")
    _write("thin", "completed", extra_records=[
        {"type": "user", "content": "hi"},
        {"type": "assistant", "content": "hello"},
    ])

    got = collect_trajectories(transcripts_dir=transcripts, sessions_dir=sessions)
    ids = {t["session_id"] for t in got}
    assert ids == {"ok"}


def test_teacher_filter(tmp_path: Path):
    transcripts = tmp_path / "transcripts"
    sessions = tmp_path / "sessions"
    transcripts.mkdir()
    sessions.mkdir()
    (transcripts / "a.jsonl").write_text(
        "\n".join(json.dumps(r) for r in _records()) + "\n", encoding="utf-8",
    )
    (sessions / "a.json").write_text(
        json.dumps({"status": "completed", "model": "mlx-community/Qwen3-8B-4bit"}),
        encoding="utf-8",
    )
    got = collect_trajectories(
        transcripts_dir=transcripts,
        sessions_dir=sessions,
        filter_sessions_by_teacher=True,
        teacher_model_id="mlx-community/Qwen3-32B-4bit",
    )
    assert got == []
    got = collect_trajectories(
        transcripts_dir=transcripts,
        sessions_dir=sessions,
        filter_sessions_by_teacher=True,
        teacher_model_id="mlx-community/Qwen3-8B-4bit",
    )
    assert len(got) == 1


def test_dedup_drops_identical_tool_sequences():
    t = extract_trajectory("a", _records(), {"status": "completed"})
    t2 = extract_trajectory("b", _records(), {"status": "completed"})
    assert len(deduplicate([t, t2])) == 1


def test_formatter_round_trip_qwen_tool_calls():
    traj = extract_trajectory("sid", _records(), {"status": "completed"})
    assert is_high_quality(traj)
    tok = qwen_probe_tokenizer()
    rendered = render_with_tokenizer(traj["messages"], tok)
    assert "<tool_call>" in rendered
    assert_round_trip(rendered, "qwen", ["search_messages", "send_message"])


def test_to_sft_jsonl_writes_messages(tmp_path: Path):
    traj = extract_trajectory("sid", _records(), {"status": "completed"})
    out = tmp_path / "sft.jsonl"
    to_sft_jsonl(
        [traj], out, tokenizer=qwen_probe_tokenizer(), family="qwen", verify_round_trip=True,
    )
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row["messages"][0]["role"] == "user"
    assert row["tools_used"] == ["search_messages", "send_message"]


def test_persist_strips_eval_blob(tmp_path: Path):
    traj = extract_trajectory(
        "sid", _records(), {"status": "completed"},
        eval_sidecar={"status": "done", "overall_score": 0.8, "pass_count": 2},
    )
    dest = persist_trajectories([traj], tmp_path / "activity" / "trajectories.jsonl")
    row = json.loads(dest.read_text(encoding="utf-8").splitlines()[0])
    assert "eval" not in row
    assert row["eval_overall_score"] == 0.8
