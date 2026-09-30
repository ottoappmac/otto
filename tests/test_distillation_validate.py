"""JSONL / ShareGPT format validator for Distill uploads."""

from __future__ import annotations

import json
from pathlib import Path

from backend.distillation.uploads import delete_upload, list_uploads, save_upload, write_sft_from_upload
from backend.distillation.validate import normalize_to_sft, validate_text


def _sft_row(user="Search for jobs", assistant="Done.", tools=None) -> dict:
    messages = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": "I'll search.", "tool_calls": tools or []},
    ]
    if tools:
        for tc in tools:
            tid = tc.get("id") or "c1"
            name = (tc.get("function") or {}).get("name") or "tool"
            messages.append({"role": "tool", "tool_call_id": tid, "name": name, "content": "ok"})
        messages.append({"role": "assistant", "content": assistant})
    else:
        messages[1]["content"] = assistant
    return {"messages": messages}


def test_valid_messages_jsonl():
    line = json.dumps(_sft_row())
    report = validate_text(line + "\n" + line + "\n", filename="ok.jsonl")
    assert report["ok"] is True
    assert report["n_valid"] == 2
    assert report["format"] == "messages"
    assert report["container"] == "jsonl"
    assert "Search for jobs" in report["sample_preview"]


def test_sharegpt_and_json_array():
    payload = [{
        "conversations": [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi there"},
        ],
    }]
    report = validate_text(json.dumps(payload), filename="sg.json")
    assert report["ok"] is True
    assert report["format"] == "sharegpt"
    assert report["container"] == "json_array"


def test_rejects_empty_and_bad_role():
    empty = validate_text("", filename="x.jsonl")
    assert empty["ok"] is False
    assert any(e["code"] == "empty" for e in empty["errors"])

    bad = validate_text(json.dumps({"messages": [{"role": "narrator", "content": "x"}]}), filename="x.jsonl")
    assert bad["ok"] is False
    codes = {e["code"] for e in bad["errors"]}
    assert "bad_role" in codes or "no_user" in codes


def test_rejects_user_only():
    row = {"messages": [{"role": "user", "content": "hi"}]}
    report = validate_text(json.dumps(row), filename="x.jsonl")
    assert report["ok"] is False
    assert any(e["code"] == "no_assistant" for e in report["errors"])


def test_malformed_jsonl_line_is_invalid_not_fatal_for_others():
    good = json.dumps(_sft_row())
    text = "not-json\n" + good + "\n"
    report = validate_text(text, filename="mix.jsonl")
    assert report["n_valid"] == 1
    assert report["n_invalid"] == 1
    assert report["ok"] is True


def test_tool_call_without_name_is_error():
    row = _sft_row(tools=[{"id": "c1", "type": "function", "function": {}}])
    report = validate_text(json.dumps(row), filename="t.jsonl")
    assert report["ok"] is False
    assert any(e["code"] == "tool_missing_name" for e in report["errors"])


def test_normalize_drops_invalid_and_keeps_sharegpt(tmp_path: Path):
    src = tmp_path / "in.jsonl"
    src.write_text(
        json.dumps({"messages": [{"role": "user", "content": "only"}]}) + "\n"
        + json.dumps(_sft_row(user="Keep me")) + "\n",
        encoding="utf-8",
    )
    dest = tmp_path / "out.jsonl"
    report = normalize_to_sft(src, dest)
    assert report["n_written"] == 1
    row = json.loads(dest.read_text(encoding="utf-8").splitlines()[0])
    assert row["messages"][0]["content"] == "Keep me"


def test_save_upload_rejects_invalid(tmp_path: Path):
    result = save_upload(b"{}", filename="bad.jsonl", app_data=tmp_path)
    assert result["saved"] is False
    assert result["upload"] is None
    assert list_uploads(app_data=tmp_path) == []


def test_save_list_delete_and_write_sft(tmp_path: Path):
    raw = (json.dumps(_sft_row()) + "\n").encode("utf-8")
    result = save_upload(raw, filename="jobs.jsonl", app_data=tmp_path)
    assert result["saved"] is True
    rec = result["upload"]
    assert rec["n_valid"] == 1
    assert rec["filename"] == "jobs.jsonl"
    listed = list_uploads(app_data=tmp_path)
    assert len(listed) == 1
    dest = tmp_path / "sft.jsonl"
    written = write_sft_from_upload(rec["id"], dest, app_data=tmp_path)
    assert written["n_written"] == 1
    assert dest.is_file()
    assert delete_upload(rec["id"], app_data=tmp_path) is True
    assert list_uploads(app_data=tmp_path) == []
