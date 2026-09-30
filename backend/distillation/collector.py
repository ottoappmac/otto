"""Extract Path A trajectories from session transcripts + session meta."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

from backend.distillation.paths import trajectories_path
from backend.distillation.quality import deduplicate, is_high_quality

logger = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Skipping malformed JSONL at %s:%d", path, line_no)
                continue
            if isinstance(rec, dict):
                records.append(rec)
    return records


def _load_session_meta(sessions_dir: Path, session_id: str) -> dict[str, Any]:
    path = sessions_dir / f"{session_id}.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _load_eval_sidecar(sessions_dir: Path, session_id: str) -> dict[str, Any] | None:
    path = sessions_dir / f"{session_id}.eval.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _flush_assistant(
    messages: list[dict[str, Any]],
    text: str,
    tool_calls: list[dict[str, Any]],
) -> None:
    if not text and not tool_calls:
        return
    entry: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        entry["tool_calls"] = tool_calls
    messages.append(entry)


def records_to_messages(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert transcript JSONL records into OpenAI-style chat messages.

    Consecutive ``assistant`` text + ``tool_call`` events become a single
    assistant message with ``tool_calls``, matching
    ``ChatMLXText._message_to_chat_dict``.
    """
    messages: list[dict[str, Any]] = []
    pending_text = ""
    pending_calls: list[dict[str, Any]] = []

    for rec in records:
        kind = rec.get("type")
        if kind == "user":
            _flush_assistant(messages, pending_text, pending_calls)
            pending_text, pending_calls = "", []
            messages.append({"role": "user", "content": rec.get("content") or ""})
        elif kind == "assistant":
            chunk = rec.get("content") or ""
            if pending_calls:
                # Text after tool calls belongs to the next assistant turn.
                _flush_assistant(messages, pending_text, pending_calls)
                pending_text, pending_calls = str(chunk), []
            else:
                pending_text = (pending_text + ("\n" if pending_text and chunk else "") + str(chunk))
        elif kind == "tool_call":
            args = rec.get("content") if isinstance(rec.get("content"), dict) else {}
            pending_calls.append({
                "id": rec.get("tool_call_id") or "",
                "type": "function",
                "function": {
                    "name": rec.get("tool") or "",
                    "arguments": args or {},
                },
            })
        elif kind == "tool_result":
            _flush_assistant(messages, pending_text, pending_calls)
            pending_text, pending_calls = "", []
            tool_msg: dict[str, Any] = {
                "role": "tool",
                "content": rec.get("content") if rec.get("content") is not None else "",
            }
            if rec.get("tool_call_id"):
                tool_msg["tool_call_id"] = rec["tool_call_id"]
            if rec.get("tool"):
                tool_msg["name"] = rec["tool"]
            messages.append(tool_msg)
        # system / unknown types are ignored — they are not SFT labels

    _flush_assistant(messages, pending_text, pending_calls)
    return messages


def _tools_used(messages: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for msg in messages:
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") if isinstance(tc, dict) else None
            name = (fn or {}).get("name") if isinstance(fn, dict) else ""
            if name and name not in names:
                names.append(name)
    return names


def extract_trajectory(
    session_id: str,
    records: list[dict[str, Any]],
    meta: dict[str, Any],
    eval_sidecar: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one trajectory dict from a session's transcript + meta."""
    messages = records_to_messages(records)
    tools = _tools_used(messages)
    timestamp = ""
    if records:
        timestamp = str(records[0].get("ts") or "")
    return {
        "session_id": session_id,
        "timestamp": timestamp,
        "messages": messages,
        "tools_used": tools,
        "status": str(meta.get("status") or ""),
        "error": meta.get("error") or "",
        "model": meta.get("model") or "",
        "steps": len(tools),
        "eval": eval_sidecar,
    }


def collect_trajectories(
    *,
    transcripts_dir: Path,
    sessions_dir: Path,
    min_tool_calls: int = 2,
    teacher_model_id: str = "",
    filter_sessions_by_teacher: bool = False,
    require_eval_pass: bool = False,
) -> list[dict[str, Any]]:
    """Scan transcripts, join session meta, apply the quality filter."""
    collected: list[dict[str, Any]] = []
    if not transcripts_dir.is_dir():
        return collected
    for path in sorted(transcripts_dir.glob("*.jsonl")):
        session_id = path.stem
        records = _read_jsonl(path)
        if not records:
            continue
        meta = _load_session_meta(sessions_dir, session_id)
        sidecar = _load_eval_sidecar(sessions_dir, session_id)
        traj = extract_trajectory(session_id, records, meta, sidecar)
        if is_high_quality(
            traj,
            min_tool_calls=min_tool_calls,
            teacher_model_id=teacher_model_id,
            filter_sessions_by_teacher=filter_sessions_by_teacher,
            require_eval_pass=require_eval_pass,
        ):
            collected.append(traj)
    return deduplicate(collected)


def persist_trajectories(
    trajectories: list[dict[str, Any]],
    output_path: Path | None = None,
    *,
    app_data: Path | None = None,
    data_dir: str = "",
) -> Path:
    """Write accepted trajectories to the durable activity JSONL."""
    dest = output_path or trajectories_path(app_data=app_data, data_dir=data_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for traj in trajectories:
            # Drop the raw eval sidecar blob — keep a compact summary.
            row = dict(traj)
            ev = row.pop("eval", None)
            if isinstance(ev, dict):
                row["eval_overall_score"] = ev.get("overall_score")
                row["eval_pass_count"] = ev.get("pass_count")
            fh.write(json.dumps(row, default=str) + "\n")
    return dest
