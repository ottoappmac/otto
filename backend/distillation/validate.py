"""Validate user-supplied JSONL for mlx_lm ChatDataset / Otto SFT training."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

MAX_BYTES = 50 * 1024 * 1024
MAX_ISSUES = 25
ALLOWED_ROLES = frozenset({"system", "user", "assistant", "tool"})
_SHAREGPT_FROM = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "function": "tool",
    "tool": "tool",
}


def validate_bytes(raw: bytes, *, filename: str = "upload.jsonl") -> dict[str, Any]:
    """Validate a file body.  Does not write anything."""
    report = _empty_report(filename)
    if len(raw) > MAX_BYTES:
        report["errors"].append({
            "line": 0,
            "code": "too_large",
            "message": f"File is {len(raw):,} bytes; max is {MAX_BYTES:,}.",
        })
        report["ok"] = False
        return report
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        report["errors"].append({
            "line": 0,
            "code": "not_utf8",
            "message": f"File is not UTF-8 ({exc}).",
        })
        report["ok"] = False
        return report
    return validate_text(text, filename=filename)


def validate_path(path: Path, *, filename: str = "") -> dict[str, Any]:
    raw = path.read_bytes() if path.is_file() else b""
    if not path.is_file():
        report = _empty_report(filename or path.name)
        report["errors"].append({
            "line": 0,
            "code": "missing",
            "message": f"File not found: {path}",
        })
        report["ok"] = False
        return report
    return validate_bytes(raw, filename=filename or path.name)


def validate_text(text: str, *, filename: str = "upload.jsonl") -> dict[str, Any]:
    """Validate JSONL (or a JSON array / single object) and collect stats."""
    report = _empty_report(filename)
    stripped = text.lstrip()
    if not stripped:
        report["errors"].append({
            "line": 0,
            "code": "empty",
            "message": "File is empty.",
        })
        report["ok"] = False
        return report

    container, rows_or_none = _parse_container(text)
    report["container"] = container
    if rows_or_none is None and container in {"json_array", "json_object"}:
        return report
    rows = rows_or_none if rows_or_none is not None else list(_iter_jsonl(text, report))

    formats = Counter()
    n_messages = 0
    n_tool_calls = 0
    tools = Counter()
    for line_no, obj in rows:
        report["n_rows"] += 1
        if not isinstance(obj, dict):
            _add_issue(report, "errors", line_no, "not_object", "Row is not a JSON object.")
            report["n_invalid"] += 1
            continue
        messages, fmt, coerce_warns = _coerce_messages(obj)
        for w in coerce_warns:
            _add_issue(report, "warnings", line_no, w["code"], w["message"])
        if messages is None:
            _add_issue(
                report, "errors", line_no, "no_messages",
                "Need a messages[] array (OpenAI / mlx_lm) or conversations[] (ShareGPT).",
            )
            report["n_invalid"] += 1
            continue
        issues = _validate_messages(messages)
        fatal = [i for i in issues if i["severity"] == "error"]
        for issue in issues:
            _add_issue(report, "errors" if issue["severity"] == "error" else "warnings",
                       line_no, issue["code"], issue["message"])
        if fatal:
            report["n_invalid"] += 1
            continue
        report["n_valid"] += 1
        formats[fmt] += 1
        if not report["sample_preview"]:
            for msg in messages:
                if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                    preview = " ".join(msg["content"].split())
                    report["sample_preview"] = (preview[:159] + "…") if len(preview) > 160 else preview
                    break
        n_messages += len(messages)
        for msg in messages:
            for tc in msg.get("tool_calls") or []:
                n_tool_calls += 1
                name = _tool_name(tc)
                if name:
                    tools[name] += 1

    if formats:
        report["format"] = formats.most_common(1)[0][0]
    report["format_histogram"] = dict(formats)
    report["n_messages"] = n_messages
    report["n_tool_calls"] = n_tool_calls
    report["tool_histogram"] = dict(tools.most_common())
    report["ok"] = report["n_valid"] > 0
    if report["n_valid"] == 0 and not report["errors"]:
        _add_issue(report, "errors", 0, "no_valid_rows", "No valid training examples.")
    return report


def normalize_to_sft(path: Path, dest: Path) -> dict[str, Any]:
    """Write valid rows from *path* as mlx_lm ChatDataset JSONL to *dest*."""
    raw = path.read_bytes()
    report = validate_bytes(raw, filename=path.name)
    text = raw.decode("utf-8")
    rows = _load_rows(text)
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with dest.open("w", encoding="utf-8") as fh:
        for _line_no, obj in rows:
            if not isinstance(obj, dict):
                continue
            messages, _fmt, _warns = _coerce_messages(obj)
            if messages is None:
                continue
            if any(i["severity"] == "error" for i in _validate_messages(messages)):
                continue
            example = {
                "messages": messages,
                "session_id": str(obj.get("session_id") or ""),
                "tools_used": list(obj.get("tools_used") or _tools_used(messages)),
            }
            fh.write(json.dumps(example, default=str) + "\n")
            written += 1
    report["n_written"] = written
    return report


def _empty_report(filename: str) -> dict[str, Any]:
    return {
        "ok": False,
        "filename": filename,
        "container": "jsonl",
        "format": "",
        "format_histogram": {},
        "n_rows": 0,
        "n_valid": 0,
        "n_invalid": 0,
        "n_messages": 0,
        "n_tool_calls": 0,
        "tool_histogram": {},
        "errors": [],
        "warnings": [],
        "sample_preview": "",
    }


def _add_issue(
    report: dict[str, Any],
    bucket: str,
    line: int,
    code: str,
    message: str,
) -> None:
    items: list[dict[str, Any]] = report[bucket]
    if len(items) >= MAX_ISSUES:
        return
    items.append({"line": line, "code": code, "message": message})


def _iter_jsonl(text: str, report: dict[str, Any]) -> list[tuple[int, Any]]:
    rows: list[tuple[int, Any]] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            _add_issue(report, "errors", line_no, "invalid_json", f"Invalid JSON: {exc.msg}")
            report["n_rows"] += 1
            report["n_invalid"] += 1
            continue
        rows.append((line_no, obj))
    return rows


def _parse_container(text: str) -> tuple[str, list[tuple[int, Any]] | None]:
    """Detect JSON array / single object vs JSONL.

    A whole-file ``json.loads`` succeeds for one object or an array.  A
    JSONL stream of objects fails that parse (extra data) and falls through.
    """
    stripped = text.lstrip()
    if stripped.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return "jsonl", None
        if isinstance(parsed, list):
            return "json_array", [(i + 1, item) for i, item in enumerate(parsed)]
        return "json_array", None
    if stripped.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return "jsonl", None
        if isinstance(parsed, dict):
            return "json_object", [(1, parsed)]
    return "jsonl", None


def _load_rows(text: str) -> list[tuple[int, Any]]:
    _container, rows = _parse_container(text)
    if rows is not None:
        return rows
    return _iter_jsonl(text, _empty_report(""))


def _coerce_messages(obj: dict[str, Any]) -> tuple[list[dict[str, Any]] | None, str, list[dict[str, str]]]:
    warns: list[dict[str, str]] = []
    messages = obj.get("messages")
    if isinstance(messages, list):
        return messages, "messages", warns
    conv = obj.get("conversations")
    if isinstance(conv, list):
        converted: list[dict[str, Any]] = []
        for turn in conv:
            if not isinstance(turn, dict):
                continue
            src = str(turn.get("from") or turn.get("role") or "").strip().lower()
            role = _SHAREGPT_FROM.get(src)
            if not role:
                warns.append({
                    "code": "unknown_sharegpt_from",
                    "message": f"Unknown ShareGPT speaker {src!r}; skipped.",
                })
                continue
            converted.append({"role": role, "content": turn.get("value") if "value" in turn else turn.get("content") or ""})
        return converted, "sharegpt", warns
    return None, "unknown", warns


def _validate_messages(messages: list[Any]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if not messages:
        issues.append({"severity": "error", "code": "empty_messages", "message": "messages[] is empty."})
        return issues
    roles: list[str] = []
    pending_calls: set[str] = set()
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            issues.append({
                "severity": "error",
                "code": "message_not_object",
                "message": f"messages[{i}] is not an object.",
            })
            continue
        role = str(msg.get("role") or "").strip()
        if role not in ALLOWED_ROLES:
            issues.append({
                "severity": "error",
                "code": "bad_role",
                "message": f"messages[{i}] role {role!r} is not one of {sorted(ALLOWED_ROLES)}.",
            })
            continue
        roles.append(role)
        content = msg.get("content")
        tool_calls = msg.get("tool_calls") or []
        if role in {"user", "system"} and not (isinstance(content, str) and content.strip()):
            issues.append({
                "severity": "error",
                "code": "empty_content",
                "message": f"messages[{i}] ({role}) needs non-empty string content.",
            })
        if role == "assistant":
            if tool_calls and not isinstance(tool_calls, list):
                issues.append({
                    "severity": "error",
                    "code": "bad_tool_calls",
                    "message": f"messages[{i}] tool_calls must be an array.",
                })
                tool_calls = []
            if not tool_calls and content is not None and not isinstance(content, str):
                issues.append({
                    "severity": "error",
                    "code": "bad_content",
                    "message": f"messages[{i}] assistant content must be a string.",
                })
            if not tool_calls and not (isinstance(content, str) and content.strip()):
                issues.append({
                    "severity": "warning",
                    "code": "empty_assistant",
                    "message": f"messages[{i}] assistant has no content and no tool_calls.",
                })
            for j, tc in enumerate(tool_calls if isinstance(tool_calls, list) else []):
                if not isinstance(tc, dict):
                    issues.append({
                        "severity": "error",
                        "code": "bad_tool_call",
                        "message": f"messages[{i}].tool_calls[{j}] is not an object.",
                    })
                    continue
                name = _tool_name(tc)
                if not name:
                    issues.append({
                        "severity": "error",
                        "code": "tool_missing_name",
                        "message": f"messages[{i}].tool_calls[{j}] has no function name.",
                    })
                tid = str(tc.get("id") or "")
                if tid:
                    pending_calls.add(tid)
        if role == "tool":
            tid = str(msg.get("tool_call_id") or "")
            if tid:
                pending_calls.discard(tid)
            elif not msg.get("name"):
                issues.append({
                    "severity": "warning",
                    "code": "tool_unlinked",
                    "message": f"messages[{i}] tool result has no tool_call_id.",
                })
    if "user" not in roles:
        issues.append({"severity": "error", "code": "no_user", "message": "Need at least one user message."})
    if "assistant" not in roles:
        issues.append({"severity": "error", "code": "no_assistant", "message": "Need at least one assistant message."})
    if pending_calls:
        issues.append({
            "severity": "warning",
            "code": "unanswered_tools",
            "message": f"{len(pending_calls)} tool call(s) have no matching tool result.",
        })
    return issues


def _tool_name(tc: dict[str, Any]) -> str:
    fn = tc.get("function") if isinstance(tc.get("function"), dict) else None
    if fn and fn.get("name"):
        return str(fn["name"])
    return str(tc.get("name") or "")


def _tools_used(messages: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for msg in messages:
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            name = _tool_name(tc)
            if name and name not in names:
                names.append(name)
    return names
