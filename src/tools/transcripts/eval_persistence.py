"""Shared eval-state and report persistence helpers.

Used by both the Claude and OpenClaw evaluator hook MCP servers to
checkpoint evaluation progress and save Markdown reports.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_REPORTS_DIR_NAME = "agent-eval-reports"


def eval_state_dir() -> Path:
    d = Path.home() / _REPORTS_DIR_NAME / ".state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def state_key(session_path: str) -> str:
    return hashlib.sha256(session_path.encode()).hexdigest()[:16]


def save_eval_state(session_path: str, state_json: str) -> dict:
    """Persist evaluation progress. Returns a status dict."""
    try:
        parsed = json.loads(state_json)
    except (json.JSONDecodeError, TypeError) as exc:
        return {"error": f"Invalid JSON: {exc}"}

    try:
        key = state_key(session_path)
        state_file = eval_state_dir() / f"eval-state-{key}.json"
        envelope = {
            "session_path": session_path,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "state": parsed,
        }
        state_file.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
        logger.info("Eval state saved to %s", state_file)
        return {"status": "ok", "path": str(state_file)}
    except Exception as exc:
        logger.exception("save_eval_state failed")
        return {"error": f"{type(exc).__name__}: {exc}"}


def load_eval_state(session_path: str) -> dict:
    """Load previously saved evaluation state. Returns the state dict or {}."""
    try:
        key = state_key(session_path)
        state_file = eval_state_dir() / f"eval-state-{key}.json"
        if not state_file.is_file():
            return {}
        envelope = json.loads(state_file.read_text(encoding="utf-8"))
        return envelope.get("state", {})
    except Exception as exc:
        logger.exception("load_eval_state failed")
        return {"error": f"{type(exc).__name__}: {exc}"}


def save_report(
    report_markdown: str,
    filename: str = "",
    output_dir: str = "",
    filename_prefix: str = "eval-report",
) -> dict:
    """Save an evaluation report as a Markdown file. Returns a status dict."""
    try:
        out = Path(output_dir).expanduser() if output_dir else Path.home() / _REPORTS_DIR_NAME
        out.mkdir(parents=True, exist_ok=True)
        if not filename:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            filename = f"{filename_prefix}-{ts}.md"
        if not filename.endswith(".md"):
            filename += ".md"
        filepath = out / filename
        filepath.write_text(report_markdown, encoding="utf-8")
        logger.info("Report saved to %s", filepath)
        return {
            "status": "ok",
            "path": str(filepath),
            "size_bytes": filepath.stat().st_size,
        }
    except Exception as exc:
        logger.exception("save_report failed")
        return {"error": f"{type(exc).__name__}: {exc}"}
