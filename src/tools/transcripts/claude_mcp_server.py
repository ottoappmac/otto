#!/usr/bin/env python3
"""Claude Evaluator Hook MCP Server.

Standalone MCP server that exposes tools for reading and parsing Claude
session transcripts.  Supports Claude Code and Claude Cowork.

Usage::

    # Streamable HTTP (default)
    python -m tools.transcripts.claude_mcp_server --port 8942

    # stdio (for direct MCP client piping)
    python -m tools.transcripts.claude_mcp_server --transport stdio
"""

import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

_src_dir = str(Path(__file__).resolve().parent.parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from mcp.server.fastmcp import FastMCP  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("claude-eval-hook")

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8942

mcp = FastMCP(
    "claude-eval-hook",
    host=_DEFAULT_HOST,
    port=_DEFAULT_PORT,
    instructions=(
        "Claude evaluator hook.  Use `list_platforms` to see supported "
        "Claude platforms, `list_projects` to discover workspaces, "
        "`list_sessions` to find sessions, `wait_for_activity` to block "
        "until new transcript data arrives (preferred for real-time "
        "monitoring), `check_session_activity` for instant metadata checks, "
        "and `parse_session_turns` / `get_session_summary` to extract "
        "structured evaluation data.  Use `save_eval_state` / "
        "`load_eval_state` to persist evaluation progress."
    ),
)


# ---------------------------------------------------------------------------
# Parser registry — lazy-loaded, one per platform
# ---------------------------------------------------------------------------

_parsers: dict[str, Any] = {}

_MAX_TIMEOUT_SECS = 120
_MAX_POLL_INTERVAL_SECS = 30

_DISABLED_MSG = (
    "Claude Evaluator Hook is disabled. "
    "Enable it in Settings → Integrations → Claude Hook."
)

_PLATFORM_INFO = {
    "claude_code": {
        "name": "Claude Code",
        "description": "Reads ~/.claude/projects/ JSONL transcripts from the Claude Code CLI.",
        "default_base_path": "~/.claude/projects",
    },
    "cowork": {
        "name": "Claude Cowork",
        "description": (
            "Reads Cowork (Desktop agent-mode) session transcripts from "
            "~/Library/Application Support/Claude/local-agent-mode-sessions/."
        ),
        "default_base_path": "~/Library/Application Support/Claude",
    },
}


def _is_claude_hook_enabled() -> bool:
    """Check whether the Claude Hook integration toggle is on."""
    try:
        from backend.config import AppConfig
        return AppConfig.load().claude_hook.enabled
    except Exception:
        return False


def _check_enabled() -> None:
    """Raise if the Claude Hook integration is disabled."""
    if not _is_claude_hook_enabled():
        raise ValueError(_DISABLED_MSG)


def _get_parser(platform: str) -> Any:
    """Return the parser instance for *platform*, creating on first use."""
    _check_enabled()
    if platform not in _parsers:
        from tools.transcripts.parsers import get_parser
        if platform not in _PLATFORM_INFO:
            raise ValueError(
                f"Unknown platform: {platform!r}. "
                f"Available: {sorted(_PLATFORM_INFO)}"
            )
        _parsers[platform] = get_parser(platform)
    return _parsers[platform]


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_platforms() -> str:
    """List all supported agent platforms.

    Returns a JSON array of platform descriptors with id, name,
    description, and default base path for transcript storage.
    """
    try:
        _check_enabled()
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    result = [
        {"id": pid, **info}
        for pid, info in _PLATFORM_INFO.items()
    ]
    return json.dumps(result, indent=2)


@mcp.tool()
async def list_projects(
    platform: str = "claude_code",
    base_path: str = "",
) -> str:
    """List all projects (workspaces) for a given agent platform.

    Args:
        platform: Agent platform identifier (run list_platforms to see options).
        base_path: Override the default transcript storage path.
            Leave empty to use the platform default.
    """
    try:
        parser = _get_parser(platform)
        projects = parser.list_projects(base_path or None)
        return json.dumps([asdict(p) for p in projects], indent=2)
    except Exception as exc:
        logger.exception("list_projects failed for %s", platform)
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@mcp.tool()
async def list_sessions(
    project_path: str,
    platform: str = "claude_code",
) -> str:
    """List all sessions within a project directory.

    Returns session metadata including timestamps, model used, size,
    and whether the session is still active (no last-prompt marker).

    Args:
        project_path: Full path to the project directory
            (from list_projects output).
        platform: Agent platform identifier.
    """
    try:
        parser = _get_parser(platform)
        sessions = parser.list_sessions(project_path)
        return json.dumps([asdict(s) for s in sessions], indent=2)
    except Exception as exc:
        logger.exception("list_sessions failed")
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@mcp.tool()
async def check_session_activity(
    session_path: str,
    platform: str = "claude_code",
    known_line_count: int = 0,
    known_size: int = 0,
) -> str:
    """Lightweight check for new session activity since last poll.

    Returns only metadata (size, timestamps, active status) — no turn
    content.  For real-time monitoring prefer ``wait_for_activity``
    which blocks server-side; use this for one-off instant checks.

    Works for both Claude Code and Cowork sessions (same JSONL schema).

    Args:
        session_path: Full path to the session JSONL file.
        platform: Agent platform identifier.
        known_line_count: Line count from your last poll (legacy).
        known_size: Byte size from your last poll (preferred — O(1) check).
            When >0, uses size-based detection instead of line counting.
    """
    try:
        _check_enabled()
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    path = Path(session_path).expanduser()
    if not path.is_file():
        return json.dumps({"error": f"File not found: {session_path}"})

    try:
        if known_size > 0:
            from tools.transcripts.parsers._utils import scan_activity_by_size
            result = scan_activity_by_size(path, known_size)
        else:
            from tools.transcripts.parsers._utils import scan_tail_for_activity
            result = scan_tail_for_activity(path, known_line_count)
        return json.dumps(result)
    except Exception as exc:
        logger.exception("check_session_activity failed")
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def _find_hook_session(session_path: str) -> str | None:
    """Resolve the Claude Code session_id for a transcript path, if the
    hook event buffer has seen events for it."""
    try:
        from tools.transcripts.hook_event_buffer import hook_buffer
        return hook_buffer.find_session_for_transcript(session_path)
    except Exception:
        return None


@mcp.tool()
async def wait_for_activity(
    session_path: str,
    known_size: int = 0,
    timeout_seconds: int = 60,
    poll_interval_seconds: int = 3,
) -> str:
    """Block until the session transcript grows or the timeout expires.

    When HTTP hooks are active for this session the call returns almost
    instantly — the hook event buffer signals new data without needing
    to poll the filesystem.  Otherwise falls back to file-size comparison
    (a single ``stat`` syscall per check).

    Returns the same activity metadata as ``check_session_activity``
    plus a ``status`` field: ``"new_activity"`` or ``"timeout"`` and
    a ``source`` field: ``"http_hooks"`` or ``"file_polling"``.

    Args:
        session_path: Full path to the session JSONL file.
        known_size: Byte size from your last check.  Pass 0 on the
            first call; thereafter use ``size_bytes`` from the response.
        timeout_seconds: Maximum seconds to wait (default 60, capped at 120).
        poll_interval_seconds: Seconds between stat checks (default 3).
    """
    try:
        _check_enabled()
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    path = Path(session_path).expanduser()
    if not path.is_file():
        return json.dumps({"error": f"File not found: {session_path}"})

    timeout_seconds = min(max(1, timeout_seconds), _MAX_TIMEOUT_SECS)
    poll_interval_seconds = max(1, min(poll_interval_seconds, _MAX_POLL_INTERVAL_SECS))

    hook_session_id = _find_hook_session(session_path)

    # --- Fast path: HTTP hook buffer has events for this session ---
    if hook_session_id is not None:
        from tools.transcripts.hook_event_buffer import hook_buffer

        snapshot = time.monotonic()
        arrived = await hook_buffer.wait_for_event(
            hook_session_id, timeout=timeout_seconds, since_mono=snapshot,
        )

        try:
            from tools.transcripts.parsers._utils import scan_activity_by_size
            result = scan_activity_by_size(path, known_size)
        except OSError as exc:
            result = {"error": f"File access error: {exc}"}

        result["status"] = "new_activity" if arrived else "timeout"
        result["source"] = "http_hooks"

        if hook_buffer.session_ended(hook_session_id):
            result["is_active"] = False

        return json.dumps(result)

    # --- Fallback: file-size polling (Cowork, or hooks not configured) ---
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds

    while loop.time() < deadline:
        try:
            current_size = path.stat().st_size
        except OSError as exc:
            return json.dumps({"error": f"File access error: {exc}"})

        if current_size > known_size:
            from tools.transcripts.parsers._utils import scan_activity_by_size

            result = scan_activity_by_size(path, known_size)
            result["status"] = "new_activity"
            result["source"] = "file_polling"
            return json.dumps(result)

        remaining = deadline - loop.time()
        sleep_time = min(poll_interval_seconds, remaining)
        if sleep_time <= 0:
            break
        await asyncio.sleep(sleep_time)

    try:
        from tools.transcripts.parsers._utils import scan_activity_by_size

        result = scan_activity_by_size(path, known_size)
        result["status"] = "timeout"
        result["source"] = "file_polling"
        return json.dumps(result)
    except OSError as exc:
        return json.dumps({"status": "timeout", "source": "file_polling", "error": str(exc)})


@mcp.tool()
async def get_hook_status(
    session_path: str = "",
) -> str:
    """Check whether HTTP hooks are active for a session.

    Returns ``hooks_active: true`` if the hook event buffer has received
    events for the session identified by *session_path*.  When hooks are
    active, ``wait_for_activity`` returns instantly on new data instead
    of polling the filesystem.

    Call this at the start of a real-time monitoring session to determine
    the data source.

    Args:
        session_path: Full path to the session JSONL file.  If empty,
            returns the global hook receiver status.
    """
    try:
        from tools.transcripts.hook_event_buffer import hook_buffer
    except Exception as exc:
        return json.dumps({"hooks_active": False, "error": str(exc)})

    if not session_path:
        try:
            from backend.config import AppConfig
            cfg = AppConfig.load()
            enabled = cfg.claude_hook.enabled and cfg.claude_hook.http_hooks_enabled
        except Exception:
            enabled = False
        return json.dumps({
            "http_hooks_enabled": enabled,
            "active_sessions": hook_buffer.active_session_count(),
        })

    hook_sid = hook_buffer.find_session_for_transcript(session_path)
    if hook_sid is None:
        return json.dumps({
            "hooks_active": False,
            "events_buffered": 0,
            "source": "file_polling",
        })

    stats = hook_buffer.session_stats(hook_sid)
    stats["source"] = "http_hooks"
    return json.dumps(stats)


# ---------------------------------------------------------------------------
# Eval-state persistence (delegates to shared module)
# ---------------------------------------------------------------------------

@mcp.tool()
async def save_eval_state(
    session_path: str,
    state_json: str,
) -> str:
    """Persist evaluation progress so it survives conversation summarization.

    Saves arbitrary JSON state (offsets, scores, eval plan) to a sidecar
    file keyed by the session path.  Call this periodically during long
    evaluations to checkpoint your work.

    Args:
        session_path: The session being evaluated (used to derive the
            state filename).
        state_json: JSON string containing the evaluation state.  Should
            include at minimum ``known_size``, ``next_line_offset``,
            ``turns_evaluated``, and any accumulated ``scores``.
    """
    from tools.transcripts.eval_persistence import save_eval_state as _save
    return json.dumps(_save(session_path, state_json))


@mcp.tool()
async def load_eval_state(
    session_path: str,
) -> str:
    """Load previously saved evaluation state for a session.

    Returns the JSON state saved by ``save_eval_state``, or an empty
    object ``{}`` if no prior state exists.

    Args:
        session_path: The session being evaluated.
    """
    from tools.transcripts.eval_persistence import load_eval_state as _load
    result = _load(session_path)
    if "error" in result:
        return json.dumps(result)
    return json.dumps(result, indent=2)


@mcp.tool()
async def parse_session_turns(
    session_path: str,
    platform: str = "claude_code",
    from_line: int = 0,
    max_turns: int = 50,
) -> str:
    """Parse structured turns from a session transcript.

    Each turn contains the user input, assistant output, thinking,
    tool calls (with results), token usage, and timing — shaped
    for direct use with the agent-evaluator's evaluate and
    evaluate_trajectory tools.

    Args:
        session_path: Full path to the session JSONL file.
        platform: Agent platform identifier.
        from_line: Line offset to start reading from (0-based).
            Use the returned next_line_offset to resume incrementally.
        max_turns: Maximum number of turns to return per call.
    """
    try:
        parser = _get_parser(platform)
        turns, next_offset = parser.parse_turns(
            session_path,
            from_line=from_line,
            max_turns=max_turns,
        )
        result = {
            "turns": [t.to_eval_dict() for t in turns],
            "turn_count": len(turns),
            "next_line_offset": next_offset,
            "from_line": from_line,
            "platform": platform,
        }
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.exception("parse_session_turns failed")
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@mcp.tool()
async def get_session_summary(
    session_path: str,
    platform: str = "claude_code",
) -> str:
    """Get aggregate statistics for a complete session.

    Returns total tokens, duration, tool usage breakdown, error
    counts, and per-turn eval-ready data.

    Args:
        session_path: Full path to the session JSONL file.
        platform: Agent platform identifier.
    """
    try:
        parser = _get_parser(platform)
        summary = parser.get_session_summary(session_path)
        return json.dumps(asdict(summary), indent=2)
    except Exception as exc:
        logger.exception("get_session_summary failed")
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@mcp.tool()
async def get_cowork_audit_info(
    session_path: str,
) -> str:
    """Get Cowork session metadata from its ``audit.jsonl`` file.

    Returns initialization info (model, tools, MCP servers, skills,
    agents), record type counts, and rate-limit data.  Only works
    for Cowork sessions — for Claude Code, use get_session_summary.

    Args:
        session_path: Full path to a Cowork session JSONL file, or
            directly to the audit.jsonl file.
    """
    try:
        from tools.transcripts.parsers.cowork import CoworkParser

        path = Path(session_path).expanduser()

        if path.name == "audit.jsonl" and path.is_file():
            audit_path = path
        else:
            audit_path = CoworkParser.find_audit_log(str(path))

        if not audit_path or not audit_path.is_file():
            return json.dumps({
                "error": (
                    "audit.jsonl not found. Provide a path to a Cowork "
                    "session JSONL or directly to audit.jsonl."
                )
            })

        info = CoworkParser.parse_audit_log(audit_path)
        result = asdict(info)
        result["audit_path"] = str(audit_path)
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.exception("get_cowork_audit_info failed")
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@mcp.tool()
async def save_report(
    report_markdown: str,
    filename: str = "",
    output_dir: str = "",
) -> str:
    """Save an evaluation report as a Markdown file.

    Args:
        report_markdown: Full Markdown content of the report.
        filename: Output filename (e.g. "eval-8e72a444.md").
            Auto-generated from timestamp if empty.
        output_dir: Directory to save into.  Defaults to
            ~/agent-eval-reports/.
    """
    from tools.transcripts.eval_persistence import save_report as _save
    return json.dumps(_save(report_markdown, filename, output_dir, filename_prefix="eval-report"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Claude Evaluator Hook MCP Server")
    parser.add_argument(
        "--transport",
        choices=["streamable-http", "stdio"],
        default="streamable-http",
    )
    parser.add_argument("--host", default=_DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT)
    args = parser.parse_args()

    logger.info(
        "Starting Claude Evaluator Hook MCP server (transport=%s, host=%s, port=%s)",
        args.transport, args.host, args.port,
    )

    mcp.settings.host = args.host
    mcp.settings.port = args.port

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
