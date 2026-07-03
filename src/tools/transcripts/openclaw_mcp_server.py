#!/usr/bin/env python3
"""OpenClaw Evaluator Hook MCP Server.

Standalone MCP server that exposes tools for reading and parsing OpenClaw
agent session transcripts.  Supports local filesystem and SSH (remote)
access modes configured through the application settings.

Usage::

    # Streamable HTTP (default)
    python -m tools.transcripts.openclaw_mcp_server --port 8943

    # stdio (for direct MCP client piping)
    python -m tools.transcripts.openclaw_mcp_server --transport stdio
"""

import asyncio
import json
import logging
import sys
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
logger = logging.getLogger("openclaw-eval-hook")

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8943

mcp = FastMCP(
    "openclaw-eval-hook",
    host=_DEFAULT_HOST,
    port=_DEFAULT_PORT,
    instructions=(
        "OpenClaw evaluator hook.  Use `list_agents` to discover agents, "
        "`list_sessions` to find sessions, `parse_session_turns` / "
        "`get_session_summary` to extract structured evaluation data, "
        "and `check_session_activity` for instant metadata checks.  "
        "Use `save_eval_state` / `load_eval_state` to persist evaluation "
        "progress.  Supports local and SSH (remote) access modes — "
        "configure in Settings → OpenClaw."
    ),
)


# ---------------------------------------------------------------------------
# Parser — lazily created from app config
# ---------------------------------------------------------------------------

_parser: Any = None
_parser_config_snapshot: str = ""

_MAX_TIMEOUT_SECS = 120
_MAX_POLL_INTERVAL_SECS = 30
_MIN_POLL_INTERVAL_SECS = 2

_DISABLED_MSG = (
    "OpenClaw integration is disabled. "
    "Enable it in Settings → Integrations → OpenClaw, "
    "then configure your connection details."
)


def _is_openclaw_enabled() -> bool:
    """Check whether the OpenClaw integration toggle is on."""
    try:
        from backend.config import AppConfig
        return AppConfig.load().openclaw.enabled
    except Exception:
        return False


def _load_openclaw_config() -> dict[str, Any]:
    """Load OpenClaw settings from the application config."""
    try:
        from backend.config import AppConfig
        cfg = AppConfig.load().openclaw
        if not cfg.enabled:
            return {}
        result: dict[str, Any] = {
            "mode": cfg.mode,
            "state_dir": cfg.state_dir,
        }
        if cfg.mode == "ssh":
            result.update(
                ssh_host=cfg.ssh_host,
                ssh_user=cfg.ssh_user,
                ssh_key_path=cfg.ssh_key_path,
                ssh_port=cfg.ssh_port,
            )
        return result
    except Exception as exc:
        logger.warning("Could not load OpenClaw config: %s", exc)
        return {}


def _get_parser() -> Any:
    """Return the OpenClaw parser, re-creating when config changes."""
    global _parser, _parser_config_snapshot
    if not _is_openclaw_enabled():
        _parser = None
        _parser_config_snapshot = ""
        raise ValueError(_DISABLED_MSG)
    kwargs = _load_openclaw_config()
    if not kwargs:
        raise ValueError(_DISABLED_MSG)
    config_key = json.dumps(kwargs, sort_keys=True)
    if _parser is None or config_key != _parser_config_snapshot:
        from tools.transcripts.parsers.openclaw import OpenClawParser
        _parser = OpenClawParser(**kwargs)
        _parser_config_snapshot = config_key
    return _parser


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_agents(base_path: str = "") -> str:
    """List all OpenClaw agents (projects).

    Each agent has its own session directory.  Returns agent id,
    session count, and path.

    Args:
        base_path: Override the default agents directory.
            Leave empty to use the configured state_dir.
    """
    try:
        parser = _get_parser()
        projects = parser.list_projects(base_path or None)
        return json.dumps([asdict(p) for p in projects], indent=2)
    except Exception as exc:
        logger.exception("list_agents failed")
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@mcp.tool()
async def list_sessions(agent_sessions_path: str) -> str:
    """List all sessions for an OpenClaw agent.

    Returns session metadata including timestamps, model used, size,
    and whether the session is still active.

    Args:
        agent_sessions_path: Path to the agent's sessions directory
            (from list_agents output).
    """
    try:
        parser = _get_parser()
        sessions = parser.list_sessions(agent_sessions_path)
        return json.dumps([asdict(s) for s in sessions], indent=2)
    except Exception as exc:
        logger.exception("list_sessions failed")
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@mcp.tool()
async def check_session_activity(
    session_path: str,
    known_line_count: int = 0,
) -> str:
    """Lightweight check for new session activity since last poll.

    Returns metadata (size, line count, timestamps, active status)
    without parsing turn content.

    Args:
        session_path: Full path to the session JSONL file.
        known_line_count: Line count from your last poll.
    """
    try:
        parser = _get_parser()
        io = parser.io
        if not io.file_exists(session_path):
            return json.dumps({"error": f"File not found: {session_path}"})

        current_lines = io.line_count(session_path)
        current_size = io.file_size(session_path)

        return json.dumps({
            "has_new_activity": current_lines > known_line_count,
            "new_lines": max(0, current_lines - known_line_count),
            "total_lines": current_lines,
            "size_bytes": current_size,
        })
    except Exception as exc:
        logger.exception("check_session_activity failed")
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def _find_buffer_session(session_path: str) -> str | None:
    """Try to find a watcher-buffered session ID for a transcript path.

    The watcher pushes events keyed by OpenClaw session ID.  We derive
    the session ID from the JSONL filename and check the buffer.
    """
    try:
        from tools.transcripts.openclaw_hook_event_buffer import oc_hook_buffer
        basename = session_path.rsplit("/", 1)[-1].replace(".jsonl", "")
        if oc_hook_buffer.has_new_events(basename):
            return basename
    except Exception:
        logger.debug("_find_buffer_session failed for %s", session_path, exc_info=True)
    return None


@mcp.tool()
async def wait_for_activity(
    session_path: str,
    known_size: int = 0,
    timeout_seconds: int = 60,
    poll_interval_seconds: int = 5,
) -> str:
    """Block until the session transcript grows or the timeout expires.

    If the session watcher is active, uses the event buffer for
    near-instant notification.  Otherwise falls back to file-size
    polling via the parser's IO layer (works for both local and SSH
    modes).

    Returns activity metadata plus a ``status`` field:
    ``"new_activity"`` or ``"timeout"``.

    Args:
        session_path: Full path to the session JSONL file.
        known_size: Byte size from your last check.  Pass 0 on the
            first call; thereafter use ``size_bytes`` from the response.
        timeout_seconds: Maximum seconds to wait (default 60, capped at 120).
        poll_interval_seconds: Seconds between size checks (default 5).
            SSH mode has network overhead per check, so avoid very short
            intervals.
    """
    import time as _time

    if not _is_openclaw_enabled():
        return json.dumps({"error": _DISABLED_MSG})

    timeout_seconds = min(max(1, timeout_seconds), _MAX_TIMEOUT_SECS)
    poll_interval_seconds = max(_MIN_POLL_INTERVAL_SECS, min(poll_interval_seconds, _MAX_POLL_INTERVAL_SECS))

    # Fast path: use the watcher event buffer if available
    buffer_sid = _find_buffer_session(session_path)
    if buffer_sid:
        from tools.transcripts.openclaw_hook_event_buffer import oc_hook_buffer
        since = _time.monotonic()
        arrived = await oc_hook_buffer.wait_for_event(
            buffer_sid, timeout=float(timeout_seconds), since_mono=since,
        )
        if arrived:
            try:
                parser = _get_parser()
                io = parser.io
                current_size = await asyncio.to_thread(io.file_size, session_path)
                current_lines = await asyncio.to_thread(io.line_count, session_path)
                return json.dumps({
                    "status": "new_activity",
                    "has_new_activity": True,
                    "size_bytes": current_size,
                    "total_lines": current_lines,
                    "source": "watcher",
                })
            except Exception:
                logger.debug("Watcher buffer hit but file read failed", exc_info=True)
                return json.dumps({
                    "status": "new_activity",
                    "has_new_activity": True,
                    "source": "watcher",
                })
        return json.dumps({
            "status": "timeout",
            "has_new_activity": False,
            "source": "watcher",
        })

    # Fallback: file-size polling
    try:
        parser = _get_parser()
        io = parser.io
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    if not io.file_exists(session_path):
        return json.dumps({"error": f"File not found: {session_path}"})

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds

    while loop.time() < deadline:
        try:
            current_size = await asyncio.to_thread(io.file_size, session_path)
        except Exception as exc:
            return json.dumps({"error": f"File access error: {exc}"})

        if current_size > known_size:
            current_lines = await asyncio.to_thread(io.line_count, session_path)
            return json.dumps({
                "status": "new_activity",
                "has_new_activity": True,
                "size_bytes": current_size,
                "total_lines": current_lines,
                "source": "file_polling",
            })

        remaining = deadline - loop.time()
        sleep_time = min(poll_interval_seconds, remaining)
        if sleep_time <= 0:
            break
        await asyncio.sleep(sleep_time)

    try:
        current_size = await asyncio.to_thread(io.file_size, session_path)
        current_lines = await asyncio.to_thread(io.line_count, session_path)
        return json.dumps({
            "status": "timeout",
            "has_new_activity": current_size > known_size,
            "size_bytes": current_size,
            "total_lines": current_lines,
            "source": "file_polling",
        })
    except Exception as exc:
        return json.dumps({"status": "timeout", "error": str(exc)})


@mcp.tool()
async def parse_session_turns(
    session_path: str,
    from_line: int = 0,
    max_turns: int = 50,
) -> str:
    """Parse structured turns from an OpenClaw session transcript.

    Each turn contains the user input, assistant output, thinking,
    tool calls (with results), token usage, and timing — shaped
    for direct use with the agent-evaluator's evaluate and
    evaluate_trajectory tools.

    Args:
        session_path: Full path to the session JSONL file.
        from_line: Line offset to start reading from (0-based).
            Use the returned next_line_offset to resume incrementally.
        max_turns: Maximum number of turns to return per call.
    """
    try:
        parser = _get_parser()
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
            "platform": "openclaw",
        }
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.exception("parse_session_turns failed")
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@mcp.tool()
async def get_session_summary(session_path: str) -> str:
    """Get aggregate statistics for a complete OpenClaw session.

    Returns total tokens, tool usage breakdown, error counts, and
    per-turn eval-ready data.

    Args:
        session_path: Full path to the session JSONL file.
    """
    try:
        parser = _get_parser()
        summary = parser.get_session_summary(session_path)
        return json.dumps(asdict(summary), indent=2)
    except Exception as exc:
        logger.exception("get_session_summary failed")
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@mcp.tool()
async def get_watcher_status(session_id: str = "") -> str:
    """Return the status of the OpenClaw session watcher event buffer.

    If *session_id* is provided, returns stats for that session.
    Otherwise returns general buffer status.

    Args:
        session_id: Optional OpenClaw session ID to inspect.
    """
    try:
        from tools.transcripts.openclaw_hook_event_buffer import oc_hook_buffer
    except Exception as exc:
        return json.dumps({"error": str(exc)})

    if session_id:
        return json.dumps(oc_hook_buffer.session_stats(session_id))

    return json.dumps({
        "active_sessions": oc_hook_buffer.active_session_count(),
    })


# ---------------------------------------------------------------------------
# Eval-state persistence (delegates to shared module)
# ---------------------------------------------------------------------------

@mcp.tool()
async def save_eval_state(session_path: str, state_json: str) -> str:
    """Persist evaluation progress so it survives conversation summarization.

    Args:
        session_path: The session being evaluated (used to derive the
            state filename).
        state_json: JSON string containing the evaluation state.
    """
    from tools.transcripts.eval_persistence import save_eval_state as _save
    return json.dumps(_save(session_path, state_json))


@mcp.tool()
async def load_eval_state(session_path: str) -> str:
    """Load previously saved evaluation state for a session.

    Returns the saved JSON state, or ``{}`` if no prior state exists.

    Args:
        session_path: The session being evaluated.
    """
    from tools.transcripts.eval_persistence import load_eval_state as _load
    result = _load(session_path)
    if "error" in result:
        return json.dumps(result)
    return json.dumps(result, indent=2)


@mcp.tool()
async def save_report(
    report_markdown: str,
    filename: str = "",
    output_dir: str = "",
) -> str:
    """Save an evaluation report as a Markdown file.

    Args:
        report_markdown: Full Markdown content of the report.
        filename: Output filename.  Auto-generated if empty.
        output_dir: Directory to save into.  Defaults to
            ~/agent-eval-reports/.
    """
    from tools.transcripts.eval_persistence import save_report as _save
    return json.dumps(_save(report_markdown, filename, output_dir, filename_prefix="openclaw-eval"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="OpenClaw Evaluator Hook MCP Server")
    parser.add_argument(
        "--transport",
        choices=["streamable-http", "stdio"],
        default="streamable-http",
    )
    parser.add_argument("--host", default=_DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT)
    args = parser.parse_args()

    logger.info(
        "Starting OpenClaw Evaluator Hook MCP server (transport=%s, host=%s, port=%s)",
        args.transport, args.host, args.port,
    )

    mcp.settings.host = args.host
    mcp.settings.port = args.port

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
