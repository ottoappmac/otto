"""Claude Code (~/.claude) JSONL transcript parser.

Reads the append-only JSONL files that Claude Code writes under
``~/.claude/projects/<encoded-workspace>/`` and reconstructs structured
:class:`Turn` objects that can be fed directly into the evaluator.

Turn boundaries are detected via the ``turn_duration`` system record
that Claude Code emits at the end of each agentic turn.  When no
``turn_duration`` record is present (e.g. mid-session tail), the parser
falls back to detecting the next ``user`` message as the boundary.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from tools.transcripts.parsers.base import (
    ModelCall,
    ProjectInfo,
    SessionInfo,
    ToolCallRecord,
    TranscriptParser,
    Turn,
)

logger = logging.getLogger(__name__)

_DEFAULT_BASE_PATH = os.path.expanduser("~/.claude/projects")


def _decode_project_dir(dirname: str) -> str:
    """Convert Claude Code's encoded dir name back to the workspace path.

    Claude Code replaces ``/`` with ``-`` in the directory name.
    The leading dash represents the root ``/``.
    """
    if dirname.startswith("-"):
        return "/" + dirname[1:].replace("-", "/")
    return dirname.replace("-", "/")


def _count_lines(path: Path) -> int:
    """Fast line count without loading the whole file into memory.

    Re-exported from ``_utils`` for backward compatibility.
    """
    from tools.transcripts.parsers._utils import count_lines
    return count_lines(path)


def _extract_user_text(content: Any) -> str:
    """Pull the user-visible text from a user message's content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    # tool_result in a user message — skip for prompt extraction
                    pass
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()
    return str(content)


def _extract_tool_result_text(content: Any) -> str:
    """Extract the textual result from a tool_result content block."""
    if isinstance(content, str):
        return content[:2000]
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                text = block.get("text", "") or block.get("content", "")
                if text:
                    return str(text)[:2000]
        return str(content)[:2000]
    return str(content)[:2000]


class ClaudeCodeParser(TranscriptParser):
    """Parser for Claude Code's ``~/.claude/projects/`` JSONL transcripts."""

    @property
    def platform(self) -> str:
        return "claude_code"

    def list_projects(self, base_path: str | None = None) -> list[ProjectInfo]:
        root = Path(base_path or _DEFAULT_BASE_PATH)
        if not root.is_dir():
            return []

        projects: list[ProjectInfo] = []
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            session_count = sum(
                1 for f in entry.iterdir()
                if f.suffix == ".jsonl" and not f.name.startswith(".")
            )
            projects.append(ProjectInfo(
                path=str(entry),
                workspace=_decode_project_dir(entry.name),
                session_count=session_count,
            ))
        return projects

    def list_sessions(self, project_path: str) -> list[SessionInfo]:
        proj = Path(project_path)
        if not proj.is_dir():
            return []

        sessions: list[SessionInfo] = []
        for f in sorted(proj.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if f.suffix != ".jsonl" or f.name.startswith("."):
                continue

            session_id = f.stem
            info = self._scan_session_meta(f, session_id, project_path)
            sessions.append(info)

        return sessions

    def _scan_session_meta(
        self, path: Path, session_id: str, project_path: str,
    ) -> SessionInfo:
        """Quick scan: read first and last few lines for metadata."""
        size = path.stat().st_size
        lines = _count_lines(path)

        first_ts = ""
        last_ts = ""
        model = ""
        version = ""
        is_active = True
        last_prompt = ""

        try:
            with open(path, "r", encoding="utf-8") as fh:
                for raw_line in fh:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        rec = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    ts = rec.get("timestamp", "")
                    if ts and not first_ts:
                        first_ts = ts

                    if rec.get("type") == "assistant":
                        msg = rec.get("message", {})
                        if not model:
                            model = msg.get("model", "")
                        if not version:
                            version = rec.get("version", "")

                    if rec.get("type") == "last-prompt":
                        is_active = False
                        last_prompt = rec.get("lastPrompt", "")

                    if ts:
                        last_ts = ts

        except Exception as exc:
            logger.debug("Error scanning %s: %s", path, exc)

        subagent_dir = path.parent / session_id / "subagents"
        has_subagents = subagent_dir.is_dir() and any(subagent_dir.iterdir())

        return SessionInfo(
            session_id=session_id,
            project_path=project_path,
            file_path=str(path),
            size_bytes=size,
            line_count=lines,
            first_timestamp=first_ts,
            last_timestamp=last_ts,
            model=model,
            version=version,
            is_active=is_active,
            last_prompt=last_prompt,
            has_subagents=has_subagents,
        )

    def parse_turns(
        self,
        session_path: str,
        *,
        from_line: int = 0,
        max_turns: int | None = None,
    ) -> tuple[list[Turn], int]:
        path = Path(session_path)
        if not path.is_file():
            return [], from_line

        records: list[dict[str, Any]] = []
        current_line = 0

        with open(path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                if current_line < from_line:
                    current_line += 1
                    continue
                current_line += 1
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    records.append(json.loads(raw_line))
                except json.JSONDecodeError:
                    logger.debug("Skipping malformed JSON at line %d", current_line)

        turns = self._records_to_turns(records, from_line)

        if max_turns is not None:
            turns = turns[:max_turns]

        return turns, current_line

    def _records_to_turns(
        self, records: list[dict[str, Any]], base_offset: int,
    ) -> list[Turn]:
        """Group raw JSONL records into Turn objects.

        Strategy: accumulate records into a buffer.  A turn is considered
        complete when we see either:
          1. A ``system`` record with ``subtype: turn_duration``
          2. A new ``user`` record with ``permissionMode`` (i.e. a fresh
             user prompt, not a tool_result fed back)
          3. A ``last-prompt`` record (session end)
        """
        turns: list[Turn] = []
        buffer: list[dict[str, Any]] = []
        turn_index = 0

        for rec in records:
            rec_type = rec.get("type", "")
            rec_subtype = rec.get("subtype", "")

            if rec_type == "last-prompt":
                if buffer:
                    turn = self._finalize_turn(buffer, turn_index)
                    if turn:
                        turns.append(turn)
                        turn_index += 1
                    buffer = []
                continue

            if rec_type == "file-history-snapshot":
                continue

            # turn_duration marks the definitive end of a turn
            if rec_type == "system" and rec_subtype == "turn_duration":
                buffer.append(rec)
                turn = self._finalize_turn(buffer, turn_index)
                if turn:
                    turns.append(turn)
                    turn_index += 1
                buffer = []
                continue

            # A new user prompt (not a tool_result) starts a new turn
            if (rec_type == "user"
                    and "permissionMode" in rec
                    and buffer
                    and self._buffer_has_assistant(buffer)):
                turn = self._finalize_turn(buffer, turn_index)
                if turn:
                    turns.append(turn)
                    turn_index += 1
                buffer = [rec]
                continue

            buffer.append(rec)

        # Remaining buffer: incomplete turn (session still active)
        if buffer and self._buffer_has_assistant(buffer):
            turn = self._finalize_turn(buffer, turn_index)
            if turn:
                turns.append(turn)

        return turns

    @staticmethod
    def _buffer_has_assistant(buffer: list[dict[str, Any]]) -> bool:
        return any(r.get("type") == "assistant" for r in buffer)

    def _finalize_turn(
        self, buffer: list[dict[str, Any]], turn_index: int,
    ) -> Turn | None:
        """Convert a buffer of raw records into a structured Turn.

        Two-pass approach: first pass collects assistant tool_use blocks
        into ``pending_tools`` and gathers metadata.  Second pass matches
        user tool_result records against pending_tools.
        """
        user_input = ""
        thinking_parts: list[str] = []
        text_parts: list[str] = []
        model_calls: list[ModelCall] = []
        tool_records: list[ToolCallRecord] = []
        api_errors: list[dict[str, Any]] = []
        duration_ms: float | None = None
        ts_start = ""
        ts_end = ""

        pending_tools: dict[str, dict[str, Any]] = {}

        msg_blocks: dict[str, list[dict[str, Any]]] = {}
        msg_meta: dict[str, dict[str, Any]] = {}

        # --- Pass 1: extract assistant content blocks + tool_use registrations ---
        for rec in buffer:
            rec_type = rec.get("type", "")
            ts = rec.get("timestamp", "")
            if ts and not ts_start:
                ts_start = ts
            if ts:
                ts_end = ts

            if rec_type == "assistant":
                msg = rec.get("message", {})
                msg_id = msg.get("id", "")
                if msg_id:
                    if msg_id not in msg_blocks:
                        msg_blocks[msg_id] = []
                        msg_meta[msg_id] = {
                            "model": msg.get("model", ""),
                            "timestamp": ts,
                            "usage": {},
                        }
                    msg_blocks[msg_id].extend(msg.get("content", []))
                    usage = msg.get("usage", {})
                    if usage:
                        msg_meta[msg_id]["usage"] = usage
                    sr = msg.get("stop_reason")
                    if sr is not None:
                        msg_meta[msg_id]["stop_reason"] = sr

                    for block in msg.get("content", []):
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            tu_id = block.get("id", "")
                            if tu_id:
                                pending_tools[tu_id] = {
                                    "name": block.get("name", ""),
                                    "input": block.get("input", {}),
                                }

            elif rec_type == "system":
                if rec.get("subtype") == "api_error":
                    api_errors.append({
                        "status": rec.get("error", {}).get("status"),
                        "message": rec.get("error", {}).get("error", {}).get("message", ""),
                        "retry_attempt": rec.get("retryAttempt"),
                        "timestamp": ts,
                    })
                elif rec.get("subtype") == "turn_duration":
                    duration_ms = rec.get("durationMs")

        # --- Pass 2: extract user prompts and match tool_results ---
        for rec in buffer:
            rec_type = rec.get("type", "")
            if rec_type != "user":
                continue

            msg = rec.get("message", {})
            content = msg.get("content", "")

            if "permissionMode" in rec:
                user_input = _extract_user_text(content)
                continue

            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_result" or "tool_use_id" in block:
                        tu_id = block.get("tool_use_id", "")
                        is_err = block.get("is_error", False)
                        result_text = _extract_tool_result_text(
                            block.get("content", "")
                        )
                        if tu_id and tu_id in pending_tools:
                            pending = pending_tools.pop(tu_id)
                            tool_records.append(ToolCallRecord(
                                tool_use_id=tu_id,
                                name=pending["name"],
                                input_parameters=pending["input"],
                                result=result_text,
                                is_error=is_err,
                            ))

        # --- Build ModelCall objects ---
        for msg_id, blocks in msg_blocks.items():
            meta = msg_meta[msg_id]
            mc_thinking = ""
            mc_text = ""

            for block in blocks:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "thinking":
                    mc_thinking = block.get("thinking", "")
                elif btype == "text":
                    mc_text = block.get("text", "")

            if mc_thinking:
                thinking_parts.append(mc_thinking)
            if mc_text:
                text_parts.append(mc_text)

            usage = meta.get("usage", {})
            model_calls.append(ModelCall(
                message_id=msg_id,
                model=meta.get("model", ""),
                thinking=mc_thinking,
                text=mc_text,
                stop_reason=meta.get("stop_reason"),
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                cache_creation_tokens=usage.get("cache_creation_input_tokens", 0),
                cache_read_tokens=usage.get("cache_read_input_tokens", 0),
                timestamp=meta.get("timestamp", ""),
            ))

        if not user_input and not model_calls:
            return None

        # Any tool_uses that never got a result (session interrupted)
        for tu_id, pending in pending_tools.items():
            tool_records.append(ToolCallRecord(
                tool_use_id=tu_id,
                name=pending["name"],
                input_parameters=pending["input"],
                result="(no result — session interrupted or still pending)",
                is_error=True,
            ))

        return Turn(
            turn_index=turn_index,
            input=user_input,
            actual_output="\n\n".join(text_parts),
            thinking="\n\n---\n\n".join(thinking_parts),
            model_calls=model_calls,
            tools_called=tool_records,
            api_errors=api_errors,
            duration_ms=duration_ms,
            timestamp_start=ts_start,
            timestamp_end=ts_end,
            total_input_tokens=sum(mc.input_tokens for mc in model_calls),
            total_output_tokens=sum(mc.output_tokens for mc in model_calls),
            total_cache_creation_tokens=sum(mc.cache_creation_tokens for mc in model_calls),
            total_cache_read_tokens=sum(mc.cache_read_tokens for mc in model_calls),
        )
