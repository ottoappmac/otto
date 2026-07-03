"""OpenClaw JSONL transcript parser.

Reads the append-only JSONL session files that OpenClaw writes under
``~/.openclaw/agents/<agentId>/sessions/<sessionId>.jsonl`` and
reconstructs structured :class:`Turn` objects for the evaluator.

Supports two access modes:

- **local** — reads files directly from the filesystem (OpenClaw
  running on the same machine).
- **ssh** — reads files from a remote host via ``asyncssh`` (e.g.
  an AWS Lightsail instance).

The SSH transport is configured through :class:`OpenClawConfig` in
``backend.config``.  When operating in SSH mode the parser runs
remote ``cat`` / ``ls`` / ``wc`` commands over a single multiplexed
connection that is lazily created and cached.

OpenClaw message schema key differences from Claude Code:

- Tool call block type is ``"toolCall"`` (not ``"tool_use"``).
- Tool call arguments field is ``"arguments"`` (not ``"input"``).
- Tool results are separate messages with ``role: "toolResult"``.
- Token usage lives inside ``message.usage`` (not record-level).
- Stop reason uses camelCase ``"stopReason"`` (not ``"stop_reason"``).
- No ``turn_duration`` marker; turns are delimited by user messages.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from pathlib import Path, PurePosixPath
from typing import Any

from tools.transcripts.parsers._utils import count_lines as _count_lines
from tools.transcripts.parsers.base import (
    ModelCall,
    ProjectInfo,
    SessionInfo,
    ToolCallRecord,
    TranscriptParser,
    Turn,
)

logger = logging.getLogger(__name__)

_DEFAULT_STATE_DIR = "~/.openclaw"


# ---------------------------------------------------------------------------
# SSH transport layer
# ---------------------------------------------------------------------------

class _SSHTransport:
    """Thin wrapper around ``asyncssh`` for running commands on a remote host.

    Lazily opens a single multiplexed connection that is reused across
    calls.  Thread-safe via an asyncio lock.
    """

    def __init__(
        self,
        host: str,
        user: str = "ubuntu",
        key_path: str = "",
        port: int = 22,
    ) -> None:
        self._host = host
        self._user = user
        self._key_path = key_path
        self._port = port
        self._conn: Any = None
        self._lock = asyncio.Lock()

    async def _ensure_connection(self) -> Any:
        if self._conn is not None:
            return self._conn

        import asyncssh  # lazy import — only needed in SSH mode

        connect_kwargs: dict[str, Any] = {
            "host": self._host,
            "port": self._port,
            "username": self._user,
            "known_hosts": None,  # skip host-key verification for now
        }
        if self._key_path:
            connect_kwargs["client_keys"] = [os.path.expanduser(self._key_path.strip())]

        self._conn = await asyncssh.connect(**connect_kwargs)
        logger.info("SSH connected to %s@%s:%s", self._user, self._host, self._port)
        return self._conn

    async def run(self, command: str) -> str:
        """Execute *command* on the remote host and return stdout."""
        async with self._lock:
            conn = await self._ensure_connection()
        result = await conn.run(command, check=True)
        return result.stdout or ""

    async def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# File I/O abstraction — local vs. SSH
# ---------------------------------------------------------------------------

class _LocalIO:
    """Filesystem access for local mode."""

    @staticmethod
    def read_text(path: str) -> str:
        return Path(path).expanduser().read_text(encoding="utf-8")

    @staticmethod
    def list_dirs(path: str) -> list[str]:
        p = Path(path).expanduser()
        if not p.is_dir():
            return []
        return [str(d) for d in sorted(p.iterdir()) if d.is_dir()]

    @staticmethod
    def list_files(path: str, suffix: str = "") -> list[str]:
        p = Path(path).expanduser()
        if not p.is_dir():
            return []
        entries = sorted(p.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True)
        if suffix:
            entries = [f for f in entries if f.suffix == suffix]
        return [str(f) for f in entries if f.is_file()]

    @staticmethod
    def file_size(path: str) -> int:
        return Path(path).expanduser().stat().st_size

    @staticmethod
    def file_exists(path: str) -> bool:
        return Path(path).expanduser().is_file()

    @staticmethod
    def line_count(path: str) -> int:
        return _count_lines(Path(path).expanduser())

    @staticmethod
    def read_from_line(path: str, from_line: int) -> str:
        lines: list[str] = []
        with open(Path(path).expanduser(), "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= from_line:
                    lines.append(line)
        return "".join(lines)


class _SSHIO:
    """Remote filesystem access via SSH.

    Maintains a dedicated background event loop so that the asyncssh
    connection is always used from the same loop, regardless of which
    thread the caller is on.
    """

    def __init__(self, transport: _SSHTransport) -> None:
        self._t = transport
        self._loop = asyncio.new_event_loop()
        self._thread: threading.Thread | None = None
        self._started = False

    def _ensure_loop_thread(self) -> None:
        """Spin up a daemon thread that runs the dedicated event loop."""
        if self._started:
            return
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True,
        )
        self._thread.start()
        self._started = True

    def _run_sync(self, command: str) -> str:
        self._ensure_loop_thread()
        future = asyncio.run_coroutine_threadsafe(
            self._t.run(command), self._loop,
        )
        return future.result(timeout=30)

    def read_text(self, path: str) -> str:
        return self._run_sync(f"cat {path}")

    def list_dirs(self, path: str) -> list[str]:
        out = self._run_sync(
            f"find {path} -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort"
        )
        return [line for line in out.strip().splitlines() if line]

    def list_files(self, path: str, suffix: str = "") -> list[str]:
        pattern = f"*{suffix}" if suffix else "*"
        out = self._run_sync(
            f"find {path} -maxdepth 1 -name '{pattern}' -type f "
            f"-printf '%T@ %p\\n' 2>/dev/null | sort -rn | cut -d' ' -f2-"
        )
        return [line for line in out.strip().splitlines() if line]

    def file_size(self, path: str) -> int:
        out = self._run_sync(f"stat -c %s {path} 2>/dev/null || echo 0")
        return int(out.strip() or "0")

    def file_exists(self, path: str) -> bool:
        out = self._run_sync(f"test -f {path} && echo yes || echo no")
        return out.strip() == "yes"

    def line_count(self, path: str) -> int:
        out = self._run_sync(f"wc -l < {path} 2>/dev/null || echo 0")
        return int(out.strip() or "0")

    def read_from_line(self, path: str, from_line: int) -> str:
        tail_start = from_line + 1  # tail -n + is 1-indexed
        return self._run_sync(f"tail -n +{tail_start} {path}")


# ---------------------------------------------------------------------------
# OpenClaw JSONL record helpers
# ---------------------------------------------------------------------------

def _extract_user_text(content: Any) -> str:
    """Pull user-visible text from an OpenClaw user message content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()
    return str(content)


def _extract_tool_result_text(content: Any) -> str:
    """Extract the textual result from an OpenClaw toolResult content."""
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


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class OpenClawParser(TranscriptParser):
    """Parser for OpenClaw ``~/.openclaw/agents/`` JSONL transcripts.

    Args:
        mode: ``"local"`` for local filesystem, ``"ssh"`` for remote.
        state_dir: Root OpenClaw state directory
            (default ``~/.openclaw``).
        ssh_host: Remote host (SSH mode only).
        ssh_user: Remote SSH user (default ``ubuntu``).
        ssh_key_path: Path to SSH private key file.
        ssh_port: SSH port (default ``22``).
    """

    def __init__(
        self,
        *,
        mode: str = "local",
        state_dir: str = _DEFAULT_STATE_DIR,
        ssh_host: str = "",
        ssh_user: str = "ubuntu",
        ssh_key_path: str = "",
        ssh_port: int = 22,
    ) -> None:
        self._mode = mode.strip()
        self._state_dir = state_dir.strip()
        self._ssh_transport: _SSHTransport | None = None

        if self._mode == "ssh":
            ssh_host = ssh_host.strip()
            if not ssh_host:
                raise ValueError("ssh_host is required in SSH mode")
            self._ssh_transport = _SSHTransport(
                host=ssh_host,
                user=ssh_user.strip(),
                key_path=ssh_key_path.strip(),
                port=ssh_port,
            )
            self._io: _LocalIO | _SSHIO = _SSHIO(self._ssh_transport)
        else:
            self._io = _LocalIO()

        self._runs_cache: dict[str, Any] | None = None

    @property
    def io(self) -> _LocalIO | _SSHIO:
        """Public accessor for the file I/O backend (local or SSH)."""
        return self._io

    @property
    def platform(self) -> str:
        return "openclaw"

    @property
    def agents_dir(self) -> str:
        base = self._state_dir
        if self._mode == "local":
            base = str(Path(base).expanduser())
        return f"{base}/agents"

    @property
    def subagents_dir(self) -> str:
        base = self._state_dir
        if self._mode == "local":
            base = str(Path(base).expanduser())
        return f"{base}/subagents"

    # ---- Subagent runs index ----

    def _load_runs(self, *, fresh: bool = False) -> dict[str, Any]:
        """Load ``~/.openclaw/subagents/runs.json``.

        Cached per instance for ``list_sessions`` efficiency.  Pass
        ``fresh=True`` to force a re-read (used by ``parse_turns``
        during real-time observation so new subagent runs are seen).
        """
        if self._runs_cache is not None and not fresh:
            return self._runs_cache
        runs_path = f"{self.subagents_dir}/runs.json"
        try:
            raw = self._io.read_text(runs_path)
            data = json.loads(raw)
            self._runs_cache = data.get("runs", {})
        except Exception:
            self._runs_cache = {}
        return self._runs_cache

    def _find_subagent_session_path(
        self, child_session_key: str, sessions_dir: str,
    ) -> str | None:
        """Resolve a childSessionKey to the subagent's JSONL file path.

        Looks up the session index to find which JSONL file belongs to
        the given session key (e.g. ``agent:main:subagent:<uuid>``).
        """
        index = self._load_sessions_index(sessions_dir)
        entry = index.get(child_session_key)
        if isinstance(entry, dict):
            sf = entry.get("sessionFile", "")
            if sf:
                return sf
            sid = entry.get("sessionId", "")
            if sid:
                return f"{sessions_dir}/{sid}.jsonl"
        return None

    def _runs_for_controller(self, controller_key: str) -> list[dict[str, Any]]:
        """Return all subagent run entries spawned by *controller_key*."""
        runs = self._load_runs()
        return [
            r for r in runs.values()
            if isinstance(r, dict) and r.get("controllerSessionKey") == controller_key
        ]

    def _session_key_for_id(
        self, session_id: str, sessions_dir: str,
    ) -> str:
        """Reverse-lookup: find the session key for a given session ID."""
        index = self._load_sessions_index(sessions_dir)
        for key, entry in index.items():
            if isinstance(entry, dict) and entry.get("sessionId") == session_id:
                return key
        return ""

    def _run_for_child_key(self, child_session_key: str) -> dict[str, Any] | None:
        """Find the subagent run record matching a childSessionKey."""
        runs = self._load_runs()
        for run in runs.values():
            if isinstance(run, dict) and run.get("childSessionKey") == child_session_key:
                return run
        return None

    # ---- Project listing (one "project" = one agent) ----

    def list_projects(self, base_path: str | None = None) -> list[ProjectInfo]:
        root = base_path or self.agents_dir
        agent_dirs = self._io.list_dirs(root)
        if not agent_dirs:
            return []

        projects: list[ProjectInfo] = []
        for agent_dir in agent_dirs:
            agent_id = PurePosixPath(agent_dir).name
            sessions_dir = f"{agent_dir}/sessions"
            session_files = self._io.list_files(sessions_dir, suffix=".jsonl")
            projects.append(ProjectInfo(
                path=sessions_dir,
                workspace=f"openclaw-agent:{agent_id}",
                session_count=len(session_files),
            ))
        return projects

    # ---- Session listing ----

    def list_sessions(self, project_path: str) -> list[SessionInfo]:
        session_files = self._io.list_files(project_path, suffix=".jsonl")
        if not session_files:
            return []

        index = self._load_sessions_index(project_path)
        sessions: list[SessionInfo] = []
        for fpath in session_files:
            fname = PurePosixPath(fpath).stem
            info = self._build_session_info(fpath, fname, project_path, index)
            sessions.append(info)
        return sessions

    def _load_sessions_index(self, sessions_dir: str) -> dict[str, Any]:
        """Load the ``sessions.json`` index file if available."""
        index_path = f"{sessions_dir}/sessions.json"
        try:
            raw = self._io.read_text(index_path)
            return json.loads(raw)
        except Exception:
            return {}

    def _build_session_info(
        self,
        file_path: str,
        session_id: str,
        project_path: str,
        index: dict[str, Any],
    ) -> SessionInfo:
        size_bytes = self._io.file_size(file_path)
        line_count = self._io.line_count(file_path)

        first_ts = ""
        last_ts = ""
        model = ""
        version = ""
        is_active = True

        index_entry = self._find_index_entry(index, session_id)
        if index_entry:
            model = index_entry.get("model", "")
            status = index_entry.get("status", "")
            is_active = status not in ("done", "error")

        try:
            raw = self._io.read_text(file_path)
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = rec.get("timestamp", "")
                if ts and not first_ts:
                    first_ts = ts
                if ts:
                    last_ts = ts
                if rec.get("type") == "session":
                    version = str(rec.get("version", ""))
                if not model and rec.get("type") == "model_change":
                    model = rec.get("modelId", "")
        except Exception as exc:
            logger.debug("Error scanning %s: %s", file_path, exc)

        session_key = self._session_key_for_id(session_id, project_path)
        has_subagents = bool(self._runs_for_controller(session_key)) if session_key else False

        return SessionInfo(
            session_id=session_id,
            project_path=project_path,
            file_path=file_path,
            size_bytes=size_bytes,
            line_count=line_count,
            first_timestamp=first_ts,
            last_timestamp=last_ts,
            model=model,
            version=version,
            is_active=is_active,
            has_subagents=has_subagents,
        )

    @staticmethod
    def _find_index_entry(
        index: dict[str, Any], session_id: str,
    ) -> dict[str, Any] | None:
        """Match a session ID against the ``sessions.json`` index."""
        for _key, entry in index.items():
            if isinstance(entry, dict) and entry.get("sessionId") == session_id:
                return entry
        return None

    # ---- Turn parsing ----

    def parse_turns(
        self,
        session_path: str,
        *,
        from_line: int = 0,
        max_turns: int | None = None,
    ) -> tuple[list[Turn], int]:
        if not self._io.file_exists(session_path):
            return [], from_line

        self._load_runs(fresh=True)

        if from_line > 0:
            raw = self._io.read_from_line(session_path, from_line)
        else:
            raw = self._io.read_text(session_path)

        records: list[dict[str, Any]] = []
        lines_read = 0
        for line in raw.splitlines():
            lines_read += 1
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.debug("Skipping malformed JSON at offset %d", from_line + lines_read)

        sessions_dir = str(PurePosixPath(session_path).parent)
        turns = self._records_to_turns(records, sessions_dir=sessions_dir)
        next_offset = from_line + lines_read

        if max_turns is not None:
            turns = turns[:max_turns]

        return turns, next_offset

    def _records_to_turns(
        self,
        records: list[dict[str, Any]],
        sessions_dir: str = "",
        _depth: int = 0,
    ) -> list[Turn]:
        """Group OpenClaw JSONL records into Turn objects.

        Turn boundary: each ``role: "user"`` message starts a new turn.
        A turn collects: user prompt, assistant messages (text + tool
        calls), tool results, and usage metadata.

        When *sessions_dir* is provided and the turn contains a
        ``sessions_spawn`` tool call, the subagent's session is
        recursively parsed and attached to the tool record.
        *_depth* guards against infinite recursion (max 3 levels).
        """
        turns: list[Turn] = []
        buffer: list[dict[str, Any]] = []
        turn_index = 0

        for rec in records:
            if rec.get("type") != "message":
                continue

            msg = rec.get("message", {})
            role = msg.get("role", "")

            if role == "user" and buffer and self._buffer_has_assistant(buffer):
                turn = self._finalize_turn(
                    buffer, turn_index, sessions_dir=sessions_dir, _depth=_depth,
                )
                if turn:
                    turns.append(turn)
                    turn_index += 1
                buffer = [rec]
            else:
                buffer.append(rec)

        if buffer and self._buffer_has_assistant(buffer):
            turn = self._finalize_turn(
                buffer, turn_index, sessions_dir=sessions_dir, _depth=_depth,
            )
            if turn:
                turns.append(turn)

        return turns

    @staticmethod
    def _buffer_has_assistant(buffer: list[dict[str, Any]]) -> bool:
        return any(
            r.get("message", {}).get("role") == "assistant"
            for r in buffer
        )

    _SUBAGENT_SPAWN_TOOLS = frozenset({"sessions_spawn"})
    _MAX_SUBAGENT_DEPTH = 3

    def _finalize_turn(
        self,
        buffer: list[dict[str, Any]],
        turn_index: int,
        sessions_dir: str = "",
        _depth: int = 0,
    ) -> Turn | None:
        """Convert a buffer of OpenClaw message records into a Turn."""
        user_input = ""
        thinking_parts: list[str] = []
        text_parts: list[str] = []
        model_calls: list[ModelCall] = []
        tool_records: list[ToolCallRecord] = []
        api_errors: list[dict[str, Any]] = []
        ts_start = ""
        ts_end = ""

        pending_tools: dict[str, dict[str, Any]] = {}
        spawn_child_keys: dict[str, str] = {}

        for rec in buffer:
            ts = rec.get("timestamp", "")
            if ts and not ts_start:
                ts_start = ts
            if ts:
                ts_end = ts

            msg = rec.get("message", {})
            role = msg.get("role", "")
            content = msg.get("content", [])

            if role == "user":
                user_input = _extract_user_text(content)

            elif role == "assistant":
                msg_id = rec.get("id", "")
                usage = msg.get("usage", {})
                stop_reason = msg.get("stopReason")
                error_msg = msg.get("errorMessage", "")

                if stop_reason == "error" and error_msg:
                    api_errors.append({
                        "message": error_msg,
                        "timestamp": ts,
                    })

                mc_thinking = ""
                mc_text_parts: list[str] = []

                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type", "")

                        if btype == "text":
                            text = block.get("text", "")
                            if text and not text.startswith("[["):
                                mc_text_parts.append(text)

                        elif btype == "thinking":
                            mc_thinking = block.get("thinking", "")

                        elif btype == "toolCall":
                            tc_id = block.get("id", "")
                            if tc_id:
                                pending_tools[tc_id] = {
                                    "name": block.get("name", ""),
                                    "arguments": block.get("arguments", {}),
                                }

                mc_text = "\n".join(mc_text_parts)
                if mc_thinking:
                    thinking_parts.append(mc_thinking)
                if mc_text:
                    text_parts.append(mc_text)

                model_calls.append(ModelCall(
                    message_id=msg_id,
                    model=msg.get("model", ""),
                    thinking=mc_thinking,
                    text=mc_text,
                    stop_reason=stop_reason,
                    input_tokens=usage.get("input", 0),
                    output_tokens=usage.get("output", 0),
                    cache_creation_tokens=usage.get("cacheWrite", 0),
                    cache_read_tokens=usage.get("cacheRead", 0),
                    timestamp=ts,
                ))

            elif role == "toolResult":
                tc_id = msg.get("toolCallId", "")
                is_error = msg.get("isError", False)
                result_text = _extract_tool_result_text(content)

                if tc_id and tc_id in pending_tools:
                    pending = pending_tools.pop(tc_id)
                    tool_records.append(ToolCallRecord(
                        tool_use_id=tc_id,
                        name=pending["name"],
                        input_parameters=pending["arguments"],
                        result=result_text,
                        is_error=is_error,
                    ))
                    if pending["name"] in self._SUBAGENT_SPAWN_TOOLS:
                        child_key = self._extract_child_session_key(content)
                        if child_key:
                            spawn_child_keys[tc_id] = child_key

        for tc_id, pending in pending_tools.items():
            tool_records.append(ToolCallRecord(
                tool_use_id=tc_id,
                name=pending["name"],
                input_parameters=pending["arguments"],
                result="(no result — session interrupted or still pending)",
                is_error=True,
            ))

        if spawn_child_keys and sessions_dir and _depth < self._MAX_SUBAGENT_DEPTH:
            tool_records = self._attach_subagent_turns(
                tool_records, spawn_child_keys, sessions_dir, _depth,
            )

        if not user_input and not model_calls:
            return None

        return Turn(
            turn_index=turn_index,
            input=user_input,
            actual_output="\n\n".join(text_parts),
            thinking="\n\n---\n\n".join(thinking_parts),
            model_calls=model_calls,
            tools_called=tool_records,
            api_errors=api_errors,
            timestamp_start=ts_start,
            timestamp_end=ts_end,
            total_input_tokens=sum(mc.input_tokens for mc in model_calls),
            total_output_tokens=sum(mc.output_tokens for mc in model_calls),
            total_cache_creation_tokens=sum(mc.cache_creation_tokens for mc in model_calls),
            total_cache_read_tokens=sum(mc.cache_read_tokens for mc in model_calls),
        )

    @staticmethod
    def _extract_child_session_key(content: Any) -> str:
        """Pull ``childSessionKey`` from a sessions_spawn tool result."""
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "") or block.get("content", "")
                    if text:
                        break
        if not text:
            return ""
        try:
            data = json.loads(text)
            return data.get("childSessionKey", "")
        except (json.JSONDecodeError, TypeError):
            return ""

    def _attach_subagent_turns(
        self,
        tool_records: list[ToolCallRecord],
        spawn_child_keys: dict[str, str],
        sessions_dir: str,
        _depth: int,
    ) -> list[ToolCallRecord]:
        """Replace spawn tool records with versions that include sub_turns."""
        updated: list[ToolCallRecord] = []
        for tr in tool_records:
            child_key = spawn_child_keys.get(tr.tool_use_id)
            if not child_key:
                updated.append(tr)
                continue

            sub_path = self._find_subagent_session_path(child_key, sessions_dir)
            if not sub_path or not self._io.file_exists(sub_path):
                logger.debug(
                    "Subagent session not found for %s (key=%s)",
                    tr.tool_use_id, child_key,
                )
                updated.append(ToolCallRecord(
                    tool_use_id=tr.tool_use_id,
                    name=tr.name,
                    input_parameters=tr.input_parameters,
                    result=tr.result,
                    is_error=tr.is_error,
                    duration_ms=tr.duration_ms,
                    subagent_pending=True,
                ))
                continue

            try:
                sub_raw = self._io.read_text(sub_path)
                sub_records: list[dict[str, Any]] = []
                for line in sub_raw.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sub_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

                sub_turns = self._records_to_turns(
                    sub_records,
                    sessions_dir=sessions_dir,
                    _depth=_depth + 1,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to parse subagent session %s: %s", sub_path, exc,
                )
                updated.append(ToolCallRecord(
                    tool_use_id=tr.tool_use_id,
                    name=tr.name,
                    input_parameters=tr.input_parameters,
                    result=tr.result,
                    is_error=tr.is_error,
                    duration_ms=tr.duration_ms,
                    subagent_pending=True,
                ))
                continue

            updated.append(ToolCallRecord(
                tool_use_id=tr.tool_use_id,
                name=tr.name,
                input_parameters=tr.input_parameters,
                result=tr.result,
                is_error=tr.is_error,
                duration_ms=tr.duration_ms,
                sub_turns=sub_turns,
            ))
        return updated
