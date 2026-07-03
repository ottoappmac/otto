"""Abstract transcript parser and shared data models.

Platform-specific parsers (Claude Code, Cursor, etc.) subclass
:class:`TranscriptParser` and implement the three abstract methods.
The MCP server is parser-agnostic — it dispatches through this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ToolCallRecord:
    """A paired tool invocation and its result."""

    tool_use_id: str
    name: str
    input_parameters: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    is_error: bool = False
    duration_ms: float | None = None
    sub_turns: list["Turn"] = field(default_factory=list)
    subagent_pending: bool = False


@dataclass(frozen=True)
class ModelCall:
    """One LLM API call within a turn (thinking + content blocks + usage)."""

    message_id: str
    model: str = ""
    thinking: str = ""
    text: str = ""
    tool_uses: list[ToolCallRecord] = field(default_factory=list)
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    timestamp: str = ""


@dataclass
class Turn:
    """A full agentic turn: user prompt → model call(s) → final response.

    Shaped so that fields map directly to evaluator inputs:
    - ``input``          → user prompt (feeds ``evaluate(input=...)``)
    - ``actual_output``  → assistant's final text (feeds ``actual_output``)
    - ``tools_called``   → list of tool records (feeds ``evaluate_trajectory``)
    """

    turn_index: int
    input: str
    actual_output: str
    thinking: str = ""
    model_calls: list[ModelCall] = field(default_factory=list)
    tools_called: list[ToolCallRecord] = field(default_factory=list)
    api_errors: list[dict[str, Any]] = field(default_factory=list)
    duration_ms: float | None = None
    timestamp_start: str = ""
    timestamp_end: str = ""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0

    def to_eval_dict(self) -> dict[str, Any]:
        """Serialize to a dict shaped for the evaluator MCP tools."""
        return {
            "input": self.input,
            "actual_output": self.actual_output,
            "thinking": self.thinking,
            "tools_called": [
                {
                    "name": t.name,
                    "input_parameters": t.input_parameters,
                    **({"sub_turns": [st.to_eval_dict() for st in t.sub_turns]}
                       if t.sub_turns else {}),
                    **({"subagent_pending": True} if t.subagent_pending else {}),
                }
                for t in self.tools_called
            ],
            "turn_index": self.turn_index,
            "duration_ms": self.duration_ms,
            "tokens": {
                "input": self.total_input_tokens,
                "output": self.total_output_tokens,
                "cache_creation": self.total_cache_creation_tokens,
                "cache_read": self.total_cache_read_tokens,
            },
            "api_errors": len(self.api_errors),
            "tool_errors": sum(1 for t in self.tools_called if t.is_error),
        }


@dataclass(frozen=True)
class ProjectInfo:
    """Metadata about a project (workspace) in the transcript store."""

    path: str
    workspace: str
    session_count: int


@dataclass(frozen=True)
class SessionInfo:
    """Metadata about a single session transcript."""

    session_id: str
    project_path: str
    file_path: str
    size_bytes: int
    line_count: int
    first_timestamp: str = ""
    last_timestamp: str = ""
    model: str = ""
    version: str = ""
    is_active: bool = False
    last_prompt: str = ""
    has_subagents: bool = False


@dataclass
class SessionSummary:
    """Aggregate statistics for a complete session."""

    session_id: str
    turn_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_duration_ms: float = 0
    tool_call_count: int = 0
    tool_error_count: int = 0
    api_error_count: int = 0
    unique_tools: list[str] = field(default_factory=list)
    model: str = ""
    version: str = ""
    first_timestamp: str = ""
    last_timestamp: str = ""
    turns: list[dict[str, Any]] = field(default_factory=list)


class TranscriptParser(ABC):
    """Interface that platform-specific parsers must implement."""

    @property
    @abstractmethod
    def platform(self) -> str:
        """Short identifier for the platform, e.g. ``"claude_code"``."""

    @abstractmethod
    def list_projects(self, base_path: str | None = None) -> list[ProjectInfo]:
        """List all projects (workspaces) found in the transcript store."""

    @abstractmethod
    def list_sessions(
        self,
        project_path: str,
    ) -> list[SessionInfo]:
        """List sessions within a project directory."""

    @abstractmethod
    def parse_turns(
        self,
        session_path: str,
        *,
        from_line: int = 0,
        max_turns: int | None = None,
    ) -> tuple[list[Turn], int]:
        """Parse complete turns from a session JSONL file.

        Returns ``(turns, next_line_offset)`` so the caller can resume
        where it left off for incremental / tail-style reading.
        """

    def get_session_summary(self, session_path: str) -> SessionSummary:
        """Compute aggregate stats for a full session.

        Default implementation parses all turns and aggregates.
        Subclasses may override for efficiency.
        """
        turns, _ = self.parse_turns(session_path)
        if not turns:
            return SessionSummary(session_id=Path(session_path).stem)

        all_tools: list[str] = []
        summary = SessionSummary(session_id=Path(session_path).stem)

        for turn in turns:
            summary.turn_count += 1
            summary.total_input_tokens += turn.total_input_tokens
            summary.total_output_tokens += turn.total_output_tokens
            summary.total_cache_creation_tokens += turn.total_cache_creation_tokens
            summary.total_cache_read_tokens += turn.total_cache_read_tokens
            if turn.duration_ms:
                summary.total_duration_ms += turn.duration_ms
            summary.tool_call_count += len(turn.tools_called)
            summary.tool_error_count += sum(1 for t in turn.tools_called if t.is_error)
            summary.api_error_count += len(turn.api_errors)
            all_tools.extend(t.name for t in turn.tools_called)
            summary.turns.append(turn.to_eval_dict())

        summary.unique_tools = sorted(set(all_tools))

        if turns:
            first_call = turns[0].model_calls[0] if turns[0].model_calls else None
            last_call = turns[-1].model_calls[-1] if turns[-1].model_calls else None
            if first_call:
                summary.model = first_call.model
                summary.first_timestamp = turns[0].timestamp_start
            if last_call:
                summary.last_timestamp = turns[-1].timestamp_end

        return summary
