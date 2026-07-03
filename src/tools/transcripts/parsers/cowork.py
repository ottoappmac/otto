"""Claude Cowork (Desktop agent-mode) JSONL transcript parser.

Cowork stores transcripts in two locations under
``~/Library/Application Support/Claude/``:

Conversation JSONL (same schema as Claude Code)::

    local-agent-mode-sessions/<session>/<org>/local_<id>/
        .claude/projects/-sessions-<name>/<uuid>.jsonl

Session metadata::

    claude-code-sessions/<session>/<org>/local_<id>.json

Audit log (unique to Cowork — queue ops, rate limits, HMAC-signed)::

    local-agent-mode-sessions/<session>/<org>/local_<id>/audit.jsonl

Because the conversation JSONL uses the same record schema as Claude
Code, this parser inherits all turn-parsing logic from
:class:`ClaudeCodeParser` and only overrides project/session discovery.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.transcripts.parsers.base import ProjectInfo, SessionInfo
from tools.transcripts.parsers.claude_code import ClaudeCodeParser

logger = logging.getLogger(__name__)


def _default_base_path() -> Path:
    """Platform-specific root for ``local-agent-mode-sessions``."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Claude"
    if sys.platform == "win32":
        appdata = Path.home() / "AppData" / "Roaming" / "Claude"
        return appdata
    return Path.home() / ".config" / "Claude"


_SESSIONS_DIR = "local-agent-mode-sessions"
_META_DIR = "claude-code-sessions"

_TRANSCRIPT_GLOB = "*/*/local_*/.claude/projects/*"


@dataclass(frozen=True)
class CoworkAuditInfo:
    """Summary of a Cowork ``audit.jsonl`` file."""

    session_id: str
    model: str = ""
    claude_code_version: str = ""
    permission_mode: str = ""
    cwd: str = ""
    tools: list[str] | None = None
    mcp_servers: list[str] | None = None
    skills: list[str] | None = None
    agents: list[str] | None = None
    record_counts: dict[str, int] | None = None
    total_records: int = 0


class CoworkParser(ClaudeCodeParser):
    """Parser for Claude Cowork (Desktop agent-mode) session transcripts.

    Inherits turn-parsing from :class:`ClaudeCodeParser` — identical JSONL
    record schema.  Overrides discovery to walk the deeper directory tree
    that Cowork uses.
    """

    @property
    def platform(self) -> str:
        return "cowork"

    def list_projects(self, base_path: str | None = None) -> list[ProjectInfo]:
        root = Path(base_path) if base_path else _default_base_path()
        sessions_root = root / _SESSIONS_DIR
        if not sessions_root.is_dir():
            return []

        projects: list[ProjectInfo] = []
        for entry in sorted(sessions_root.glob(_TRANSCRIPT_GLOB)):
            if not entry.is_dir():
                continue

            session_count = sum(
                1 for f in entry.iterdir()
                if f.suffix == ".jsonl" and not f.name.startswith(".")
            )
            workspace = _decode_session_name(entry.name)

            projects.append(ProjectInfo(
                path=str(entry),
                workspace=workspace,
                session_count=session_count,
            ))

        return projects

    def list_sessions(self, project_path: str) -> list[SessionInfo]:
        sessions = super().list_sessions(project_path)
        meta = self._load_session_meta(project_path)

        if meta:
            for s in sessions:
                if not s.model and "model" in meta:
                    object.__setattr__(s, "model", meta["model"])

        return sessions

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    @staticmethod
    def find_audit_log(session_path: str) -> Path | None:
        """Locate ``audit.jsonl`` relative to a conversation JSONL path.

        Walks upward from the ``.claude/projects/-sessions-…/`` dir to
        the ``local_*`` root where ``audit.jsonl`` lives.
        """
        path = Path(session_path)
        cursor = path.parent if path.is_file() else path

        for _ in range(6):
            candidate = cursor / "audit.jsonl"
            if candidate.is_file():
                return candidate
            if cursor.name.startswith("local_"):
                candidate = cursor / "audit.jsonl"
                return candidate if candidate.is_file() else None
            cursor = cursor.parent

        return None

    @staticmethod
    def parse_audit_log(audit_path: str | Path) -> CoworkAuditInfo:
        """Extract structured metadata from an ``audit.jsonl`` file."""
        path = Path(audit_path)
        counts: dict[str, int] = {}
        session_id = ""
        model = ""
        cc_version = ""
        perm_mode = ""
        cwd = ""
        tools: list[str] | None = None
        mcp_servers: list[str] | None = None
        skills: list[str] | None = None
        agents: list[str] | None = None
        total = 0

        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                total += 1
                rtype = rec.get("type", "unknown")
                subtype = rec.get("subtype", "")
                key = f"{rtype}:{subtype}" if subtype else rtype
                counts[key] = counts.get(key, 0) + 1

                if rtype == "system" and subtype == "init":
                    session_id = rec.get("session_id", "")
                    model = rec.get("model", "")
                    cc_version = rec.get("claude_code_version", "")
                    perm_mode = rec.get("permissionMode", "")
                    cwd = rec.get("cwd", "")
                    tools = rec.get("tools")
                    mcp_servers = [
                        s.get("name", "") for s in rec.get("mcp_servers", [])
                    ]
                    skills = rec.get("skills")
                    agents = rec.get("agents")

        return CoworkAuditInfo(
            session_id=session_id,
            model=model,
            claude_code_version=cc_version,
            permission_mode=perm_mode,
            cwd=cwd,
            tools=tools,
            mcp_servers=mcp_servers,
            skills=skills,
            agents=agents,
            record_counts=counts,
            total_records=total,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_session_meta(self, project_path: str) -> dict[str, Any] | None:
        """Load companion metadata from ``claude-code-sessions/``."""
        meta_root = _default_base_path() / _META_DIR
        if not meta_root.is_dir():
            return None

        try:
            for meta_file in meta_root.rglob("*.json"):
                return json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("Failed to load session meta for %s", project_path)

        return None


def _decode_session_name(dirname: str) -> str:
    """Convert ``-sessions-elegant-inspiring-fermi`` → ``elegant-inspiring-fermi``."""
    prefix = "-sessions-"
    if dirname.startswith(prefix):
        return dirname[len(prefix):]
    return dirname
