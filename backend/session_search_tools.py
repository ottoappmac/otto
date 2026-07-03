"""LangChain tools: ``search_sessions`` and ``get_session_messages``.

Lets the orchestrator search through past sessions by title keyword and
read the message history of any session it finds.

Both tools are synchronous — they read JSON files directly from disk
using the same primitives as the HTTP session API.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_MAX_SEARCH_RESULTS = 10
_MAX_MESSAGES = 20       # max messages returned per session
_SNIPPET_CHARS = 500     # max chars per message content


def _extract_text(content: Any) -> str:
    """Normalise a message content value to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content or "")


def build_session_search_tools() -> list:
    """Return session-search tools for injection into the agent graph."""

    @tool
    def search_sessions(query: str, limit: int = 5) -> str:
        """Search past agent sessions by title keyword.

        Returns matching session titles, dates, agents, and status so you
        can identify which session to look at. Use get_session_messages to
        read the actual conversation from a session found here.

        Args:
            query: Case-insensitive substring to match against session titles.
                   Use an empty string to list the most recent sessions.
            limit: Maximum number of results to return (1–10, default 5).
        """
        from backend.state import session_mgr  # type: ignore[import]

        limit = max(1, min(limit, _MAX_SEARCH_RESULTS))

        try:
            all_sessions = session_mgr.list_history()
        except Exception:
            logger.debug("[session_search] list_history failed", exc_info=True)
            return "Could not read session history."

        q = query.strip().lower()
        if q:
            matches = [s for s in all_sessions if q in (s.title or "").lower()]
        else:
            matches = list(all_sessions)

        matches = matches[:limit]

        if not matches:
            return f"No sessions found matching '{query}'." if q else "No sessions found."

        lines: list[str] = [f"{len(matches)} session(s) found:\n"]
        for s in matches:
            age = ""
            try:
                delta = datetime.now(timezone.utc) - s.created_at
                days = delta.days
                if days == 0:
                    mins = int(delta.total_seconds() / 60)
                    age = f"{mins}m ago" if mins < 60 else f"{mins // 60}h ago"
                else:
                    age = f"{days}d ago"
            except Exception:
                pass

            duration = ""
            try:
                secs = int((s.updated_at - s.created_at).total_seconds())
                if secs >= 60:
                    duration = f", ran {secs // 60}m"
            except Exception:
                pass

            tools_str = ""
            if s.tools_used:
                tools_str = f", tools: {', '.join(s.tools_used[:4])}"

            lines.append(
                f"- id: `{s.id}`\n"
                f"  title: {s.title or 'Untitled'}\n"
                f"  date: {s.created_at.strftime('%Y-%m-%d %H:%M')} ({age})"
                f"  agent: {s.agent_name or 'default'}"
                f"  status: {s.status or 'unknown'}"
                f"{duration}{tools_str}"
            )

        return "\n".join(lines)

    @tool
    def get_session_messages(session_id: str, last_n: int = 10) -> str:
        """Read the message history of a past session.

        Use search_sessions first to find the session ID you want.
        Returns the last N messages (user + assistant turns) so you can
        understand what was discussed and what conclusions were reached.

        Args:
            session_id: The session UUID (from search_sessions results).
            last_n: How many of the most recent messages to return (1–20,
                    default 10).
        """
        from backend.session_manager import load_messages  # type: ignore[import]

        last_n = max(1, min(last_n, _MAX_MESSAGES))

        try:
            msgs = load_messages(session_id)
        except Exception:
            logger.debug("[session_search] load_messages failed", exc_info=True)
            return f"Could not read messages for session '{session_id}'."

        if not msgs:
            return f"No messages found for session '{session_id}'."

        tail = msgs[-last_n:]

        lines: list[str] = [
            f"Last {len(tail)} message(s) from session `{session_id}`:\n"
        ]
        for i, msg in enumerate(tail, 1):
            role = (msg.get("role") or msg.get("type") or "unknown").lower()
            display_role = {
                "human": "User",
                "user": "User",
                "ai": "Assistant",
                "assistant": "Assistant",
                "agent": "Assistant",
            }.get(role, role.capitalize())

            content = _extract_text(msg.get("content", ""))
            content = " ".join(content.split())  # collapse whitespace
            if len(content) > _SNIPPET_CHARS:
                content = content[:_SNIPPET_CHARS] + "…"

            lines.append(f"[{i}] **{display_role}:** {content}")

        return "\n".join(lines)

    return [search_sessions, get_session_messages]
