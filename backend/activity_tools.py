"""LangChain tool wrappers around the activity timeline.

Lets the agent answer questions like "what was I working on last
Tuesday afternoon" or "find when I was researching flights to Tokyo"
by querying the local activity DB.

The tool deliberately returns short rendered strings (not raw JSON) so
small models can use them effectively without further parsing.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any

from langchain_core.tools import tool

from backend.activity_tracker import (
    daily_summary,
    list_apps,
    search_activity,
)


def _parse_when(s: str | None) -> int | None:
    """Best-effort parse of a natural-ish date string into a unix timestamp.

    Accepts:
      - ``"today"``, ``"yesterday"``
      - ``"3d"`` / ``"24h"`` / ``"30m"`` (offsets back from now)
      - ISO date ``"2026-04-29"`` (start-of-day local)
      - Bare epoch ints
    """
    if s is None or s == "":
        return None
    s = s.strip().lower()
    now = int(time.time())

    if s == "today":
        d = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return int(d.timestamp())
    if s == "yesterday":
        d = (datetime.now() - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return int(d.timestamp())
    if s.endswith("d") and s[:-1].isdigit():
        return now - int(s[:-1]) * 86400
    if s.endswith("h") and s[:-1].isdigit():
        return now - int(s[:-1]) * 3600
    if s.endswith("m") and s[:-1].isdigit():
        return now - int(s[:-1]) * 60
    try:
        return int(s)
    except ValueError:
        pass
    try:
        d = datetime.fromisoformat(s)
        return int(d.timestamp())
    except ValueError:
        return None


def _fmt_row(row: dict[str, Any]) -> str:
    when = datetime.fromtimestamp(row["ts"]).strftime("%Y-%m-%d %H:%M")
    parts = [when, row["app"]]
    if row.get("title"):
        parts.append(f"— {row['title']}")
    if row.get("url"):
        parts.append(f"({row['url']})")
    if row.get("file_path"):
        parts.append(f"<{row['file_path']}>")
    if row.get("duration_s"):
        mins = row["duration_s"] // 60
        if mins >= 1:
            parts.append(f"[{mins}m]")
    head = "  ".join(parts)
    ctx = row.get("context") or ""
    if ctx:
        # Keep the rendered context one line so the agent's tool output
        # stays scannable even with 25+ rows returned.
        ctx_one_line = " ".join(ctx.split())
        if len(ctx_one_line) > 200:
            ctx_one_line = ctx_one_line[:200] + "…"
        head += f"\n      ↳ {ctx_one_line}"
    return head


def build_activity_tools() -> list:
    """Return the agent tools that read the local activity timeline."""

    @tool
    def search_screen_history(
        query: str = "",
        when_from: str = "",
        when_to: str = "",
        app: str = "",
        limit: int = 25,
    ) -> str:
        """Search the user's local activity timeline (apps, window titles,
        URLs) recorded by the on-device tracker.  Use this when the user
        asks 'when did I look at X', 'what was I doing on Tuesday', or
        wants to recall past activity.  All data is local — nothing is
        sent over the network.

        Args:
            query: Free-text search terms (FTS5).  Empty returns latest
                   rows in the date range.  Examples: "tokyo flights",
                   "session_manager", "slack deploy".
            when_from: Lower bound on time.  Accepts "today", "yesterday",
                       "7d" (7 days ago), "24h", "30m", or ISO date
                       "2026-04-29", or epoch seconds.  Empty = no lower
                       bound.
            when_to:   Upper bound on time, same formats as when_from.
            app:       Restrict to a single app (e.g. "Safari", "Slack").
            limit:     Max rows to return (default 25).
        """
        rows = search_activity(
            query=query or None,
            date_from=_parse_when(when_from or None),
            date_to=_parse_when(when_to or None),
            app=app or None,
            limit=limit,
        )
        if not rows:
            return "No activity found for those filters."
        header = f"Found {len(rows)} entries:"
        return "\n".join([header, *[f"  • {_fmt_row(r)}" for r in rows]])

    @tool
    def list_recent_apps(days: int = 7) -> str:
        """List apps the user has used over the last N days, ranked by
        time spent.  Useful for 'what have I been doing this week'."""
        rows = list_apps(days=days)
        if not rows:
            return f"No activity recorded in the last {days} days."
        lines = [f"Apps used in the last {days} days:"]
        for r in rows[:25]:
            mins = r["seconds"] // 60
            lines.append(f"  • {r['app']:30s}  {mins:>5d}m   ({r['samples']} samples)")
        return "\n".join(lines)

    @tool
    def activity_summary(when_from: str = "today", when_to: str = "") -> str:
        """Give a per-app time breakdown for a window.  Defaults to today.

        Args:
            when_from: Start of window (e.g. "today", "yesterday", "7d").
            when_to:   End of window.  Empty = now.
        """
        f = _parse_when(when_from) or 0
        t = _parse_when(when_to) or int(time.time())
        s = daily_summary(f, t)
        if not s["apps"]:
            return "No activity recorded in that window."
        total_min = s["total_seconds"] // 60
        lines = [
            f"Activity {datetime.fromtimestamp(f).date()} → "
            f"{datetime.fromtimestamp(t).date()}  ({total_min}m total):",
        ]
        for a in s["apps"][:25]:
            mins = a["seconds"] // 60
            lines.append(f"  • {a['app']:30s}  {mins:>5d}m")
        return "\n".join(lines)

    return [search_screen_history, list_recent_apps, activity_summary]
