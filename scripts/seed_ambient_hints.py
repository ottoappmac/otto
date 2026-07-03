#!/usr/bin/env python3
"""Seed fake ambient hints for UI testing.

Usage
-----
# Add one of each kind with no real LLM or model needed
uv run python scripts/seed_ambient_hints.py

# Flush all existing hints first, then seed
uv run python scripts/seed_ambient_hints.py --reset

Options
-------
--reset     Wipe the hints table before seeding.
--url URL   Backend base URL (default: http://localhost:8000).
            If the backend is running, also fires POST /api/ambient/run
            so the UI polling loop picks up the new hints immediately.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

_FAKE_HINTS = [
    {
        "title": "Summarise last week's research sessions",
        "rationale": (
            "You ran 6 research sessions in the past 7 days covering "
            "LangChain, RAG pipelines, and embedding models. A summary "
            "document could help consolidate what you learned."
        ),
        "proposed_prompt": (
            "Please read my recent session transcripts from the last 7 days "
            "and write a concise summary of the key findings and open questions."
        ),
        "suggested_agent": None,
        "kind": "task",
        "confidence": 0.87,
        "sources": ["sessions", "memory"],
    },
    {
        "title": "Automate daily standup note generation",
        "rationale": (
            "Based on your activity (VS Code, Slack, GitHub) between 9–11 am "
            "every weekday, you could auto-generate a standup note each morning."
        ),
        "proposed_prompt": (
            "Set up a scheduled task that reads my macOS activity from 9 am to "
            "now and generates a short standup note: what I worked on and any blockers."
        ),
        "suggested_agent": None,
        "kind": "automation",
        "confidence": 0.79,
        "sources": ["activity", "history"],
    },
    {
        "title": "Review open TODO items from memory",
        "rationale": (
            "Your long-term memory contains 4 topics tagged as TODO or in-progress. "
            "It has been over a week since you last touched them."
        ),
        "proposed_prompt": (
            "Show me all the open TODO or in-progress items from my memory "
            "and help me prioritise which ones to tackle today."
        ),
        "suggested_agent": None,
        "kind": "task",
        "confidence": 0.92,
        "sources": ["memory"],
    },
]


async def _seed(reset: bool) -> None:
    import sys
    sys.path.insert(0, "src")   # so backend imports resolve

    from backend.ambient_store import get_store
    from backend.config import get_app_data_dir

    store = await get_store()

    if reset:
        db_path = get_app_data_dir() / "ambient.db"
        import aiosqlite
        async with aiosqlite.connect(str(db_path)) as db:
            await db.execute("DELETE FROM hints")
            await db.commit()
        print("✓ Flushed existing hints")

    ids = await store.add_hints(_FAKE_HINTS, cooldown_hours=0, max_per_day=50)
    print(f"✓ Seeded {len(ids)} hint(s):")
    for hint_id in ids:
        h = await store.get(hint_id)
        if h:
            print(f"  [{h.kind}] {h.title}")

    if not ids:
        print("  (none added — they may already exist in the cooldown window; use --reset to force)")


def _notify_backend(base_url: str) -> None:
    """Best-effort: tell the running backend to re-read its hints list."""
    try:
        import urllib.request
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/ambient/run",
            data=b"",
            method="POST",
        )
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=3) as resp:
            body = json.loads(resp.read())
        print(f"✓ Backend sweep triggered: {body}")
    except Exception as exc:
        print(f"  (could not notify backend at {base_url}: {exc} — hints will appear on next poll)")


def main() -> None:
    reset = "--reset" in sys.argv
    url_flag = next((a for a in sys.argv if a.startswith("--url=")), None)
    base_url = url_flag.split("=", 1)[1] if url_flag else "http://localhost:8000"

    asyncio.run(_seed(reset))
    _notify_backend(base_url)
    print("\nOpen the app → Suggestions tab (or navigate to /ambient) to see the hints.")


if __name__ == "__main__":
    main()
