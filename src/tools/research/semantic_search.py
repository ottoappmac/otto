"""Semantic search tool — query the local embedding index.

Searches across OTTO's indexed artifacts (memory topic files, session
transcripts, session upload files, and user-pinned directories) using
dense vector retrieval backed by sqlite-vec and nomic-embed-text.

Requires ``memory.enabled`` and ``memory.embedding.enabled`` in the
app config.  On non-Apple-Silicon machines the tool returns a helpful
error rather than crashing the agent.

Usage by the agent::

    semantic_search("how do I authenticate with the Stripe API?")
    semantic_search("project goals for Q3", source_type="memory")
    semantic_search("deployment error last week", source_type="transcript")
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_SOURCE_TYPES = ("memory", "transcript", "file")


@tool
async def semantic_search(
    query: str,
    k: int = 10,
    source_type: str = "",
) -> str:
    """Search OTTO's local knowledge base using semantic (meaning-based) similarity.

    Unlike ``doc_research`` which searches a specific document you provide,
    this tool searches across everything OTTO has indexed: memory topics,
    past session transcripts, and files you have shared.

    Args:
        query: Natural-language question or topic to search for.
        k: Number of results to return (default 10, max 20).
        source_type: Optional filter — one of "memory", "transcript",
            "file".  Leave empty to search all indexed content.
    """
    from backend.config import AppConfig

    cfg = await AppConfig.aload()
    if not cfg.memory.embedding_enabled:
        return (
            "Semantic search is disabled. "
            "Enable Memory (Settings → Agent Memory) to activate it."
        )

    stype: str | None = source_type.strip().lower() or None
    if stype and stype not in _SOURCE_TYPES:
        return (
            f"Unknown source_type '{stype}'. "
            f"Valid values: {', '.join(_SOURCE_TYPES)} (or leave empty for all)."
        )

    k = max(1, min(k, 20))

    try:
        from backend.embedding_index import get_embedding_index

        idx = await get_embedding_index()
        results = await idx.search(query, k=k, source_type=stype)
    except RuntimeError as exc:
        return f"Semantic search unavailable: {exc}"
    except Exception as exc:
        logger.warning("[semantic_search] search failed: %s", exc, exc_info=True)
        return f"Search failed: {exc}"

    if not results:
        filter_note = f" (filter: source_type={stype})" if stype else ""
        return f"No results found for: {query!r}{filter_note}"

    parts: list[str] = [f"Found {len(results)} result(s) for: {query!r}\n"]
    for i, r in enumerate(results, 1):
        source = r["source_path"]
        stype_label = r["source_type"]
        score = r["score"]
        text = r["text"].strip()
        parts.append(
            f"[{i}] [{stype_label}] {source} (score: {score:.3f})\n{text}"
        )

    # Activity timeline correlation — if any results are transcripts, surface
    # what the user was doing in the OS during those sessions (FTS5, no extra
    # embedding cost).
    transcript_sources = [r for r in results if r["source_type"] == "transcript"]
    if transcript_sources:
        activity_note = _correlate_activity(transcript_sources)
        if activity_note:
            parts.append(f"\n{activity_note}")

    return "\n\n---\n\n".join(parts)


def _correlate_activity(transcript_results: list[dict]) -> str:
    """Return a brief note of what the user was doing during transcript sessions.

    Queries the activity tracker FTS5 index for the session timestamps
    to add context without any additional embedding cost.
    """
    try:
        import re
        import sqlite3

        from backend.config import get_app_data_dir
        from backend.session_transcript import _transcript_path

        activity_db = get_app_data_dir() / "activity.db"
        if not activity_db.exists():
            return ""

        notes: list[str] = []
        seen_sessions: set[str] = set()
        for r in transcript_results[:3]:  # cap at 3 to keep context brief
            path_str = r["source_path"]
            # Extract session_id from path like .../transcripts/<uuid>.jsonl
            m = re.search(r"([0-9a-f\-]{36})\.jsonl$", path_str)
            if not m or m.group(1) in seen_sessions:
                continue
            session_id = m.group(1)
            seen_sessions.add(session_id)

            tp = _transcript_path(session_id)
            if not tp.exists():
                continue
            mtime = tp.stat().st_mtime

            db = sqlite3.connect(str(activity_db))
            db.row_factory = sqlite3.Row
            try:
                rows = db.execute(
                    "SELECT DISTINCT app, title FROM activity"
                    " WHERE ts BETWEEN ? AND ?"
                    " AND app != '' LIMIT 5",
                    (int((mtime - 3600) * 1000), int(mtime * 1000)),
                ).fetchall()
                if rows:
                    apps = ", ".join({r["app"] for r in rows})
                    notes.append(f"During session {session_id[:8]}: {apps}")
            except Exception:
                pass
            finally:
                db.close()

        if notes:
            return "Activity context:\n" + "\n".join(f"  • {n}" for n in notes)
    except Exception:
        pass
    return ""
