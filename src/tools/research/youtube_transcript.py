"""YouTube transcript tools — find videos and search their transcripts.

Two composable LangChain tools:

- ``youtube_search`` — find candidate YouTube videos for a query (via the
  shared DuckDuckGo helper) and return their video IDs / URLs.
- ``youtube_transcript`` — fetch a video's transcript with
  ``langchain_community``'s ``YoutubeLoader`` (CHUNKS format, so each chunk
  carries a start timestamp + deep link in its metadata), then return the
  passages most relevant to a query ranked with BM25 — or the whole
  transcript when ``full_transcript=True``.

No vectorstore or embeddings model required: ranking reuses the BM25
``rank_chunks`` helper shared with ``doc_research``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _extract_video_id(value: str) -> Optional[str]:
    """Return the 11-char video ID for a URL/ID, or ``None`` if not parseable."""
    from langchain_community.document_loaders import YoutubeLoader

    candidate = value.strip()
    if not candidate:
        return None
    # Bare 11-char IDs aren't URLs, so extract_video_id can't parse them.
    if len(candidate) == 11 and "/" not in candidate and "." not in candidate:
        return candidate
    try:
        return YoutubeLoader.extract_video_id(candidate)
    except ValueError:
        return None


@tool
async def youtube_search(query: str, max_results: int = 5) -> list[dict]:
    """Find YouTube videos matching a query and return their IDs and URLs.

    Searches the web (DuckDuckGo) and keeps only YouTube watch links,
    de-duplicated by video ID. Use this to pick a video, then pass the
    chosen ``video_id`` or ``url`` to ``youtube_transcript``.

    Args:
        query: What to search for (e.g. "3blue1brown attention transformers").
        max_results: Maximum number of videos to return (default 5).

    Returns:
        List of dicts with 'video_id', 'url', 'title', and 'snippet' keys.
    """
    from tools.research._loaders import ddg_search_urls

    try:
        # Over-fetch since many results won't be watch pages (channels,
        # playlists, non-YouTube sites).
        raw = await ddg_search_urls(
            f"{query} site:youtube.com",
            max_results=max(max_results * 4, 10),
        )
    except Exception as exc:
        return [{"error": f"YouTube search failed: {exc}"}]

    results: list[dict] = []
    seen: set[str] = set()
    for item in raw:
        url = item.get("url", "")
        video_id = _extract_video_id(url)
        if not video_id or video_id in seen:
            continue
        seen.add(video_id)
        results.append({
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
        })
        if len(results) >= max_results:
            break

    if not results:
        return [{"error": f"No YouTube videos found for: {query}"}]
    return results


def _load_transcript_chunks(
    video_id: str,
    *,
    chunk_size_seconds: int,
    language: str,
) -> list:
    """Load a transcript as timestamped chunk Documents (blocking)."""
    from langchain_community.document_loaders import YoutubeLoader
    from langchain_community.document_loaders.youtube import TranscriptFormat

    loader = YoutubeLoader(
        video_id=video_id,
        add_video_info=False,  # avoids flaky pytube metadata scraping
        language=[language] if language else ["en"],
        transcript_format=TranscriptFormat.CHUNKS,
        chunk_size_seconds=chunk_size_seconds,
    )
    return loader.load()


def _format_passages(docs: list, video_id: str) -> str:
    """Format ranked transcript chunks using their timestamp metadata."""
    if not docs:
        return f"No matching passages found in video {video_id}."
    parts: list[str] = []
    for doc in docs:
        ts = doc.metadata.get("start_timestamp", "00:00:00")
        link = doc.metadata.get(
            "source", f"https://www.youtube.com/watch?v={video_id}"
        )
        parts.append(f"[{ts}] {link}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


@tool
async def youtube_transcript(
    video: str,
    query: str = "",
    k: int = 8,
    start_seconds: Optional[int] = None,
    end_seconds: Optional[int] = None,
    chunk_size_seconds: int = 30,
    language: str = "en",
    full_transcript: bool = False,
) -> str:
    """Fetch a YouTube transcript and return the passages relevant to a query.

    Each returned passage is timestamped and includes a deep link that
    jumps to that moment in the video. By default only the top-``k``
    passages matching ``query`` (BM25-ranked) are returned, so long
    transcripts don't flood the context.

    Args:
        video: A YouTube URL, an 11-char video ID, or a free-text search
            query (the top search hit is used when it isn't a URL/ID).
        query: What to look for inside the transcript. Leave empty to get
            the opening passages (or the windowed passages with start/end).
        k: Number of passages to return when searching (default 8).
        start_seconds: If set, only include passages at/after this time.
        end_seconds: If set, only include passages before this time.
        chunk_size_seconds: Granularity of each passage (default 30s).
        language: Preferred transcript language code (default "en"; falls
            back to English, then any available transcript).
        full_transcript: Return the entire transcript text instead of
            ranked passages (large results are auto-offloaded for grep).
    """
    video_id = _extract_video_id(video)
    if video_id is None:
        # Treat the input as a search query and pick the best hit.
        from tools.research._loaders import ddg_search_urls

        try:
            raw = await ddg_search_urls(
                f"{video} site:youtube.com", max_results=10
            )
        except Exception as exc:
            return f"Could not search for a video matching '{video}': {exc}"
        for item in raw:
            video_id = _extract_video_id(item.get("url", ""))
            if video_id:
                break
        if video_id is None:
            return f"Could not find a YouTube video for: {video}"

    try:
        chunks = await asyncio.to_thread(
            _load_transcript_chunks,
            video_id,
            chunk_size_seconds=chunk_size_seconds,
            language=language,
        )
    except Exception as exc:
        return f"Could not load transcript for video {video_id}: {exc}"

    if not chunks:
        return (
            f"No transcript available for video {video_id} "
            "(captions may be disabled)."
        )

    if start_seconds is not None or end_seconds is not None:
        lo = start_seconds if start_seconds is not None else 0
        hi = end_seconds if end_seconds is not None else float("inf")
        chunks = [
            c for c in chunks
            if lo <= c.metadata.get("start_seconds", 0) < hi
        ]
        if not chunks:
            return (
                f"No transcript passages between {lo}s and {hi}s "
                f"for video {video_id}."
            )

    if full_transcript:
        header = f"Full transcript for video {video_id}:\n\n"
        return header + "\n".join(c.page_content.strip() for c in chunks)

    if query and len(chunks) > k:
        from tools.research._loaders import rank_chunks

        relevant = await asyncio.to_thread(rank_chunks, chunks, query, k)
    else:
        relevant = chunks[:k]

    logger.info(
        "youtube_transcript: video=%s chunks=%d returned=%d query=%r",
        video_id, len(chunks), len(relevant), query,
    )
    return _format_passages(relevant, video_id)
