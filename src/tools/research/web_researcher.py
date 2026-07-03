"""Web researcher tool — search the web and return page content.

Combines DuckDuckGo search with page fetching: searches for URLs, fetches the
top results as markdown, and returns the content for the agent to reason over.

No vectorstore, no embeddings, no sub-query generation.
"""

from __future__ import annotations

import asyncio
import logging
import time

from langchain_core.tools import tool

from tools.research._loaders import (
    ddg_search_urls,
    fetch_url_as_markdown,
)

logger = logging.getLogger(__name__)


@tool
async def web_research(
    query: str,
    max_results: int = 5,
    max_pages_to_fetch: int = 3,
) -> str:
    """Search the web and return content from the top results.

    Searches DuckDuckGo for URLs matching *query*, then fetches and converts
    the top pages to markdown so the agent can read them directly.

    Args:
        query: The search query.
        max_results: Number of DuckDuckGo results to retrieve.
        max_pages_to_fetch: Number of pages to actually fetch and read.
    """
    t0 = time.monotonic()
    logger.info("[web_research] START query=%r max_results=%d", query, max_results)
    try:
        results = await ddg_search_urls(
            query, max_results=max_results,
        )
    except Exception as exc:
        logger.warning("[web_research] DDG search failed after %.1fs: %s", time.monotonic() - t0, exc)
        return f"Web search failed: {exc}"

    logger.debug("[web_research] DDG returned %d results in %.1fs", len(results), time.monotonic() - t0)

    if not results:
        return f"No search results found for: {query}"

    to_fetch = results[:max_pages_to_fetch]
    urls = [r["url"] for r in to_fetch]
    logger.debug("[web_research] fetching %d pages: %s", len(urls), urls)
    t1 = time.monotonic()
    pages = await asyncio.gather(
        *[_safe_fetch(r["url"]) for r in to_fetch],
    )
    logger.debug("[web_research] all %d pages fetched in %.1fs", len(pages), time.monotonic() - t1)

    parts: list[str] = []
    for i, (result, page) in enumerate(
        zip(to_fetch, pages), 1,
    ):
        title = result.get("title", "Untitled")
        url = result["url"]
        if page.startswith("Error:"):
            parts.append(
                f"## [{i}] {title}\nURL: {url}\n{page}"
            )
        else:
            parts.append(
                f"## [{i}] {title}\nURL: {url}\n\n{page}"
            )

    if len(results) > max_pages_to_fetch:
        extras = results[max_pages_to_fetch:]
        lines = ["\n## Additional results (not fetched):"]
        for r in extras:
            lines.append(f"- [{r['title']}]({r['url']})")
        parts.append("\n".join(lines))

    logger.info("[web_research] DONE query=%r — %d results, %d pages fetched (%.1fs)", query, len(results), len(to_fetch), time.monotonic() - t0)
    return "\n\n---\n\n".join(parts)


async def _safe_fetch(url: str) -> str:
    """Fetch a URL, returning an error string on failure."""
    t0 = time.monotonic()
    try:
        result = await fetch_url_as_markdown(url)
        logger.debug("[web_research] fetch OK %s (%.1fs, %d chars)", url, time.monotonic() - t0, len(result))
        return result
    except Exception as exc:
        logger.warning("[web_research] fetch FAIL %s (%.1fs): %s", url, time.monotonic() - t0, exc)
        return f"Error: could not fetch page — {exc}"
