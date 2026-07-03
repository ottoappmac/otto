"""DuckDuckGo web search tool.

Returns search result URLs, titles, and snippets using the ``ddgs``
package — no browser automation required.
"""

from __future__ import annotations

from langchain_core.tools import tool


@tool
async def duckduckgo_search(
    query: str,
    max_results: int = 10,
) -> list[dict]:
    """Search DuckDuckGo and return result URLs with titles and snippets.

    Args:
        query: The search query.
        max_results: Maximum number of results (default 10).

    Returns:
        List of dicts with 'url', 'title', and 'snippet' keys.
    """
    from tools.research._loaders import ddg_search_urls

    try:
        return await ddg_search_urls(
            query, max_results=max_results,
        )
    except Exception as exc:
        return [{"error": f"DuckDuckGo search failed: {exc}"}]
