"""Wikipedia search tool.

Lightweight wrapper around the ``wikipedia`` package for looking up factual
information about people, places, events, concepts, etc.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def wikipedia_search(query: str, sentences: int = 5) -> str:
    """Search Wikipedia and return a summary.

    Args:
        query: The search query.
        sentences: Number of sentences to include in the summary (default 5).
    """
    import wikipedia

    try:
        search_results = wikipedia.search(query, results=3)
        if not search_results:
            return f"No Wikipedia results found for: {query}"

        try:
            summary = wikipedia.summary(search_results[0], sentences=sentences)
            return f"Wikipedia ({search_results[0]}): {summary}"
        except wikipedia.DisambiguationError as exc:
            if exc.options:
                summary = wikipedia.summary(
                    exc.options[0], sentences=sentences,
                )
                return (
                    f"Wikipedia ({exc.options[0]}): {summary}"
                )
            return (
                f"Disambiguation error for '{query}'. "
                f"Options: {exc.options[:5]}"
            )
        except wikipedia.PageError:
            for result in search_results[1:]:
                try:
                    summary = wikipedia.summary(result, sentences=sentences)
                    return f"Wikipedia ({result}): {summary}"
                except (wikipedia.PageError, wikipedia.DisambiguationError):
                    continue
            return f"Could not find a Wikipedia page for: {query}"

    except Exception as exc:
        logger.error("Wikipedia search error: %s", exc)
        return f"Error searching Wikipedia: {exc}"
