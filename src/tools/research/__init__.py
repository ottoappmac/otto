"""Research tools — web search, document reading, and knowledge lookup.

All tools are LangChain-compatible and can be passed directly to
``create_deep_agent(tools=[...])``.
"""

from tools.research.duckduckgo_search import duckduckgo_search
from tools.research.web_researcher import web_research
from tools.research.doc_researcher import build_doc_research, doc_research
from tools.research.doc_reader import DocReader
from tools.research.wikipedia import wikipedia_search
from tools.research.youtube_transcript import youtube_search, youtube_transcript

__all__ = [
    "duckduckgo_search",
    "web_research",
    "doc_research",
    "build_doc_research",
    "DocReader",
    "wikipedia_search",
    "youtube_search",
    "youtube_transcript",
]
