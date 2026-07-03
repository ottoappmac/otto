"""Enum definitions for selecting tools and subagents."""

from __future__ import annotations

from enum import Enum


class ToolOption(str, Enum):
    """Selectable direct tools for the orchestrator."""

    WIKIPEDIA = "wikipedia"
    DUCKDUCKGO = "duckduckgo"
    DOC_READER = "doc_reader"
    DOC_RESEARCHER = "doc_researcher"
    WEB_RESEARCHER = "web_researcher"
    YOUTUBE = "youtube"
    PLAYWRIGHT_MCP = "playwright_mcp"
    SEMANTIC_SEARCH = "semantic_search"
    AMBIENT_TOGGLE = "ambient_toggle"


class SubAgentOption(str, Enum):
    """Selectable subagents delegated via the ``task`` tool."""

    WEB_VOYAGER = "web-voyager"
    COMPUTER_VOYAGER = "computer-voyager"
