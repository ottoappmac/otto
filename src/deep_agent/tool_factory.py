"""Tool initialization based on :class:`ToolOption` selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from deep_agent.options import ToolOption
from tools.loop_guard import (
    ToolLoopGuard,
    guard_all_tools,
    wrap_with_loop_guard,
)
from utilities.environment import Environment

logger = logging.getLogger(__name__)


# Backwards-compatible alias.  ``ToolLoopGuard``, ``ToolLoopDetected`` and
# the wrap helper now live in :mod:`tools.loop_guard` so the backend can
# attach them to MCP-loaded tools without importing from ``deep_agent``.
# Older imports of ``_wrap_with_loop_guard`` from this module still work.
_wrap_with_loop_guard = wrap_with_loop_guard


# ── Playwright-specific recovery hint for ToolLoopGuard ──────────────────────
#
# The Playwright `ref` argument is the canonical trigger of the
# identical-args failure mode this guard exists to bound: the model
# sometimes passes a CSS selector like ``"[role='combobox']"`` instead
# of a snapshot id like ``"e23"``, the MCP server rejects it, and the
# KV prefix cache keeps re-predicting the same bad token sequence.
# Stamping this hint into the guard makes the recovery message
# specifically actionable for browser tools rather than generic.

_PLAYWRIGHT_RECOVERY_HINT = (
    "For Playwright browser tools, the `ref` argument MUST be a literal "
    "snapshot id like \"e23\" (NOT a CSS selector, NOT an XPath). "
    "Call `browser_snapshot()` first, pick a different element id, or "
    "use a different tool."
)


@dataclass
class ToolSet:
    """Holds the tool list consumed by the orchestrator."""

    deep_agent_tools: List[BaseTool] = field(default_factory=list)
    _mcp_helper: Any = field(default=None, repr=False)
    _loop_guard: Optional[ToolLoopGuard] = field(default=None, repr=False)

    async def close(self) -> None:
        """Clean up MCP connections."""
        if self._mcp_helper is not None:
            await self._mcp_helper.close()
            self._mcp_helper = None


def create_tools(
    options: list[ToolOption],
    llm: BaseChatModel,
    *,
    browser_headless: bool = False,
    provider: str = "",
) -> ToolSet:
    """Initialise only the requested tools and return a :class:`ToolSet`.

    Research tools (``WIKIPEDIA``, ``DUCKDUCKGO``, ``WEB_RESEARCHER``,
    ``DOC_RESEARCHER``, ``DOC_READER``) are pure-Python and need no
    embedding model or vector store: web/doc research returns full
    page/document content and ``DOC_RESEARCHER`` ranks chunks with BM25.
    ``SEMANTIC_SEARCH`` uses the local sqlite-vec index; it soft-fails
    on non-Apple-Silicon machines with a clear error message.
    """
    ts = ToolSet()
    opt = set(options)

    if ToolOption.WIKIPEDIA in opt:
        _load_wikipedia(ts)

    if ToolOption.DUCKDUCKGO in opt:
        _load_duckduckgo(ts)

    if ToolOption.WEB_RESEARCHER in opt:
        _load_web_researcher(ts)

    if ToolOption.YOUTUBE in opt:
        _load_youtube(ts)

    if ToolOption.DOC_RESEARCHER in opt:
        _load_doc_researcher(ts)

    if ToolOption.DOC_READER in opt:
        _load_doc_reader(ts, llm=llm)

    if ToolOption.SEMANTIC_SEARCH in opt:
        _load_semantic_search(ts)

    if ToolOption.PLAYWRIGHT_MCP in opt:
        _load_playwright_mcp_tools(ts, provider=provider)

    if ToolOption.AMBIENT_TOGGLE in opt:
        _load_ambient_settings(ts)

    # Universal loop-guard chokepoint: wrap every directly-loaded tool with
    # one shared guard.  Idempotent for the Playwright MCP tools already
    # guarded above, so they keep their per-connection guard.
    try:
        guard_all_tools(
            ts.deep_agent_tools,
            recovery_hint=(
                "You are repeating tool calls without making progress. Stop "
                "and either try a fundamentally different approach/tool or "
                "report your best answer from what you already have."
            ),
            window=Environment.get_loop_guard_window(),
            max_no_progress=Environment.get_loop_guard_max_no_progress(),
            max_identical_success=Environment.get_loop_guard_max_success(),
            recovery_temperature=Environment.get_loop_recovery_temperature(),
            recovery_temperature_turns=(
                Environment.get_loop_recovery_temperature_turns()
            ),
            max_escalations=Environment.get_loop_guard_max_escalations(),
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("Universal loop guard (direct agent): could not apply — %s", exc)

    return ts


# ── Research tools (pure-Python, no vector store) ────────────────────────────


def _load_wikipedia(ts: ToolSet) -> None:
    try:
        from tools.research.wikipedia import wikipedia_search
        ts.deep_agent_tools.append(wikipedia_search)
        logger.info("Wikipedia search tool loaded")
    except Exception as exc:
        logger.warning("Wikipedia search: could not load — %s", exc)


def _load_duckduckgo(ts: ToolSet) -> None:
    try:
        from tools.research.duckduckgo_search import duckduckgo_search
        ts.deep_agent_tools.append(duckduckgo_search)
        logger.info("DuckDuckGo search tool loaded")
    except Exception as exc:
        logger.warning("DuckDuckGo search: could not load — %s", exc)


def _load_web_researcher(ts: ToolSet) -> None:
    try:
        from tools.research.web_researcher import web_research
        ts.deep_agent_tools.append(web_research)
        logger.info("Web researcher tool loaded")
    except Exception as exc:
        logger.warning("Web researcher: could not load — %s", exc)


def _load_youtube(ts: ToolSet) -> None:
    try:
        from tools.research.youtube_transcript import (
            youtube_search,
            youtube_transcript,
        )
        ts.deep_agent_tools.append(youtube_search)
        ts.deep_agent_tools.append(youtube_transcript)
        logger.info("YouTube transcript tools loaded")
    except Exception as exc:
        logger.warning("YouTube transcript tools: could not load — %s", exc)


def _load_doc_researcher(ts: ToolSet) -> None:
    try:
        from tools.research.doc_researcher import doc_research
        ts.deep_agent_tools.append(doc_research)
        logger.info("Doc researcher tool loaded")
    except Exception as exc:
        logger.warning("Doc researcher: could not load — %s", exc)


def _load_semantic_search(ts: ToolSet) -> None:
    try:
        from tools.research.semantic_search import semantic_search
        ts.deep_agent_tools.append(semantic_search)
        logger.info("Semantic search tool loaded")
    except Exception as exc:
        logger.warning("Semantic search: could not load — %s", exc)


def _load_ambient_settings(ts: ToolSet) -> None:
    try:
        from tools.settings.ambient_settings import configure_ambient_agent
        ts.deep_agent_tools.append(configure_ambient_agent)
        logger.info("Ambient settings tool loaded")
    except Exception as exc:
        logger.warning("Ambient settings tool: could not load — %s", exc)


def _load_doc_reader(ts: ToolSet, *, llm: Optional[BaseChatModel]) -> None:
    try:
        from tools.research.doc_reader import DocReader

        if llm is None:
            logger.warning(
                "Doc reader: skipped — no LLM provided. "
                "Pass llm= to create_tools() to enable."
            )
            return

        tool = DocReader.from_llm(llm)
        ts.deep_agent_tools.append(tool)
        logger.info("Doc reader tool loaded")
    except Exception as exc:
        logger.warning("Doc reader: could not load — %s", exc)


# ── MCP wrappers / patches (Playwright-specific quirks) ──────────────────────


def _strip_none_args(tool: BaseTool) -> None:
    """Wrap an MCP tool's coroutine to drop ``None``-valued arguments.

    Small models sometimes hallucinate optional parameters as ``null``.
    Stripping them prevents validation errors from the MCP server.
    """
    from functools import wraps
    from langchain_core.tools import StructuredTool

    if not isinstance(tool, StructuredTool) or tool.coroutine is None:
        return

    original = tool.coroutine

    @wraps(original)
    async def _cleaned(*args: Any, **kwargs: Any) -> Any:
        cleaned = {k: v for k, v in kwargs.items() if v is not None}
        return await original(*args, **cleaned)

    tool.coroutine = _cleaned


def _strip_null_image_content(tool: BaseTool) -> None:
    """Wrap an MCP tool's coroutine to drop image content blocks with null data."""
    from functools import wraps
    from langchain_core.tools import StructuredTool

    if not isinstance(tool, StructuredTool) or tool.coroutine is None:
        return

    original = tool.coroutine

    @wraps(original)
    async def _filtered(*args: Any, **kwargs: Any) -> Any:
        import langchain_mcp_adapters.tools as _lma_tools

        _orig_convert = _lma_tools._convert_call_tool_result

        def _safe_convert(result: Any) -> Any:
            if hasattr(result, "content") and isinstance(result.content, list):
                filtered = []
                for item in result.content:
                    if getattr(item, "type", None) == "image" and not getattr(item, "data", None):
                        logger.debug("MCP: dropping null-data image content block")
                        continue
                    filtered.append(item)
                result = type(result)(
                    content=filtered,
                    isError=getattr(result, "isError", False),
                )
            return _orig_convert(result)

        _lma_tools._convert_call_tool_result = _safe_convert
        try:
            return await original(*args, **kwargs)
        finally:
            _lma_tools._convert_call_tool_result = _orig_convert

    tool.coroutine = _filtered


_IMAGE_ONLY_TOOLS = frozenset({"browser_take_screenshot"})

# Playwright MCP tools whose `ref` argument is the classic CSS-selector
# confusion target.  We patch their descriptions with an explicit
# positive/negative example pair so small models see the hint in the
# tool schema itself, which is weighted more heavily than system prompt
# text on most ReAct stacks.
_REF_TAKING_BROWSER_TOOLS = frozenset({
    "browser_click",
    "browser_type",
    "browser_fill_form",
    "browser_hover",
    "browser_select_option",
    "browser_drag",
})

_REF_HINT = (
    "\n\n"
    "IMPORTANT: `ref` MUST be a literal element id from the latest "
    "accessibility snapshot (e.g. \"e23\", \"e147\"). It is NOT a CSS "
    "selector, XPath, or role expression. "
    "Correct:   {\"ref\": \"e23\"}. "
    "Incorrect: {\"ref\": \"[role='combobox']\"}, {\"ref\": \"#search\"}, "
    "{\"ref\": \".btn-primary\"}. "
    "If you don't have a current ref, call `browser_snapshot()` first."
)


_BROWSER_TYPE_SUBMIT_HINT = (
    "\n\n"
    "IMPORTANT: pass `submit=true` to press Enter right after typing whenever "
    "the field commits on Enter — search boxes, URL/address bars, chat inputs, "
    "and single-line forms. Only omit `submit=true` when the user said not to "
    "press Enter, or for a multi-field form with a separate submit button "
    "(use `browser_fill_form` + `browser_click`)."
)


def _patch_ref_tool_description(tool: BaseTool) -> None:
    """Append the ref-vs-selector hint to ref-taking browser tools.

    Most agent frameworks embed ``tool.description`` directly in the
    schema the model sees during tool selection, so this is the highest-
    signal place to put the guidance.  Idempotent: checks for a marker
    substring before appending so repeated tool loads don't stack hints.

    Also appends the ``submit=true`` hint to ``browser_type`` so single-line
    fields (search boxes, URL bars) get an Enter press in the same call.
    """
    if tool.name == "browser_type":
        current = tool.description or ""
        if "pass `submit=true` to press Enter" not in current:
            tool.description = current.rstrip() + _BROWSER_TYPE_SUBMIT_HINT
    if tool.name not in _REF_TAKING_BROWSER_TOOLS:
        return
    current = tool.description or ""
    if "literal element id from the latest accessibility snapshot" in current:
        return
    tool.description = current.rstrip() + _REF_HINT


def _load_playwright_mcp_tools(ts: ToolSet, *, provider: str = "") -> None:
    """Connect to the Playwright MCP service and add its tools to *ts*.

    Uses ``nest_asyncio`` to run the async MCP handshake from a sync
    context (works in Jupyter and regular scripts).  If the service is
    unreachable the error is logged and no tools are added.

    When *provider* is ``"mlx"`` (text-only LLM), tools that return images
    (e.g. ``browser_take_screenshot``) are excluded since the model cannot
    interpret them.
    """
    try:
        import asyncio
        import nest_asyncio
        nest_asyncio.apply()

        from tools.navigation.web.playwright_mcp import create_playwright_mcp_client
        from tools.anthropic.mcps import MCPHelper

        mcps = create_playwright_mcp_client()
        helper = MCPHelper(mcps)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(helper.connect_all())

        mcp_tools = helper.get_tools()
        if provider == "mlx" and Environment.get_deep_agent_mlx_model_type() == "llm":
            excluded = [t.name for t in mcp_tools if t.name in _IMAGE_ONLY_TOOLS]
            mcp_tools = [t for t in mcp_tools if t.name not in _IMAGE_ONLY_TOOLS]
            if excluded:
                logger.info("Playwright MCP: excluded image tools for text-only MLX: %s", excluded)

        # One guard per ToolSet (≈ one per DeepAgent instance).  Scope is
        # process-local and per-agent so concurrent agents never share
        # failure history, but all Playwright calls within a single
        # agent's lifetime feed the same deque.
        if ts._loop_guard is None:
            ts._loop_guard = ToolLoopGuard(
                recovery_hint=_PLAYWRIGHT_RECOVERY_HINT,
                recovery_temperature=Environment.get_loop_recovery_temperature(),
                recovery_temperature_turns=(
                    Environment.get_loop_recovery_temperature_turns()
                ),
            )
        guard = ts._loop_guard

        for t in mcp_tools:
            t.handle_tool_error = True
            _strip_none_args(t)
            _patch_ref_tool_description(t)
            wrap_with_loop_guard(t, guard)
        ts.deep_agent_tools.extend(mcp_tools)
        ts._mcp_helper = helper

        logger.info(
            "Playwright MCP: %d tools loaded — %s",
            len(mcp_tools),
            [t.name for t in mcp_tools],
        )
    except Exception as e:
        logger.warning(
            "Playwright MCP: could not connect — %s. "
            "Start the service with: ./scripts/start_playwright_mcp.sh",
            e,
        )
