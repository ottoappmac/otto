"""Middleware to compact stale Playwright MCP tool results in the LLM context.

Playwright MCP tool results embed massive accessibility-tree YAML snapshots
(10K+ tokens each) that become stale after every browser action.  This
middleware aggressively compacts older tool results down to just the
``### Ran Playwright code`` block needed for script generation, while keeping
the most recent snapshot intact so the model can see the current page.

Two strategies applied in order:

1. **Aggressive compaction** — For every ``ToolMessage`` *except* the most
   recent one containing a snapshot:

   - ``browser_snapshot`` / ``browser_take_screenshot`` results are replaced
     entirely with a short placeholder (they contain no code blocks).
   - All other tool results (``browser_click``, ``browser_navigate``,
     ``browser_fill_form``, etc.) are compacted to **only** their
     ``### Ran Playwright code`` block.  Everything else (``### Snapshot``
     YAML, ``### Open tabs``, ``### Page`` metadata) is dropped.

2. **Max-messages safety net** — Only the last *max_messages* non-system
   messages are sent to the LLM, capping context growth in extremely long
   sessions.  The default (40) is high enough to rarely trigger.

Usage::

    from middleware.playwright_pruning import PlaywrightSnapshotPruningMiddleware

    agent = create_deep_agent(
        model=llm,
        tools=[...],
        middleware=[PlaywrightSnapshotPruningMiddleware()],
    )
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)


_SNAPSHOT_TOOLS = frozenset({
    "browser_snapshot",
    "browser_take_screenshot",
})

_SUPERSEDED_PLACEHOLDER = "[snapshot superseded — see latest snapshot below]"

_PW_CODE_RE = re.compile(
    r"(### Ran Playwright code\s*```(?:js|javascript)?\s*.*?```)",
    re.DOTALL,
)

_DEFAULT_MAX_MESSAGES = 40
_MAX_RETRIES_SAME_REF = 2


class PlaywrightSnapshotPruningMiddleware(AgentMiddleware):
    """Compact stale Playwright tool results to just their code blocks.

    Only the most recent snapshot-bearing tool result is kept intact (the
    model needs it to see element refs for the next action).  All older
    results are compacted to their ``### Ran Playwright code`` block, or
    replaced with a short placeholder if they have no code block.

    The full message history is preserved in the graph state — only the
    view sent to the LLM is modified.

    Args:
        max_messages: Safety-net cap on total messages sent to the LLM.
            Only triggers for very long sessions.  Sourced from
            ``Environment.get_playwright_max_messages()`` via ``DeepAgent``.
    """

    name: str = "playwright_snapshot_pruning"

    def __init__(self, max_messages: int = _DEFAULT_MAX_MESSAGES) -> None:
        self.max_messages = max_messages

    async def awrap_model_call(self, request, handler):
        original = request.messages
        pruned = _prune(original, self.max_messages)
        orig_chars = sum(len(_content_text(m.content)) for m in original if isinstance(m, ToolMessage))
        pruned_chars = sum(len(_content_text(m.content)) for m in pruned if isinstance(m, ToolMessage))
        logger.info(
            "PW pruning: %d msgs, tool_content %d→%d chars (%.0f%% reduction)",
            len(pruned), orig_chars, pruned_chars,
            (1 - pruned_chars / orig_chars) * 100 if orig_chars else 0,
        )
        return await handler(request.override(messages=pruned))

    def wrap_model_call(self, request, handler):
        original = request.messages
        pruned = _prune(original, self.max_messages)
        orig_chars = sum(len(_content_text(m.content)) for m in original if isinstance(m, ToolMessage))
        pruned_chars = sum(len(_content_text(m.content)) for m in pruned if isinstance(m, ToolMessage))
        logger.info(
            "PW pruning: %d msgs, tool_content %d→%d chars (%.0f%% reduction)",
            len(pruned), orig_chars, pruned_chars,
            (1 - pruned_chars / orig_chars) * 100 if orig_chars else 0,
        )
        return handler(request.override(messages=pruned))


def _content_text(content: Any) -> str:
    """Extract plain text from a ToolMessage content (str or list-of-blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


def _has_snapshot(content: Any) -> bool:
    """Return True if the message content contains a ``### Snapshot`` section."""
    return "### Snapshot" in _content_text(content)


def _extract_pw_code(content: Any) -> str | None:
    """Extract the ``### Ran Playwright code`` block from content.

    Returns the matched section (header + fenced code) or ``None``.
    """
    text = _content_text(content)
    match = _PW_CODE_RE.search(text)
    return match.group(1) if match else None


def _compact_tool_result(msg: ToolMessage) -> ToolMessage:
    """Replace a tool result with only its Playwright code block.

    If no code block is found, returns a short placeholder.
    """
    tool_name = getattr(msg, "name", "")

    if tool_name in _SNAPSHOT_TOOLS:
        return ToolMessage(
            content=_SUPERSEDED_PLACEHOLDER,
            name=msg.name,
            tool_call_id=msg.tool_call_id,
        )

    pw_code = _extract_pw_code(msg.content)
    compacted = pw_code if pw_code else "[action completed]"
    return ToolMessage(
        content=compacted,
        name=msg.name,
        tool_call_id=msg.tool_call_id,
    )


def _is_compactable(msg: ToolMessage) -> bool:
    """Return True if the message should be compacted when it's not the latest snapshot.

    Any ToolMessage that contains extra context beyond just a code block is a
    candidate: snapshot-bearing messages, and any result with ``### Page``,
    ``### Open tabs``, ``### Events``, or ``### Result`` sections that are
    stale after the next action.
    """
    text = _content_text(msg.content)
    if msg.content == _SUPERSEDED_PLACEHOLDER:
        return True
    return bool(
        "### Snapshot" in text
        or "### Page" in text
        or "### Open tabs" in text
        or "### Events" in text
        or "### Result" in text
    )


def _prune_snapshots(messages: list) -> list:
    """Compact all but the newest snapshot-bearing ToolMessage.

    Walks messages in reverse to find the most recent tool result that
    contains a ``### Snapshot`` section and keeps it fully intact.  All
    older ``ToolMessage`` objects that carry stale context (snapshots, page
    metadata, tab lists, events, or result sections) are compacted to just
    their ``### Ran Playwright code`` block (or a placeholder).
    """
    latest_snapshot_idx: int | None = None
    copied = False

    for i in reversed(range(len(messages))):
        msg = messages[i]
        if not isinstance(msg, ToolMessage):
            continue
        if not _is_compactable(msg):
            continue

        if _has_snapshot(msg.content) and latest_snapshot_idx is None:
            latest_snapshot_idx = i
            continue

        if not copied:
            messages = list(messages)
            copied = True

        messages[i] = _compact_tool_result(msg)

    return messages


def _trim_to_window(messages: list, max_messages: int) -> list:
    """Keep only the last *max_messages* messages, preserving System/Human messages."""
    if len(messages) <= max_messages:
        return messages
    prefix = [m for m in messages if isinstance(m, (SystemMessage, HumanMessage))]
    remaining = [m for m in messages if not isinstance(m, (SystemMessage, HumanMessage))]
    budget = max_messages - len(prefix)
    if budget <= 0:
        # Degenerate: the System/Human prefix alone exceeds the window.
        # Fail open (no trim) rather than dropping every AI/Tool message,
        # which would orphan tool calls and break the conversation.
        return messages
    sliced = remaining[-budget:]

    # Never start with a ToolMessage — its AIMessage partner was cut off,
    # which would leave orphaned tool-result blocks that Bedrock rejects.
    # Advance forward until the slice starts on a clean AIMessage boundary.
    while sliced and isinstance(sliced[0], ToolMessage):
        sliced = sliced[1:]

    return prefix + sliced


def _detect_error_loop(messages: list) -> list:
    """If the model is retrying the same failing ref, nudge it to snapshot.

    Scans the tail of messages for consecutive ``(AIMessage, error ToolMessage)``
    pairs that target the same element ref.  After ``_MAX_RETRIES_SAME_REF``
    consecutive failures on the same ref, appends a ``HumanMessage`` telling the
    model to ``browser_snapshot()`` and pick a different element.
    """
    consecutive = 0
    target_ref: str | None = None

    i = len(messages) - 1
    while i >= 1:
        tool_msg = messages[i]
        ai_msg = messages[i - 1]

        if not (isinstance(tool_msg, ToolMessage)
                and getattr(tool_msg, "status", None) == "error"
                and isinstance(ai_msg, AIMessage)
                and ai_msg.tool_calls):
            break

        ref = (ai_msg.tool_calls[-1].get("args") or {}).get("ref")
        if ref is None:
            break
        if target_ref is None:
            target_ref = ref
        if ref != target_ref:
            break

        consecutive += 1
        i -= 2

    if consecutive >= _MAX_RETRIES_SAME_REF:
        messages = list(messages)
        messages.append(HumanMessage(
            content=(
                f"SYSTEM: You have failed {consecutive} times on ref {target_ref}. "
                "STOP retrying this ref. Call browser_snapshot() NOW to get fresh "
                "refs, then pick a DIFFERENT element (look for a `textbox`, not "
                "a `generic` or `link`)."
            )
        ))
    return messages


def _prune(messages: list, max_messages: int) -> list:
    """Apply aggressive compaction, error-loop detection, then safety-net trim."""
    messages = _prune_snapshots(messages)
    messages = _detect_error_loop(messages)
    messages = _trim_to_window(messages, max_messages)
    return messages
