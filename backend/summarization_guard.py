"""Monkey-patch for SummarizationMiddleware to prevent orphaned tool_result messages.

When the deepagents SummarizationMiddleware picks a cutoff boundary that falls
between an AIMessage (with tool_calls) and its corresponding ToolMessage(s), the
ToolMessages end up at the start of the kept portion with no preceding tool_use
block.  Anthropic's API rejects such requests with:

    "unexpected `tool_use_id` found in `tool_result` blocks"

This module patches two methods on the class at import time so ALL instances
(including those created internally by ``create_deep_agent``) benefit:

1. ``_apply_event_to_messages`` (staticmethod) — heals already-persisted bad
   states on every model call.  If the first kept message after the summary is
   an orphaned ToolMessage, it advances the effective cutoff until a non-orphan
   message is reached.

2. ``_determine_cutoff_index`` (instance method) — prevents new bad states from
   being stored.  After the upstream helper computes a cutoff, it advances past
   any leading ToolMessages so the first kept message is never an orphan.

Import this module once at application startup (``backend.server``) before any
session is created.  It is idempotent — repeated imports are no-ops.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_PATCHED = False


def _advance_past_orphaned_tool_messages(
    msgs: list[Any],
    start: int,
) -> int:
    """Return the smallest index >= *start* that is not an orphaned ToolMessage.

    A ToolMessage is "orphaned" at position *start* when its parent AIMessage
    was removed from the list (i.e. the AIMessage is before *start*).  Because
    a ToolMessage always immediately follows its parent AIMessage in LangChain
    message lists, any ToolMessage at the very start of the kept slice has no
    surviving parent and must be skipped.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    # Collect tool_call_ids that still have a parent AIMessage within msgs[start:]
    valid_tc_ids: set[str] = set()
    for m in msgs[start:]:
        if isinstance(m, AIMessage):
            for tc in (getattr(m, "tool_calls", None) or []):
                tid = tc.get("id") or ""
                if tid:
                    valid_tc_ids.add(tid)

    idx = start
    while idx < len(msgs) and isinstance(msgs[idx], ToolMessage):
        tc_id = getattr(msgs[idx], "tool_call_id", None) or ""
        if tc_id not in valid_tc_ids:
            idx += 1
        else:
            break

    if idx > start:
        logger.warning(
            "summarization_guard: advanced cutoff by %d to skip orphaned "
            "ToolMessage(s) at boundary (tool_call_ids: %s)",
            idx - start,
            [getattr(msgs[i], "tool_call_id", None) for i in range(start, idx)],
        )

    return idx


def _install_subagent_state_guard() -> None:
    """Stop subagents from writing ``_summarization_event`` into the parent graph.

    deepagents' ``task`` tool copies every non-excluded key from a finished
    subagent's state back into the parent graph via ``Command(update=...)``.
    The ``_summarization_event`` key is *not* in ``_EXCLUDED_STATE_KEYS``, and
    the parent's channel for that key has no merge reducer (it is annotated only
    with ``PrivateStateAttr``, a schema marker — not a reducer).  When several
    ``task`` subagents run in parallel and more than one auto-summarizes, each
    returns a ``_summarization_event`` update in the same LangGraph superstep,
    triggering::

        At key '_summarization_event': Can receive only one value per step.
        (INVALID_CONCURRENT_GRAPH_UPDATE)

    A subagent's internal summarization bookkeeping has no meaning for the
    parent, so we add the key to the exclusion set.  The set is read at call
    time by the ``task`` tool, so mutating it in place takes effect for all
    subsequent invocations.
    """
    try:
        from deepagents.middleware import subagents as _subagents
    except ImportError:
        logger.debug("summarization_guard: deepagents.subagents unavailable, skipping")
        return

    excluded = getattr(_subagents, "_EXCLUDED_STATE_KEYS", None)
    if isinstance(excluded, set) and "_summarization_event" not in excluded:
        excluded.add("_summarization_event")
        logger.info(
            "summarization_guard: excluded _summarization_event from subagent "
            "state propagation to prevent INVALID_CONCURRENT_GRAPH_UPDATE"
        )


def _install_guard() -> None:
    """Patch SummarizationMiddleware in-place.  Safe to call multiple times."""
    global _PATCHED
    if _PATCHED:
        return

    # Guard parallel-subagent summarization writes regardless of whether the
    # SummarizationMiddleware import below succeeds.
    _install_subagent_state_guard()

    try:
        from deepagents.middleware.summarization import SummarizationMiddleware
    except ImportError:
        logger.debug("summarization_guard: deepagents not available, skipping patch")
        return

    # ── 1. Patch _apply_event_to_messages (heals stored bad states) ──────────
    # In Python 3, a staticmethod accessed from the class is a plain function.
    _orig_apply = SummarizationMiddleware._apply_event_to_messages

    @staticmethod  # type: ignore[misc]
    def _safe_apply_event_to_messages(
        messages: list[Any],
        event: Any,
    ) -> list[Any]:
        result = _orig_apply(messages, event)

        if event is not None and len(result) >= 2:
            # result[0] is the summary message; result[1:] are the kept messages.
            # Advance past any orphaned ToolMessages at the start of the kept slice.
            kept = result[1:]
            safe_start = _advance_past_orphaned_tool_messages(kept, 0)
            if safe_start > 0:
                result = [result[0]] + kept[safe_start:]

        return result

    SummarizationMiddleware._apply_event_to_messages = _safe_apply_event_to_messages  # type: ignore[method-assign]

    # ── 2. Patch _determine_cutoff_index (prevents new bad states) ───────────
    _orig_cutoff = SummarizationMiddleware._determine_cutoff_index

    def _safe_determine_cutoff_index(self: Any, messages: list[Any]) -> int:
        cutoff = _orig_cutoff(self, messages)
        safe = _advance_past_orphaned_tool_messages(messages, cutoff)
        return safe

    SummarizationMiddleware._determine_cutoff_index = _safe_determine_cutoff_index  # type: ignore[method-assign]

    _PATCHED = True
    logger.info("summarization_guard: SummarizationMiddleware patched successfully")


# Apply the guard immediately on import.
_install_guard()
