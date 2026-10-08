"""Decide whether a turn runs Quick or Deep.

A mode the user picked (``quick`` or ``deep``) is used as-is. ``auto`` asks
the session's own model, with tools off, to answer ``QUICK`` or ``DEEP``
before the real run starts. The limits for that choice are applied by
middleware, so they are in place before the first tool-calling turn.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

RUN_DEPTHS = ("quick", "deep")
REQUESTED_DEPTHS = ("auto", "quick", "deep")

FALLBACK_REASON = (
    "Couldn't classify this turn, so it's running Quick. "
    "Switch to Deep to go further."
)
UNATTENDED_REASON = "Scheduled and trigger runs always use Deep."

_JUDGE_SYSTEM = (
    "You decide how much work one chat turn needs. "
    "Reply with exactly two lines and nothing else:\n"
    "QUICK or DEEP\n"
    "one sentence saying why\n"
    "QUICK is a short answer, a lookup, a conversion, or a single small action. "
    "DEEP is research, comparison, planning, a multi-part request, creating or "
    "editing files, or anything that needs several steps. "
    "A follow-up that continues an earlier DEEP task is DEEP."
)

_MODE_RE = re.compile(r"\b(QUICK|DEEP)\b", re.IGNORECASE)


def depth_notice(mode: str, source: str, reason: str) -> str:
    """One line for the chat transcript and the run timeline.

    A mode the user pinned reads ``Running Quick`` or ``Running Deep``.
    Auto names the choice and keeps the judge's sentence.
    """
    label = "Quick" if mode == "quick" else "Deep"
    lead = f"Auto chose {label}" if source == "agent" else f"Running {label}"
    reason = (reason or "").strip()
    if reason:
        return f"{lead}. {reason}"
    return f"{lead}."


def parse_depth_reply(text: str) -> tuple[str, str] | None:
    """Parse ``QUICK`` or ``DEEP`` plus a one-line reason.

    Returns ``None`` when the reply does not name a mode.
    """
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return None
    match = _MODE_RE.search(lines[0])
    if match is None and len(lines) > 1:
        match = _MODE_RE.search(lines[1])
        reason_lines = lines[2:]
    else:
        reason_lines = lines[1:]
    if match is None:
        return None
    reason = " ".join(reason_lines).strip()
    if not reason:
        reason = "Chosen by the model."
    return match.group(1).lower(), reason[:400]


def message_has_attachments(query: str) -> bool:
    return "[Uploaded files:" in (query or "") or "[Context folders:" in (query or "")


def prior_turn(messages: list[Any]) -> tuple[str, str]:
    """Last genuine user text and the depth chosen for that turn."""
    from langchain_core.messages import HumanMessage

    from middleware._react_core import content_to_text

    nudges = _nudge_texts()
    last_user = ""
    last_depth = ""
    for message in messages:
        if not isinstance(message, HumanMessage):
            continue
        text = content_to_text(message.content).strip()
        if not text or text in nudges or text.startswith("Date check failed."):
            continue
        last_user = text
        depth = (getattr(message, "additional_kwargs", None) or {}).get("run_depth")
        if depth in RUN_DEPTHS:
            last_depth = depth
    return last_user[:800], last_depth


def _nudge_texts() -> set[str]:
    texts: set[str] = set()
    try:
        from middleware.run_depth import QUICK_SOFT_NUDGE
        from middleware.tool_call_budget import _SOFT_NUDGE_TEXT

        texts.add(_SOFT_NUDGE_TEXT)
        texts.add(QUICK_SOFT_NUDGE)
    except Exception:  # pragma: no cover — defensive
        pass
    return texts


async def judge_run_depth(
    llm: Any,
    *,
    query: str,
    has_attachments: bool,
    prior_user: str,
    prior_depth: str,
) -> tuple[str, str, Any]:
    """Ask *llm* to pick a mode. Returns ``(mode, reason, last_response)``.

    Retries once when the reply is not ``QUICK`` or ``DEEP``. On a second
    failure, or when the call errors, falls back to Quick.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    user = f"New message:\n{query}"
    if has_attachments:
        user += "\n\nThe user attached files or folders."
    if prior_user:
        user += f"\n\nPrevious user message:\n{prior_user}"
        if prior_depth:
            user += f"\nPrevious depth: {prior_depth.upper()}"
    messages: list[Any] = [
        SystemMessage(content=_JUDGE_SYSTEM),
        HumanMessage(content=user),
    ]
    last_response: Any = None
    for attempt in range(2):
        try:
            last_response = await _invoke(llm, messages)
        except Exception:
            logger.warning("Run-depth judge failed", exc_info=True)
            return "quick", FALLBACK_REASON, last_response
        parsed = parse_depth_reply(_response_text(last_response))
        if parsed is not None:
            return parsed[0], parsed[1], last_response
        if attempt == 0:
            messages = [
                *messages,
                HumanMessage(content="Reply with QUICK or DEEP on the first line, then one sentence."),
            ]
    return "quick", FALLBACK_REASON, last_response


async def resolve_run_depth(
    *,
    requested: str | None,
    query: str,
    llm: Any | None,
    unattended: bool,
    default_mode: str,
    has_attachments: bool,
    prior_user: str,
    prior_depth: str,
) -> tuple[str, str, str, Any]:
    """Return ``(mode, source, reason, judge_response)``.

    ``source`` is ``user`` when the mode was chosen without a model call,
    and ``agent`` when the session model judged an Auto turn. The judge
    response is ``None`` when the model was not called.
    """
    if unattended:
        return "deep", "user", UNATTENDED_REASON, None

    req = (requested or "").strip().lower()
    if req in RUN_DEPTHS:
        return req, "user", "", None

    fallback = (default_mode or "auto").strip().lower()
    if req not in ("", "auto"):
        req = ""
    if req == "" and fallback in RUN_DEPTHS:
        return fallback, "user", "", None

    if llm is None:
        logger.warning("Run-depth judge skipped: session has no model")
        return "quick", "agent", FALLBACK_REASON, None

    mode, reason, response = await judge_run_depth(
        llm,
        query=query,
        has_attachments=has_attachments,
        prior_user=prior_user,
        prior_depth=prior_depth,
    )
    return mode, "agent", reason, response


def accumulate_usage(session: Any, response: Any) -> None:
    """Fold the judge call's token counts into the session totals."""
    if response is None or session is None:
        return
    usage = getattr(response, "usage_metadata", None) or {}
    if not usage:
        meta = getattr(response, "response_metadata", None) or {}
        usage = meta.get("token_usage") or meta.get("usage") or {}
    try:
        session.input_tokens += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        session.output_tokens += int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    except (TypeError, ValueError, AttributeError):
        return


async def _invoke(llm: Any, messages: list[Any]) -> Any:
    model = llm
    bind = getattr(llm, "bind", None)
    if callable(bind):
        try:
            model = bind(max_tokens=80)
        except Exception:
            model = llm
    return await model.ainvoke(messages)


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    try:
        from middleware._react_core import content_to_text

        return content_to_text(content)
    except Exception:  # pragma: no cover — defensive
        return str(content)
