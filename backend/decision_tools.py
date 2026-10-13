"""The ``jev`` tool, attached to every agent when Jev is enabled.

The tool is advisory. It asks a Jev-compatible model which of a few
options fits, or whether a yes/no is true, and returns a confidence
the agent can act on. It never approves a side effect.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from backend.ask_user_tools import _coerce_options

logger = logging.getLogger(__name__)

_counts: dict[str, int] = {}
_counts_lock = threading.Lock()


class _DecideInput(BaseModel):
    situation: str = Field(
        description=(
            "The evidence for this decision, in your own words. Do not ask "
            "the tool to look anything up; it only classifies what you pass."
        ),
    )
    question: str = Field(description="The decision, as one question.")
    options: list[str] | dict[str, str] | None = Field(
        default=None,
        description=(
            "Choices. A list of labels, or an object of label to description. "
            "Omit for a yes/no question. At most 12."
        ),
    )
    scale: list[str] | None = Field(
        default=None,
        description="Ordered levels, lowest to highest, for a score. Omit unless you want a rating.",
    )

    @field_validator("options", "scale", mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return value
        return _coerce_options(value)


def reset_decision_turn(session_id: str) -> None:
    """Zero the per-turn call count. Called at the start of each user turn."""
    with _counts_lock:
        _counts[session_id] = 0


def _consume(session_id: str, limit: int) -> str | None:
    with _counts_lock:
        count = _counts.get(session_id, 0) + 1
        _counts[session_id] = count
    if count > max(1, limit):
        return (
            f"jev call cap reached ({limit} this turn). "
            "Use what you already have, or call ask_user if a person is available."
        )
    return None


def _unattended(session_id: str) -> bool:
    if not session_id:
        return False
    try:
        from backend.state import session_mgr
        session = session_mgr.get_session(session_id)
    except Exception:
        return False
    if session is None:
        return False
    return (getattr(session, "trigger_source", None) or "") in {"schedule", "trigger"}


def _failure(message: str) -> str:
    return (
        f"{message} Decide yourself from the evidence you have. "
        "Tell the user to open Settings → Jev if they want this tool working."
    )


def build_decision_tools(session_id: str | None = None) -> list:
    """One ``jev`` tool bound to ``session_id`` for the per-turn cap."""
    sid = session_id or ""

    @tool(args_schema=_DecideInput)
    def jev(
        situation: str,
        question: str,
        options: list[str] | dict[str, str] | None = None,
        scale: list[str] | None = None,
    ) -> str:
        """Classify a decision you are unsure about.

        Use this when you are torn between concrete options, or unsure
        whether to proceed. Put the evidence in ``situation``. Omit
        ``options`` for yes/no. Pass ``scale`` for an ordered rating.

        Do not use it for facts, for a trivial choice, or to avoid reading
        a file. The answer is advisory: a confident result is not permission
        for a destructive, external, or irreversible action. Approvals still
        apply. If the verdict is UNSURE, call ``ask_user``.
        """
        from backend.config import AppConfig
        from backend.decision_client import format_answer, source_label
        from backend.decision_runtime import effective_base_url
        from backend.decision_service import DecisionUnavailable, ask_once

        try:
            cfg = AppConfig.load()
        except Exception as exc:
            return _failure(f"Could not read settings ({exc}).")
        decision = cfg.decision
        if not decision.enabled:
            return _failure("Jev is turned off.")

        capped = _consume(sid, int(decision.max_calls_per_turn))
        if capped:
            return capped

        try:
            result = ask_once(cfg, situation, question, options, scale)
        except DecisionUnavailable as exc:
            return _failure(str(exc))
        return format_answer(
            result,
            confident_threshold=float(decision.confident_threshold),
            unsure_threshold=float(decision.unsure_threshold),
            unattended=_unattended(sid),
            source=source_label(effective_base_url(decision)),
        )

    return [jev]
