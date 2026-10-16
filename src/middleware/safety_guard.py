"""Stop a run when Jev judges the user's message or the agent's action unsafe.

Two checks, both fail closed:

* **Input** (orchestrator only): the latest genuine user message, judged once
  at the start of a turn, before the model is called.
* **Action** (orchestrator and every subagent): the model's reply and tool
  calls, judged after the model answers and before any tool runs.

An unsafe verdict, or a judge that cannot answer, replaces the step with a
terminal ``AIMessage`` that has no tool calls. The agent loop then ends the
same way it does for the tool-call budget guard.

The judge is injected, so this module knows nothing about Jev, HTTP, or
settings. See :mod:`backend.safety_guard` for the real judge.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage

from middleware._react_core import content_to_text as _content_to_text
from middleware.tool_call_budget import (
    is_genuine_user_message,
    latest_genuine_user_text,
)

logger = logging.getLogger(__name__)

STAGE_INPUT = "input"
STAGE_ACTION = "action"

_STAGE_LABEL = {
    STAGE_INPUT: "user request",
    STAGE_ACTION: "proposed action",
}


@dataclass(frozen=True)
class SafetyCheck:
    """What the judge is asked to evaluate."""

    stage: str
    user_text: str
    assistant_text: str = ""
    tool_calls: tuple[dict[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SafetyVerdict:
    unsafe: bool
    p_unsafe: float


class SafetyCheckUnavailable(Exception):
    """The judge could not decide. ``str(exc)`` says why."""


SafetyJudge = Callable[[SafetyCheck], SafetyVerdict]


def _stop_text(stage: str, *, verdict: SafetyVerdict | None, error: str = "") -> str:
    label = _STAGE_LABEL.get(stage, stage)
    if verdict is not None:
        return (
            f"Stopped by the Jev safety guardrail: the {label} was judged unsafe "
            f"({verdict.p_unsafe:.0%}). Nothing further was run. If this is a "
            "mistake, turn the guardrail off in Settings → Jev."
        )
    return (
        f"Stopped by the Jev safety guardrail: the {label} could not be checked "
        f"({error}). Nothing further was run. Fix Jev in Settings → Jev, or "
        "turn the guardrail off."
    )


class SafetyGuardMiddleware(AgentMiddleware):
    """Judge the user's message and each proposed action; stop the run if unsafe."""

    def __init__(self, judge: SafetyJudge, *, check_input: bool) -> None:
        self._judge = judge
        self._check_input = check_input

    # ── AgentMiddleware hooks ───────────────────────────────────────────────

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        check = self._input_check(request)
        if check is not None:
            stop = self._stop(check, self._evaluate(check))
            if stop is not None:
                return stop
        response = handler(request)
        check = self._action_check(request, response)
        if check is None:
            return response
        return self._stop(check, self._evaluate(check)) or response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        # The judge blocks on HTTP, and on a cold start on the model load.
        check = self._input_check(request)
        if check is not None:
            outcome = await asyncio.to_thread(self._evaluate, check)
            stop = self._stop(check, outcome)
            if stop is not None:
                return stop
        response = await handler(request)
        check = self._action_check(request, response)
        if check is None:
            return response
        outcome = await asyncio.to_thread(self._evaluate, check)
        return self._stop(check, outcome) or response

    # ── What to judge ───────────────────────────────────────────────────────

    def _input_check(self, request: ModelRequest) -> Optional[SafetyCheck]:
        """The user's message, only on the first model call after they send it."""
        if not self._check_input:
            return None
        messages = list(getattr(request, "messages", None) or [])
        if not messages or not is_genuine_user_message(messages[-1]):
            return None
        text = _content_to_text(messages[-1].content).strip()
        if not text:
            return None
        return SafetyCheck(stage=STAGE_INPUT, user_text=text)

    def _action_check(
        self, request: ModelRequest, response: ModelResponse
    ) -> Optional[SafetyCheck]:
        """The model's reply and tool calls, or ``None`` when it said nothing."""
        replies = [m for m in (response.result or []) if isinstance(m, AIMessage)]
        text = "\n".join(
            part for part in (_content_to_text(m.content).strip() for m in replies) if part
        )
        calls = tuple(
            {"name": str(c.get("name") or ""), "args": c.get("args") or {}}
            for m in replies
            for c in (getattr(m, "tool_calls", None) or [])
        )
        if not text and not calls:
            return None
        messages = list(getattr(request, "messages", None) or [])
        return SafetyCheck(
            stage=STAGE_ACTION,
            user_text=latest_genuine_user_text(messages),
            assistant_text=text,
            tool_calls=calls,
        )

    # ── Judging and stopping ────────────────────────────────────────────────

    def _evaluate(self, check: SafetyCheck) -> SafetyVerdict | Exception:
        """The verdict, or the error that stopped the judge. Never raises."""
        try:
            return self._judge(check)
        except Exception as exc:  # fail closed: any failure is a stop
            return exc

    @staticmethod
    def _stop(
        check: SafetyCheck, outcome: SafetyVerdict | Exception
    ) -> Optional[ModelResponse]:
        """A terminal response when the run must stop, else ``None``."""
        if isinstance(outcome, SafetyVerdict):
            if not outcome.unsafe:
                return None
            logger.warning(
                "SafetyGuard: %s judged unsafe (p=%.2f) — stopping run.",
                check.stage, outcome.p_unsafe,
            )
            text = _stop_text(check.stage, verdict=outcome)
            meta = {"stage": check.stage, "reason": "unsafe", "p_unsafe": outcome.p_unsafe}
        else:
            error = str(outcome) or type(outcome).__name__
            logger.warning(
                "SafetyGuard: %s could not be checked (%s) — stopping run.",
                check.stage, error,
            )
            text = _stop_text(check.stage, verdict=None, error=error)
            meta = {"stage": check.stage, "reason": "unavailable", "error": error}
        return ModelResponse(
            result=[AIMessage(content=text, additional_kwargs={"safety_guard": meta})],
            structured_response=None,
        )
