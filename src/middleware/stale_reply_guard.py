"""Stop a chat from re-sending its previous answer as if it were new work.

In a long session a model can answer a fresh "it's still broken" message by
repeating its last summary ("I've successfully fixed ...") word for word,
without reading a file or running a command. The user sees a confident fix
report, nothing changed, and the only way out is a new session.

The other loop guards do not see this. ``RepeatedThoughtGuardMiddleware``
counts repeats inside one run, and ``RepetitionGuardMiddleware`` looks for a
sentence repeated inside one message. Here the repeat is *across* turns: the
final answer matches an earlier turn's final answer.

This middleware runs at ``wrap_model_call``. When the model produces a final
answer (no tool calls) that nearly matches one of the last few final answers
from earlier turns, it:

1. asks once more, with a transient nudge and a temperature bump, telling the
   model that its earlier answer did not solve the problem and to re-check the
   real state with tools;
2. if the retry is still an echo, replaces the reply with an honest message
   that says nothing was done, instead of letting a false "fixed" report stand.

The nudge is added to that one request only, never to saved state.
"""

from __future__ import annotations

import difflib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

logger = logging.getLogger(__name__)

# Short answers ("Done.", "Yes") legitimately repeat; only long ones count.
_MIN_CHARS = 200
# How many earlier final answers to compare against.
_LOOKBACK = 3
# Normalised text similarity at or above which a reply counts as an echo.
_SIMILARITY = 0.85

_NUDGE_TEXT = (
    "Your reply repeats your previous answer almost word for word. The user is "
    "telling you the problem is still there, so that earlier answer did not "
    "solve it. Do not restate it. Re-check the real state first: read the "
    "relevant files and re-run the failing command, quote what you actually "
    "see, then change something different from last time. Only say it is "
    "fixed after a check you ran this turn passes."
)

_TERMINAL_TEXT = (
    "I started to repeat my previous answer without checking anything, so I am "
    "not going to tell you the problem is fixed. Nothing was changed this turn. "
    "Paste the exact error you see now, or switch to Debug mode and I will "
    "reproduce it first."
)


def _text_of(msg: BaseMessage) -> str:
    try:
        from middleware._react_core import content_to_text

        return content_to_text(msg.content)
    except Exception:  # pragma: no cover — defensive
        content = msg.content
        return content if isinstance(content, str) else str(content)


def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def _is_final(msg: BaseMessage) -> bool:
    return isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None)


def previous_final_answers(messages: list[BaseMessage]) -> list[str]:
    """Normalised final answers from turns *before* the latest user message.

    Newest first, at most ``_LOOKBACK``. Answers shorter than ``_MIN_CHARS``
    are skipped.
    """
    last_human = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human = i
            break
    out: list[str] = []
    for msg in reversed(messages[: max(last_human, 0)]):
        if not _is_final(msg):
            continue
        text = _normalize(_text_of(msg))
        if len(text) < _MIN_CHARS:
            continue
        out.append(text)
        if len(out) >= _LOOKBACK:
            break
    return out


def is_echo(reply: str, previous: list[str], *, threshold: float = _SIMILARITY) -> bool:
    """Is *reply* a near copy of any of the *previous* final answers?"""
    text = _normalize(reply)
    if len(text) < _MIN_CHARS:
        return False
    for old in previous:
        matcher = difflib.SequenceMatcher(None, text, old, autojunk=False)
        if matcher.real_quick_ratio() < threshold or matcher.quick_ratio() < threshold:
            continue
        if matcher.ratio() >= threshold:
            return True
    return False


def _final_reply_text(response: ModelResponse) -> str | None:
    """Text of the model's final answer, or ``None`` when it called tools."""
    result = list(getattr(response, "result", None) or [])
    if not result:
        return None
    last = result[-1]
    if not _is_final(last):
        return None
    return _text_of(last)


class StaleReplyGuardMiddleware(AgentMiddleware):
    """Retry, then refuse, a final answer that echoes an earlier turn's."""

    def __init__(self, *, similarity: float = _SIMILARITY) -> None:
        super().__init__()
        self._similarity = similarity

    # ── AgentMiddleware hooks ───────────────────────────────────────────────

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        response = handler(request)
        previous = self._previous(request)
        if not previous or not self._echoes(response, previous):
            return response
        self._log("retrying")
        self._bump_temperature()
        retry = handler(self._with_nudge(request))
        if not self._echoes(retry, previous):
            return retry
        self._log("refusing")
        return self._terminal_response()

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        response = await handler(request)
        previous = self._previous(request)
        if not previous or not self._echoes(response, previous):
            return response
        self._log("retrying")
        self._bump_temperature()
        retry = await handler(self._with_nudge(request))
        if not self._echoes(retry, previous):
            return retry
        self._log("refusing")
        return self._terminal_response()

    # ── Private helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _previous(request: ModelRequest) -> list[str]:
        return previous_final_answers(list(getattr(request, "messages", None) or []))

    def _echoes(self, response: ModelResponse, previous: list[str]) -> bool:
        text = _final_reply_text(response)
        return text is not None and is_echo(text, previous, threshold=self._similarity)

    @staticmethod
    def _log(action: str) -> None:
        logger.warning(
            "StaleReplyGuard: final answer repeats an earlier turn's — %s.", action
        )

    @staticmethod
    def _with_nudge(request: ModelRequest) -> ModelRequest:
        messages: list[Any] = list(getattr(request, "messages", None) or [])
        messages.append(HumanMessage(content=_NUDGE_TEXT))
        try:
            return request.override(messages=messages)
        except Exception:  # pragma: no cover — defensive
            return request

    @staticmethod
    def _bump_temperature() -> None:
        """Ask the next local-MLX generation to sample hotter (no-op elsewhere)."""
        try:
            from chat_models.mlx._shared import request_temperature_bump
            from utilities.environment import Environment

            temperature = Environment.get_loop_recovery_temperature()
            if temperature > 0.0:
                request_temperature_bump(
                    temperature, Environment.get_loop_recovery_temperature_turns()
                )
        except Exception:  # pragma: no cover — diagnostic path
            logger.debug("StaleReplyGuard: temperature bump failed", exc_info=True)

    @staticmethod
    def _terminal_response() -> ModelResponse:
        return ModelResponse(
            result=[AIMessage(content=_TERMINAL_TEXT)],
            structured_response=None,
        )
