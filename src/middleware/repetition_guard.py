"""Provider-agnostic guard against degenerate intra-message repetition.

The :class:`~middleware.repeated_thought_guard.RepeatedThoughtGuardMiddleware`
catches a model that re-emits the *same thought + action across turns*.  This
guard catches the orthogonal failure mode: a single model generation that
collapses into repeating the *same sentence over and over inside one message*
until it hits the generation-token cap (observed in the wild: ~8192 tokens of
"The charts were generated in the virtual filesystem...").

The in-process MLX path has two partial defences (``repetition_penalty`` and a
truncation-recovery branch in :mod:`chat_models.mlx.chat_mlx_text`), but the
oMLX / exo / OpenAI-compatible clients have neither, so the degenerate blob is
streamed straight through and persisted verbatim as the final ``agent`` message.

This middleware runs at ``wrap_model_call`` so it sees the model's *output*.  If
the produced ``AIMessage`` has **no tool calls** and its text is a degenerate
repetition, the content is replaced with a short recovery signal (mirroring the
MLX ``_TRUNCATION_RECOVERY_MSG``).  Because the replacement carries no tool
calls, the ReAct router ends the turn cleanly — the user never sees the blob and
the model can take a smaller next step on resume.

Before giving up on the turn the guard asks the model again, up to ``retries``
times, with the recovery note added as a message (on a throwaway copy of the
request, so nothing extra is persisted). Ending the turn is only safe when
someone will resume it: an unattended agent (a benchmark run, a scheduled task)
has nobody to say "go on", and four OSWorld runs died mid-task with 16 to 45
steps left after one degenerate reply.

We deliberately only act when there are **no tool calls**: a model that emitted
a usable tool call is making progress, and an oversized tool *argument* is a
separate concern already handled on the MLX path.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)

_RECOVERY_MSG = (
    "[system: your previous response was discarded because it collapsed into "
    "repeating the same text over and over without making progress. Do NOT "
    "repeat that text. Take a different, smaller step: re-read the most recent "
    "observation and either act differently or, if you cannot make progress, "
    "summarise what you have so far and finish.]"
)

# Sent to the model, after a degenerate reply was dropped, when the guard asks it again.
_RETRY_MSG = (
    "[system: your last reply was discarded because it repeated the same text "
    "over and over. Do not repeat it. Re-read the most recent tool result and "
    "take one small, concrete next step: call a tool, or give a short final "
    "answer if the work is done.]"
)

# Segments shorter than this (after normalisation) are ignored so trivial
# fragments ("ok", "-", numbers) never drive detection.
_MIN_SEGMENT_CHARS = 10


def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def is_degenerate_repetition(
    text: str,
    *,
    min_chars: int = 400,
    min_repeats: int = 5,
    dominant_ratio: float = 0.5,
    distinct_max: float = 0.25,
) -> bool:
    """Heuristic: does *text* look like a collapsed repetition loop?

    Two complementary signals, tuned to fire on genuine loops while leaving
    normal (even repetitive-but-valid) prose alone:

    * **Dominant segment** — one normalised sentence/line accounts for at least
      ``dominant_ratio`` of all segments and repeats ``>= min_repeats`` times
      (catches a single-sentence loop).
    * **Low diversity** — the set of distinct normalised segments is tiny
      relative to their count (``<= distinct_max``) with ``>= min_repeats``
      segments (catches a short multi-sentence *cycle* repeating forever).

    Short outputs (``< min_chars``) never trip, to avoid false positives on
    legitimately concise answers.
    """
    if not text or len(text) < min_chars:
        return False

    segments = [s for s in re.split(r"[\n.!?]+", text) if s.strip()]
    norm = [_normalize(s) for s in segments]
    norm = [s for s in norm if len(s) >= _MIN_SEGMENT_CHARS]
    if len(norm) < min_repeats:
        return False

    counts = Counter(norm)
    _, top_n = counts.most_common(1)[0]
    distinct_ratio = len(counts) / len(norm)

    if top_n < min_repeats:
        return False
    if top_n / len(norm) >= dominant_ratio:
        return True
    if distinct_ratio <= distinct_max:
        return True
    return False


class RepetitionGuardMiddleware(AgentMiddleware):
    """Replace a degenerate, tool-call-free repetition with a recovery message.

    Provider-agnostic: works for MLX, oMLX, exo, OpenAI, and Anthropic because
    it inspects the produced ``AIMessage`` rather than any provider internals.
    """

    def __init__(
        self,
        *,
        min_chars: int | None = None,
        min_repeats: int | None = None,
        dominant_ratio: float | None = None,
        distinct_max: float | None = None,
        retries: int | None = None,
    ) -> None:
        # Defaults are intentionally conservative; explicit values are mainly
        # for tests.  Kept as plain attributes (no Environment dependency) so
        # the guard can be added unconditionally and stays cheap.
        self._min_chars = 400 if min_chars is None else min_chars
        self._min_repeats = 5 if min_repeats is None else min_repeats
        self._dominant_ratio = 0.5 if dominant_ratio is None else dominant_ratio
        self._distinct_max = 0.25 if distinct_max is None else distinct_max
        # Times the model is asked again after a degenerate reply before the turn ends with the recovery note.
        self._retries = 1 if retries is None else max(0, int(retries))

    # ── AgentMiddleware hooks ───────────────────────────────────────────────

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        response = handler(request)
        for _ in range(self._retries):
            retry = self._retry_request(request, response)
            if retry is None:
                break
            response = handler(retry)
        return self._sanitize(response)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        response = await handler(request)
        for _ in range(self._retries):
            retry = self._retry_request(request, response)
            if retry is None:
                break
            response = await handler(retry)
        return self._sanitize(response)

    # ── Private helpers ─────────────────────────────────────────────────────

    def _degenerate(self, response: ModelResponse) -> AIMessage | None:
        """The response's final message when it is a degenerate, tool-call-free repetition, else None."""
        result = list(getattr(response, "result", None) or [])
        if not result:
            return None
        last = result[-1]
        if not isinstance(last, AIMessage):
            return None
        # A model that emitted a usable tool call is making progress; only the
        # final-answer (no tool calls) case is in scope here.
        if getattr(last, "tool_calls", None):
            return None
        text = self._content_to_text(last.content)
        if not is_degenerate_repetition(
            text,
            min_chars=self._min_chars,
            min_repeats=self._min_repeats,
            dominant_ratio=self._dominant_ratio,
            distinct_max=self._distinct_max,
        ):
            return None
        return last

    def _retry_request(self, request: ModelRequest, response: ModelResponse) -> ModelRequest | None:
        """The request to ask the model again with, when ``response`` is degenerate; None otherwise (or when the
        request cannot be copied)."""
        if self._degenerate(response) is None or not hasattr(request, "override"):
            return None
        logger.warning("RepetitionGuard: degenerate reply dropped; asking the model again.")
        try:
            return request.override(messages=[*list(request.messages), HumanMessage(content=_RETRY_MSG)])
        except Exception:  # pragma: no cover — defensive: fall back to ending the turn
            return None

    def _sanitize(self, response: ModelResponse) -> ModelResponse:
        last = self._degenerate(response)
        if last is None:
            return response
        result = list(response.result)
        text = self._content_to_text(last.content)

        logger.warning(
            "RepetitionGuard: discarded a degenerate repetition (%d chars, no "
            "tool calls) and substituted a recovery message.",
            len(text),
        )
        recovery = AIMessage(
            content=_RECOVERY_MSG,
            additional_kwargs=dict(getattr(last, "additional_kwargs", {}) or {}),
        )
        return ModelResponse(
            result=[*result[:-1], recovery],
            structured_response=getattr(response, "structured_response", None),
        )

    @staticmethod
    def _content_to_text(content: object) -> str:
        if isinstance(content, str):
            return content
        try:
            from middleware._react_core import content_to_text

            return content_to_text(content)
        except Exception:  # pragma: no cover — defensive fallback
            return content if isinstance(content, str) else str(content)
