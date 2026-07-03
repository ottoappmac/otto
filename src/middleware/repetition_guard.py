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
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

_RECOVERY_MSG = (
    "[system: your previous response was discarded because it collapsed into "
    "repeating the same text over and over without making progress. Do NOT "
    "repeat that text. Take a different, smaller step: re-read the most recent "
    "observation and either act differently or, if you cannot make progress, "
    "summarise what you have so far and finish.]"
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
    ) -> None:
        # Defaults are intentionally conservative; explicit values are mainly
        # for tests.  Kept as plain attributes (no Environment dependency) so
        # the guard can be added unconditionally and stays cheap.
        self._min_chars = 400 if min_chars is None else min_chars
        self._min_repeats = 5 if min_repeats is None else min_repeats
        self._dominant_ratio = 0.5 if dominant_ratio is None else dominant_ratio
        self._distinct_max = 0.25 if distinct_max is None else distinct_max

    # ── AgentMiddleware hooks ───────────────────────────────────────────────

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return self._sanitize(handler(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return self._sanitize(await handler(request))

    # ── Private helpers ─────────────────────────────────────────────────────

    def _sanitize(self, response: ModelResponse) -> ModelResponse:
        result = list(getattr(response, "result", None) or [])
        if not result:
            return response

        last = result[-1]
        if not isinstance(last, AIMessage):
            return response
        # A model that emitted a usable tool call is making progress; only the
        # final-answer (no tool calls) case is in scope here.
        if getattr(last, "tool_calls", None):
            return response

        text = self._content_to_text(last.content)
        if not is_degenerate_repetition(
            text,
            min_chars=self._min_chars,
            min_repeats=self._min_repeats,
            dominant_ratio=self._dominant_ratio,
            distinct_max=self._distinct_max,
        ):
            return response

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
