"""Repeated-thought / repeated-action loop guard middleware.

The provider-agnostic :class:`tools.loop_guard.ToolLoopGuard` keys off
``(tool_name, canonical_args)`` *outcomes*, so it misses a loop whose tool
calls keep *succeeding* with a result that jitters every turn (e.g. a
``browser_click`` on the same element whose snapshot path carries a timestamp
and whose console error count keeps climbing).  The invariant signal in that
failure mode is the model re-emitting an *identical thought + identical action*
turn after turn.

This middleware detects that pattern by counting how many times a short
*cycle* of ``AIMessage``s (period ``1..max_period``, keyed on each message's
normalised thought text *and* tool-call signature) repeats at the tail of the
current invocation.  Period-1 is the classic "same thought + action every
turn"; higher periods catch alternating loops — e.g. ``scroll`` then ``read a
stale snapshot`` forever — that a consecutive-only count scores as 1 and never
trips.  Once the cycle has repeated enough times it:

* at :meth:`Environment.get_repeat_guard_nudge_at` consecutive repeats — bumps
  the local-MLX recovery temperature (a no-op for API providers) and injects a
  one-shot corrective nudge for that turn, encouraging the model to change
  approach;
* at :meth:`Environment.get_repeat_guard_abort_at` consecutive repeats — ends
  the run gracefully by short-circuiting the model call with a terminal
  ``AIMessage`` (no tool calls), so the agent loop exits with whatever partial
  result it has.  When the looping agent is a subagent, ending it lets the
  ``task`` tool resolve so the orchestrator can proceed with the other
  subagents' results.

**Why ``wrap_model_call`` and not ``after_model``**

``after_model`` runs *before* the tool node executes, so the message history at
that point ends with the just-produced ``AIMessage`` (which usually still has
pending ``tool_calls``).  Returning a nudge message from there would insert it
between a ``tool_use`` and its ``tool_result``, corrupting the sequence for
providers (Anthropic, OpenAI) that require those to be adjacent.
``wrap_model_call`` instead lets us (a) inject a *transient* nudge into the
request for a single turn without mutating persisted state, and (b)
short-circuit the turn that *would* produce the next repeat with a clean
terminal message — no dangling tool calls.

The guard is correctly scoped per graph invocation: each subagent run sees its
own ``request.messages``, so parallel subagents that share one session id are
never conflated, and ending one never aborts the others.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, Optional

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

logger = logging.getLogger(__name__)

# Separator between the thought-text and tool-call halves of a signature.  A
# NUL byte never appears in normalised text, so it cannot cause a collision.
_SIG_SEP = "\x00"

# How often to re-emit the nudge once past the nudge threshold, in repeat
# steps.  Nudging on every step from ``nudge_at`` up to ``abort_at`` would
# spam the context; spacing it out keeps the corrective signal present without
# flooding.
_RENUDGE_EVERY = 3

_NUDGE_TEXT = (
    "You have repeated the same reasoning and the same action several times "
    "without making progress. Stop repeating it. Re-read the latest "
    "observation, then either take a DIFFERENT action (a different tool, "
    "different arguments, or a different target) or, if you cannot make "
    "progress, report what you have found so far and finish."
)

_TERMINAL_TEXT = (
    "Stopping: I detected a repeated-thought loop — the same reasoning and "
    "action were produced too many times without progress. Returning the "
    "partial results gathered so far rather than continuing to loop."
)


class RepeatedThoughtGuard:
    """Pure helper: signatures and consecutive-repeat counting.

    Separated from the middleware so it can be unit-tested without the agent
    framework and reused by other call sites if needed.
    """

    @staticmethod
    def _canonicalize(arguments: Any) -> str:
        """Stable string form of tool-call args (reuses ToolLoopGuard's)."""
        from tools.loop_guard import ToolLoopGuard

        return ToolLoopGuard._canonicalize(arguments)

    @classmethod
    def signature(cls, msg: BaseMessage) -> Optional[str]:
        """Return a stable signature for *msg*, or ``None`` when empty.

        The signature combines the normalised thought text (whitespace
        collapsed, lower-cased — so trivial formatting differences don't
        defeat detection, while genuinely different reasoning still differs)
        with each tool call's ``name`` + canonical args.  An ``AIMessage``
        with neither text nor tool calls returns ``None`` so degenerate empty
        chunks never count towards a loop.
        """
        if not isinstance(msg, AIMessage):
            return None

        from middleware._react_core import content_to_text

        text = content_to_text(msg.content)
        norm_text = " ".join(text.split()).lower()

        tool_calls = getattr(msg, "tool_calls", None) or []
        tool_parts = sorted(
            f"{tc.get('name', '')}:{cls._canonicalize(tc.get('args', {}))}"
            for tc in tool_calls
        )
        tool_sig = ";".join(tool_parts)

        if not norm_text and not tool_sig:
            return None
        return norm_text + _SIG_SEP + tool_sig

    @classmethod
    def consecutive_repeats(
        cls, messages: list[BaseMessage]
    ) -> tuple[int, Optional[str]]:
        """Count trailing ``AIMessage``s sharing the newest signature.

        Walks the history backwards, considering only ``AIMessage``s (so
        interleaved ``ToolMessage``s and injected nudge ``HumanMessage``s are
        skipped, letting the count keep climbing across nudges).  Stops at the
        first ``AIMessage`` whose signature differs from the most recent one.

        Returns ``(count, signature)`` where ``count`` is the number of
        consecutive matching ``AIMessage``s (``0`` when there are none).
        """
        latest_sig: Optional[str] = None
        count = 0
        for msg in reversed(messages):
            if not isinstance(msg, AIMessage):
                continue
            sig = cls.signature(msg)
            if sig is None:
                continue
            if latest_sig is None:
                latest_sig = sig
                count = 1
            elif sig == latest_sig:
                count += 1
            else:
                break
        return count, latest_sig

    # Separator used to fold a multi-signature cycle into a single stable key.
    # Double NUL can't collide with a real signature (which never contains NUL).
    _CYCLE_SEP = "\x00\x00"

    @classmethod
    def cyclic_repeats(
        cls, messages: list[BaseMessage], max_period: int = 1
    ) -> tuple[int, Optional[str]]:
        """Count trailing repetitions of a short ``AIMessage`` *cycle*.

        Generalises :meth:`consecutive_repeats` (which only sees period-1
        streaks) to short cycles of length ``1..max_period``.  An agent stuck
        alternating between two distinct thought+action steps — e.g.
        ``scroll`` then ``read stale snapshot`` — forms a period-2 cycle that
        ``consecutive_repeats`` scores as ``1`` forever; here it is detected as
        the cycle repeating ``r`` times.

        Walks the history backwards over ``AIMessage``s only (skipping
        interleaved ``ToolMessage``/nudge ``HumanMessage``s and ``None``
        signatures, exactly like :meth:`consecutive_repeats`), then for each
        period ``p`` in ``1..max_period`` counts how many times the last block
        of ``p`` signatures repeats consecutively at the tail.  Returns the
        ``(count, key)`` for the period with the most repeats, tie-breaking to
        the *smallest* period so a pure period-1 loop returns the same count as
        :meth:`consecutive_repeats`.  ``key`` is a stable fold of the winning
        cycle's signatures (non-``None`` whenever ``count >= 1``).
        """
        # Trailing run of AIMessage signatures, newest LAST (chronological).
        sigs: list[str] = []
        for msg in reversed(messages):
            if not isinstance(msg, AIMessage):
                continue
            sig = cls.signature(msg)
            if sig is None:
                continue
            sigs.append(sig)
        sigs.reverse()

        if not sigs:
            return 0, None

        max_period = max(1, max_period)
        best_count = 0
        best_period = 0
        best_block: list[str] = []
        for period in range(1, max_period + 1):
            if len(sigs) < period:
                break
            block = sigs[-period:]
            reps = 0
            idx = len(sigs)
            while idx - period >= 0 and sigs[idx - period:idx] == block:
                reps += 1
                idx -= period
            # Prefer more repeats; on a tie keep the smaller period (found
            # first), so e.g. [A,A,A,A] reports period-1 count 4 rather than
            # period-2 count 2.
            if reps > best_count:
                best_count = reps
                best_period = period
                best_block = block

        if best_count < 1:
            return 0, None
        return best_count, cls._CYCLE_SEP.join(best_block)


class RepeatedThoughtGuardMiddleware(AgentMiddleware):
    """Nudge, then gracefully end, an agent stuck repeating the same thought.

    See the module docstring for the detection model and the rationale for
    using ``wrap_model_call`` instead of ``after_model``.
    """

    def __init__(
        self,
        *,
        nudge_at: int | None = None,
        abort_at: int | None = None,
        max_period: int | None = None,
        recovery_temperature: float | None = None,
        recovery_temperature_turns: int | None = None,
    ) -> None:
        """Construct the guard.

        All thresholds default to the corresponding ``Environment`` getters so
        the middleware can be added unconditionally; explicit values are mainly
        for tests.
        """
        from utilities.environment import Environment

        self._nudge_at = (
            nudge_at if nudge_at is not None
            else Environment.get_repeat_guard_nudge_at()
        )
        self._abort_at = (
            abort_at if abort_at is not None
            else Environment.get_repeat_guard_abort_at()
        )
        self._max_period = (
            max_period if max_period is not None
            else Environment.get_repeat_guard_max_period()
        )
        self._recovery_temperature = (
            recovery_temperature if recovery_temperature is not None
            else Environment.get_loop_recovery_temperature()
        )
        self._recovery_temperature_turns = (
            recovery_temperature_turns if recovery_temperature_turns is not None
            else Environment.get_loop_recovery_temperature_turns()
        )

    # ── AgentMiddleware hooks ───────────────────────────────────────────────

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        decision = self._decide(request)
        if decision == "abort":
            return self._terminal_response()
        if decision == "nudge":
            request = self._with_nudge(request)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        decision = self._decide(request)
        if decision == "abort":
            return self._terminal_response()
        if decision == "nudge":
            request = self._with_nudge(request)
        return await handler(request)

    # ── Private helpers ─────────────────────────────────────────────────────

    def _decide(self, request: ModelRequest) -> Optional[str]:
        """Return ``"abort"``, ``"nudge"`` or ``None`` for this turn.

        ``count`` is how many times the detected thought+action *cycle*
        (period ``1..max_period``) already repeats in the history; we are about
        to produce the next turn.  Aborting when ``count >= abort_at`` means
        exactly ``abort_at`` cycle repeats were produced before the run stops.
        For a period-1 loop this is identical to the consecutive-repeat count.
        """
        messages = list(getattr(request, "messages", None) or [])
        count, sig = RepeatedThoughtGuard.cyclic_repeats(
            messages, max_period=self._max_period
        )
        if sig is None or count < 1:
            return None

        if self._abort_at is not None and count >= self._abort_at:
            logger.warning(
                "RepeatedThoughtGuard: thought+action cycle repeated %d times "
                "(abort_at=%d) — ending run with partial result.",
                count, self._abort_at,
            )
            return "abort"

        if self._nudge_at is not None and count >= self._nudge_at:
            # Re-nudge only every few steps so we don't flood the context with
            # corrective messages on every turn between nudge_at and abort_at.
            if (count - self._nudge_at) % _RENUDGE_EVERY == 0:
                self._request_temperature_bump()
                logger.warning(
                    "RepeatedThoughtGuard: thought+action cycle repeated %d "
                    "times (nudge_at=%d) — nudging + temperature bump.",
                    count, self._nudge_at,
                )
                return "nudge"
            # Still in the looping band but between nudge points: bump temp
            # (cheap, helps local MLX escape greedy decoding) without injecting
            # another message.
            self._request_temperature_bump()
        return None

    def _with_nudge(self, request: ModelRequest) -> ModelRequest:
        """Return *request* with a transient corrective nudge appended.

        The nudge is appended after the trailing ``ToolMessage`` (the normal
        end of a ReAct turn), so ``tool_use``/``tool_result`` adjacency is
        preserved.  It is transient — added only to this request, never to
        persisted state — so it cannot corrupt the saved message sequence.
        """
        messages = list(getattr(request, "messages", None) or [])
        messages.append(HumanMessage(content=_NUDGE_TEXT))
        try:
            return request.override(messages=messages)
        except Exception:  # pragma: no cover — defensive
            return request

    def _terminal_response(self) -> ModelResponse:
        """A model response that ends the agent loop with a partial result.

        The terminal ``AIMessage`` carries no tool calls, so the ReAct router
        sends the graph to ``end`` instead of the tool node — no dangling
        ``tool_calls`` are left in the final state.
        """
        return ModelResponse(
            result=[AIMessage(content=_TERMINAL_TEXT)],
            structured_response=None,
        )

    def _request_temperature_bump(self) -> None:
        """Ask the next MLX generation(s) to sample hotter (no-op off MLX)."""
        if self._recovery_temperature <= 0.0:
            return
        try:
            from chat_models.mlx._shared import request_temperature_bump
        except Exception:
            return
        try:
            request_temperature_bump(
                self._recovery_temperature,
                self._recovery_temperature_turns,
            )
        except Exception as exc:  # pragma: no cover — diagnostic path
            logger.debug(
                "RepeatedThoughtGuard: temperature bump failed: %s", exc
            )
