"""Provider-agnostic tool-call loop guard.

Used by every MCP / tool loader to bound the worst-case "model emits
the same failing tool call forever" failure mode.  Detection is
provider-neutral — it just counts ``(tool_name, canonical_args)``
failures across a small recent-history window — so it works
unchanged for Anthropic, OpenAI, MLX, vLLM, or anything else
plugged into LangChain's tool layer.

The recovery side-effect is best-effort and includes a one-shot
``mlx_turbo`` KV-cache eviction that no-ops on every non-MLX path.
That eviction is what addresses the empirically-observed second
half of the failure mode: when KV-cache reuse is high, the model's
argmax can be pinned into re-emitting the same bad tokens; flushing
the cache forces a fresh prefill on the recovery turn.

See :mod:`backend.mcp_manager` for the wiring that attaches one
:class:`ToolLoopGuard` per :class:`backend.mcp_manager.MCPConnection`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import deque
from typing import Any, Callable, Deque, Optional, Tuple

from langchain_core.tools import BaseTool, ToolException

logger = logging.getLogger(__name__)

__all__ = [
    "ToolLoopDetected",
    "ToolLoopGuard",
    "wrap_with_loop_guard",
    "guard_all_tools",
    "build_default_guard",
    "OBSERVATION_TOOLS",
    "DEFAULT_GUARD_WINDOW",
    "DEFAULT_MAX_IDENTICAL",
    "DEFAULT_MAX_IDENTICAL_SUCCESS",
    "DEFAULT_MAX_NO_PROGRESS",
    "DEFAULT_HIGH_COST_TOOLS",
    "DEFAULT_MAX_HIGH_COST_REPEATS",
]


# Provider-neutral exemption set: read-only / observation / poll tools whose
# *successful* repeated calls (and identical results) are a normal part of an
# agent's loop — re-scanning a screen, re-reading a file, re-listing apps.
# These are exempt from success-loop and no-progress detection (but NOT from
# the failure-loop guard).  Per-loader guards (e.g. the macOS-native guard in
# ``backend.mcp_manager``) extend this set with their own tool names.
OBSERVATION_TOOLS: frozenset[str] = frozenset({
    # macOS desktop observation
    "get_screen_controls",
    "wait_for_controls",
    "list_apps",
    "take_screenshot",
    "capture_app_screenshot",
    "get_control_value",
    "read_clipboard",
    "read_app_dom",
    "read_screen",
    "find_text_on_screen",
    # filesystem / generic read tools
    "read_file",
    "ls",
    "glob",
    "grep",
    "view_image",
    "grep_large_results",
})

# Default knobs for the universal (per-agent) guard.  Tuned so that a model
# alternating between two distinct calls (e.g. ``doc_research(a)`` /
# ``doc_research(b)``) still trips per-key within the window, while leaving
# headroom for legitimate short retry/refine sequences.
DEFAULT_GUARD_WINDOW = 8
DEFAULT_MAX_IDENTICAL = 3
DEFAULT_MAX_IDENTICAL_SUCCESS = 3
DEFAULT_MAX_NO_PROGRESS = 4

# High-cost navigation / research tools.  Unlike cheap reads, *dozens* of these
# per run — re-navigating the same URL, re-snapshotting the same page, re-running
# the same web search — are pure waste and made up the bulk of the observed
# 185-call thrash.  They remain exempt from the small-window success/no-progress
# detectors (a few legitimate re-scans within a window are fine), but are subject
# to a generous *cumulative per-run ceiling* on identical calls, so a model that
# keeps re-issuing the same navigation/search gets curbed rather than running
# free up to the recursion limit.  The window-based detectors (maxlen ~8) cannot
# see "dozens", which is why this is a separate lifetime counter.
DEFAULT_HIGH_COST_TOOLS: frozenset[str] = frozenset({
    "browser_navigate",
    "browser_snapshot",
    "browser_take_screenshot",
    "web_research",
    "doc_research",
})
DEFAULT_MAX_HIGH_COST_REPEATS = 12


class ToolLoopDetected(ToolException):
    """Raised when the same tool is invoked repeatedly with identical
    arguments (whether or not the calls succeeded).

    MUST subclass :class:`langchain_core.tools.ToolException`: ``BaseTool``
    (and LangGraph's ``ToolNode``) only route ``ToolException`` through the
    ``handle_tool_error=True`` path that turns the error into a corrective
    tool *message* fed back to the model on the next turn.  Any other
    exception type is re-raised instead and crashes the whole agent/scheduled
    run — which is exactly the failure this guard exists to prevent."""


class ToolLoopGuard:
    """Detects identical-args tool-call loops within a recent window.

    The detection is provider-agnostic: it only looks at
    ``(tool_name, canonical_args)`` tuples and their success / failure
    status across the last ``window`` calls.

    Two independent thresholds are supported:

    * **Failure loop** (``max_identical``): trips when the same call has
      failed ``max_identical`` times in the window.  Covers the classic
      "model retries a broken call forever" pattern.
    * **Success loop** (``max_identical_success``): trips when the same
      call has *succeeded* ``max_identical_success`` times in the window
      without any observable effect — e.g. a UI action tool whose click
      lands but never changes the screen.  Observation-only tools (reads,
      screenshots) can be exempted via ``success_exempt_tools`` so that
      legitimate re-scans don't trigger false positives.
    * **No-progress loop** (``max_no_progress``): trips when the last
      ``max_no_progress`` non-exempt calls all returned the *same result*
      regardless of their arguments.  Covers the "different args, same
      useless output" pattern that identical-args detection misses — e.g.
      a research tool that returns the same boilerplate page for every
      distinct query, or a model that alternates between two distinct
      calls that each keep yielding identical results.  Uses the same
      ``success_exempt_tools`` exemption set as the success-loop guard.

    The first time a key trips either threshold, the guard also evicts
    every live ``mlx_turbo`` prompt-cache singleton.  This breaks the
    deterministic argmax that high KV-cache reuse can pin onto a
    repeating bad output, forcing a fresh prefill on the recovery turn.
    The eviction is a no-op when MLX isn't loaded, so the guard works
    unchanged on Anthropic / OpenAI / vLLM / etc.

    Parameters
    ----------
    max_identical:
        Number of *failed* calls with the same ``(name, canonical args)``
        tuple in the window required to trip the failure-loop guard.
    window:
        Size of the recent-history ring buffer (shared by both thresholds).
    recovery_hint:
        Domain-specific guidance appended to every raised exception message.
    max_identical_success:
        Number of *successful* calls with the same ``(name, canonical args)``
        tuple in the window required to trip the success-loop guard.
        ``None`` (default) disables success-loop detection.
    success_exempt_tools:
        Tool names whose successful calls are *not* counted towards
        ``max_identical_success`` *or* ``max_no_progress``.  Use this to
        exclude read-only observation tools (e.g. ``get_screen_controls``)
        that may legitimately be called many times between action steps
        and legitimately return the same content.
    max_no_progress:
        Number of consecutive *non-exempt* calls that must all return the
        same result (regardless of arguments) to trip the no-progress
        guard.  ``None`` (default) disables no-progress detection.
    recovery_temperature:
        Sampling temperature to request for the recovery turn(s) when a
        key first trips.  Greedy decoding (``temp == 0``) is a common
        proximate cause of identical-call loops, so perturbing sampling
        gives the model a way out.  ``0.0`` (default) disables the bump.
        No-op for non-MLX providers (see
        :func:`chat_models.mlx._shared.request_temperature_bump`).
    recovery_temperature_turns:
        Number of subsequent generations the temperature bump applies to
        before it auto-expires (default 1).
    max_escalations:
        Total number of trips (across all keys/detectors) after which the
        guard stops trying to coax the model back on track and instead emits
        a terminal "stop now, give your final answer" directive and fires
        ``on_escalate``.  ``None`` (default) disables escalation; the guard
        just keeps returning corrective messages.
    on_escalate:
        Optional callback invoked once (with a short reason string) when the
        escalation limit is reached.  Used by the run host to set a
        cooperative-abort flag so a model that ignores every corrective
        message still terminates gracefully.
    """

    _GENERIC_HINT = (
        "Change your arguments based on the error message, or pick a "
        "different tool."
    )

    def __init__(
        self,
        *,
        max_identical: int = 3,
        window: int = 5,
        recovery_hint: str = "",
        max_identical_success: int | None = None,
        success_exempt_tools: frozenset[str] = frozenset(),
        max_no_progress: int | None = None,
        high_cost_tools: frozenset[str] = frozenset(),
        max_high_cost_repeats: int | None = None,
        recovery_temperature: float = 0.0,
        recovery_temperature_turns: int = 1,
        max_escalations: int | None = None,
        on_escalate: Optional[Callable[[str], None]] = None,
    ) -> None:
        # Each entry is (key, ok, result_hash).  ``result_hash`` is "" when
        # the result wasn't captured (e.g. a failed call) — only non-empty
        # hashes count towards no-progress detection.
        self._history: Deque[Tuple[Tuple[str, str], bool, str]] = deque(
            maxlen=window
        )
        self._max_identical = max_identical
        self._max_identical_success = max_identical_success
        self._success_exempt_tools = success_exempt_tools
        self._max_no_progress = max_no_progress
        self._high_cost_tools = high_cost_tools
        self._max_high_cost_repeats = max_high_cost_repeats
        # Cumulative (lifetime, not windowed) success count per high-cost key,
        # so "dozens of redundant same-URL navigations" can be detected.
        self._high_cost_counts: dict[Tuple[str, str], int] = {}
        self._tripped_keys: set[Tuple[str, str]] = set()
        self._recovery_hint = recovery_hint or self._GENERIC_HINT
        self._recovery_temperature = recovery_temperature
        self._recovery_temperature_turns = recovery_temperature_turns
        # Escalation: count every trip; once it reaches ``max_escalations``
        # the corrective message becomes a terminal "stop now" directive and
        # the (optional) ``on_escalate`` callback fires once so the run host
        # can cooperatively abort a model that ignores the guidance.
        self._max_escalations = max_escalations
        self._on_escalate = on_escalate
        self._trip_count = 0
        self._escalated = False

    @staticmethod
    def _canonicalize(arguments: Any) -> str:
        """Return a stable string representation of *arguments* so that
        ``{"ref": "e23"}`` and ``{ "ref": "e23" }`` hash the same."""
        if not isinstance(arguments, dict):
            return repr(arguments)
        try:
            return json.dumps(
                arguments, sort_keys=True, separators=(",", ":"), default=str,
            )
        except (TypeError, ValueError):
            return repr(sorted(arguments.items(), key=lambda kv: kv[0]))

    @staticmethod
    def _hash_result(result: Any) -> str:
        """Return a stable short hash of *result* for no-progress detection.

        Empty / ``None`` results hash to ``""`` so they never count towards a
        no-progress loop (a tool that returns nothing isn't "making the same
        progress" in a meaningful sense, and counting it would be noisy)."""
        if result is None:
            return ""
        try:
            text = json.dumps(
                result, sort_keys=True, separators=(",", ":"), default=str,
            )
        except (TypeError, ValueError):
            text = repr(result)
        text = text.strip()
        if not text:
            return ""
        return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()

    def check_before(self, tool_name: str, arguments: Any) -> None:
        """Raise :class:`ToolLoopDetected` if either loop threshold has
        been reached for ``(tool_name, arguments)``.  Call *before* the tool."""
        key = (tool_name, self._canonicalize(arguments))

        # Cumulative per-run ceiling for high-cost navigation/research tools.
        # Checked first because it counts across the whole run, not just the
        # recent window, so it catches slow-burn redundant churn.
        if (
            self._max_high_cost_repeats is not None
            and tool_name in self._high_cost_tools
            and self._high_cost_counts.get(key, 0) >= self._max_high_cost_repeats
        ):
            self._trip(
                key,
                tool_name,
                f"Tool {tool_name!r} was called with identical arguments "
                f"{self._max_high_cost_repeats}+ times in this run — this is "
                f"redundant churn. Stop repeating it: use the results you "
                f"already have, or take a fundamentally different step. "
                f"{self._recovery_hint}",
            )

        failed_hits = sum(
            1 for k, ok, _ in self._history if k == key and not ok
        )
        if failed_hits >= self._max_identical:
            self._trip(
                key,
                tool_name,
                f"Tool {tool_name!r} was called {self._max_identical} times "
                f"with identical failing arguments. "
                f"{self._recovery_hint}",
            )

        if (
            self._max_identical_success is not None
            and tool_name not in self._success_exempt_tools
        ):
            success_hits = sum(
                1 for k, ok, _ in self._history if k == key and ok
            )
            if success_hits >= self._max_identical_success:
                self._trip(
                    key,
                    tool_name,
                    f"Tool {tool_name!r} was called "
                    f"{self._max_identical_success} times with identical "
                    f"arguments but had no visible effect. Try a different "
                    f"approach or target a different control. "
                    f"{self._recovery_hint}",
                )

        if self._max_no_progress is not None:
            # Hashes of recent *non-exempt* calls, in chronological order.
            recent_hashes = [
                h
                for k, _ok, h in self._history
                if h and k[0] not in self._success_exempt_tools
            ]
            if len(recent_hashes) >= self._max_no_progress:
                tail = recent_hashes[-self._max_no_progress:]
                if len(set(tail)) == 1:
                    self._trip(
                        ("<no_progress>", tail[-1]),
                        tool_name,
                        f"The last {self._max_no_progress} tool calls all "
                        f"returned the same result despite differing "
                        f"arguments — you are not making progress. Stop "
                        f"repeating this approach: try a fundamentally "
                        f"different tool or strategy, or report what you "
                        f"have found so far. {self._recovery_hint}",
                    )

    _TERMINAL_DIRECTIVE = (
        "You have repeatedly looped despite corrective guidance. STOP NOW — "
        "do not call any more tools. Provide your best final answer using the "
        "information you have already gathered."
    )

    def _trip(
        self, key: Tuple[str, str], tool_name: str, message: str
    ) -> None:
        """Run the recovery side-effects for a detected loop and raise.

        The first time a given *key* trips, evict the MLX prompt cache and
        request a temperature bump.  Every trip increments the escalation
        counter; once it reaches ``max_escalations`` the message becomes a
        terminal stop directive and ``on_escalate`` fires once."""
        if key not in self._tripped_keys:
            self._tripped_keys.add(key)
            self._evict_mlx_turbo_cache(tool_name)
            self._request_temperature_bump(tool_name)

        self._trip_count += 1
        if (
            self._max_escalations is not None
            and self._trip_count >= self._max_escalations
        ):
            if not self._escalated:
                self._escalated = True
                reason = (
                    f"loop guard tripped {self._trip_count} times "
                    f"(limit {self._max_escalations})"
                )
                logger.warning(
                    "ToolLoopGuard: escalation limit reached — %s; "
                    "requesting cooperative run abort.",
                    reason,
                )
                if self._on_escalate is not None:
                    try:
                        self._on_escalate(reason)
                    except Exception as exc:  # pragma: no cover
                        logger.debug(
                            "ToolLoopGuard: on_escalate callback failed: %s",
                            exc,
                        )
            raise ToolLoopDetected(self._TERMINAL_DIRECTIVE)

        raise ToolLoopDetected(message)

    def record_result(
        self, tool_name: str, arguments: Any, ok: bool, result: Any = None
    ) -> None:
        """Record the outcome of a tool call in the history buffer.

        ``result`` is hashed for no-progress detection; pass it whenever the
        call succeeded so "different args, same output" loops can be caught."""
        key = (tool_name, self._canonicalize(arguments))
        result_hash = self._hash_result(result) if ok else ""
        self._history.append((key, ok, result_hash))
        if (
            self._max_high_cost_repeats is not None
            and ok
            and tool_name in self._high_cost_tools
        ):
            self._high_cost_counts[key] = self._high_cost_counts.get(key, 0) + 1

    @staticmethod
    def _evict_mlx_turbo_cache(tool_name: str) -> None:
        """Drop every live mlx_turbo singleton once.  Breaks the
        deterministic argmax that high cache reuse can pin onto a
        repeating bad output.  No-op when turbo isn't installed or
        isn't active, so the guard works unchanged on every other
        provider."""
        try:
            from chat_models.mlx_turbo import _registry as _turbo_registry
        except Exception:
            return
        try:
            before = _turbo_registry.size()
            if before == 0:
                return
            _turbo_registry.evict_all()
            logger.warning(
                "ToolLoopGuard: evicted %d mlx_turbo singleton(s) after "
                "identical failing %s calls — KV prefix cache reset for "
                "recovery turn.",
                before,
                tool_name,
            )
        except Exception as exc:  # pragma: no cover — diagnostic path
            logger.debug("ToolLoopGuard: turbo evict failed: %s", exc)

    def _request_temperature_bump(self, tool_name: str) -> None:
        """Ask the next MLX generation(s) to sample at a higher temperature.

        Greedy decoding (``temp == 0``) pins the model's argmax onto the same
        bad output every turn; perturbing the temperature for the recovery
        turn(s) lets it pick a different action.  Complements the KV-cache
        eviction above and is a no-op when MLX isn't loaded (the bump is only
        consumed by ``ChatMLXText._sampler_kwargs``), so the guard works
        unchanged on every other provider."""
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
            logger.warning(
                "ToolLoopGuard: requested loop-recovery temperature bump "
                "(temp=%.2f for %d turn(s)) after identical %s calls.",
                self._recovery_temperature,
                self._recovery_temperature_turns,
                tool_name,
            )
        except Exception as exc:  # pragma: no cover — diagnostic path
            logger.debug("ToolLoopGuard: temperature bump failed: %s", exc)


def wrap_with_loop_guard(tool: BaseTool, guard: ToolLoopGuard) -> None:
    """Wrap *tool*'s coroutine with :class:`ToolLoopGuard`.

    The guard checks the incoming arguments against recent history
    *before* the tool runs and, after the tool completes, records
    success/failure based on whether the coroutine raised.  Failures
    include :class:`langchain_core.tools.ToolException` (the error
    channel MCP adapters use when ``isError=True``).

    When the guard trips it raises :class:`ToolLoopDetected`, which
    LangChain's ``handle_tool_error=True`` converts to a tool message
    visible to the model on the next turn.

    Idempotent — re-wrapping a tool that's already guarded is a no-op.
    Both the deep-agent direct loader (``_load_playwright_mcp_tools``)
    and the backend's ``MCPManager`` loaders attach guards, and a
    given tool list could in theory flow through both.
    """
    from functools import wraps
    from langchain_core.tools import StructuredTool

    if not isinstance(tool, StructuredTool) or tool.coroutine is None:
        return

    if getattr(tool.coroutine, "__loop_guard_wrapped__", False):
        return

    original = tool.coroutine
    tool_name = tool.name

    @wraps(original)
    async def _guarded(*args: Any, **kwargs: Any) -> Any:
        guard.check_before(tool_name, kwargs)
        try:
            result = await original(*args, **kwargs)
        except Exception:
            guard.record_result(tool_name, kwargs, ok=False)
            raise
        guard.record_result(tool_name, kwargs, ok=True, result=result)
        return result

    setattr(_guarded, "__loop_guard_wrapped__", True)
    tool.coroutine = _guarded


def build_default_guard(
    *,
    recovery_hint: str = "",
    window: int = DEFAULT_GUARD_WINDOW,
    max_identical: int = DEFAULT_MAX_IDENTICAL,
    max_identical_success: int | None = DEFAULT_MAX_IDENTICAL_SUCCESS,
    max_no_progress: int | None = DEFAULT_MAX_NO_PROGRESS,
    success_exempt_tools: frozenset[str] = OBSERVATION_TOOLS,
    high_cost_tools: frozenset[str] = DEFAULT_HIGH_COST_TOOLS,
    max_high_cost_repeats: int | None = DEFAULT_MAX_HIGH_COST_REPEATS,
    recovery_temperature: float = 0.0,
    recovery_temperature_turns: int = 1,
    max_escalations: int | None = None,
    on_escalate: Optional[Callable[[str], None]] = None,
) -> ToolLoopGuard:
    """Construct a :class:`ToolLoopGuard` with the universal default policy.

    Enables all three detectors (failure / success / no-progress) with the
    shared :data:`OBSERVATION_TOOLS` exemption set so read-only tools that
    legitimately repeat don't trip the success/no-progress guards, plus a
    cumulative per-run ceiling on identical :data:`DEFAULT_HIGH_COST_TOOLS`
    calls so redundant navigation/research churn is curbed."""
    return ToolLoopGuard(
        max_identical=max_identical,
        window=window,
        recovery_hint=recovery_hint,
        max_identical_success=max_identical_success,
        success_exempt_tools=success_exempt_tools,
        max_no_progress=max_no_progress,
        high_cost_tools=high_cost_tools,
        max_high_cost_repeats=max_high_cost_repeats,
        recovery_temperature=recovery_temperature,
        recovery_temperature_turns=recovery_temperature_turns,
        max_escalations=max_escalations,
        on_escalate=on_escalate,
    )


def guard_all_tools(
    tools: list[BaseTool],
    *,
    guard: ToolLoopGuard | None = None,
    **guard_kwargs: Any,
) -> ToolLoopGuard:
    """Universal chokepoint: wrap every tool in *tools* with one shared guard.

    This is the single place that makes "unguarded tool" impossible by
    construction.  Apply it wherever a final tool list is handed to an agent.

    ``wrap_with_loop_guard`` is idempotent (it skips any tool whose coroutine
    is already wrapped), so tools that a per-loader path already guarded —
    e.g. the macOS-native or Playwright MCP tools, which carry specialised
    per-connection guards — keep their existing guard and are left untouched.
    Only the previously-unguarded tools get attached to the shared *guard*.

    Returns the guard used, so callers can hold a reference (e.g. to install
    an escalation/abort hook).
    """
    if guard is None:
        guard = build_default_guard(**guard_kwargs)
    for t in tools:
        try:
            wrap_with_loop_guard(t, guard)
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug(
                "guard_all_tools: could not wrap %r: %s",
                getattr(t, "name", t),
                exc,
            )
    return guard
