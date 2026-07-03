"""Ambient assistant engine — context gathering and lightweight LLM hint generation.

The engine is intentionally simple and self-contained:

1. Gather a bounded context bundle from memory, sessions, activity, and history.
2. Build a short JSON-requesting prompt.
3. Call a cheap, small model (the user's configured ambient LLM — defaults to
   ``mlx-community/Qwen3-1.7B-4bit`` on Apple Silicon).
4. Parse the JSON response into structured hints.
5. Persist accepted hints via :mod:`backend.ambient_store`.

The public surface is a single coroutine :func:`run_sweep` that is
called from the scheduler and from session-completion hooks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import platform
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from backend.config import AppConfig, AmbientConfig, get_app_data_dir
from backend.ambient_store import get_store

# ---------------------------------------------------------------------------
# Last-sweep timestamp
# ---------------------------------------------------------------------------


def _sweep_ts_path() -> Path:
    return get_app_data_dir() / "ambient_last_sweep.txt"


def get_last_sweep_time() -> Optional[float]:
    """Return the Unix timestamp of the most recent completed sweep, or None."""
    try:
        txt = _sweep_ts_path().read_text(encoding="utf-8").strip()
        return float(txt)
    except Exception:
        return None


def _record_sweep_time() -> None:
    try:
        _sweep_ts_path().write_text(str(time.time()), encoding="utf-8")
    except Exception:
        pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-progress flag
# ---------------------------------------------------------------------------
# Set while an LLM-backed sweep is actively generating suggestions so the
# frontend can surface a "Generating suggestions…" indicator. This is a simple
# in-process counter (sweeps can briefly overlap across triggers).
_active_sweeps = 0


def is_sweep_running() -> bool:
    """Return True while at least one suggestion-generating sweep is active."""
    return _active_sweeps > 0

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_MEMORY_TOPICS = 8
_MAX_SESSIONS = 10
_MAX_ACTIVITY_ROWS = 15   # each row is now multi-line (title + url + context)
_MAX_PROMPT_CHARS = 12000

_HINT_SCHEMA = """\
Return ONLY a JSON array (no markdown fence, no prose before/after) of hint
objects.  Each object must have exactly these keys:

  title           (string, ≤ 12 words — be specific, not generic)
  rationale       (string, 1-2 sentences: WHY now, what evidence from context)
  proposed_prompt (string, ready-to-send chat message — include all details
                   needed to act without follow-up questions)
  suggested_agent (string | null, agent name if one is clearly relevant)
  kind            ("task" | "schedule" | "trigger" | "automation")
                  task       — one-off action to do right now
                  schedule   — recurring cron job (include schedule_cron)
                  trigger    — event-driven automation (file, shell, HTTP, git)
                  automation — general multi-step workflow
  schedule_cron   (string | null, standard cron expression, only for kind=schedule)
  confidence      (float 0.0-1.0)
  sources         (array of strings from: "memory" | "sessions" | "activity"
                   | "history" | "schedules" | "triggers")

Rules:
- Prefer "schedule" when the task naturally repeats on a time interval.
- Prefer "trigger" when the task should react to an event (new file, app state,
  URL change, git push, system condition).
- For schedule hints, set schedule_cron to a valid cron expression (e.g. "0 9 * * 1-5").
- For trigger hints, describe the event condition clearly in proposed_prompt.
- The proposed_prompt for schedule/trigger kinds should ask the agent to CREATE
  the automation, not just perform the action once.
- NEVER suggest a schedule or trigger that already exists (see "Existing automations"
  section in context).
- Return an empty array [] if you have no high-quality suggestions.
"""

# ---------------------------------------------------------------------------
# Quiet-hours guard
# ---------------------------------------------------------------------------


def _in_quiet_hours(cfg: AmbientConfig) -> bool:
    """Return True if the current local time falls inside the quiet window."""
    hour = datetime.now().hour
    start, end = cfg.quiet_hours_start, cfg.quiet_hours_end
    if start <= end:
        return start <= hour < end
    # Wraps midnight: e.g. 22–8 means 22,23,0,1,...,7
    return hour >= start or hour < end


# ---------------------------------------------------------------------------
# Model resolver (mirrors the memory consolidation pattern)
# ---------------------------------------------------------------------------


async def _resolve_model(ambient: AmbientConfig) -> Optional[Any]:
    """Return an instantiated LangChain chat model for the ambient family.

    Returns None when the resolved stack is unavailable (e.g. MLX on a
    non-Apple-Silicon Mac), allowing a graceful no-op.
    """
    family = ambient.llm_family

    is_apple = platform.system() == "Darwin" and platform.machine() == "arm64"

    if family == "mlx" and not is_apple:
        # Silently fall back to follow_main on non-Apple hardware.
        family = "follow_main"

    try:
        cfg = await AppConfig.aload()

        # Don't load a second in-process MLX model while the main chat
        # provider is a cloud/frontier one.  Doing so pulls a fresh copy of
        # weights (plus a warmup pass) into unified Metal memory on the sweep
        # schedule even though the user deliberately switched away from MLX —
        # the exact combination that caused GPU OOM aborts.  Ride along on the
        # main provider instead until the user switches back to a local one.
        _LOCAL_PROVIDERS = {"mlx", "omlx", "exo"}
        if family == "mlx" and (cfg.llm.provider or "").lower() not in _LOCAL_PROVIDERS:
            logger.info(
                "[ambient] llm_family='mlx' but main provider='%s' is non-local "
                "— following main provider to avoid a second MLX load.",
                cfg.llm.provider,
            )
            family = "follow_main"

        if family == "mlx":
            repo = ambient.mlx_model or cfg.llm.mlx.hf_llm_model_id
            if not repo:
                return None
            from deep_agent.model_factory import create_llm  # type: ignore[import]
            # Temporarily override to use the ambient MLX repo.
            import os
            orig = os.environ.get("MLX_LLM_MODEL_ID")
            os.environ["MLX_LLM_MODEL_ID"] = repo
            try:
                return await asyncio.to_thread(create_llm, "mlx")
            finally:
                if orig is None:
                    os.environ.pop("MLX_LLM_MODEL_ID", None)
                else:
                    os.environ["MLX_LLM_MODEL_ID"] = orig

        elif family == "frontier":
            from deep_agent.model_factory import create_llm  # type: ignore[import]
            provider = cfg.llm.provider
            if provider not in ("anthropic", "openai"):
                provider = "anthropic"
            return await asyncio.to_thread(create_llm, provider)

        elif family == "exo":
            from deep_agent.model_factory import create_llm  # type: ignore[import]
            return await asyncio.to_thread(create_llm, "exo")

        else:  # follow_main
            from deep_agent.model_factory import create_llm  # type: ignore[import]
            return await asyncio.to_thread(create_llm, cfg.llm.provider)

    except Exception:
        logger.debug("[ambient] model resolve failed", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Context gatherers
# ---------------------------------------------------------------------------


async def _gather_memory(cfg: AmbientConfig) -> str:
    """Return a compact summary of top memory topics."""
    if not cfg.use_memory:
        return ""
    try:
        mem_dir = get_app_data_dir() / "memory"
        index_path = mem_dir / "MEMORY.md"
        if not index_path.exists():
            return ""

        lines = index_path.read_text(encoding="utf-8").splitlines()
        # The MEMORY.md index lists topic headlines — take the first N lines.
        summary = "\n".join(lines[:60])
        return f"## Long-term memory (index)\n{summary}"
    except Exception:
        logger.debug("[ambient] memory gather failed", exc_info=True)
        return ""


def _last_assistant_snippet(session_id: str, max_chars: int = 300) -> str:
    """Return a short snippet of the final assistant message in a session, or ''."""
    try:
        from backend.session_manager import load_messages  # type: ignore[import]
        msgs = load_messages(session_id)
        # Walk backwards to find the last AI/assistant turn.
        for msg in reversed(msgs):
            role = msg.get("role") or msg.get("type") or ""
            if role in ("ai", "assistant", "agent"):
                content = msg.get("content") or ""
                if isinstance(content, list):
                    # LangChain content blocks — extract text parts.
                    content = " ".join(
                        c.get("text", "") for c in content if isinstance(c, dict)
                    )
                content = " ".join(str(content).split())  # collapse whitespace
                return content[:max_chars]
    except Exception:
        pass
    return ""


async def _gather_sessions(cfg: AmbientConfig) -> str:
    """Return titles + agents of the most recent sessions within the lookback window."""
    if not cfg.use_sessions:
        return ""
    try:
        from backend.state import session_mgr  # type: ignore[import]
        from datetime import timezone
        cutoff = datetime.now(timezone.utc).timestamp() - cfg.lookback_hours * 3600
        sessions = await asyncio.to_thread(session_mgr.list_history)
        sessions = [
            s for s in sessions
            if s.created_at.timestamp() >= cutoff
        ]
        sessions = sorted(sessions, key=lambda s: s.created_at, reverse=True)[:_MAX_SESSIONS]
        if not sessions:
            return ""
        lines: list[str] = []
        for s in sessions:
            duration_str = ""
            try:
                secs = int((s.updated_at - s.created_at).total_seconds())
                if secs >= 60:
                    duration_str = f", {secs // 60}m"
            except Exception:
                pass
            tools_str = ""
            if s.tools_used:
                tools_str = f", tools: {', '.join(s.tools_used[:5])}"
            msgs_str = f", {s.message_count} msg{'s' if s.message_count != 1 else ''}" if s.message_count else ""
            header = (
                f"- [{s.created_at.strftime('%Y-%m-%d %H:%M')}] {s.title or 'Untitled'}"
                f" (agent: {s.agent_name or 'default'}{duration_str}{msgs_str}{tools_str})"
            )
            snippet = await asyncio.to_thread(_last_assistant_snippet, s.id)
            if snippet:
                lines.append(f"{header}\n  last output: {snippet}")
            else:
                lines.append(header)
        return "## Recent sessions\n" + "\n".join(lines)
    except Exception:
        logger.debug("[ambient] sessions gather failed", exc_info=True)
        return ""


async def _gather_activity(cfg: AmbientConfig) -> str:
    """Return the last lookback_hours of macOS activity as a text block."""
    if not cfg.use_activity:
        return ""
    try:
        import time as _time
        from backend.activity_tracker import search_activity  # type: ignore[import]
        date_from = int(_time.time()) - cfg.lookback_hours * 3600
        rows = await asyncio.to_thread(
            search_activity, None, limit=_MAX_ACTIVITY_ROWS, date_from=date_from
        )
        if not rows:
            return ""
        lines: list[str] = []
        for r in rows:
            app = r.get("app", "?")
            title = (r.get("title") or "")[:80]
            duration = int(r.get("duration_s", 0) or 0)
            url = (r.get("url") or "")[:120]
            context = (r.get("context") or "")[:200]

            parts = [f"- {app} | {title} ({duration}s)"]
            if url:
                parts.append(f"  url: {url}")
            if context:
                # Collapse newlines so the snippet stays on one visual line.
                snippet = " ".join(context.split())[:200]
                parts.append(f"  context: {snippet}")
            lines.append("\n".join(parts))
        return f"## Recent macOS activity (last {cfg.lookback_hours}h)\n" + "\n".join(lines)
    except Exception:
        logger.debug("[ambient] activity gather failed", exc_info=True)
        return ""


async def _gather_history(cfg: AmbientConfig) -> str:
    """Return a summary of transcript patterns (session count, tool usage) within the lookback window."""
    if not cfg.use_history:
        return ""
    try:
        from backend.state import session_mgr  # type: ignore[import]
        from datetime import timezone
        cutoff = datetime.now(timezone.utc).timestamp() - cfg.lookback_hours * 3600
        all_sessions = await asyncio.to_thread(session_mgr.list_history)
        all_sessions = [s for s in all_sessions if s.created_at.timestamp() >= cutoff]
        all_sessions = sorted(all_sessions, key=lambda s: s.created_at, reverse=True)[:30]
        if not all_sessions:
            return ""

        # Aggregate tools used across recent sessions.
        tool_freq: dict[str, int] = {}
        for s in all_sessions:
            for t in s.tools_used:
                tool_freq[t] = tool_freq.get(t, 0) + 1

        top_tools = sorted(tool_freq.items(), key=lambda x: x[1], reverse=True)[:8]
        tool_str = ", ".join(f"{t}({c})" for t, c in top_tools)
        return (
            f"## Usage history\n"
            f"Total recent sessions: {len(all_sessions)}\n"
            f"Top tools used: {tool_str or 'none'}"
        )
    except Exception:
        logger.debug("[ambient] history gather failed", exc_info=True)
        return ""


async def _gather_existing_automations() -> str:
    """Return a compact list of existing schedules and triggers.

    This is injected into the prompt so the LLM can avoid suggesting
    automations that already exist.
    """
    lines: list[str] = []
    try:
        from backend.scheduler import load_all_schedules  # type: ignore[import]
        schedules = await asyncio.to_thread(load_all_schedules)
        # Only include enabled schedules — disabled ones should not prevent
        # the LLM from suggesting similar automations.
        active_schedules = [s for s in schedules if s.enabled]
        if active_schedules:
            lines.append("### Existing schedules (do NOT suggest these again)")
            for s in active_schedules:
                lines.append(f"- [{s.id}] cron={s.cron_expression}: {s.prompt[:80]}")
    except Exception:
        logger.debug("[ambient] schedule gather failed", exc_info=True)

    try:
        from backend.trigger_manager import load_all_triggers  # type: ignore[import]
        triggers = await asyncio.to_thread(load_all_triggers)
        # Only include enabled triggers — disabled builtin/custom triggers
        # should not prevent the LLM from suggesting similar automations.
        active_triggers = [t for t in triggers if t.enabled]
        if active_triggers:
            lines.append("### Existing triggers (do NOT suggest these again)")
            for t in active_triggers:
                detail = ""
                if t.type == "fileos":
                    detail = f" path={t.path} watch={t.watch}"
                elif t.type in ("macostool", "shell"):
                    script_snip = (t.script or t.command or "")[:50]
                    detail = f" script={script_snip!r}"
                elif t.type == "http":
                    detail = f" url={t.url}"
                elif t.type == "git":
                    detail = f" repo={t.repo_path}"
                lines.append(
                    f"- [{t.id}] type={t.type}{detail}: {t.prompt[:80]}"
                )
    except Exception:
        logger.debug("[ambient] trigger gather failed", exc_info=True)

    if not lines:
        return ""
    return "## Existing automations\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _build_prompt(sections: list[str]) -> str:
    context = "\n\n".join(s for s in sections if s.strip())
    # Trim to hard character cap so we never exhaust a small model's context.
    if len(context) > _MAX_PROMPT_CHARS:
        context = context[:_MAX_PROMPT_CHARS] + "\n[…context trimmed…]"

    return (
        "You are an ambient assistant that analyses a user's work patterns and "
        "surfaces targeted, high-value suggestions — including one-off tasks, "
        "recurring scheduled jobs, and event-driven triggers.\n\n"
        "When you spot a pattern that repeats (e.g. the user manually runs the "
        "same thing every morning, or checks a URL periodically), prefer suggesting "
        "a SCHEDULE or TRIGGER automation so it runs automatically.\n\n"
        "Trigger types available: fileos (watch files/dirs), shell (run a shell "
        "command and react to its output), macostool (AppleScript/JXA), http "
        "(poll a URL), git (react to new commits).\n\n"
        "Context gathered right now:\n\n"
        f"{context}\n\n"
        "Based on this context, generate up to 3 high-quality, specific suggestions. "
        "Each suggestion must be grounded in evidence from the context above — "
        "do not invent suggestions that have no basis in the data.\n\n"
        f"{_HINT_SCHEMA}"
    )


# ---------------------------------------------------------------------------
# LLM invocation + JSON parsing
# ---------------------------------------------------------------------------


async def _call_llm(model: Any, prompt: str) -> str:
    """Invoke the model and return the raw text response as a plain string.

    Some on-prem / MLX models return a list of content blocks
    (``[{"type": "text", "text": "..."}]``) rather than a bare string.
    We normalise that here so the JSON parser always receives a str.
    """
    from langchain_core.messages import HumanMessage  # type: ignore[import]

    try:
        response = await model.ainvoke([HumanMessage(content=prompt)])
        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            # Flatten LangChain content blocks → plain text.
            content = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        return content
    except Exception:
        logger.debug("[ambient] LLM call failed", exc_info=True)
        return ""


_VALID_KINDS = frozenset({"task", "schedule", "trigger", "automation"})


def _parse_hints(raw: str, min_confidence: float) -> list[dict[str, Any]]:
    """Extract and validate JSON hints from the LLM response."""
    # Strip markdown fences if the model ignored the instruction.
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().strip("`")

    # Find the first [ ... ] block.
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1:
        logger.debug("[ambient] no JSON array found in LLM response")
        return []

    try:
        items: list[Any] = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        logger.debug("[ambient] JSON parse failed: %s", raw[start:end + 1][:200])
        return []

    valid: list[dict[str, Any]] = []
    required = {"title", "rationale", "proposed_prompt", "kind"}
    for item in items:
        if not isinstance(item, dict):
            continue
        if not required.issubset(item.keys()):
            continue
        try:
            confidence = float(item["confidence"]) if "confidence" in item else 0.8
        except (TypeError, ValueError):
            confidence = 0.8
        if confidence < min_confidence:
            continue
        kind = item.get("kind", "task")
        if kind not in _VALID_KINDS:
            kind = "task"
        # Carry schedule_cron forward when the model fills it in.
        schedule_cron: Optional[str] = None
        if kind == "schedule":
            raw_cron = item.get("schedule_cron") or None
            if raw_cron and isinstance(raw_cron, str):
                schedule_cron = raw_cron.strip()
        valid.append({
            "title": str(item["title"])[:120],
            "rationale": str(item["rationale"])[:500],
            "proposed_prompt": str(item["proposed_prompt"])[:2000],
            "suggested_agent": item.get("suggested_agent") or None,
            "kind": kind,
            "schedule_cron": schedule_cron,
            "confidence": confidence,
            "sources": [str(s) for s in item.get("sources", [])],
        })

    return valid[:3]  # hard cap: never more than 3 hints per sweep


# ---------------------------------------------------------------------------
# Idle gate
# ---------------------------------------------------------------------------


async def _is_idle(cfg: AmbientConfig) -> bool:
    """Return True when the user appears idle (or idle check is disabled)."""
    if not cfg.idle_only:
        return True
    try:
        from backend.activity_tracker import tracker  # type: ignore[import]
        idle_secs = await asyncio.to_thread(tracker.seconds_since_last_input)
        # Consider "idle" when no input for 3× the activity poll interval
        # or for more than 2 minutes, whichever is greater.
        threshold = max(cfg.interval_mins * 60 * 0.1, 120)
        return idle_secs >= threshold
    except Exception:
        # If we can't check, allow the sweep to proceed.
        return True


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_sweep(*, is_manual: bool = False) -> dict[str, Any]:
    """Run a full ambient sweep.

    Returns a summary dict:  ``{"hints_added": int, "skipped": str | None}``

    This is intentionally best-effort — every error is caught and logged so
    it never crashes the scheduler or the session cleanup path.
    """
    t0 = time.monotonic()
    try:
        cfg = await AppConfig.aload()
        ambient = cfg.ambient

        if not ambient.enabled and not is_manual:
            return {"hints_added": 0, "skipped": "disabled"}

        # Idle gate — only check for automated sweeps.
        if not is_manual:
            if not await _is_idle(ambient):
                logger.debug("[ambient] user not idle, skipping sweep")
                return {"hints_added": 0, "skipped": "not_idle"}

        # Resolve model — if unavailable, skip gracefully.
        model = await _resolve_model(ambient)
        if model is None:
            logger.info("[ambient] no model available for sweep")
            return {"hints_added": 0, "skipped": "no_model"}

        global _active_sweeps  # noqa: PLW0603
        _active_sweeps += 1
        try:
            # Gather context sections in parallel.
            sections = await asyncio.gather(
                _gather_memory(ambient),
                _gather_sessions(ambient),
                _gather_activity(ambient),
                _gather_history(ambient),
                _gather_existing_automations(),
            )

            non_empty = [s for s in sections if s.strip()]
            if not non_empty:
                logger.debug("[ambient] no context available, skipping sweep")
                return {"hints_added": 0, "skipped": "no_context"}

            prompt = _build_prompt(non_empty)

            raw = await _call_llm(model, prompt)
            if not raw:
                return {"hints_added": 0, "skipped": "empty_response"}

            hints = _parse_hints(raw, ambient.min_confidence)
            if not hints:
                logger.debug("[ambient] no valid hints parsed from LLM response")
                return {"hints_added": 0, "skipped": "no_valid_hints"}

            store = await get_store()
            added_ids = await store.add_hints(
                hints,
                cooldown_hours=ambient.cooldown_hours,
                max_per_day=ambient.max_hints_per_day,
            )

            _record_sweep_time()
            elapsed = time.monotonic() - t0
            logger.info(
                "[ambient] sweep complete in %.1fs — %d hint(s) added (parsed %d)",
                elapsed, len(added_ids), len(hints),
            )
            return {"hints_added": len(added_ids), "skipped": None}
        finally:
            _active_sweeps = max(0, _active_sweeps - 1)

    except Exception:
        logger.exception("[ambient] sweep error")
        return {"hints_added": 0, "skipped": "error"}
