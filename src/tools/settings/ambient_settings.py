"""Ambient assistant configuration tool.

Allows the orchestrator (and the user speaking to it) to enable, disable,
configure, or query the ambient suggestion agent at runtime.

Changes are persisted immediately to the app config file so they survive
server restarts. The ambient engine reads the config on every sweep, so a
save is the only coordination needed — no restart or signal required.

Usage by the agent::

    # Query current state
    configure_ambient_agent(action="status")

    # Simple toggle
    configure_ambient_agent(action="enable")
    configure_ambient_agent(action="disable")

    # Fine-grained configuration
    configure_ambient_agent(
        action="configure",
        interval_mins=60,
        idle_only=True,
        quiet_hours_start=22,
        quiet_hours_end=8,
        max_hints_per_day=5,
        allow_auto_run=False,
    )
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_VALID_ACTIONS = ("status", "enable", "disable", "configure")
_VALID_LLM_FAMILIES = ("mlx", "frontier", "exo", "follow_main")


@tool
async def configure_ambient_agent(
    action: str,
    enabled: Optional[bool] = None,
    interval_mins: Optional[int] = None,
    idle_only: Optional[bool] = None,
    llm_family: Optional[str] = None,
    mlx_model: Optional[str] = None,
    max_hints_per_day: Optional[int] = None,
    cooldown_hours: Optional[int] = None,
    quiet_hours_start: Optional[int] = None,
    quiet_hours_end: Optional[int] = None,
    allow_auto_run: Optional[bool] = None,
    use_memory: Optional[bool] = None,
    use_sessions: Optional[bool] = None,
    use_activity: Optional[bool] = None,
    use_history: Optional[bool] = None,
) -> str:
    """Enable, disable, configure, or query the ambient suggestion agent.

    The ambient agent runs periodic sweeps in the background and surfaces
    proactive hints based on the user's memory, recent sessions, macOS
    activity, and usage history.

    Actions
    -------
    ``status``     — Return the current configuration and pending hint count.
    ``enable``     — Turn the ambient agent on.
    ``disable``    — Turn the ambient agent off (existing hints are kept).
    ``configure``  — Update one or more settings (all fields are optional;
                     omitted fields are left unchanged).

    Configuration fields (used with ``action="configure"``)
    -------------------------------------------------------
    enabled           (bool)  — Master on/off switch.
    interval_mins     (int)   — Minutes between automated sweeps (default 30).
    idle_only         (bool)  — Only sweep when the user appears idle.
    llm_family        (str)   — Model family: "mlx" | "frontier" | "exo" |
                                "follow_main".
    mlx_model         (str)   — HuggingFace repo id for the MLX model
                                (e.g. "mlx-community/Qwen3-1.7B-4bit").
    max_hints_per_day (int)   — Daily hint cap (default 10).
    cooldown_hours    (int)   — Min hours between hints on the same topic (4).
    quiet_hours_start (int)   — Hour (0-23) when quiet hours begin (22).
    quiet_hours_end   (int)   — Hour (0-23) when quiet hours end (8).
    allow_auto_run    (bool)  — Allow "Approve & run" to auto-spawn sessions.
    use_memory        (bool)  — Include long-term memory in sweep context.
    use_sessions      (bool)  — Include recent sessions in sweep context.
    use_activity      (bool)  — Include macOS activity in sweep context.
    use_history       (bool)  — Include usage history in sweep context.
    """
    if action not in _VALID_ACTIONS:
        return (
            f"Unknown action '{action}'. "
            f"Valid actions: {', '.join(_VALID_ACTIONS)}."
        )

    try:
        from backend.config import AppConfig

        cfg = await AppConfig.aload()
        ambient = cfg.ambient

        # ── status ──────────────────────────────────────────────────────────
        if action == "status":
            try:
                from backend.ambient_store import get_store
                store = await get_store()
                pending = await store.pending_count()
            except Exception:
                pending = "unknown"

            return (
                f"Ambient agent status:\n"
                f"  enabled:           {ambient.enabled}\n"
                f"  llm_family:        {ambient.llm_family}\n"
                f"  mlx_model:         {ambient.mlx_model or '(not set)'}\n"
                f"  interval_mins:     {ambient.interval_mins}\n"
                f"  idle_only:         {ambient.idle_only}\n"
                f"  max_hints_per_day: {ambient.max_hints_per_day}\n"
                f"  cooldown_hours:    {ambient.cooldown_hours}\n"
                f"  quiet_hours:       {ambient.quiet_hours_start:02d}:00–{ambient.quiet_hours_end:02d}:00\n"
                f"  allow_auto_run:    {ambient.allow_auto_run}\n"
                f"  sources:           memory={ambient.use_memory}, "
                f"sessions={ambient.use_sessions}, "
                f"activity={ambient.use_activity}, "
                f"history={ambient.use_history}\n"
                f"  pending hints:     {pending}"
            )

        # ── enable ──────────────────────────────────────────────────────────
        if action == "enable":
            if ambient.enabled:
                return "Ambient agent is already enabled."
            ambient.enabled = True
            await cfg.asave()
            return (
                "Ambient agent enabled. It will run its first sweep on the "
                f"next {ambient.interval_mins}-minute interval (or after the "
                "current session ends)."
            )

        # ── disable ─────────────────────────────────────────────────────────
        if action == "disable":
            if not ambient.enabled:
                return "Ambient agent is already disabled."
            ambient.enabled = False
            await cfg.asave()
            return "Ambient agent disabled. Existing unread hints are preserved."

        # ── configure ───────────────────────────────────────────────────────
        # action == "configure"
        changed: list[str] = []

        if enabled is not None:
            ambient.enabled = enabled
            changed.append(f"enabled={enabled}")

        if interval_mins is not None:
            if not 1 <= interval_mins <= 1440:
                return "interval_mins must be between 1 and 1440."
            ambient.interval_mins = interval_mins
            changed.append(f"interval_mins={interval_mins}")

        if idle_only is not None:
            ambient.idle_only = idle_only
            changed.append(f"idle_only={idle_only}")

        if llm_family is not None:
            if llm_family not in _VALID_LLM_FAMILIES:
                return (
                    f"Unknown llm_family '{llm_family}'. "
                    f"Valid values: {', '.join(_VALID_LLM_FAMILIES)}."
                )
            ambient.llm_family = llm_family
            changed.append(f"llm_family={llm_family}")

        if mlx_model is not None:
            ambient.mlx_model = mlx_model
            changed.append(f"mlx_model={mlx_model!r}")

        if max_hints_per_day is not None:
            if not 1 <= max_hints_per_day <= 100:
                return "max_hints_per_day must be between 1 and 100."
            ambient.max_hints_per_day = max_hints_per_day
            changed.append(f"max_hints_per_day={max_hints_per_day}")

        if cooldown_hours is not None:
            if not 0 <= cooldown_hours <= 168:
                return "cooldown_hours must be between 0 and 168."
            ambient.cooldown_hours = cooldown_hours
            changed.append(f"cooldown_hours={cooldown_hours}")

        if quiet_hours_start is not None:
            if not 0 <= quiet_hours_start <= 23:
                return "quiet_hours_start must be 0–23."
            ambient.quiet_hours_start = quiet_hours_start
            changed.append(f"quiet_hours_start={quiet_hours_start}")

        if quiet_hours_end is not None:
            if not 0 <= quiet_hours_end <= 23:
                return "quiet_hours_end must be 0–23."
            ambient.quiet_hours_end = quiet_hours_end
            changed.append(f"quiet_hours_end={quiet_hours_end}")

        if allow_auto_run is not None:
            ambient.allow_auto_run = allow_auto_run
            changed.append(f"allow_auto_run={allow_auto_run}")

        if use_memory is not None:
            ambient.use_memory = use_memory
            changed.append(f"use_memory={use_memory}")

        if use_sessions is not None:
            ambient.use_sessions = use_sessions
            changed.append(f"use_sessions={use_sessions}")

        if use_activity is not None:
            ambient.use_activity = use_activity
            changed.append(f"use_activity={use_activity}")

        if use_history is not None:
            ambient.use_history = use_history
            changed.append(f"use_history={use_history}")

        if not changed:
            return (
                "No fields were specified. "
                "Provide at least one field to update, or use action='status' "
                "to view the current configuration."
            )

        await cfg.asave()
        return (
            "Ambient agent updated:\n"
            + "\n".join(f"  • {c}" for c in changed)
        )

    except Exception as exc:
        logger.warning("[configure_ambient_agent] failed: %s", exc, exc_info=True)
        return f"Failed to update ambient agent settings: {exc}"
