"""LangChain tools that let the orchestrator agent inspect and adjust
application settings at runtime.

Design principles
-----------------
* **Read first, write narrow.**  ``get_settings`` returns a filtered,
  credential-free view of the full config so the agent can reason about
  the current state before changing anything.
* **Scoped mutations.**  Each write tool touches only a well-defined
  subset of ``AppConfig`` fields.  Credentials, provider selection, and
  cluster-provisioning config are intentionally excluded.
* **Changes take effect immediately** for subsequent sessions via
  ``AppConfig.save()`` + ``apply_to_environ()``.  Active session LLMs
  are already instantiated, so provider-level changes (e.g.
  ``orchestrator.llm_family``) apply on the next session rebuild or
  after calling ``spawn_followup_session``.
* **MCP server toggling** reconnects / disconnects the server in the
  running process so the next message in the current session already
  reflects the change — same as the Settings UI.

Excluded from agent write access
---------------------------------
* API keys / AWS credentials — never flow through LLM context.
* ``exo.*`` cluster provisioning / remote management — human action.
* ``claude_hook`` / ``openclaw`` / ``observability`` — no useful
  orchestrator reason to change mid-run; high accidental-damage risk.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from backend.config import AppConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_VALID_ORCHESTRATOR_FAMILIES = frozenset({"follow_main", "frontier", "mlx", "exo"})
_VALID_PROMPT_MODES = frozenset({"auto", "full", "lite"})
_VALID_MEMORY_FAMILIES = frozenset({"follow_main", "frontier", "mlx"})
# Every provider create_llm understands (kept in sync with
# ``deep_agent.model_factory.create_llm`` / the LLM_PROVIDER raise message).
_VALID_LLM_PROVIDERS = frozenset({"anthropic", "openai", "mlx", "exo", "omlx"})


def _safe_settings_view(cfg: AppConfig) -> dict:
    """Return a credential-free dict of settings the agent can reason about."""
    orch = cfg.orchestrator
    mem = cfg.memory
    llm = cfg.llm
    act = cfg.activity

    return {
        "llm": {
            "provider": llm.provider,
            "anthropic": {
                "model_name": llm.anthropic.model_name,
                "model_provider": llm.anthropic.model_provider,
                "max_tokens": llm.anthropic.max_tokens,
                "thinking_enabled": llm.anthropic.thinking_enabled,
                "thinking_budget": llm.anthropic.thinking_budget,
                "tool_efficient": llm.anthropic.tool_efficient,
            },
            "mlx": {
                "hf_llm_model_id": llm.mlx.hf_llm_model_id,
                "mlx_max_tokens": llm.mlx.mlx_max_tokens,
                "mlx_temp": llm.mlx.mlx_temp,
                "mlx_thinking": llm.mlx.mlx_thinking,
                "mlx_prompt_cache_max_tokens": llm.mlx.mlx_prompt_cache_max_tokens,
            },
        },
        "orchestrator": {
            "llm_family": orch.llm_family,
            "mlx_model": orch.mlx_model,
            "mlx_model_type": orch.mlx_model_type,
            "prompt_mode": orch.prompt_mode,
        },
        "memory": {
            "enabled": mem.enabled,
            "inject_on_session_start": mem.inject_on_session_start,
            "inject_realtime": mem.inject_realtime,
            "llm_family": mem.llm_family,
            "mlx_model": mem.mlx_model,
            "min_hours": mem.min_hours,
            "min_sessions": mem.min_sessions,
            "retention_days": mem.retention_days,
        },
        "activity": {
            "enabled": act.enabled,
            "interval_secs": act.interval_secs,
            "retain_days": act.retain_days,
            "exclude_apps": list(act.exclude_apps),
            "idle_threshold_secs": act.idle_threshold_secs,
        },
        "exo": {
            "enabled": cfg.exo.enabled,
            "model_name": cfg.exo.model_name,
            "max_tokens": cfg.exo.max_tokens,
            "enable_thinking": cfg.exo.enable_thinking,
            "sharding": cfg.exo.sharding,
            "instance_meta": cfg.exo.instance_meta,
            "min_nodes": cfg.exo.min_nodes,
        },
        "omlx": {
            "enabled": cfg.omlx.enabled,
            "model_name": cfg.omlx.model_name,
            "max_tokens": cfg.omlx.max_tokens,
            "thinking_enabled": cfg.omlx.thinking_enabled,
            "max_context_window": cfg.omlx.max_context_window,
        },
        "ambient_suggest_recurrence": cfg.ambient_suggest_recurrence,
        "mcp_servers": [
            {
                "id": s.id,
                "name": s.name,
                "enabled": s.enabled,
                "builtin": s.builtin,
                "transport": s.transport,
            }
            for s in cfg.mcp_servers
        ],
    }


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_settings_tools() -> list:
    """Return tools that let the orchestrator inspect and adjust settings."""

    @tool
    def get_settings() -> str:
        """Return the current application settings relevant to the orchestrator:
        LLM provider, model names, generation parameters, orchestrator config,
        memory settings, and MCP server list.

        Credentials and sensitive fields are excluded. Use this before calling
        any update_* settings tool to understand the current configuration.
        """
        import json

        cfg = AppConfig.load()
        return json.dumps(_safe_settings_view(cfg), indent=2)

    @tool
    def update_orchestrator_settings(
        llm_family: str | None = None,
        mlx_model: str | None = None,
        mlx_model_type: str | None = None,
        prompt_mode: str | None = None,
    ) -> str:
        """Adjust the ORCHESTRATOR SUBAGENT's LLM stack and prompt configuration.

        IMPORTANT: This does NOT change the main chat LLM that the user talks to.
        To switch the main chat model, use switch_model_provider instead.

        The orchestrator is an internal planning subagent that runs tool calls
        and coordinates work. Its LLM can be set independently from the main
        chat provider (e.g. run orchestration on local MLX while chat uses
        Anthropic, or vice versa).

        Changes are persisted and take effect on the next session build.

        Args:
            llm_family: Which LLM stack the orchestrator subagent uses.
                "follow_main"  — same provider as the main chat LLM (default).
                "frontier"     — force Anthropic API regardless of main provider.
                "mlx"          — local MLX; optionally set mlx_model.
                "exo"          — distributed exo cluster.
            mlx_model: HuggingFace repo id used when llm_family is "mlx"
                (e.g. "mlx-community/Qwen3-8B-4bit"). Empty string clears
                the override and falls back to the global MLX text model.
            mlx_model_type: "llm" (text-only) or "vlm" (vision-language).
                Only relevant when llm_family is "mlx".
            prompt_mode: Size of the orchestrator system prompt.
                "auto"  — lite when on mlx/exo, full otherwise (default).
                "full"  — always use the full Claude-tuned prompt.
                "lite"  — always use the compact prompt (saves context).
        """
        if llm_family is not None and llm_family not in _VALID_ORCHESTRATOR_FAMILIES:
            return (
                f"Error: llm_family must be one of {sorted(_VALID_ORCHESTRATOR_FAMILIES)}, "
                f"got '{llm_family}'."
            )
        if prompt_mode is not None and prompt_mode not in _VALID_PROMPT_MODES:
            return (
                f"Error: prompt_mode must be one of {sorted(_VALID_PROMPT_MODES)}, "
                f"got '{prompt_mode}'."
            )
        if mlx_model_type is not None and mlx_model_type not in ("llm", "vlm"):
            return "Error: mlx_model_type must be 'llm' or 'vlm'."

        cfg = AppConfig.load()
        orch = cfg.orchestrator

        if llm_family is not None:
            orch.llm_family = llm_family
        if mlx_model is not None:
            orch.mlx_model = mlx_model
        if mlx_model_type is not None:
            orch.mlx_model_type = mlx_model_type
        if prompt_mode is not None:
            orch.prompt_mode = prompt_mode

        cfg.save()
        cfg.apply_to_environ()

        parts = []
        if llm_family is not None:
            parts.append(f"llm_family={orch.llm_family}")
        if mlx_model is not None:
            parts.append(f"mlx_model={orch.mlx_model or '(global default)'}")
        if mlx_model_type is not None:
            parts.append(f"mlx_model_type={orch.mlx_model_type}")
        if prompt_mode is not None:
            parts.append(f"prompt_mode={orch.prompt_mode}")

        return (
            f"Orchestrator settings updated: {', '.join(parts)}. "
            "Takes full effect on the next session build."
        )

    @tool
    def update_generation_params(
        max_tokens: int | None = None,
        thinking_enabled: bool | None = None,
        thinking_budget: int | None = None,
        tool_efficient: bool | None = None,
        mlx_max_tokens: int | None = None,
        mlx_temp: float | None = None,
        mlx_thinking: bool | None = None,
        exo_max_tokens: int | None = None,
        exo_enable_thinking: bool | None = None,
        omlx_max_tokens: int | None = None,
        omlx_thinking_enabled: bool | None = None,
    ) -> str:
        """Tune inference generation parameters for any LLM provider.

        Anthropic parameters (max_tokens, thinking_enabled, thinking_budget,
        tool_efficient) apply when the provider is "anthropic" or "bedrock".
        MLX parameters (mlx_max_tokens, mlx_temp, mlx_thinking) apply when
        the provider is "mlx".
        EXO parameters (exo_max_tokens, exo_enable_thinking) apply when the
        provider is "exo".
        oMLX parameters (omlx_max_tokens, omlx_thinking_enabled) apply when
        the provider is "omlx".

        All changes are persisted and applied to the environment immediately.
        The currently running session LLM is already instantiated, so these
        take full effect on the next session build.

        Args:
            max_tokens: Maximum output tokens for Anthropic models (e.g. 16384).
            thinking_enabled: Enable extended thinking for Claude models that
                support it (e.g. claude-3-7-sonnet).
            thinking_budget: Token budget for thinking when enabled (e.g. 4096).
            tool_efficient: Use the token-efficient tool-use beta header.
            mlx_max_tokens: Maximum output tokens for local MLX models.
            mlx_temp: Sampling temperature for MLX models (0.0 = greedy).
            mlx_thinking: Enable chain-of-thought for MLX models that support it.
            exo_max_tokens: Maximum output tokens for exo cluster requests.
                Caps runaway generations — exo otherwise falls back to the
                model's full context window (can cause multi-minute completions).
                Default 8192. Increase for long document tasks.
            exo_enable_thinking: Enable chain-of-thought reasoning for exo
                models that support it (Qwen3, DeepSeek V3.1, GLM-4.x).
                False (default) = fastest; True adds reasoning tokens.
            omlx_max_tokens: Maximum output tokens per oMLX response.
                Default 8192. Increase for long document tasks.
            omlx_thinking_enabled: Enable chain-of-thought for oMLX models
                that support it (e.g. Qwen3). Sent per request as
                chat_template_kwargs.enable_thinking.
        """
        if max_tokens is not None and max_tokens < 1:
            return "Error: max_tokens must be a positive integer."
        if thinking_budget is not None and thinking_budget < 1:
            return "Error: thinking_budget must be a positive integer."
        if mlx_max_tokens is not None and mlx_max_tokens < 1:
            return "Error: mlx_max_tokens must be a positive integer."
        if mlx_temp is not None and not (0.0 <= mlx_temp <= 2.0):
            return "Error: mlx_temp must be between 0.0 and 2.0."
        if exo_max_tokens is not None and exo_max_tokens < 1:
            return "Error: exo_max_tokens must be a positive integer."
        if omlx_max_tokens is not None and omlx_max_tokens < 1:
            return "Error: omlx_max_tokens must be a positive integer."

        cfg = AppConfig.load()
        a = cfg.llm.anthropic
        m = cfg.llm.mlx

        if max_tokens is not None:
            a.max_tokens = max_tokens
        if thinking_enabled is not None:
            a.thinking_enabled = thinking_enabled
        if thinking_budget is not None:
            a.thinking_budget = thinking_budget
        if tool_efficient is not None:
            a.tool_efficient = tool_efficient
        if mlx_max_tokens is not None:
            m.mlx_max_tokens = mlx_max_tokens
        if mlx_temp is not None:
            m.mlx_temp = mlx_temp
        if mlx_thinking is not None:
            m.mlx_thinking = mlx_thinking
        if exo_max_tokens is not None:
            cfg.exo.max_tokens = exo_max_tokens
        if exo_enable_thinking is not None:
            cfg.exo.enable_thinking = exo_enable_thinking
        if omlx_max_tokens is not None:
            cfg.omlx.max_tokens = omlx_max_tokens
        if omlx_thinking_enabled is not None:
            cfg.omlx.thinking_enabled = omlx_thinking_enabled

        cfg.save()
        cfg.apply_to_environ()

        changed: list[str] = []
        if max_tokens is not None:
            changed.append(f"max_tokens={a.max_tokens}")
        if thinking_enabled is not None:
            changed.append(f"thinking_enabled={a.thinking_enabled}")
        if thinking_budget is not None:
            changed.append(f"thinking_budget={a.thinking_budget}")
        if tool_efficient is not None:
            changed.append(f"tool_efficient={a.tool_efficient}")
        if mlx_max_tokens is not None:
            changed.append(f"mlx_max_tokens={m.mlx_max_tokens}")
        if mlx_temp is not None:
            changed.append(f"mlx_temp={m.mlx_temp}")
        if mlx_thinking is not None:
            changed.append(f"mlx_thinking={m.mlx_thinking}")
        if exo_max_tokens is not None:
            changed.append(f"exo_max_tokens={cfg.exo.max_tokens}")
        if exo_enable_thinking is not None:
            changed.append(f"exo_enable_thinking={cfg.exo.enable_thinking}")
        if omlx_max_tokens is not None:
            changed.append(f"omlx_max_tokens={cfg.omlx.max_tokens}")
        if omlx_thinking_enabled is not None:
            changed.append(f"omlx_thinking_enabled={cfg.omlx.thinking_enabled}")

        return (
            f"Generation parameters updated: {', '.join(changed)}. "
            "Takes full effect on the next session build."
        )

    @tool
    def update_memory_settings(
        enabled: bool | None = None,
        inject_on_session_start: bool | None = None,
        inject_realtime: bool | None = None,
        llm_family: str | None = None,
        mlx_model: str | None = None,
        min_hours: int | None = None,
        min_sessions: int | None = None,
        retention_days: int | None = None,
    ) -> str:
        """Adjust memory consolidation and injection settings.

        Memory operates on two independent injection layers:
        - Layer 1 (inject_on_session_start): injects MEMORY.md into the
          system prompt once at session start. Zero extra LLM calls; stable.
        - Layer 2 (inject_realtime): runs a per-turn relevance ranker and
          injects the top matching memory topics. Adaptive but costs one
          extra LLM call per turn.

        Changes take effect on the next session build.

        Args:
            enabled: Master toggle for memory consolidation (background job).
            inject_on_session_start: Enable Layer 1 memory injection.
            inject_realtime: Enable Layer 2 per-turn memory injection.
            llm_family: LLM stack for consolidation side-queries.
                "follow_main" | "frontier" | "mlx".
            mlx_model: HF repo id for the consolidation model when
                llm_family is "mlx". Empty string clears the override.
            min_hours: Minimum hours between consolidation runs.
            min_sessions: Minimum sessions before consolidation triggers.
            retention_days: Days before old memory files are pruned.
        """
        if llm_family is not None and llm_family not in _VALID_MEMORY_FAMILIES:
            return (
                f"Error: llm_family must be one of {sorted(_VALID_MEMORY_FAMILIES)}, "
                f"got '{llm_family}'."
            )
        if min_hours is not None and min_hours < 0:
            return "Error: min_hours must be non-negative."
        if min_sessions is not None and min_sessions < 0:
            return "Error: min_sessions must be non-negative."
        if retention_days is not None and retention_days < 1:
            return "Error: retention_days must be at least 1."

        cfg = AppConfig.load()
        mem = cfg.memory

        if enabled is not None:
            mem.enabled = enabled
        if inject_on_session_start is not None:
            mem.inject_on_session_start = inject_on_session_start
        if inject_realtime is not None:
            mem.inject_realtime = inject_realtime
        if llm_family is not None:
            mem.llm_family = llm_family
        if mlx_model is not None:
            mem.mlx_model = mlx_model
        if min_hours is not None:
            mem.min_hours = min_hours
        if min_sessions is not None:
            mem.min_sessions = min_sessions
        if retention_days is not None:
            mem.retention_days = retention_days

        cfg.save()

        changed: list[str] = []
        if enabled is not None:
            changed.append(f"enabled={mem.enabled}")
        if inject_on_session_start is not None:
            changed.append(f"inject_on_session_start={mem.inject_on_session_start}")
        if inject_realtime is not None:
            changed.append(f"inject_realtime={mem.inject_realtime}")
        if llm_family is not None:
            changed.append(f"llm_family={mem.llm_family}")
        if mlx_model is not None:
            changed.append(f"mlx_model={mem.mlx_model or '(global default)'}")
        if min_hours is not None:
            changed.append(f"min_hours={mem.min_hours}")
        if min_sessions is not None:
            changed.append(f"min_sessions={mem.min_sessions}")
        if retention_days is not None:
            changed.append(f"retention_days={mem.retention_days}")

        return (
            f"Memory settings updated: {', '.join(changed)}. "
            "Takes full effect on the next session build."
        )

    @tool
    def update_activity_settings(
        enabled: bool | None = None,
        interval_secs: int | None = None,
        retain_days: int | None = None,
        exclude_apps: list[str] | None = None,
        idle_threshold_secs: int | None = None,
    ) -> str:
        """Adjust the macOS activity tracker (foreground app, window title, browser URL)
        without restarting the backend.

        The tracker re-reads its config every poll cycle, so changes take effect on the
        next tick (within ``interval_secs``).  Toggling ``enabled`` simply pauses or
        resumes capture — the existing SQLite database is preserved either way.

        macOS Accessibility permission is required for the tracker to work.  Without
        it the tracker silently records empty rows; flip ``enabled`` and check
        ``GET /api/activity/status`` to see whether real data is landing.

        Args:
            enabled: Master on/off for the tracker. False pauses capture immediately;
                the next tick will skip recording and the timeline stops growing.
            interval_secs: Seconds between polls. Lower = finer granularity, higher CPU.
                Sensible range 5-60. The tracker enforces a floor of 2s internally so
                you can't accidentally pin a CPU.
            retain_days: How many days of activity to keep on disk.  Older rows are
                pruned by a daily cleanup pass.  ``0`` keeps everything forever.
            exclude_apps: List of app names that should NEVER be recorded.
                Case-insensitive substring match against the app's localised name —
                e.g. ``["1Password", "Keychain Access", "Banking"]``.  Pass an empty
                list to clear all exclusions; pass ``None`` to keep the current list.
            idle_threshold_secs: Skip ticks when no keyboard/mouse input has happened
                for this many seconds.  Prevents recording an 8-hour overnight span
                where the user wasn't actually present.  ``0`` disables idle detection.
        """
        if interval_secs is not None and interval_secs < 1:
            return "Error: interval_secs must be at least 1."
        if retain_days is not None and retain_days < 0:
            return "Error: retain_days must be non-negative (0 = keep forever)."
        if idle_threshold_secs is not None and idle_threshold_secs < 0:
            return "Error: idle_threshold_secs must be non-negative."
        if exclude_apps is not None:
            if not isinstance(exclude_apps, list):
                return "Error: exclude_apps must be a list of strings."
            cleaned: list[str] = []
            for x in exclude_apps:
                if not isinstance(x, str):
                    return "Error: every entry in exclude_apps must be a string."
                stripped = x.strip()
                if stripped:
                    cleaned.append(stripped)
            exclude_apps = cleaned

        cfg = AppConfig.load()
        a = cfg.activity

        if enabled is not None:
            a.enabled = enabled
        if interval_secs is not None:
            a.interval_secs = interval_secs
        if retain_days is not None:
            a.retain_days = retain_days
        if exclude_apps is not None:
            a.exclude_apps = exclude_apps
        if idle_threshold_secs is not None:
            a.idle_threshold_secs = idle_threshold_secs

        cfg.save()
        cfg.apply_to_environ()

        changed: list[str] = []
        if enabled is not None:
            changed.append(f"enabled={a.enabled}")
        if interval_secs is not None:
            changed.append(f"interval_secs={a.interval_secs}")
        if retain_days is not None:
            changed.append(f"retain_days={a.retain_days}")
        if exclude_apps is not None:
            changed.append(f"exclude_apps={a.exclude_apps}")
        if idle_threshold_secs is not None:
            changed.append(f"idle_threshold_secs={a.idle_threshold_secs}")

        if not changed:
            return (
                "No changes specified. Pass at least one parameter to update. "
                "Call get_settings to see the current activity configuration."
            )

        # The tracker reads cfg live on every tick (max delay = the OLD
        # interval_secs); no explicit reload required.
        return (
            f"Activity tracker updated: {', '.join(changed)}. "
            "Changes take effect on the next poll cycle."
        )

    @tool
    def toggle_mcp_server(server_id: str, enabled: bool) -> str:
        """Enable or disable an MCP tool server by its ID.

        When enabling a server, the backend will attempt to connect to it
        immediately so tools become available in this session. When disabling,
        the server is disconnected.

        Use get_settings to see the list of available server IDs and their
        current enabled state. Built-in servers can be toggled but not removed.

        Do not use this to add new MCP servers — use the mcp_builder tools
        for that.

        Args:
            server_id: The ID of the MCP server to toggle
                (e.g. "playwright-mcp", "agent-eval-service").
            enabled: True to enable and connect, False to disable and disconnect.
        """
        import asyncio

        from backend.mcp_manager import reset_circuit_breaker
        from backend.state import mcp_mgr, session_mgr

        cfg = AppConfig.load()
        server = next((s for s in cfg.mcp_servers if s.id == server_id), None)
        if server is None:
            available = [s.id for s in cfg.mcp_servers]
            return (
                f"Error: MCP server '{server_id}' not found. "
                f"Available: {available}"
            )

        if server.enabled == enabled:
            state = "enabled" if enabled else "disabled"
            return f"MCP server '{server_id}' is already {state}."

        server.enabled = enabled
        cfg.save()

        action = "enable" if enabled else "disable"
        try:
            loop = asyncio.new_event_loop()
            try:
                if enabled:
                    reset_circuit_breaker(server_id)
                    loop.run_until_complete(mcp_mgr.ensure_process(server))
                    loop.run_until_complete(mcp_mgr.connect(server, skip_process_start=True))
                else:
                    loop.run_until_complete(mcp_mgr.stop_process(server_id))
                    loop.run_until_complete(mcp_mgr.disconnect(server_id))
                loop.run_until_complete(session_mgr.refresh_tools(cfg))
            finally:
                loop.close()
        except Exception as exc:
            logger.warning("Failed to %s MCP server '%s': %s", action, server_id, exc)
            return (
                f"MCP server '{server_id}' config saved as {'enabled' if enabled else 'disabled'}, "
                f"but live reconnect failed: {exc}. The change will take full effect on the next "
                f"session build."
            )

        verb = "enabled and connected" if enabled else "disabled and disconnected"
        return f"MCP server '{server_id}' {verb} successfully."

    @tool
    def switch_model_provider(
        provider: str,
        model_name: str | None = None,
    ) -> str:
        """Switch the main chat LLM provider and optionally the model name.

        Use this to change what model the user is talking to. This is the
        correct tool when the user asks to "switch to MLX", "use the 35B model",
        "switch to Anthropic", etc.

        Do NOT use update_orchestrator_settings for this — that only changes
        the internal orchestrator subagent, not the main chat model.

        Call get_settings first to see the current provider and model.
        Credentials must already be configured in Settings; this tool does not set API keys.

        Args:
            provider: The LLM provider to switch to.
                "anthropic" — Anthropic API or AWS Bedrock (uses existing
                              credentials from Settings).
                "mlx"       — Local MLX inference (requires HF_LLM_MODEL_ID
                              to be set in Settings).
                "exo"       — Distributed exo cluster (cluster must be
                              running; use exo_start if needed).
            model_name: Optional model identifier for the new provider.
                For "anthropic": e.g. "claude-sonnet-4-6",
                    "us.anthropic.claude-sonnet-4-20250514-v1:0" (Bedrock).
                For "mlx": HuggingFace repo id, e.g.
                    "mlx-community/Qwen3-8B-4bit".
                For "exo": model id served by the cluster.
                Omit to keep the currently configured model for that provider.
        """
        if provider not in _VALID_LLM_PROVIDERS:
            return (
                f"Error: provider must be one of {sorted(_VALID_LLM_PROVIDERS)}, "
                f"got '{provider}'."
            )

        cfg = AppConfig.load()

        cfg.llm.provider = provider
        if model_name:
            if provider == "anthropic":
                cfg.llm.anthropic.model_name = model_name
            elif provider == "mlx":
                cfg.llm.mlx.hf_llm_model_id = model_name
            elif provider == "exo":
                cfg.exo.model_name = model_name
            elif provider == "omlx":
                cfg.omlx.model_name = model_name

        cfg.save()
        cfg.apply_to_environ()

        model_label = model_name or "(current model)"

        # EXO needs an explicit place_instance call before inference can route
        # to the model — saving the config is not enough.  Auto-trigger a
        # preload so the agent doesn't have to chain a second tool call (and
        # so a missing call doesn't silently fall back to the previously
        # loaded instance).
        if provider == "exo":
            import asyncio as _asyncio

            from backend import exo_provisioner as _ep

            target_model = cfg.exo.model_name
            if not target_model:
                return (
                    "Switched provider to 'exo' but no model is configured. "
                    "Pass model_name to load one, or call exo_load_model directly."
                )

            # Check if the target model is already loaded — if so, skip the
            # preload entirely.  ``_preload_model_sync`` defaults to
            # ``replace_existing=True`` which DELETEs any current instance
            # (including the same-model one we want to keep) before
            # re-placing it, so a redundant preload would needlessly tear
            # down and reload a working model.
            try:
                _loop = _asyncio.new_event_loop()
                try:
                    listing = _loop.run_until_complete(_ep.alist_models(cfg.exo))
                finally:
                    _loop.close()
                already_loaded = any(
                    m.get("id") == target_model and m.get("loaded")
                    for m in (listing.get("models") or [])
                )
            except Exception:
                already_loaded = False

            if already_loaded:
                return (
                    f"Switched to provider 'exo', model {model_label}. "
                    "Model is already loaded in cluster memory — ready for inference."
                )

            try:
                _loop = _asyncio.new_event_loop()
                try:
                    preload_result = _loop.run_until_complete(
                        _ep.apreload_model(
                            cfg.exo,
                            target_model,
                            timeout=1800.0,
                            min_nodes=int(cfg.exo.min_nodes or 1),
                        )
                    )
                finally:
                    _loop.close()
            except Exception as exc:
                logger.warning("exo preload after switch failed: %s", exc)
                return (
                    f"Switched config to provider 'exo', model {model_label}, "
                    f"but the cluster preload call failed: {exc}.\n"
                    "Call exo_load_model explicitly, or check exo_status / exo_tail_log."
                )

            if preload_result.get("ok"):
                elapsed = preload_result.get("elapsed_seconds", 0)
                return (
                    f"Switched to provider 'exo', model {model_label}. "
                    f"Loaded into cluster memory in {elapsed}s — ready for inference."
                )
            err = preload_result.get("error", "unknown error")
            return (
                f"Switched config to provider 'exo', model {model_label}, "
                f"but failed to load into cluster memory: {err}.\n"
                "The model is configured but inference will fail until it's "
                "loaded. Try exo_load_model with a longer timeout, or check "
                "exo_status / exo_tail_log."
            )

        return (
            f"Switched to provider '{provider}', model {model_label}. "
            "The change is saved — start a new session (or send the next message) "
            "to use the new model."
        )

    # -----------------------------------------------------------------------
    # Frontier models — curated list by active provider / model_provider
    # -----------------------------------------------------------------------

    # Anthropic direct API models (mid-2026 catalogue)
    _ANTHROPIC_MODELS = [
        {"id": "claude-opus-4-5", "name": "Claude Opus 4.5", "notes": "Most capable, highest cost"},
        {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6", "notes": "Balanced — recommended default"},
        {"id": "claude-haiku-3-5", "name": "Claude Haiku 3.5", "notes": "Fastest, lowest cost"},
        {"id": "claude-3-7-sonnet-20250219", "name": "Claude 3.7 Sonnet", "notes": "Extended thinking support"},
    ]

    # AWS Bedrock cross-region inference profile ids (us.* prefix)
    _BEDROCK_MODELS = [
        {"id": "us.anthropic.claude-opus-4-5-20250514-v1:0", "name": "Claude Opus 4.5 (Bedrock)"},
        {"id": "us.anthropic.claude-sonnet-4-20250514-v1:0", "name": "Claude Sonnet 4 (Bedrock)"},
        {"id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0", "name": "Claude 3.7 Sonnet (Bedrock)", "notes": "Extended thinking"},
        {"id": "us.anthropic.claude-3-5-haiku-20241022-v1:0", "name": "Claude 3.5 Haiku (Bedrock)"},
    ]

    # OpenAI models
    _OPENAI_MODELS = [
        {"id": "gpt-5", "name": "GPT-5", "notes": "Most capable"},
        {"id": "gpt-4o", "name": "GPT-4o", "notes": "Balanced — recommended default"},
        {"id": "gpt-4o-mini", "name": "GPT-4o mini", "notes": "Fast and cheap"},
        {"id": "o3", "name": "o3", "notes": "Extended reasoning"},
        {"id": "o4-mini", "name": "o4-mini", "notes": "Fast reasoning"},
    ]

    @tool
    def list_frontier_models() -> str:
        """List available frontier (cloud) model IDs for the currently configured
        provider and model_provider sub-selection.

        Returns the model IDs you can pass to switch_model_provider or
        update_orchestrator_settings.  The currently active model is marked
        with an asterisk (*).

        This is a curated catalogue — it does not require a live API call.
        """
        cfg = AppConfig.load()
        provider = cfg.llm.provider
        a = cfg.llm.anthropic
        o = cfg.llm.openai

        if provider == "anthropic":
            sub = a.model_provider  # "anthropic" | "bedrock" | "anthropic_bedrock"
            if sub in ("bedrock", "anthropic_bedrock"):
                models = _BEDROCK_MODELS
                active = a.model_name
            else:
                models = _ANTHROPIC_MODELS
                active = a.model_name
            label = f"Anthropic ({sub})"
        elif provider == "openai":
            models = _OPENAI_MODELS
            active = o.model_name
            label = "OpenAI"
        else:
            return (
                f"Provider '{provider}' is not a frontier provider (mlx / exo use "
                "list_mlx_models / list_exo_models instead)."
            )

        rows = []
        for m in models:
            marker = " *" if m["id"] == active else "  "
            notes = f"  — {m['notes']}" if m.get("notes") else ""
            rows.append(f"{marker}{m['id']}  ({m['name']}){notes}")

        return (
            f"{label} models (* = currently active):\n"
            + "\n".join(rows)
            + "\n\nPass the id to switch_model_provider(provider='"
            + ("anthropic" if provider == "anthropic" else "openai")
            + "', model_name='<id>') to switch."
        )

    # -----------------------------------------------------------------------
    # MLX — locally downloaded models in the HF Hub cache
    # -----------------------------------------------------------------------

    @tool
    def list_mlx_models() -> str:
        """List MLX models already downloaded to the local Hugging Face Hub cache.

        Shows the currently active main chat LLM (marked [ACTIVE CHAT MODEL]) and
        VLM (marked [ACTIVE VLM]).

        To switch the main chat model to one of these, call:
            switch_model_provider(provider='mlx', model_name='<repo_id>')

        Do NOT use update_orchestrator_settings for this — that only changes the
        orchestrator subagent's model, not the main chat LLM.
        """
        from pathlib import Path

        cfg = AppConfig.load()
        mlx = cfg.llm.mlx
        active_llm = mlx.hf_llm_model_id
        active_vlm = mlx.hf_vlm_model_id

        from backend.mlx_hub_paths import resolve_hf_hub_cache_dir
        hub = resolve_hf_hub_cache_dir(mlx.hf_hub_cache)
        hub_path = Path(hub)

        if not hub_path.is_dir():
            return f"Hub cache directory does not exist: {hub}\nNo models downloaded yet."

        try:
            from huggingface_hub import scan_cache_dir
            info = scan_cache_dir(hub_path)
        except Exception as exc:
            return f"Failed to scan Hub cache: {exc}"

        bookmark_labels = {b.repo_id: b.label for b in mlx.mlx_bookmarks if b.label}
        rows = []
        for repo in sorted(info.repos, key=lambda r: r.repo_id.lower()):
            rid = repo.repo_id
            size_gb = round(repo.size_on_disk / (1024 ** 3), 1) if repo.size_on_disk else 0
            label = bookmark_labels.get(rid, "")
            markers = []
            if rid == active_llm:
                markers.append("ACTIVE CHAT MODEL")
            if rid == active_vlm:
                markers.append("ACTIVE VLM")
            flag = f"  <-- {', '.join(markers)}" if markers else ""
            name_part = f"  ({label})" if label else ""
            rows.append(f"  {rid}{name_part}  {size_gb} GB{flag}")

        if not rows:
            return (
                f"No models found in Hub cache: {hub}\n"
                "Download one from the On-Device page or ask the agent to start a download."
            )

        # Active model summary at the top so it's visible even if the list is long.
        active_summary = f"Active chat model : {active_llm}\n"
        if active_vlm:
            active_summary += f"Active VLM        : {active_vlm}\n"
        active_in_cache = any(r.repo_id == active_llm for r in info.repos)
        if not active_in_cache:
            active_summary += f"WARNING: active model '{active_llm}' is NOT in the local cache — it may be missing or misconfigured.\n"
        active_summary += "\n"

        header = f"MLX models in Hub cache ({hub}):\n"
        return active_summary + header + "\n".join(rows)

    # -----------------------------------------------------------------------
    # EXO — cluster model catalogue with downloaded / loaded flags
    # -----------------------------------------------------------------------

    @tool
    def list_exo_models(show_all: bool = False) -> str:
        """List models available on the exo distributed inference cluster.

        By default shows only models that have been downloaded to at least one
        node. Pass show_all=True to see the full ~120-model catalogue including
        models not yet on disk.

        Each model is flagged:
          [loaded]     — an instance is in cluster memory right now
          [downloaded] — weights are on disk (ready to load instantly)
          (neither)    — in the catalogue but not yet downloaded

        The cluster must be running; call exo_start if it is not reachable.
        """
        import asyncio

        from backend import exo_provisioner as ep

        cfg = AppConfig.load().exo
        loop = asyncio.new_event_loop()
        try:
            data = loop.run_until_complete(ep.alist_models(cfg))
        finally:
            loop.close()

        if not data.get("reachable"):
            err = data.get("error", "unknown error")
            return (
                f"EXO cluster is not reachable ({err}).\n"
                "Call exo_start to bring the cluster up, then retry."
            )

        active_model = cfg.model_name
        models = data.get("models") or []

        if not show_all:
            models = [m for m in models if m.get("downloaded") or m.get("loaded")]

        if not models:
            if show_all:
                return "Cluster is reachable but returned an empty model catalogue."
            return (
                "No models downloaded on the cluster yet.\n"
                "Call list_exo_models(show_all=True) to see the full catalogue, "
                "or load one from the EXO page in the app."
            )

        rows = []
        for m in sorted(models, key=lambda x: (not x.get("loaded"), not x.get("downloaded"), x.get("id", ""))):
            mid = m.get("id", "?")
            name = m.get("name") or mid
            flags = []
            if m.get("loaded"):
                flags.append("loaded")
            if m.get("downloaded"):
                flags.append("downloaded")
            active_marker = " *" if mid == active_model else ""
            flag_str = f"  [{', '.join(flags)}]" if flags else ""
            name_str = f"  ({name})" if name != mid else ""
            rows.append(f"  {mid}{active_marker}{name_str}{flag_str}")

        total = len(data.get("models") or [])
        shown = len(models)
        footer = "" if show_all else f"\n\n({shown} downloaded/{total} total — pass show_all=True to see full catalogue)"

        return (
            "EXO cluster models (* = configured active model):\n"
            + "\n".join(rows)
            + footer
            + "\n\nPass the id to switch_model_provider(provider='exo', model_name='<id>') to activate."
        )

    _VALID_SHARDING = frozenset({"Pipeline", "Tensor"})
    _VALID_INSTANCE_META = frozenset({"MlxRing", "MlxJaccl"})

    @tool
    def update_exo_cluster_params(
        sharding: str | None = None,
        instance_meta: str | None = None,
        min_nodes: int | None = None,
    ) -> str:
        """Adjust exo cluster placement parameters that govern how a model is
        split and communicated across nodes.

        Changes are persisted immediately. They take effect the next time a
        model is placed (loaded) onto the cluster — an already-loaded instance
        keeps its current strategy until reloaded. Call exo_load_model after
        changing these to apply them to the active model right away.

        Call get_settings first to see the current values.

        Args:
            sharding: How the model is split across nodes.
                "Pipeline" (default) — layers split across devices; best
                    single-request latency; works for every model.
                "Tensor" — each layer split across devices; up to 1.8x (2
                    nodes) / 3.2x (4 nodes) throughput for concurrent
                    requests, but only tensor-capable models support it.
                    Only beneficial when running multiple requests in parallel.
            instance_meta: The inter-node collective / transport backend.
                "MlxRing" (default) — ring all-reduce; works over any
                    network; universally compatible.
                "MlxJaccl" — lower latency but requires Thunderbolt 5 /
                    RDMA hardware (M4 Pro/Max, macOS 26.2+). Only set this
                    on a qualified Thunderbolt cluster — it will fail on
                    ordinary network connections.
            min_nodes: Minimum nodes that must hold a shard of a placed
                instance. 1 (default) = let the scheduler pick the cheapest
                single-node placement. Set 2+ to force a multi-node split
                even when the model fits on one machine.
        """
        if sharding is not None and sharding not in _VALID_SHARDING:
            return (
                f"Error: sharding must be one of {sorted(_VALID_SHARDING)}, "
                f"got '{sharding}'."
            )
        if instance_meta is not None and instance_meta not in _VALID_INSTANCE_META:
            return (
                f"Error: instance_meta must be one of {sorted(_VALID_INSTANCE_META)}, "
                f"got '{instance_meta}'."
            )
        if min_nodes is not None and min_nodes < 1:
            return "Error: min_nodes must be at least 1."

        cfg = AppConfig.load()

        if sharding is not None:
            cfg.exo.sharding = sharding
        if instance_meta is not None:
            cfg.exo.instance_meta = instance_meta
        if min_nodes is not None:
            cfg.exo.min_nodes = min_nodes

        cfg.save()
        cfg.apply_to_environ()

        changed: list[str] = []
        if sharding is not None:
            changed.append(f"sharding={cfg.exo.sharding}")
        if instance_meta is not None:
            changed.append(f"instance_meta={cfg.exo.instance_meta}")
        if min_nodes is not None:
            changed.append(f"min_nodes={cfg.exo.min_nodes}")

        return (
            f"EXO cluster params updated: {', '.join(changed)}. "
            "Takes effect on the next model placement — call exo_load_model "
            "to reload the active model with the new strategy."
        )

    @tool
    def toggle_ambient_scheduling(enabled: bool) -> str:
        """Enable or disable ambient scheduling suggestions.

        When enabled, Otto will notice at the end of responses when a task looks
        repeatable and ask whether the user would like to automate it — offering
        to create a cron schedule, set a file/event trigger, or run it once more
        at a later time.  Otto uses its scheduling tools directly when the user
        says yes, with no extra steps.

        The setting is opt-in (off by default) so it never interrupts users who
        haven't asked for it.  It only applies to interactive sessions; scheduled
        and trigger-spawned runs are never prompted.

        The change is persisted immediately and takes effect on the next session
        (the current session's system prompt was already built at startup).

        Args:
            enabled: True to turn on ambient scheduling suggestions, False to
                turn them off.
        """
        cfg = AppConfig.load()

        if cfg.ambient_suggest_recurrence == enabled:
            state = "already enabled" if enabled else "already disabled"
            return f"Ambient scheduling is {state}."

        cfg.ambient_suggest_recurrence = enabled
        cfg.save()

        if enabled:
            return (
                "Ambient scheduling enabled. "
                "Starting from your next session, Otto will ask at the end of "
                "repeatable-task responses whether you'd like to automate them."
            )
        return (
            "Ambient scheduling disabled. "
            "Otto will no longer suggest automating tasks at the end of responses."
        )

    return [
        get_settings,
        update_orchestrator_settings,
        update_generation_params,
        update_exo_cluster_params,
        update_memory_settings,
        update_activity_settings,
        toggle_mcp_server,
        switch_model_provider,
        list_frontier_models,
        list_mlx_models,
        list_exo_models,
        toggle_ambient_scheduling,
    ]
