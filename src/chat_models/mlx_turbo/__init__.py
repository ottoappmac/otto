"""Opt-in oMLX-derived turbo path for local MLX inference.

Enabled via ``MlxHfConfig.turbo_level`` in Settings → LLM.  The agents
factory (``deep_agent.model_factory._build_mlx_chat``) reads the config
and calls :func:`build_turbo_chat` when ``turbo_level != "off"``; any
failure here (e.g. ``mlx_lm`` missing on a non-Apple host) is caught
upstream and falls back to the classic :class:`ChatMLXText` path, so
flipping the dropdown is always safe.

Supported levels
----------------

``basic``
    Every MLX GPU operation runs on the process-wide single-thread
    executor (:mod:`chat_models.mlx_turbo._executor`).  Gives FIFO
    ordering on Metal and removes the per-call lock dance classic uses.

``cache``
    ``basic`` plus cross-session KV prefix sharing: TurboMLXChat is
    instantiated as a singleton per ``(model_path, draft_path, …)``
    key via :mod:`chat_models.mlx_turbo._registry`, with prompt +
    system-prefix caching force-enabled so the shared instance's KV
    cache carries state across sessions.

``ssd``
    ``cache`` plus a disk-backed cold tier
    (:class:`chat_models.mlx_turbo._ssd_cache.SSDPrefixStore`).  The
    in-memory singleton is primed from disk on its first request, and
    every successful turn writes the updated cache back to disk keyed
    by the prompt token hash.  Fresh processes therefore reuse the
    system-prompt prefill paid by earlier runs, and the on-disk budget
    is capped at ``turbo_ssd_max_gb`` with a global LRU eviction.

Higher levels (``max``) are not yet implemented; the UI hides those
options and the factory raises ``ValueError`` if asked for them.
"""

from __future__ import annotations

import logging
from typing import Any

from chat_models.mlx_turbo.chat import TurboMLXChat

logger = logging.getLogger(__name__)

__all__ = ["TurboMLXChat", "build_turbo_chat", "SUPPORTED_LEVELS"]


SUPPORTED_LEVELS = ("basic", "cache", "ssd")


def build_turbo_chat(
    *,
    turbo_level: str,
    model_path: str,
    draft_model_path: str = "",
    num_draft_tokens: int = 3,
    max_tokens: int = 8192,
    thinking: bool = False,
    enable_prompt_cache: bool = False,
    enable_system_prompt_cache: bool = False,
    kv_bits: Any = None,
    kv_group_size: int = 64,
    repetition_penalty: float = 1.1,
    prompt_cache_max_tokens: int = 32768,
    turbo_ssd_dir: str = "",
    turbo_ssd_max_gb: int = 50,
) -> TurboMLXChat:
    """Construct (or fetch) a :class:`TurboMLXChat` for *turbo_level*.

    Raises :class:`ValueError` for unsupported levels so the caller's
    factory surface ("try turbo, fall back to classic") catches it as
    an explicit, actionable error rather than a silent mis-dispatch.
    """
    if turbo_level not in SUPPORTED_LEVELS:
        raise ValueError(
            f"Unsupported turbo level {turbo_level!r}. "
            f"Implemented levels: {SUPPORTED_LEVELS}. "
            f"The classic path will be used instead."
        )

    # Normalise the optional draft id so None and "" hash the same
    # singleton key — otherwise cache mode would end up with two
    # identically-configured singletons, defeating the prefix share.
    draft = draft_model_path or None

    init_kwargs: dict[str, Any] = {
        "model_path": model_path,
        "draft_model_path": draft,
        "num_draft_tokens": num_draft_tokens,
        "max_tokens": max_tokens,
        "thinking": thinking,
        "enable_prompt_cache": enable_prompt_cache,
        "enable_system_prompt_cache": enable_system_prompt_cache,
        "kv_bits": kv_bits,
        "kv_group_size": kv_group_size,
        "repetition_penalty": repetition_penalty,
        "prompt_cache_max_tokens": prompt_cache_max_tokens,
        "turbo_level": turbo_level,
        "turbo_ssd_dir": turbo_ssd_dir,
        "turbo_ssd_max_gb": turbo_ssd_max_gb,
    }

    # Both cache-and-above levels reuse a process-wide singleton so the
    # in-memory KV state is genuinely shared across sessions.  ``ssd``
    # layers a disk-backed cold tier on top, but the hot path is still
    # the same shared instance — otherwise a fresh session would miss
    # the in-memory prefix in the common case even though the SSD cache
    # already knows about it.
    if turbo_level in ("cache", "ssd"):
        # Force-enable the trim-to-prefix logic so the shared instance
        # actually reuses KV state across sessions; a user leaving the
        # individual cache toggles off in Settings would otherwise make
        # cache-mode a no-op versus basic.
        init_kwargs["enable_prompt_cache"] = True
        init_kwargs["enable_system_prompt_cache"] = True

        from chat_models.mlx_turbo._registry import get_or_create

        # Key on every tuning knob that would change observable behaviour.
        # turbo_ssd_dir and turbo_ssd_max_gb are included so that updating
        # the SSD directory or budget in Settings immediately creates a new
        # singleton with the correct on-disk store; without them the old
        # singleton (and its old SSD path) would be reused until restart.
        key = (
            turbo_level,
            model_path,
            draft,
            num_draft_tokens,
            max_tokens,
            thinking,
            kv_bits,
            kv_group_size,
            round(repetition_penalty, 4),
            turbo_ssd_dir or "",
            turbo_ssd_max_gb,
        )
        logger.info(
            "Turbo %s mode: fetching / creating singleton for %s%s",
            turbo_level,
            model_path,
            f" (draft={draft})" if draft else "",
        )
        return get_or_create(key, lambda: TurboMLXChat(**init_kwargs))

    # basic: fresh instance per call.  Weight tensors are still shared
    # via chat_models.mlx._shared._LOADED_MODELS, so this is cheap.
    logger.info(
        "Turbo basic mode: constructing TurboMLXChat for %s%s",
        model_path,
        f" (draft={draft})" if draft else "",
    )
    return TurboMLXChat(**init_kwargs)
