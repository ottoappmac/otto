"""Process-wide MLX state shared across chat model implementations.

This module centralises the globals that both the existing
:mod:`chat_models.mlx.chat_mlx_text` path and the upcoming turbo path
(:mod:`chat_models.mlx_turbo`) must agree on:

* :data:`MLX_GEN_LOCK` — serialises every Metal ``stream_generate`` call to
  avoid the ``command encoder is already encoding`` assertion.
* :data:`_LOADED_MODELS` — cache of ``(model_path, draft_path) -> (model,
  tokenizer, draft_model)`` triples so additional ``ChatMLXText`` instances
  (and, later, ``TurboMLXChat``) re-use already-loaded weights instead of
  re-pulling them into unified memory.
* :data:`_WARMED_UP` — set of cache keys that have already paid the graph
  compilation cost (see ``ChatMLXText._warmup``).
* :func:`_load_or_reuse` — thread-safe factory for the triple.
* :func:`loaded_mlx_models` — introspection helper for diagnostics.

Nothing here changes behaviour on its own; it's a pure refactor so that a
future turbo engine can share the same lock and weight cache.  The original
names remain re-exported from :mod:`chat_models.mlx.chat_mlx_text` for
backward-compatibility with modules that do
``from chat_models.mlx.chat_mlx_text import MLX_GEN_LOCK``.
"""

from __future__ import annotations

import threading
from typing import Any, List, Optional, Tuple


# ── Shared model registry ─────────────────────────────────────────────────────
#
# MLX model weights are the dominant cost of a chat model instance: a 4-bit
# Qwen3-Coder-30B occupies ~17 GB of unified Metal memory.  Without a cache,
# every new chat session, every ``refresh_tools`` call, and every
# memory-relevance ranking turn would call ``mlx_lm.load()`` and pull a fresh
# copy of the same weights into Metal — quickly blowing past the GPU memory
# budget on machines running more than one session.

_ModelTriple = Tuple[Any, Any, Optional[Any]]
_LOADED_MODELS: dict[Tuple[str, Optional[str]], _ModelTriple] = {}
_WARMED_UP: set[Tuple[str, Optional[str]]] = set()
_LOAD_LOCK = threading.Lock()


# ── Process-wide MLX generation lock ──────────────────────────────────────────
#
# Apple's Metal API does NOT permit two threads to encode commands into the
# same command buffer simultaneously — it aborts the process with::
#
#     [AGXG16XFamilyCommandBuffer tryCoalescingPreviousComputeCommandEncoder…]:
#         failed assertion `A command encoder is already encoding to this
#         command buffer'
#
# All MLX work (text + VLM, every loaded model) runs through the same Metal
# device and the same default ``mx.gpu`` stream, so the encoder collision can
# happen across *any* pair of concurrent ``stream_generate`` calls — not just
# two callers sharing the same model.  Two sessions running tools that each
# trigger an LLM turn at roughly the same time is enough to crash the process.
#
# We serialise generation explicitly with a process-wide lock.  Since the GPU
# already serialises GPU work end-to-end, this lock costs nothing in throughput
# — it just makes the queueing happen at the Python layer instead of inside
# Metal's encoder, which avoids the assertion.
#
# ``MLX_GEN_LOCK`` is shared by both the classic (``ChatMLXText`` /
# ``MLXVLChatModel``) and turbo paths so the two can safely coexist in the
# same process.

MLX_GEN_LOCK = threading.Lock()


# ── Loop-recovery temperature bump ────────────────────────────────────────────
#
# When a :class:`tools.loop_guard.ToolLoopGuard` detects the model re-emitting
# an identical tool call, greedy decoding (``temp == 0.0``) is frequently the
# proximate cause: the same logits yield the same argmax token sequence every
# turn, so the model can never escape the loop on its own.  Evicting the KV
# prefix cache (see :mod:`chat_models.mlx_turbo._registry`) is one half of the
# recovery; the other half is to *perturb sampling* for the immediate recovery
# turn(s) so the model explores a different action instead of re-deriving the
# same one.
#
# This holder lets a provider-agnostic loop guard request a one-shot
# temperature override that the next MLX generation(s) honour and then clear,
# with zero import coupling between the two modules.  It is best-effort and
# process-wide, mirroring :func:`evict_all_mlx_models` /
# ``mlx_turbo._registry.evict_all``: it perturbs whichever MLX chat instance
# runs next.  Non-MLX providers never call :func:`consume_temperature_bump`, so
# the request is a harmless no-op for them.

_RECOVERY_LOCK = threading.Lock()
_recovery_temp: float = 0.0
_recovery_turns: int = 0


def request_temperature_bump(temp: float, turns: int = 1) -> None:
    """Ask the next *turns* MLX generations to sample at no less than *temp*.

    Called by the loop guard's recovery path.  Overlapping requests take the
    stronger temperature and the longer remaining-turn count so concurrent
    trips can't weaken each other.  A ``temp <= 0`` or ``turns <= 0`` request
    is ignored.
    """
    global _recovery_temp, _recovery_turns
    if temp <= 0.0 or turns <= 0:
        return
    with _RECOVERY_LOCK:
        _recovery_temp = max(_recovery_temp, float(temp))
        _recovery_turns = max(_recovery_turns, int(turns))


def consume_temperature_bump() -> Optional[float]:
    """Return the pending recovery temperature for this generation, or ``None``.

    Decrements the remaining-turn counter so the bump auto-expires after the
    configured number of generations.  Called once per MLX generation from
    ``ChatMLXText._sampler_kwargs``.
    """
    global _recovery_temp, _recovery_turns
    with _RECOVERY_LOCK:
        if _recovery_turns <= 0:
            return None
        _recovery_turns -= 1
        temp = _recovery_temp
        if _recovery_turns == 0:
            _recovery_temp = 0.0
        return temp


def _resolve_local_path(model_path: str) -> str:
    """Resolve *model_path* to an absolute local filesystem path.

    For HuggingFace repo IDs (``owner/name`` form) this looks up the local
    Hub cache snapshot and returns the on-disk directory so that callers can
    pass a local path directly to ``mlx_lm.load``.  Passing a local path to
    mlx_lm bypasses its internal ``snapshot_download`` call entirely — if any
    weight file is missing, mlx_lm raises ``FileNotFoundError`` immediately
    instead of silently starting a network download.

    Raises ``FileNotFoundError`` when:
    * the repo is not present in the local Hub cache at all, or
    * the cached snapshot contains no ``.safetensors`` / ``.npz`` weight files
      (i.e. the download was interrupted before the weights arrived).
    """
    from pathlib import Path as _Path

    if _Path(model_path).exists():
        return model_path  # already a local filesystem path — pass through

    if "/" not in model_path:
        return model_path  # single-segment id (unusual), let mlx_lm handle it

    try:
        from huggingface_hub import snapshot_download as _sd

        local = _sd(repo_id=model_path, local_files_only=True)
    except Exception:
        raise FileNotFoundError(
            f"MLX model '{model_path}' is not in the local Hub cache. "
            "Download it from the model catalog first."
        ) from None

    # Guard against partial downloads: config.json arrives first, weights last.
    local_path = _Path(local)
    has_weights = (
        any(local_path.glob("*.safetensors"))
        or any(local_path.glob("*.npz"))
    )
    if not has_weights:
        raise FileNotFoundError(
            f"MLX model '{model_path}' has no weight files in the local cache "
            "(download may be incomplete). "
            "Download it fully from the model catalog first."
        )

    return local


def _load_or_reuse(
    model_path: str,
    draft_model_path: Optional[str],
) -> Tuple[_ModelTriple, bool]:
    """Return cached ``(model, tokenizer, draft_model)`` or load on miss.

    Returns ``(triple, freshly_loaded)`` so the caller can gate one-time
    side effects (logging, warmup) on the first load.  Thread-safe under
    the initial-load race via double-checked locking.
    """
    key = (model_path, draft_model_path)
    cached = _LOADED_MODELS.get(key)
    if cached is not None:
        return cached, False

    with _LOAD_LOCK:
        cached = _LOADED_MODELS.get(key)
        if cached is not None:
            return cached, False

        # Resolve repo IDs to absolute local paths before calling load().
        # Passing a local path to mlx_lm.load skips its internal
        # snapshot_download entirely, so if any weight file is missing the call
        # raises FileNotFoundError immediately instead of triggering a download.
        resolved_path = _resolve_local_path(model_path)
        resolved_draft_path = (
            _resolve_local_path(draft_model_path) if draft_model_path else None
        )

        from mlx_lm import load  # lazy import — only required on Apple Silicon

        model, tokenizer = load(resolved_path)
        draft_model: Optional[Any] = None
        if resolved_draft_path:
            draft_model, draft_tokenizer = load(resolved_draft_path)
            if draft_tokenizer.vocab_size != tokenizer.vocab_size:
                raise ValueError(
                    f"Draft model vocab size ({draft_tokenizer.vocab_size}) "
                    f"does not match main model ({tokenizer.vocab_size})"
                )
        triple = (model, tokenizer, draft_model)
        _LOADED_MODELS[key] = triple
        return triple, True


def loaded_mlx_models() -> List[Tuple[str, Optional[str]]]:
    """Return the list of currently cached ``(model_path, draft_path)`` keys.

    Exposed for diagnostics / settings UI.  Not part of the hot path.
    """
    return list(_LOADED_MODELS.keys())


def evict_all_mlx_models() -> int:
    """Drop every cached ``(model, tokenizer, draft_model)`` triple.

    This is the *real* weight cache that keeps MLX models resident in
    unified Metal memory for the life of the process (a 4-bit 30B model is
    ~17 GB).  Calling :func:`loaded_mlx_models` after this returns an empty
    list.

    NOTE: clearing this dict only drops *this module's* strong references to
    the weights.  Any live ``ChatMLXText`` / ``TurboMLXChat`` instance still
    holds its own reference (``self._model``) and keeps the weights alive
    until it too is released (e.g. when ``refresh_tools`` rebuilds a session
    graph onto a different provider).  Callers that want memory actually
    returned to the OS must therefore (1) drop those instances, (2) call
    this, and only then (3) ``gc.collect()`` + ``mx.clear_cache()``.

    Returns the number of cache entries that were evicted (diagnostics).
    """
    with _LOAD_LOCK:
        count = len(_LOADED_MODELS)
        _LOADED_MODELS.clear()
        _WARMED_UP.clear()
    return count
