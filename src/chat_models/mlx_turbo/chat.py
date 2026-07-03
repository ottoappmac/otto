"""TurboMLXChat — opt-in MLX chat model with oMLX-derived optimisations.

Inherits every behaviour of :class:`chat_models.mlx.ChatMLXText` (native
tool calling, ReAct fallback, speculative decoding, KV quantisation,
thinking, prefix-trim caching, per-turn metadata) and layers three
opt-in features on top:

* **Single-thread executor** — all MLX work (warmup + generation) is
  dispatched onto :func:`_executor.get_mlx_executor`, so turbo requests
  are strictly FIFO on Metal.  Classic ChatMLXText / VLM instances in
  the same process still run on their caller threads and coordinate
  with turbo via :data:`chat_models.mlx._shared.MLX_GEN_LOCK` — no
  behaviour change for the classic path.

* **Cross-session prefix cache** (``turbo_level == "cache"``) — the
  :mod:`chat_models.mlx_turbo._registry` hands out a process-wide
  singleton per ``(model_path, draft_path, …)`` key.  With
  ``enable_prompt_cache=True`` + ``enable_system_prompt_cache=True``
  force-enabled, the inherited ``_find_common_prefix`` / ``_trim_cache_to``
  logic in :class:`ChatMLXText` now spans sessions: the first session
  pays the system-prompt prefill, every subsequent session re-uses the
  KV cache for that prefix.

* **SSD cold-tier cache** (``turbo_level == "ssd"``) — the "cache" tier
  plus :class:`chat_models.mlx_turbo._ssd_cache.SSDPrefixStore`.  After
  every successful generation we persist the current prompt cache to
  disk keyed by the sha256 of the prompt tokens; on a cold start (new
  process, or first request to a model the in-memory singleton has
  never seen) we probe the SSD store for the longest matching prefix
  and prime the in-memory cache from it before falling into the
  standard prefix-trim path.  Net effect: the expensive system-prompt
  prefill is paid once per machine, not once per process.

Anything a future level needs (paged allocator, TurboQuant attention)
plugs in here as additional overrides.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.outputs import ChatResult

from chat_models.mlx._shared import MLX_GEN_LOCK
from chat_models.mlx.chat_mlx_text import ChatMLXText
from chat_models.mlx_turbo._executor import run_on_mlx_thread

logger = logging.getLogger(__name__)


class TurboMLXChat(ChatMLXText):
    """MLX chat model that runs every GPU op on the turbo executor.

    All new Pydantic fields must be declared with defaults so a
    ``ChatMLXText.__init__(model_path=…)`` super call doesn't require
    them; ``turbo_level`` is informational for logging and metadata.
    """

    turbo_level: str = "basic"

    # SSD-tier configuration — all ignored when turbo_level != "ssd".
    turbo_ssd_dir: str = ""
    turbo_ssd_max_gb: int = 50

    # Private runtime state for the SSD tier.  ``_ssd_store`` is the
    # live SSDPrefixStore instance (or None for non-ssd levels),
    # ``_ssd_primed`` tracks whether we've already tried to warm the
    # in-memory cache from disk for the current singleton.  A bool
    # rather than a "did load" flag on purpose: we only want to probe
    # the disk once per singleton lifetime.  On subsequent turns the
    # normal in-memory prefix-trim path is faster and handles
    # incremental reuse correctly.
    _ssd_store: Any = None
    _ssd_primed: bool = False

    def __init__(self, model_path: str, **kwargs: Any) -> None:
        super().__init__(model_path=model_path, **kwargs)
        if self.turbo_level == "ssd":
            self._init_ssd_store()

    # ── SSD store lifecycle ───────────────────────────────────────────

    def _init_ssd_store(self) -> None:
        """Construct the on-disk KV store for the ``ssd`` turbo level.

        Failures are demoted to a warning (and the store is left unset)
        so an unwritable cache directory never breaks generation — the
        chat model keeps working as though turbo were in "cache" mode.
        """
        try:
            from chat_models.mlx_turbo._ssd_cache import (
                SSDPrefixStore,
                compute_model_fingerprint,
                kv_signature,
                resolve_root,
            )

            root = resolve_root(self.turbo_ssd_dir)
            fingerprint = compute_model_fingerprint(self.model_path)
            kv_sig = kv_signature(self.kv_bits, self.kv_group_size)
            max_bytes = max(1, int(self.turbo_ssd_max_gb)) * (1024 ** 3)

            self._ssd_store = SSDPrefixStore(
                global_root=root,
                model_path=self.model_path,
                model_fingerprint=fingerprint,
                kv_sig=kv_sig,
                max_bytes=max_bytes,
            )
            logger.info(
                "Turbo ssd mode: SSD KV store active for %s at %s",
                self.model_path, self._ssd_store.global_root,
            )
        except Exception as exc:
            # Non-fatal: the chat will behave like the "cache" level.
            # We log at WARNING so operators notice, but don't abort —
            # the model factory already has a try/except that would
            # fall back to the classic path, which would be a worse
            # user experience than just skipping the SSD tier.
            logger.warning(
                "Turbo ssd init failed (%s) — continuing without SSD "
                "tier; falling back to cache-only behaviour.",
                exc,
            )
            self._ssd_store = None

    # ── Warmup / generation dispatch ──────────────────────────────────

    def _warmup(self) -> None:
        """Run the inherited warmup on the executor thread.

        Keeping warmup on the same thread that will do generation means
        the first real user turn pays zero graph-tracing cost and any
        per-thread Metal state warmed here stays hot.
        """
        run_on_mlx_thread(ChatMLXText._warmup, self)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        # ``run_manager`` (LangChain callbacks) is not used by the
        # inherited implementation, and callbacks should fire on the
        # caller's thread — not the executor thread — so pass ``None``
        # through to the parent call.  We capture ``tools`` before
        # dispatch because the parent pops it from ``kwargs`` during
        # generation and we need it to re-render the system-only prefix
        # at save time (so tool-aware templates stamp the same tool
        # section into the save key as the live prompt).
        tools = kwargs.get("tools")

        def _work() -> ChatResult:
            self._maybe_prime_from_ssd(messages, **kwargs)
            result = ChatMLXText._generate(
                self, messages, stop=stop, run_manager=None, **kwargs,
            )
            self._maybe_save_to_ssd(messages, tools=tools)
            return result

        return run_on_mlx_thread(_work)

    # ── SSD cold-tier hooks ───────────────────────────────────────────

    def _maybe_prime_from_ssd(
        self,
        messages: List[BaseMessage],
        **kwargs: Any,
    ) -> None:
        """Warm the in-memory prompt cache from disk on first request.

        Only fires once per singleton lifetime; subsequent turns rely on
        the inherited prefix-trim path which is strictly faster than
        reading a fresh snapshot from SSD.  A lookup hit loads the
        longest previously-saved prefix of the upcoming prompt tokens,
        replaces ``self._prompt_cache`` with it, and stamps
        ``_last_prompt_tokens`` so the parent ``_generate`` picks up
        the common-prefix path for free.
        """
        if (
            self.turbo_level != "ssd"
            or self._ssd_store is None
            or self._ssd_primed
            or self._prompt_cache is None
        ):
            return
        self._ssd_primed = True

        try:
            # Reuse the parent's prompt construction + tokenisation so
            # the hash we probe with is byte-identical to what parent
            # ``_generate`` will encode in a moment.  Don't execute the
            # tokenizer twice on the hot path — we stash the tokens
            # back onto ``_last_prompt_tokens`` if we prime, which
            # makes parent's ``_find_common_prefix`` return the full
            # loaded length immediately.
            tools = kwargs.get("tools")
            prompt_str = self._to_prompt(messages, tools=tools)
            tokens = self._tokenizer.encode(prompt_str)
        except Exception as exc:
            logger.debug("SSD prime: tokenisation failed (%s)", exc)
            return

        match = self._ssd_store.find_longest_match(tokens)
        if match is None:
            # INFO (not DEBUG): a miss here is the single most useful
            # signal for diagnosing "why isn't SSD helping?" — e.g. the
            # save path saved a full-prompt key that no future query
            # could match, or the kv/fingerprint namespace flipped.
            logger.info(
                "SSD prime: no saved prefix for %s (%d prompt tokens)",
                self.model_path, len(tokens),
            )
            return

        prefix_len, path = match
        try:
            with MLX_GEN_LOCK:
                cache, _meta = self._ssd_store.load(path)
        except Exception as exc:
            logger.warning(
                "SSD prime: load failed (%s) for %s — ignoring.",
                exc, path,
            )
            return

        if not cache:
            return

        # Replace the in-memory cache with the one we just read back.
        # The cache's KV offset equals the saved token length (with
        # potentially trailing generated tokens — we trim to prefix_len
        # below so the parent ``_generate`` sees a clean state).
        self._prompt_cache = cache
        self._last_prompt_tokens = list(tokens[:prefix_len])

        # If the saved snapshot extends past the prompt prefix (because
        # we saved after a full turn that included generated tokens),
        # roll it back to the prefix length.  This is the standard
        # prompt-cache trim path; it's cheap when the cache layer type
        # supports trim() and skipped otherwise.
        try:
            current_offset = self._cache_offset()
            excess = current_offset - prefix_len
            if excess > 0:
                for c in self._prompt_cache:
                    if c.is_trimmable():
                        c.trim(excess)
        except Exception:
            # Trim failure just means the parent's ``_find_common_prefix``
            # path will recompute the right state.  No data loss.
            pass

        logger.info(
            "SSD prime: warmed cache for %s with %d-token prefix from %s",
            self.model_path, prefix_len, Path(path).name,
        )

    def _maybe_save_to_ssd(
        self,
        messages: List[BaseMessage],
        tools: Optional[list] = None,
    ) -> None:
        """Persist the current prompt cache to disk after a successful turn.

        The save key is the **system-prompt prefix** of the current
        turn, not the full prompt: every query sharing the same system
        message will then produce the same on-disk key, so a brand-new
        user question on a cold process can find and reuse the stored
        KV state for the system block.  Saving by the full prompt (the
        previous behaviour) keyed every turn under a unique hash and
        guaranteed zero cross-session hits on any query the machine
        hadn't seen verbatim before.

        The in-memory prompt cache still holds ``prompt + generated``
        tokens — we don't touch it.  ``_maybe_prime_from_ssd`` trims
        the loaded cache back to the saved prefix length on restore,
        so the length mismatch between the key and the cache is safe.

        If the chat template doesn't produce a strict token-prefix for
        the system-only render (unusual but possible for exotic
        templates) we fall back to saving under the full prompt — no
        cross-session hit, but no correctness risk either.
        """
        if (
            self.turbo_level != "ssd"
            or self._ssd_store is None
            or self._prompt_cache is None
            or not self._last_prompt_tokens
        ):
            return

        current = list(self._last_prompt_tokens)
        save_tokens: Optional[List[int]] = None
        strategy = "bootstrap"

        # Strategy 1: template-based system-only prefix.  Fast and
        # keyed deterministically, but depends on the chat template
        # producing a strict token-prefix when we render with system
        # messages only — which a few exotic templates don't.
        sys_tokens = self._system_prefix_save_key(messages, tools)
        if sys_tokens:
            save_tokens = sys_tokens
            strategy = "template-system-prefix"

        # Strategy 2: dynamic longest-common-prefix against anything
        # already on disk for this model namespace.  This is the path
        # that unblocks benchmarks where the template render silently
        # fails the prefix check: turn 0 saves the full prompt
        # (nothing to compare against yet), turn 1 finds LCP(turn-0,
        # turn-1) = the stable system prefix and saves at that
        # length, turn 2+ prime from that entry.
        if save_tokens is None:
            common = self._ssd_store.find_best_common_prefix(current)
            # Guard against the degenerate "LCP == full current" case:
            # that just means we've already seen this exact prompt, in
            # which case the full-length save short-circuits on hash
            # match inside SSDPrefixStore.save().
            if 0 < common < len(current):
                save_tokens = list(current[:common])
                strategy = "dynamic-common-prefix"

        # Strategy 3: bootstrap — first save on this machine or a
        # mismatch that yielded no overlap.  Save the full prompt so
        # the next save has something to compute LCP against.
        if save_tokens is None:
            save_tokens = current

        logger.info(
            "SSD save: strategy=%s len=%d (full prompt=%d)",
            strategy, len(save_tokens), len(current),
        )

        try:
            with MLX_GEN_LOCK:
                self._ssd_store.save(
                    tokens=save_tokens,
                    cache=self._prompt_cache,
                    extra_meta={
                        "turbo_level": self.turbo_level,
                        "save_strategy": strategy,
                    },
                )
        except Exception as exc:
            logger.warning(
                "SSD cache save skipped after turn: %s", exc,
            )

    def _system_prefix_save_key(
        self,
        messages: List[BaseMessage],
        tools: Optional[list],
    ) -> Optional[List[int]]:
        """Return the system-only token prefix to key the SSD save under.

        Renders the chat template with **only** the SystemMessage
        entries (plus any ``tools`` block, which most tool-aware
        templates inject into the system region), without the
        generation opener, then tokenises the result.  Returns ``None``
        if there are no system messages, tokenisation fails, or the
        rendered tokens aren't a strict prefix of the full prompt
        tokens — in which case the caller falls back to saving the
        full prompt.

        The prefix check is the correctness gate: if we keyed a save
        under tokens that aren't an actual prefix of the live prompt,
        the loaded KV state would be placed at an offset the tokenizer
        never really saw, and downstream generation would produce
        garbage.
        """
        system_only = [m for m in messages if isinstance(m, SystemMessage)]
        if not system_only:
            return None

        try:
            prefix_str = self._render_system_prefix(system_only, tools)
            prefix_tokens = list(self._tokenizer.encode(prefix_str))
        except Exception as exc:
            logger.info(
                "SSD save: system-prefix render failed (%s) — "
                "falling back to dynamic common-prefix strategy.", exc,
            )
            return None

        if not prefix_tokens:
            logger.info(
                "SSD save: system-prefix render produced 0 tokens — "
                "falling back to dynamic common-prefix strategy.",
            )
            return None
        full = self._last_prompt_tokens
        if len(prefix_tokens) > len(full):
            logger.info(
                "SSD save: system-only render (%d tok) longer than live "
                "prompt (%d tok) — falling back to dynamic strategy.",
                len(prefix_tokens), len(full),
            )
            return None
        if full[:len(prefix_tokens)] != prefix_tokens:
            # Find where they first diverge so logs tell us what's off
            # (tokenizer whitespace, default system injection, etc.).
            divergence = 0
            limit = min(len(prefix_tokens), len(full))
            for i in range(limit):
                if prefix_tokens[i] != full[i]:
                    divergence = i
                    break
            logger.info(
                "SSD save: system-only render (%d tok) is not a strict "
                "prefix of live prompt (%d tok); diverged at index %d "
                "(sys=%s vs full=%s) — falling back to dynamic strategy.",
                len(prefix_tokens), len(full), divergence,
                prefix_tokens[divergence:divergence + 4],
                full[divergence:divergence + 4],
            )
            return None
        return prefix_tokens

    def _render_system_prefix(
        self,
        system_messages: List[BaseMessage],
        tools: Optional[list],
    ) -> str:
        """Render the chat template with system messages only.

        ``add_generation_prompt=False`` is the key knob: every common
        chat template (ChatML/Qwen, Llama 3, Mistral, DeepSeek) ends
        the system block with a closing control token *before* any
        subsequent user block opens.  Dropping the generation opener
        therefore yields a render that is byte-identical to the first
        N bytes of the full-prompt render — which means its token
        sequence is a strict prefix of the full-prompt token sequence,
        which is what the SSD prefix-cache contract requires.
        """
        chat_messages = [
            self._message_to_chat_dict(m) for m in system_messages
        ]
        if hasattr(self._tokenizer, "apply_chat_template"):
            base: dict = {
                "tokenize": False,
                "add_generation_prompt": False,
            }
            if tools:
                base["tools"] = tools
            try:
                return self._tokenizer.apply_chat_template(
                    chat_messages,
                    enable_thinking=self.thinking,
                    **base,
                )
            except TypeError:
                return self._tokenizer.apply_chat_template(
                    chat_messages, **base,
                )
        return "\n".join(
            f"{m['role'].capitalize()}: {m['content']}"
            for m in chat_messages
        )

    # ── Diagnostics ───────────────────────────────────────────────────

    @property
    def _llm_type(self) -> str:
        return "mlx-text-chat-turbo"
