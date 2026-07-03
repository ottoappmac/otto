"""Text-only MLX chat model wrapper.

Calls ``mlx_lm`` directly instead of going through
``langchain_community.llms.MLXPipeline``, which passes a ``formatter``
keyword argument that newer versions of ``mlx_lm`` no longer accept.

Usage::

    from chat_models.mlx import ChatMLXText

    llm = ChatMLXText(
        model_path="mlx-community/Qwen3-8B-4bit",
        draft_model_path="mlx-community/Qwen3-0.6B-4bit",  # optional speculative decoding
        max_tokens=4096,
        temp=0.0,
    )
"""

import asyncio
import json
import logging
from typing import Any, List, Optional, Sequence, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict

from chat_models.mlx._native_tool_parsing import (
    detect_native_tool_support,
    parse_native_tool_calls,
    stop_tokens_for,
    strip_tool_call_markup,
)

# Process-wide MLX state (weight cache, warmup set, generation lock) lives
# in ``_shared`` so the turbo engine and the classic path serialise on the
# same Metal lock and re-use the same loaded weights.  Only import what
# this module itself touches — callers that need ``MLX_GEN_LOCK``,
# ``_LOAD_LOCK`` or ``loaded_mlx_models`` should go to ``_shared`` directly.
from chat_models.mlx._shared import (
    MLX_GEN_LOCK,
    _LOADED_MODELS,
    _WARMED_UP,
    _load_or_reuse,
)

logger = logging.getLogger(__name__)

__all__ = ["ChatMLXText"]


def _action_block_complete(text: str) -> bool:
    """Whether *text* contains a complete, balanced ``Action:`` JSON object.

    Used for early-stopping during streaming: once the model has emitted a
    full JSON object after an ``Action:`` marker, further generation is wasted
    (and often duplicates the same action).

    Brace counting is used instead of a regex because the action JSON commonly
    nests another object inside ``action_input`` — most notably the empty-args
    case ``{"action": "X", "action_input": {}}``.  A non-greedy
    ``\\{[\\s\\S]*?\\}`` match terminates at the *first* closing brace (the
    inner ``{}``), stopping generation one brace early and yielding invalid
    JSON that the downstream parser then rejects.  Counting braces (while
    ignoring those inside strings) detects the true end of the outermost
    object regardless of nesting.
    """
    idx = text.lower().rfind("action:")
    if idx == -1:
        return False
    brace_start = text.find("{", idx)
    if brace_start == -1:
        return False
    depth = 0
    in_string = False
    escape_next = False
    for ch in text[brace_start:]:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return True
    return False


# Chat-template end-of-turn / control tokens.  When ``stream_generate`` does
# not honour the tokenizer's eos_token_id (varies by model + mlx_lm version),
# the model happily continues past its turn boundary and starts emitting a
# fake "user" or "assistant" turn from the chat template.  We watch the
# decoded text for these markers and break out of the stream the moment one
# appears so the spurious continuation never reaches the user.
_STOP_TOKENS: tuple[str, ...] = (
    "<|im_end|>",      # Qwen / ChatML
    "<|im_start|>",    # Qwen — model started hallucinating the next turn
    "<|eot_id|>",      # Llama 3 family
    "<|end_of_text|>",  # Llama
    "<|endoftext|>",   # GPT family / DeepSeek
    "</s>",            # Mistral / Llama 2
)

# Substituted as the assistant content when generation is truncated at
# ``max_tokens`` and no executable tool call could be extracted (e.g. the model
# fell into a repetition loop building a huge argument and never closed its
# tool-call JSON).  Without this, the truncated garbage is silently treated as a
# final answer and the run stalls.  This short, explicit signal replaces the
# garbage so the model can recover with a smaller step on the next turn.
_TRUNCATION_RECOVERY_MSG = (
    "[system: your previous response was cut off because it reached the "
    "{max_tokens}-token generation limit before completing. This usually means "
    "you started repeating text or built an oversized tool argument. Do NOT "
    "repeat long patterns. Take a smaller step: if you were searching a large "
    "result, use a short search pattern or read_file with offset/limit to page "
    "through it.]"
)

# Intra-generation repetition early-abort.  Rather than burn the full
# ``max_tokens`` budget streaming the same sentence over and over, sample the
# growing text every few tokens and stop as soon as it collapses into a
# degenerate repetition.  The aborted output flows into the same
# truncation-recovery branch below.  Cheap because we only start checking once
# the text is already long and only every N tokens.
_REPETITION_CHECK_EVERY = 64
_REPETITION_MIN_CHARS = 800


def _looks_repetitive(text: str) -> bool:
    """Provider-agnostic degenerate-repetition check (lazy, fail-open)."""
    try:
        from middleware.repetition_guard import is_degenerate_repetition

        return is_degenerate_repetition(text, min_chars=_REPETITION_MIN_CHARS)
    except Exception:  # pragma: no cover — never break generation on a guard
        return False


class ChatMLXText(BaseChatModel):
    """``BaseChatModel`` backed by a local ``mlx_lm`` text model.

    Bypasses ``MLXPipeline`` entirely, which fixes the
    ``generate_step() got an unexpected keyword argument 'formatter'``
    error caused by a version mismatch between ``langchain-community``
    and newer ``mlx_lm`` releases.

    Args:
        model_path:         HuggingFace model ID or local path
                            (e.g. ``"mlx-community/Qwen3-8B-4bit"``).
        draft_model_path:   Optional HuggingFace model ID for a small draft
                            model used for speculative decoding.  Must share
                            the same tokenizer/vocab as the main model.
                            ``None`` disables speculative decoding.
        num_draft_tokens:   Number of tokens the draft model proposes per
                            speculative step (default 3).
        max_tokens:         Maximum tokens to generate (default 4096).
        temp:               Sampling temperature — 0.0 for greedy (default).
        repetition_penalty: Repetition penalty applied during generation
                            (default 1.1).
        verbose:            Print token stream to stdout while generating
                            (default False).
        kv_bits:            Quantize the KV cache to this many bits (4 or 8).
                            ``None`` keeps full-precision cache.  Equivalent to
                            ``mlx_lm``'s ``--cache-8bit`` / ``--cache-4bit``.
    """

    model_path: str
    draft_model_path: Optional[str] = None
    num_draft_tokens: int = 3
    max_tokens: int = 4096
    temp: float = 0.0
    repetition_penalty: float = 1.2
    # Window (in tokens) the repetition penalty considers.  mlx_lm's default of
    # 20 is too small to catch medium-period loops; a wider window penalises
    # repeated alternations sooner.
    repetition_context_size: int = 60
    verbose: bool = False
    thinking: bool = False
    enable_prompt_cache: bool = False
    enable_system_prompt_cache: bool = False
    kv_bits: Optional[int] = None
    kv_group_size: int = 64
    # Soft cap on the KV prefix cache, in tokens.  After each generation
    # the cumulative cache offset is checked against this value; if it's
    # over, the cache is trimmed to roughly half the cap so the next turn
    # has headroom and we don't trim every single call.  ``0`` disables
    # the cap and reverts to the legacy unbounded behaviour.  Default
    # 32 768 tokens ≈ 1 GB on a 7B 4-bit model.
    prompt_cache_max_tokens: int = 32768

    # Exposes the effective input budget to framework helpers such as
    # ``compute_summarization_defaults`` (deepagents) and
    # ``_model_input_budget`` (session_manager).  Without this, those
    # helpers fall back to a 170 000-token trigger which exceeds the
    # model's actual context window, causing "prompt too long" errors.
    # Set lazily in ``__init__`` so it reflects the configured
    # ``prompt_cache_max_tokens`` rather than a hard-coded constant.
    profile: dict = {}

    # mlx objects are not JSON-serialisable — allow arbitrary types
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _model: Any = None
    _tokenizer: Any = None
    _draft_model: Any = None
    _prompt_cache: Any = None
    _last_prompt_tokens: Optional[List[int]] = None
    # Native tool-calling state — populated in __init__ once the tokenizer is loaded.
    # ``_native_tools_supported`` gates ``bind_tools``; ``_tool_family`` selects the
    # parser used by ``_generate`` to extract structured tool calls from output.
    _native_tools_supported: bool = False
    _tool_family: str = "unknown"

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path=model_path, **kwargs)
        # Populate profile so summarization middleware can use fraction-based
        # trigger (85 % of budget) instead of the 170 k-token fallback which
        # exceeds most local model context windows and never fires.
        if not self.profile:
            self.profile = {"max_input_tokens": self.prompt_cache_max_tokens}
        cache_key = (self.model_path, self.draft_model_path)
        logger.info(
            "Initialising ChatMLXText: %s (max_tokens=%d, temp=%.2f, kv_bits=%s, "
            "prompt_cache=%s, system_prompt_cache=%s, thinking=%s)",
            self.model_path,
            self.max_tokens,
            self.temp,
            self.kv_bits,
            self.enable_prompt_cache,
            self.enable_system_prompt_cache,
            self.thinking,
        )
        if cache_key in _LOADED_MODELS:
            logger.info(
                "MLX model %s reused from process cache (no reload)",
                self.model_path,
            )
        else:
            logger.info("Loading MLX model: %s", self.model_path)
            if self.draft_model_path:
                logger.info("Loading MLX draft model: %s", self.draft_model_path)

        try:
            triple, freshly_loaded = _load_or_reuse(
                self.model_path, self.draft_model_path,
            )
        except Exception as exc:  # noqa: BLE001
            # A memory error while pulling weights into unified Metal memory
            # would otherwise leave the allocator pool bloated with a partial
            # allocation.  Free it and re-raise so the caller (create_llm →
            # session build) fails cleanly and falls back to the previous
            # graph/provider instead of the process aborting later.
            msg = str(exc).lower()
            if "memory" in msg or "insufficient" in msg or "alloc" in msg:
                logger.warning(
                    "MLX weight load hit a memory error for %s (reason: %s). "
                    "Unload the previous provider or pick a smaller model.",
                    self.model_path, exc,
                )
                try:
                    import mlx.core as mx
                    mx.clear_cache()
                except Exception:  # noqa: BLE001
                    pass
            raise
        self._model, self._tokenizer, self._draft_model = triple

        if freshly_loaded:
            try:
                import mlx.core as mx
                mem_gb = mx.get_active_memory() / (1024**3)
                logger.info(
                    "MLX model loaded: %s (active GPU memory: %.2f GB)",
                    self.model_path, mem_gb,
                )
            except Exception:
                logger.info("MLX model loaded: %s", self.model_path)
            if self.draft_model_path:
                logger.info(
                    "Speculative decoding enabled: draft=%s",
                    self.draft_model_path,
                )

        if self.enable_prompt_cache:
            from mlx_lm.models.cache import make_prompt_cache
            self._prompt_cache = make_prompt_cache(self._model)

        # Inspect the chat template to decide whether the model was fine-tuned
        # for native tool calling.  Models without a tool-aware template
        # silently fall back to the ReAct text shim (MLXReActWrapper /
        # MLXReActMiddleware) — full backward compatibility.
        self._native_tools_supported, self._tool_family = detect_native_tool_support(
            self._tokenizer
        )
        if self._native_tools_supported:
            logger.info(
                "Native tool calling enabled for %s (family=%s)",
                self.model_path, self._tool_family,
            )
        else:
            logger.info(
                "Native tool calling NOT detected for %s — ReAct shim will be used",
                self.model_path,
            )

        if cache_key not in _WARMED_UP:
            self._warmup()
            _WARMED_UP.add(cache_key)

    # ── Warmup ────────────────────────────────────────────────────────────────

    def _warmup(self) -> None:
        """Trigger MLX graph compilation with a short dummy generation.

        The first call to ``stream_generate`` traces and compiles the
        computation graph.  Running a short warmup at init time pays that
        cost once so that the first real user request is not penalised.

        When speculative decoding is enabled the draft-verify graph is also
        traced here, so the first real call does not pay that compilation cost
        either.
        """
        try:
            from mlx_lm import stream_generate

            warmup_kwargs: dict = {"max_tokens": 1, **self._sampler_kwargs()}
            if self._draft_model is not None:
                warmup_kwargs["draft_model"] = self._draft_model
                warmup_kwargs["num_draft_tokens"] = self.num_draft_tokens

            with MLX_GEN_LOCK:
                for _ in stream_generate(
                    model=self._model,
                    tokenizer=self._tokenizer,
                    prompt="Hi",
                    **warmup_kwargs,
                ):
                    pass
                import mlx.core as mx
                mx.clear_cache()
            logger.debug(
                "MLX warmup complete (%s%s)",
                self.model_path,
                f" + draft={self.draft_model_path}" if self._draft_model is not None else "",
            )
        except Exception as exc:  # noqa: BLE001
            # Warmup is best-effort — never let it abort model construction.
            # A failure here is most often memory-related (the warmup forward
            # pass needs a transient allocation on top of the just-loaded
            # weights).  Free whatever the partial pass allocated so the
            # allocator pool isn't left bloated, and surface the reason at
            # warning level so a recurring OOM is visible rather than hidden
            # behind a debug log.
            msg = str(exc).lower()
            if "memory" in msg or "insufficient" in msg or "alloc" in msg:
                logger.warning(
                    "MLX warmup hit a memory error for %s — skipping warmup "
                    "(reason: %s). Consider unloading the previous provider or "
                    "using a smaller model.",
                    self.model_path, exc,
                )
            else:
                logger.debug("MLX warmup skipped", exc_info=True)
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:  # noqa: BLE001
                pass

    # ── Native tool-calling API ──────────────────────────────────────────────

    def supports_native_tools(self) -> bool:
        """Return True when the loaded model has a tool-aware chat template.

        Callers (e.g. ``MLXReActWrapper``) use this to decide whether to
        bypass the ReAct text shim and let the model emit structured tool
        calls directly.
        """
        return self._native_tools_supported

    def bind_tools(
        self,
        tools: Sequence[Union[dict, type, BaseTool, Any]],
        *,
        tool_choice: Optional[Union[str, dict]] = None,
        **kwargs: Any,
    ) -> Runnable[Any, BaseMessage]:
        """Bind tools to this model using its native chat-template tool slot.

        Mirrors LangChain's standard ``bind_tools`` contract.  Tools are
        converted to OpenAI JSON-schema dicts and stored on the returned
        ``RunnableBinding`` so they flow through to ``_generate`` as kwargs
        and are forwarded to ``apply_chat_template(tools=...)``.

        Raises ``NotImplementedError`` when the loaded model does NOT have a
        tool-aware chat template, so the caller can fall back to the ReAct
        text shim.
        """
        if not self._native_tools_supported:
            raise NotImplementedError(
                f"Model {self.model_path!r} does not have a tool-aware chat "
                f"template (family={self._tool_family!r}). Use MLXReActWrapper "
                f"or MLXReActMiddleware for ReAct-style tool calling instead."
            )

        formatted_tools = [convert_to_openai_tool(t) for t in tools]
        return self.bind(tools=formatted_tools, tool_choice=tool_choice, **kwargs)

    # ── Prompt formatting ─────────────────────────────────────────────────────

    def _message_to_chat_dict(self, msg: BaseMessage) -> dict:
        """Convert a single LangChain message into the dict shape expected by
        ``tokenizer.apply_chat_template``.

        For native tool-calling models this preserves ``AIMessage.tool_calls``
        as OpenAI-format ``tool_calls`` and translates ``ToolMessage`` into a
        ``tool`` role — the chat template handles family-specific rendering
        (e.g. Qwen wraps tool results in ``<tool_response>...</tool_response>``).

        For non-native models (or any non-tool message) the legacy
        ``{role, content}`` shape is preserved.
        """
        # Anthropic-style multi-part content (list of {"type": "text", ...} blocks)
        # is emitted by middleware like ``langchain_anthropic.prompt_caching`` and
        # cannot be passed straight to text-only HF chat templates — many
        # templates do ``messages[0].content + '\n'`` which raises TypeError on
        # a list.  Flatten to plain text; image parts in multimodal lists are
        # dropped (this is the text-only chat model — VLM lives in chat_vlm.py).
        def _flatten(content) -> str:
            if isinstance(content, list):
                return "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            return content or ""

        if isinstance(msg, SystemMessage):
            return {"role": "system", "content": _flatten(msg.content)}
        if isinstance(msg, HumanMessage):
            return {"role": "user", "content": _flatten(msg.content)}
        if isinstance(msg, ToolMessage):
            entry: dict = {
                "role": "tool",
                "content": _flatten(msg.content),
            }
            if msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id
            if getattr(msg, "name", None):
                entry["name"] = msg.name
            return entry
        if isinstance(msg, AIMessage):
            entry = {"role": "assistant", "content": _flatten(msg.content)}
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls and self._native_tools_supported:
                # ``arguments`` is passed as a dict (not a JSON-encoded string)
                # because some HF chat templates iterate it with the Jinja
                # ``|items`` filter (e.g. Hermes/Functionary XML used by
                # ``Qwen3.5-4B-OptiQ-4bit``), which requires a mapping.
                # Standard OpenAI-style templates parse a stringified value
                # back into a dict internally — but they also accept a dict
                # directly, so this form is the more portable choice.
                entry["tool_calls"] = [
                    {
                        "id": tc.get("id") or "",
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": (
                                json.loads(tc["args"])
                                if isinstance(tc["args"], str)
                                else tc["args"]
                            ),
                        },
                    }
                    for tc in tool_calls
                ]
            return entry
        return {"role": "user", "content": str(msg.content)}

    def _to_prompt(
        self,
        messages: List[BaseMessage],
        tools: Optional[list[dict]] = None,
    ) -> str:
        """Convert LangChain messages to a single prompt string.

        Uses the tokenizer's ``apply_chat_template`` when available,
        falling back to a simple role-prefixed concatenation.

        When *tools* is provided the OpenAI-format tool list is forwarded to
        the chat template, which renders the family-appropriate tool section
        (e.g. Qwen's ``<tools>...</tools>`` block).  This is only meaningful
        for tool-aware templates — see :meth:`supports_native_tools`.
        """
        chat_messages = [self._message_to_chat_dict(m) for m in messages]

        if hasattr(self._tokenizer, "apply_chat_template"):
            base_kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
            if tools:
                base_kwargs["tools"] = tools
            # Qwen3 passes enable_thinking through **kwargs to its Jinja chat template,
            # so it never appears in inspect.signature — try/except is the only reliable way.
            try:
                return self._tokenizer.apply_chat_template(
                    chat_messages, enable_thinking=self.thinking, **base_kwargs
                )
            except TypeError:
                return self._tokenizer.apply_chat_template(chat_messages, **base_kwargs)

        # Plain fallback — works for models without a chat template
        return "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in chat_messages
        )

    # ── Sampling kwargs ───────────────────────────────────────────────────────

    def _sampler_kwargs(self) -> dict:
        """Return sampler/logits_processors kwargs for mlx_lm >= 0.18 (0.31 pinned).

        Consumes any pending loop-recovery temperature bump (see
        :func:`chat_models.mlx._shared.request_temperature_bump`).  When a
        ToolLoopGuard has tripped, the next generation(s) sample at the bumped
        temperature instead of the configured (often greedy) ``self.temp`` so
        the model can break out of a deterministic identical-call loop.
        """
        from mlx_lm.sample_utils import make_logits_processors, make_sampler
        from chat_models.mlx._shared import consume_temperature_bump

        temp = self.temp
        bump = consume_temperature_bump()
        if bump is not None and bump > temp:
            logger.warning(
                "ChatMLXText: loop-recovery temperature bump active — sampling "
                "at temp=%.2f for this turn (configured temp=%.2f).",
                bump, temp,
            )
            temp = bump
        return {
            "sampler": make_sampler(temp=temp),
            "logits_processors": make_logits_processors(
                repetition_penalty=self.repetition_penalty,
                repetition_context_size=self.repetition_context_size,
            ),
        }

    # ── Cache stats ───────────────────────────────────────────────────────────

    def _cache_offset(self) -> int:
        """Return the current KV cache offset (0 when caching is disabled or unsupported).

        Not all cache types expose an ``offset`` (e.g. ``ArraysCache`` used by
        GatedDeltaNet / state-space models).  Scans for the first layer that
        tracks an offset and falls back to 0 gracefully.
        """
        if self._prompt_cache is None:
            return 0
        for c in self._prompt_cache:
            offset = getattr(c, "offset", None)
            if offset is not None and isinstance(offset, int):
                return offset
        return 0

    def _build_response_metadata(self, last_response: Any, cache_offset_before: int) -> dict:
        """Build response metadata attached to ``AIMessage.response_metadata``.

        LangChain's standard location for model-level stats (token counts, TPS,
        memory, cache metrics).  Visible in LangSmith's Metadata panel and
        preserved through any number of wrapper layers.
        """
        tokens_prefilled = last_response.prompt_tokens
        tokens_from_cache = cache_offset_before
        total = tokens_from_cache + tokens_prefilled
        generation_tokens = last_response.generation_tokens
        # ``finish_reason == "length"`` flags a hard truncation: generation ran
        # right up to ``max_tokens`` instead of stopping on an EOS/control
        # token.  Surfaced so callers can detect cut-off output (see _generate).
        finish_reason = "length" if generation_tokens >= self.max_tokens else "stop"
        return {
            "tokens_from_cache": tokens_from_cache,
            "tokens_prefilled": tokens_prefilled,
            "cache_hit_ratio": round(tokens_from_cache / total, 3) if total else 0.0,
            "prompt_tps": round(last_response.prompt_tps, 1),
            "generation_tokens": generation_tokens,
            "generation_tps": round(last_response.generation_tps, 1),
            "cache_offset_after": self._cache_offset(),
            "peak_memory_gb": round(last_response.peak_memory, 3),
            "finish_reason": finish_reason,
        }

    # ── Prefix-aware caching helpers ──────────────────────────────────────────

    def _find_common_prefix(self, tokens: List[int]) -> int:
        """Return the length of the longest common token prefix with the previous turn."""
        if not self._last_prompt_tokens:
            return 0
        limit = min(len(self._last_prompt_tokens), len(tokens))
        for i in range(limit):
            if tokens[i] != self._last_prompt_tokens[i]:
                return i
        return limit

    def _trim_cache_to(self, target: int) -> None:
        """Roll the KV cache back to *target* tokens by trimming the excess.

        Trims each layer individually so that mixed caches (e.g. Qwen3.5's
        ArraysCache + KVCache) are handled gracefully — non-trimmable layers
        are skipped.
        """
        current = self._cache_offset()
        excess = current - target
        if excess <= 0:
            return
        trimmed = 0
        for c in self._prompt_cache:
            if c.is_trimmable():
                n = c.trim(excess)
                if not trimmed:
                    trimmed = n  # all trimmable layers return the same value; log first
        logger.debug(
            "Cache trimmed by %d tokens (%d → %d)",
            trimmed, current, self._cache_offset(),
        )

    def _enforce_cache_budget(self) -> None:
        """Trim (or rebuild) the KV prompt cache when it exceeds the budget.

        Long autonomous sessions can otherwise grow the cache without bound
        and trip macOS into swap or OOM.  Strategy:

        * No cap (``prompt_cache_max_tokens == 0``) → no-op (legacy behaviour).
        * Cache empty / disabled → no-op.
        * Cache at or under cap → no-op.
        * Cache over cap → trim down to roughly half the cap, so the next
          turn has headroom and we don't trim on every call.  If trimming
          fails (e.g. all-non-trimmable cache layers like ``ArraysCache``),
          rebuild the cache from scratch via ``make_prompt_cache`` so we
          drop the memory in one decisive step rather than leaking it.

        Releasing the Metal allocator pool (``mx.clear_cache``) is left to
        the caller — :meth:`_generate` already does it once per turn.
        """
        budget = int(self.prompt_cache_max_tokens or 0)
        if budget <= 0 or self._prompt_cache is None:
            return
        current = self._cache_offset()
        if current <= budget:
            return

        target = max(1, budget // 2)
        logger.info(
            "MLX KV cache exceeded budget (%d > %d tokens) — trimming to %d",
            current, budget, target,
        )
        self._trim_cache_to(target)

        after = self._cache_offset()
        if after > budget:
            # Trim was a no-op (e.g. non-trimmable cache type) — rebuild.
            try:
                from mlx_lm.models.cache import make_prompt_cache
                self._prompt_cache = make_prompt_cache(self._model)
                self._last_prompt_tokens = None
                logger.warning(
                    "MLX KV cache could not be trimmed (%d → %d) — rebuilt "
                    "from scratch. Next turn will pay a full prefill.",
                    current, after,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("MLX KV cache rebuild failed: %s", exc)
        else:
            # Keep _last_prompt_tokens consistent with the trimmed cache so
            # the next call's prefix matcher doesn't claim a hit beyond the
            # actual cache offset.
            if self._last_prompt_tokens is not None and len(self._last_prompt_tokens) > after:
                self._last_prompt_tokens = self._last_prompt_tokens[:after]

    # ── Synchronous generation ────────────────────────────────────────────────

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        from mlx_lm import stream_generate  # lazy import — only required on Apple Silicon

        # ``tools`` is injected into kwargs by ``self.bind_tools(...)``'s
        # RunnableBinding.  Pull it out so we forward it to the chat template
        # (and so it's not passed to mlx_lm.stream_generate which would reject
        # the unknown kwarg).
        tools: Optional[list[dict]] = kwargs.pop("tools", None)
        kwargs.pop("tool_choice", None)
        native_mode = bool(tools) and self._native_tools_supported

        prompt_str = self._to_prompt(messages, tools=tools)

        prompt: Any = prompt_str
        if self.enable_system_prompt_cache and self._prompt_cache is not None:
            full_tokens = self._tokenizer.encode(prompt_str)
            common = self._find_common_prefix(full_tokens)
            new_tokens = len(full_tokens) - common
            if common > 0:
                self._trim_cache_to(common)
                if self._cache_offset() == common:
                    prompt = full_tokens[common:]
                else:
                    logger.warning(
                        "KV prefix trim failed (wanted %d, got %d) — sending full prompt",
                        common, self._cache_offset(),
                    )
                    common = 0
                    new_tokens = len(full_tokens)
            logger.info(
                "KV prefix cache: %d total tokens, %d reused, %d new (%.0f%% hit)",
                len(full_tokens), common, new_tokens,
                (common / len(full_tokens)) * 100 if full_tokens else 0,
            )
            self._last_prompt_tokens = full_tokens

        cache_offset_before = self._cache_offset()

        gen_kwargs: dict = {"max_tokens": self.max_tokens, **self._sampler_kwargs()}
        if self._prompt_cache is not None:
            gen_kwargs["prompt_cache"] = self._prompt_cache
        if self.kv_bits is not None:
            gen_kwargs["kv_bits"] = self.kv_bits
            gen_kwargs["kv_group_size"] = self.kv_group_size
        if self._draft_model is not None:
            gen_kwargs["draft_model"] = self._draft_model
            gen_kwargs["num_draft_tokens"] = self.num_draft_tokens

        text = ""
        last_response = None
        repetition_aborted = False
        _tokens_since_repeat_check = 0
        # In native tool mode the model emits structured calls (e.g.
        # ``<tool_call>{...}</tool_call>``) and we must let the stream run to
        # turn-end so all parallel calls land in the buffer.  In ReAct mode
        # we keep the early-break on a complete ``Action:`` block to save
        # wall time.
        active_stop_tokens: tuple[str, ...] = _STOP_TOKENS
        if native_mode:
            active_stop_tokens = stop_tokens_for(self._tool_family) or _STOP_TOKENS

        # Hold the process-wide MLX generation lock for the entire stream.
        # Releasing between tokens would let another thread sneak in a
        # ``stream_generate`` call and trigger the Metal command-buffer
        # assertion mid-decode.  See ``MLX_GEN_LOCK`` for the full rationale.
        with MLX_GEN_LOCK:
            for response in stream_generate(
                model=self._model,
                tokenizer=self._tokenizer,
                prompt=prompt,
                **gen_kwargs,
            ):
                text += response.text
                last_response = response
                if not native_mode and _action_block_complete(text):
                    break
                # Some mlx_lm / tokenizer combinations don't honour every stop
                # token (e.g. Qwen3 models often skip <|im_end|> when streaming),
                # which lets the model continue past its turn boundary and start
                # hallucinating the next user/assistant turn from the chat
                # template.  Belt-and-braces: break on any known control token.
                if any(tok in text for tok in active_stop_tokens):
                    for tok in active_stop_tokens:
                        idx = text.find(tok)
                        if idx != -1:
                            text = text[:idx]
                            break
                    break
                # Intra-generation repetition early-abort: stop before burning
                # the whole token budget on a degenerate loop.  Checked sparsely
                # and only once the text is already long, so it adds negligible
                # cost to healthy generations.
                _tokens_since_repeat_check += 1
                if (
                    _tokens_since_repeat_check >= _REPETITION_CHECK_EVERY
                    and len(text) >= _REPETITION_MIN_CHARS
                ):
                    _tokens_since_repeat_check = 0
                    if _looks_repetitive(text):
                        repetition_aborted = True
                        logger.warning(
                            "MLX generation aborted early: degenerate repetition "
                            "detected at %d chars (max_tokens=%d).",
                            len(text), self.max_tokens,
                        )
                        break

        response_metadata = self._build_response_metadata(last_response, cache_offset_before)

        # Soft cap on the KV cache size — the only defence against unbounded
        # memory growth in long autonomous sessions.  Done BEFORE clear_cache
        # so the freed cache buffers are released by the same Metal sweep.
        self._enforce_cache_budget()

        # Release metal buffer pool to prevent unbounded memory growth across
        # turns.  Without this, each stream_generate call's KV-cache buffers
        # stay in MLX's metal allocator pool even after Python GC, causing
        # memory pressure and TPS degradation on subsequent turns.
        try:
            import mlx.core as mx
            # mlx >= 0.18 moved clear_cache to the top-level module; the older
            # mx.metal.clear_cache() still works but emits a deprecation warning.
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            else:
                mx.metal.clear_cache()
        except Exception:
            pass
        logger.info(
            "MLX generate: %d prompt tokens (%.0f%% cached), %d generated | "
            "prompt %.1f t/s, gen %.1f t/s | %.3f GB peak",
            response_metadata["tokens_from_cache"] + response_metadata["tokens_prefilled"],
            response_metadata["cache_hit_ratio"] * 100,
            response_metadata["generation_tokens"],
            response_metadata["prompt_tps"],
            response_metadata["generation_tps"],
            response_metadata["peak_memory_gb"],
        )

        # Native tool-call extraction.  Only attempted when the caller bound
        # tools AND the loaded model supports them — otherwise we leave the
        # raw text alone and the upstream ReAct shim parses it.
        ai_tool_calls: list[dict] = []
        clean_content = text
        if native_mode:
            # Strip <think> blocks so thinking-mode draft tool calls inside
            # the scratch reasoning are not executed as real calls.
            from middleware._react_core import strip_think_tags
            search_text = strip_think_tags(text)
            parsed_calls = parse_native_tool_calls(search_text, self._tool_family)
            if parsed_calls:
                ai_tool_calls = parsed_calls
                # The user-visible content should not include the raw tool-call
                # JSON markup; the tool_calls list carries the executable form.
                clean_content = strip_tool_call_markup(search_text, self._tool_family)

        # ── Truncation guard ──────────────────────────────────────────────────
        # If generation hit max_tokens (finish_reason == "length") AND we could
        # not extract a usable tool call, the output is almost certainly garbage
        # cut off mid-stream (e.g. a repetition loop building an oversized tool
        # argument).  Replace it with a short recovery signal so it is not
        # mistaken for a final answer and the model can take a smaller next step.
        if response_metadata.get("finish_reason") == "length" or repetition_aborted:
            logger.warning(
                "MLX generation %s (max_tokens=%d); tool_calls extracted: %d, "
                "content chars: %d",
                "aborted on repetition" if repetition_aborted
                else "truncated at max_tokens (finish_reason=length)",
                self.max_tokens, len(ai_tool_calls), len(clean_content),
            )
            if not ai_tool_calls:
                clean_content = _TRUNCATION_RECOVERY_MSG.format(max_tokens=self.max_tokens)

        return ChatResult(
            generations=[ChatGeneration(
                message=AIMessage(
                    content=clean_content,
                    tool_calls=ai_tool_calls,
                    response_metadata=response_metadata,
                ),
            )],
        )

    # ── Asynchronous generation ───────────────────────────────────────────────

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        """Run generation in a thread pool to avoid blocking the event loop."""
        return await asyncio.to_thread(
            self._generate, messages, stop=stop, **kwargs
        )

    @property
    def _llm_type(self) -> str:
        return "mlx-text-chat"
