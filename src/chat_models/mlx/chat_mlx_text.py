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
import time
from hashlib import sha256
from typing import Any, List, NamedTuple, Optional, Sequence, Union

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
    weights_fingerprint,
)
from chat_models.mlx import _prefix_disk, _prefix_store

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


# Static prompt prefixes shorter than this aren't snapshotted into the
# process-wide store: prefilling them costs a few seconds at most, and every
# entry — however short — holds a model's whole recurrent state (~50 MB for
# Qwen3.5-9B) and could evict a long, valuable prefix.
_STATIC_PREFIX_MIN_TOKENS = 1024


# How the "KV prefix cache" log line names where the reused tokens came from
# (``_CachePlan.source``, also ``response_metadata["prefix_cache_source"]``).
_REUSE_SOURCES = {
    "session": " from this session's cache",
    "ram": " (static prefix) from the RAM store",
    "ssd": " (static prefix) from SSD",
    "none": "",
}


class _CachePlan(NamedTuple):
    """What :meth:`ChatMLXText._prepare_prompt_cache` put in the prompt cache."""

    reused: int  # restored: from this instance's cache or a stored snapshot
    prefilled: int  # fed before ``stream_generate`` runs...
    seconds: float  # ...in this many seconds
    source: str  # where *reused* came from: a ``_REUSE_SOURCES`` key


def _common_prefix(a: str, b: str) -> str:
    """The longest common prefix of *a* and *b*."""
    n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return a[:n]


def _tools_key(tools: list[dict]) -> str:
    """A stable fingerprint of a bound tool set: its static snapshots' family."""
    return sha256(json.dumps(tools, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _snapshot_nbytes(layers: List[Any]) -> int:
    """Bytes held by the arrays of a prompt-cache snapshot."""
    from mlx.utils import tree_flatten

    return sum(x.nbytes for layer in layers for _, x in tree_flatten(layer.state) if hasattr(x, "nbytes"))


def _trimmable(layer: Any) -> bool:
    """Whether a prompt-cache layer can be rolled back by trimming.

    True for KV caches (``KVCache``, ``QuantizedKVCache``, a
    ``RotatingKVCache`` that hasn't wrapped yet).  False for layers whose
    state folds in every processed token — e.g. the ``ArraysCache`` holding
    the GatedDeltaNet recurrent state of Qwen3.5 / Qwen3-Next linear-attention
    layers — which can only be restored from a snapshot.
    """
    return layer.is_trimmable() and isinstance(getattr(layer, "offset", None), int)


def _clone_cache_layer(layer: Any) -> Any:
    """Return an evaluated, independent copy of a prompt-cache layer.

    ``mx.array`` copies, so in-place writes to one copy (``RotatingKVCache``
    updates its buffers in place) never reach the other.  Evaluating right
    away matters: MLX cannot evaluate a lazy array on a thread other than the
    one that built it, and consecutive generations may run on different
    ``asyncio.to_thread`` workers.
    """
    import mlx.core as mx
    from mlx.utils import tree_map

    state = tree_map(lambda x: mx.array(x) if isinstance(x, mx.array) else x, layer.state)
    mx.eval(state)
    return type(layer).from_state(state, layer.meta_state)


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
    # over, the generated tail is dropped (the cache rolls back to the
    # reusable prompt prefix), or the cache is rebuilt when that prefix
    # alone exceeds the cap.  ``0`` disables the cap and reverts to the
    # legacy unbounded behaviour.  Default 32 768 tokens ≈ 1 GB on a 7B
    # 4-bit model.  Static prompt prefixes longer than the cap aren't
    # snapshotted for other sessions either (see ``_static_prefixes``).
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
    # The longest token prefix the prompt cache can be rolled back to exactly
    # (see ``_rollback_cache_to``).  ``_reuse_points`` maps each shorter or
    # equal length it can also return to → copies of the non-trimmable layers
    # taken at exactly that many tokens (see ``_add_reuse_point``);
    # ``_query_point`` is the one where the latest user turn's generation
    # prompt started.
    _last_prompt_tokens: Optional[List[int]] = None
    _reuse_points: dict = {}
    _query_point: int = 0
    # ``_static_prefixes`` memo: the leading system messages and tools it last
    # probed, the static token prefixes they render to and the tools' key.
    _static_probe: Optional[tuple] = None
    # ``_turn_start`` memo: the tokens of the template's generation prompt.
    _scaffold: Optional[List[int]] = None
    # Static-prefix snapshots to write to SSD once the reply is generated:
    # ``(key, layers, n_tokens)`` (see ``_keep_on_ssd``).
    _ssd_writes: list = []
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
        add_generation_prompt: bool = True,
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
            base_kwargs: dict = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
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

    def _build_response_metadata(
        self,
        last_response: Any,
        cache_offset_before: int,
        prefilled_before: int = 0,
        prefill_seconds_before: float = 0.0,
        source: Optional[str] = None,
    ) -> dict:
        """Build response metadata attached to ``AIMessage.response_metadata``.

        LangChain's standard location for model-level stats (token counts, TPS,
        memory, cache metrics).  Visible in LangSmith's Metadata panel and
        preserved through any number of wrapper layers.

        *prefilled_before* tokens were prefilled in *prefill_seconds_before*
        before ``stream_generate`` ran (static prefixes being snapshotted, the
        turn start of a hybrid model, see :meth:`_prepare_prompt_cache`); they
        count as prefilled and in the prompt throughput, which the session
        stats turn back into prefill time.
        *source* says where the cached tokens came from (see ``_CachePlan``).
        """
        tokens_prefilled = last_response.prompt_tokens + prefilled_before
        prompt_tps = last_response.prompt_tps
        if prefilled_before and prompt_tps > 0:
            prompt_tps = tokens_prefilled / (
                prefill_seconds_before + last_response.prompt_tokens / prompt_tps
            )
        tokens_from_cache = cache_offset_before
        total = tokens_from_cache + tokens_prefilled
        generation_tokens = last_response.generation_tokens
        # ``finish_reason == "length"`` flags a hard truncation: generation ran
        # right up to ``max_tokens`` instead of stopping on an EOS/control
        # token.  Surfaced so callers can detect cut-off output (see _generate).
        finish_reason = "length" if generation_tokens >= self.max_tokens else "stop"
        metadata = {
            "tokens_from_cache": tokens_from_cache,
            "tokens_prefilled": tokens_prefilled,
            "cache_hit_ratio": round(tokens_from_cache / total, 3) if total else 0.0,
            "prompt_tps": round(prompt_tps, 1),
            "generation_tokens": generation_tokens,
            "generation_tps": round(last_response.generation_tps, 1),
            "cache_offset_after": self._cache_offset(),
            "peak_memory_gb": round(last_response.peak_memory, 3),
            "finish_reason": finish_reason,
        }
        if source is not None:
            metadata["prefix_cache_source"] = source
        return metadata

    # ── Prefix-aware caching helpers ──────────────────────────────────────────

    def _find_common_prefix(self, tokens: List[int]) -> int:
        """Return the length of the longest common token prefix with the cache's reusable prefix."""
        if not self._last_prompt_tokens:
            return 0
        limit = min(len(self._last_prompt_tokens), len(tokens))
        for i in range(limit):
            if tokens[i] != self._last_prompt_tokens[i]:
                return i
        return limit

    def _reset_prompt_cache(self) -> None:
        """Replace the prompt cache with an empty one; nothing is reusable."""
        from mlx_lm.models.cache import make_prompt_cache

        self._prompt_cache = make_prompt_cache(self._model)
        self._last_prompt_tokens = None
        self._reuse_points = {}
        self._query_point = 0

    def _resume_point(self, common: int) -> int:
        """The longest prefix of at most *common* tokens the cache can return to exactly.

        KV layers trim back to any length.  A cache with other layers — e.g.
        the ``ArraysCache`` recurrent state of a hybrid model's
        linear-attention layers — can only return to a reuse point (see
        :meth:`_add_reuse_point`).
        """
        if all(_trimmable(layer) for layer in self._prompt_cache):
            return common
        return max((n for n in self._reuse_points if n <= common), default=0)

    def _rollback_cache_to(self, target: int) -> bool:
        """Roll the prompt cache back to exactly its first *target* tokens.

        Trimmable (KV) layers are trimmed.  Other layers are restored from the
        copies taken at the reuse point *target*, so they can only return to
        one (see :meth:`_resume_point`).  A cache that can't be rolled back
        exactly is rebuilt instead: trimming only some layers leaves them
        describing different token sequences, which corrupts every later
        generation.

        Returns ``True`` when the cache now holds *target* tokens, ``False``
        when it was rebuilt empty.
        """
        snapshot = self._reuse_points.get(target, {})
        exact = True
        for i, layer in enumerate(self._prompt_cache):
            if not exact:
                break
            if i in snapshot:
                self._prompt_cache[i] = _clone_cache_layer(snapshot[i])
            elif _trimmable(layer) and layer.offset >= target:
                layer.trim(layer.offset - target)
                exact = layer.offset == target
            else:
                exact = False
        if not exact:
            logger.info(
                "KV prefix cache: can't roll back exactly to %d tokens (reuse points %s) "
                "— rebuilding the cache",
                target, sorted(self._reuse_points),
            )
            self._reset_prompt_cache()
        return exact

    def _reuse_prefix(self, tokens: List[int]) -> int:
        """Prepare the prompt cache for *tokens*; return how many are already cached.

        The cache is rolled back to the longest common prefix it can restore
        exactly — on a hybrid model the longest reuse point inside it — or
        rebuilt empty: stale context never leaks into a prompt.  At least one
        token is always left to feed: ``stream_generate`` needs it to produce
        the first logits.
        """
        common = min(self._find_common_prefix(tokens), len(tokens) - 1)
        target = self._resume_point(common) if common > 0 else 0
        if target > 0 and self._rollback_cache_to(target):
            if target < common:
                logger.info(
                    "KV prefix cache: prompt diverges after %d tokens — resuming "
                    "from the %d-token reuse point", common, target,
                )
            # Until the boundary hook records this prompt, ``target`` is where
            # a failed or cancelled generation can still be resumed from.
            self._reuse_points = {n: s for n, s in self._reuse_points.items() if n <= target}
            self._last_prompt_tokens = tokens[:target]
            return target
        if common > 0:
            logger.info(
                "KV prefix cache: prompt diverges after %d tokens, before any reuse "
                "point (%s) — rebuilding the cache", common, sorted(self._reuse_points),
            )
        self._reset_prompt_cache()
        return 0

    def _add_reuse_point(self, tokens: List[int], query: bool = False) -> None:
        """Record that the prompt cache holds exactly *tokens* and can return to them.

        KV layers can always be trimmed back later; every other layer (e.g.
        the GatedDeltaNet ``ArraysCache`` of Qwen3.5 / Qwen3-Next, whose state
        folds in every token processed — generated ones included) is copied
        now: ~50 MB for Qwen3.8-9B.  *query* marks where the generation prompt
        of a prompt ending with a user turn starts (see :meth:`_turn_start`).
        """
        snapshot = {
            i: _clone_cache_layer(layer)
            for i, layer in enumerate(self._prompt_cache)
            if not _trimmable(layer)
        }
        # A new dict: ``copy.copy`` of this model shares the old one.
        self._reuse_points = {**self._reuse_points, len(tokens): snapshot}
        if query:
            self._query_point = len(tokens)
        if len(tokens) >= len(self._last_prompt_tokens or ()):
            self._last_prompt_tokens = tokens

    def _prompt_boundary_hook(self, tokens: List[int]):
        """Return a ``prompt_progress_callback`` that records the reusable prompt boundary.

        ``generate_step`` prefills every prompt token but the last, evaluating
        the cache and reporting ``(processed, total)`` after each chunk; the
        last token goes in with the first decode step.  At ``processed ==
        total - 1`` the cache therefore holds exactly ``tokens[:-1]`` — the
        furthest point the next call can resume from.

        Two reuse points are kept: this boundary, from which the next agent
        step (a tool result appended) resumes, and the start of the latest
        user turn's generation prompt, from which a follow-up resumes — it
        re-renders every assistant turn after that user turn (see
        :meth:`_turn_start`).  At ~50 MB each for Qwen3.8-9B, the others
        (static prefixes: the process-wide store has those) are dropped.
        """
        boundary = len(tokens) - 1
        cache = self._prompt_cache

        def _on_prompt_progress(processed: int, total: int) -> None:
            # Also bail if another call on this instance replaced the cache.
            if processed != total - 1 or cache is not self._prompt_cache:
                return
            self._add_reuse_point(tokens[:boundary])
            keep = (boundary, self._query_point)
            self._reuse_points = {n: s for n, s in self._reuse_points.items() if n in keep}

        return _on_prompt_progress

    def _turn_start(self, tokens: List[int]) -> int:
        """Where the generation prompt — the assistant-turn scaffold — starts in *tokens*; 0 if unknown.

        Once the turn is in the history the template may render it
        differently.  The Qwen3.5 template drops the empty reasoning block
        ``<think>\\n\\n</think>\\n\\n`` from assistant turns before the last user
        query, so the prompt after a direct answer (a follow-up, or an agent
        observation sent as a user message) diverges 3 tokens after this point
        — before the ``len(prompt) - 1`` boundary.  The scaffold is found by
        rendering a probe with and without the generation prompt, and counts
        only if it tokenizes the same at the end of *tokens*.
        """
        if self._scaffold is None:
            self._scaffold = self._probe_scaffold()
        start = len(tokens) - len(self._scaffold)
        if self._scaffold and start > 0 and tokens[start:] == self._scaffold:
            return start
        return 0

    def _probe_scaffold(self) -> List[int]:
        """Tokenize the template's generation prompt (``[]`` if it has none)."""
        probe = [HumanMessage("x")]
        try:
            with_prompt = self._to_prompt(probe)
            without = self._to_prompt(probe, add_generation_prompt=False)
        except Exception:  # noqa: BLE001 — a template the probe doesn't fit
            logger.debug("KV prefix cache: generation prompt probe failed", exc_info=True)
            return []
        if len(with_prompt) <= len(without) or not with_prompt.startswith(without):
            return []
        return self._tokenizer.encode(with_prompt[len(without):], add_special_tokens=False)

    def _static_prefixes(
        self,
        messages: List[BaseMessage],
        tools: Optional[list[dict]],
        tokens: List[int],
    ) -> tuple[Optional[str], List[int]]:
        """The tools' key and the lengths of the static prefixes of *tokens* worth a stored snapshot.

        A tool-calling agent's prompt opens with a block the conversation
        doesn't change.  Two points in it are worth a snapshot:

        * where the system prompt starts.  Qwen3.5 / Qwen3-Next templates
          render the tool block first, and it is shared by every session —
          unlike the system prompt, which in OTTO names the session's files
          directory and the current time;
        * where the first user message starts.  The session's other prompts —
          e.g. the one for its next user message — share the whole system turn.

        Each point is found by rendering the prompt twice through
        ``_to_prompt`` with only the text after it changed, and keeping the
        common part — no knowledge of the template needed (nor possible via a
        system-only render: Qwen3.5 templates reject a prompt without a user
        query).  A point counts only if that text tokenizes to an exact prefix
        of *tokens* (tokens merging across it would leave a snapshot of other
        tokens), is at least ``_STATIC_PREFIX_MIN_TOKENS`` long, leaves a token
        to feed and fits the cache budget.  Prompts without tools — title
        generation, memory ranking — have none, so they never evict an
        agent's snapshots.  The lengths are ascending; the key (see
        ``_tools_key``) names the agent's family in the store.
        """
        if not tools:
            return None, []
        lead: List[BaseMessage] = []
        for message in messages:
            if not isinstance(message, SystemMessage):
                break
            lead.append(message)
        probe_key = ([self._message_to_chat_dict(m) for m in lead], tools)
        if self._static_probe is None or self._static_probe[0] != probe_key:
            # Rendering and tokenizing ~35k tokens takes ~0.1 s: only redo it
            # when the system messages or the tools change.
            self._static_probe = (
                probe_key, self._probe_static_prefixes(lead, tools), _tools_key(tools),
            )
        budget = int(self.prompt_cache_max_tokens or 0)
        lengths: List[int] = []
        for prefix in self._static_probe[1]:
            n = len(prefix)
            if (
                _STATIC_PREFIX_MIN_TOKENS <= n < len(tokens)
                and (not budget or n <= budget)
                and (not lengths or n > lengths[-1])
                and tokens[:n] == prefix
            ):
                lengths.append(n)
        return self._static_probe[2], lengths

    def _probe_static_prefixes(
        self, lead: List[BaseMessage], tools: list[dict],
    ) -> List[List[int]]:
        """Tokenize the prompt up to its system prompt and up to its first user message."""
        # Each pair differs only after the point it locates.
        probes = (
            ([SystemMessage("a"), HumanMessage("x")], [SystemMessage("b"), HumanMessage("x")]),
            (lead + [HumanMessage("a")], lead + [HumanMessage("b")]),
        )
        try:
            prefixes = [
                _common_prefix(self._to_prompt(a, tools=tools), self._to_prompt(b, tools=tools))
                for a, b in probes
            ]
        except Exception:  # noqa: BLE001 — a template the probes don't fit
            logger.debug("KV prefix cache: static prefix probe failed", exc_info=True)
            return []
        return [self._tokenizer.encode(prefix) for prefix in prefixes]

    def _prepare_prompt_cache(
        self,
        tokens: List[int],
        static_lengths: List[int],
        tools_key: Optional[str] = None,
        turn_start: int = 0,
    ) -> _CachePlan:
        """Prepare the prompt cache for *tokens*.

        The cache ends up holding ``tokens[:reused + prefilled]``.  *reused*
        came from this instance's own cache (:meth:`_reuse_prefix`) or a
        static-prefix snapshot restored from the process-wide RAM store or,
        after a restart, from SSD; *prefilled* were fed here, before
        ``stream_generate`` runs.

        Of the prompt's static prefixes (*static_lengths*, see
        :meth:`_static_prefixes`) the longest one the instance's cache doesn't
        cover is restored if stored; each longer one is prefilled and stored,
        at exactly its length, for the next session — in the family of the
        agent's tools (*tools_key*); the ones worth a file are queued for SSD
        (see :meth:`_keep_on_ssd`).  On a cache that can't be trimmed (a
        hybrid model), a prompt ending with a user turn is prefilled up to its
        *turn_start* as well and gets a reuse point there: the next prompt
        re-renders this turn (see :meth:`_turn_start`).  Every restored or
        stored point is a reuse point of the instance too.  Called under
        ``MLX_GEN_LOCK``: rolling back, restoring and prefilling are Metal
        work.
        """
        reused = self._reuse_prefix(tokens)
        source = "session" if reused else "none"
        for n in reversed(static_lengths):
            if n <= reused:
                break
            where = self._restore_static_prefix(tokens[:n])
            if where is not None:
                self._add_reuse_point(tokens[:n])
                reused, source = n, where
                break
        cached, started = reused, time.perf_counter()
        for n in static_lengths:
            if n <= cached:
                continue
            self._prefill(tokens[cached:n])
            self._store_static_prefix(tokens[:n], tools_key, first=(n == static_lengths[0]))
            self._add_reuse_point(tokens[:n])
            cached = n
        if cached < turn_start and not all(_trimmable(layer) for layer in self._prompt_cache):
            self._prefill(tokens[cached:turn_start])
            self._add_reuse_point(tokens[:turn_start], query=True)
            cached = turn_start
        logger.info(
            "KV prefix cache: %d total tokens, %d reused%s, %d new (%.0f%% hit)",
            len(tokens), reused, _REUSE_SOURCES[source], len(tokens) - reused,
            (reused / len(tokens)) * 100 if tokens else 0,
        )
        return _CachePlan(reused, cached - reused, time.perf_counter() - started, source)

    def _restore_static_prefix(self, prefix: List[int]) -> Optional[str]:
        """Make the prompt cache the stored snapshot of exactly *prefix*.

        Returns where it came from — ``"ram"``, ``"ssd"`` — or ``None`` when
        it isn't stored.  A RAM hit is copied (the store's snapshot is
        shared) and counts as a use on SSD too (see :meth:`_keep_on_ssd`).  A
        snapshot loaded from SSD is this instance's own: it becomes the cache
        as is and stays out of the RAM store, where it would take another
        ~0.37 GB although the next session can load it again in ~0.1-0.3 s.
        """
        layers = _prefix_store.STATIC_PREFIXES.get(
            self._model, (self.kv_bits, self.kv_group_size), prefix,
        )
        key = self._ssd_key(prefix)
        if layers is not None:
            # Private copies: this instance writes into its cache.
            self._prompt_cache = [_clone_cache_layer(layer) for layer in layers]
            self._keep_on_ssd(key, layers, len(prefix))
            return "ram"
        if key is None:
            return None
        started = time.perf_counter()
        layers = _prefix_disk.SSD_PREFIXES.load(key, len(prefix), len(self._prompt_cache))
        if layers is None:
            return None
        self._prompt_cache = layers
        logger.info(
            "KV prefix cache: loaded the %d-token static prefix from SSD in %.2f s",
            len(prefix), time.perf_counter() - started,
        )
        return "ssd"

    def _store_static_prefix(
        self, prefix: List[int], tools_key: Optional[str], first: bool,
    ) -> None:
        """Snapshot the prompt cache, which holds exactly *prefix*, into the RAM store.

        *first*: *prefix* is the prompt's first static prefix, its tool block
        — written to SSD as well (see :meth:`_keep_on_ssd`).
        """
        store = _prefix_store.STATIC_PREFIXES
        layers = [_clone_cache_layer(layer) for layer in self._prompt_cache]
        store.put(
            self._model, (self.kv_bits, self.kv_group_size), prefix, layers,
            family=tools_key, nbytes=_snapshot_nbytes(layers),
        )
        self._keep_on_ssd(self._ssd_key(prefix), layers, len(prefix), first)
        logger.info(
            "KV prefix cache: stored the %d-token static prefix for other sessions "
            "(%d in RAM across %d agents)",
            len(prefix), len(store), store.family_count,
        )

    def _keep_on_ssd(
        self, key: Optional[str], layers: List[Any], n_tokens: int, first: bool = False,
    ) -> None:
        """Count a use of the static-prefix snapshot *layers* (*key*: its SSD file) on SSD.

        Its file is touched — the SSD tier evicts least recently used first —
        or, when it has none yet and is worth one (see
        ``DiskPrefixStore.wants``; *first*: the prompt's tool block), it is
        queued for :meth:`_write_ssd_snapshots`.
        """
        if key is not None and _prefix_disk.SSD_PREFIXES.wants(key, first):
            self._ssd_writes = [*self._ssd_writes, (key, layers, n_tokens)]

    def _write_ssd_snapshots(self) -> None:
        """Write the snapshots :meth:`_keep_on_ssd` queued.

        Runs once the reply is generated and outside ``MLX_GEN_LOCK``, so a
        write (~0.37 GB for OTTO's tool block) holds up neither this call's
        first token nor another agent.  The layers are evaluated copies that
        nothing writes into, built on this thread, so writing them doesn't
        need the lock.
        """
        writes, self._ssd_writes = self._ssd_writes, []
        for key, layers, n_tokens in writes:
            started = time.perf_counter()
            if _prefix_disk.SSD_PREFIXES.save(key, layers, n_tokens):
                logger.info(
                    "KV prefix cache: wrote the %d-token static prefix to SSD in %.2f s",
                    n_tokens, time.perf_counter() - started,
                )

    def _ssd_key(self, prefix: List[int]) -> Optional[str]:
        """The SSD file key of the snapshot of *prefix*; ``None`` without an SSD tier or local weights."""
        if _prefix_disk.SSD_PREFIXES.directory is None:
            return None
        fingerprint = weights_fingerprint(self._model)
        if fingerprint is None:
            return None
        return _prefix_disk.snapshot_key(fingerprint, (self.kv_bits, self.kv_group_size), prefix)

    def _prefill(self, tokens: List[int]) -> None:
        """Feed *tokens* into the prompt cache without generating anything.

        ``generate_step`` with ``max_tokens=0`` runs exactly the prefill of a
        normal generation — chunked, with the same KV quantisation — then stops
        before its first decode step, so the cache ends at exactly *tokens*.
        """
        import mlx.core as mx
        from mlx_lm.generate import generate_step, generation_stream, wired_limit

        kv_kwargs: dict = {}
        if self.kv_bits is not None:
            kv_kwargs = {"kv_bits": self.kv_bits, "kv_group_size": self.kv_group_size}
        with wired_limit(self._model, [generation_stream]):
            for _ in generate_step(
                mx.array(tokens), self._model,
                max_tokens=0, prompt_cache=self._prompt_cache, **kv_kwargs,
            ):
                pass

    def _enforce_cache_budget(self) -> None:
        """Keep the prompt cache under the budget without breaking prefix reuse.

        Long autonomous sessions can otherwise grow the cache without bound
        and trip macOS into swap or OOM.  Strategy:

        * No cap (``prompt_cache_max_tokens == 0``) → no-op (legacy behaviour).
        * Cache empty / disabled, or at or under the cap → no-op.
        * Over the cap, but the reusable prompt prefix fits → roll back to
          that prefix, dropping only the generated tail.  Trimming below it
          would throw away exactly what the next call reuses (and can't be
          done exactly on hybrid caches at all).
        * Prefix itself over the cap (or unknown) → rebuild the cache from
          scratch so the memory is actually released; the next call pays a
          full prefill.

        Releasing the Metal allocator pool (``mx.clear_cache``) is left to
        the caller — :meth:`_generate` already does it once per turn.
        """
        budget = int(self.prompt_cache_max_tokens or 0)
        if budget <= 0 or self._prompt_cache is None:
            return
        current = self._cache_offset()
        if current <= budget:
            return

        prefix = self._last_prompt_tokens or []
        if prefix and len(prefix) <= budget and self._rollback_cache_to(len(prefix)):
            logger.info(
                "MLX KV cache exceeded budget (%d > %d tokens) — dropped the "
                "generated tail, keeping the %d-token reusable prompt prefix",
                current, budget, len(prefix),
            )
            return
        self._reset_prompt_cache()
        logger.warning(
            "MLX KV cache exceeded budget (%d > %d tokens) and the reusable "
            "prompt prefix (%d tokens) doesn't fit — rebuilt from scratch. "
            "Next turn will pay a full prefill; raise prompt_cache_max_tokens "
            "above the prompt size to keep prefix reuse.",
            current, budget, len(prefix),
        )

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
        boundary_hook = None
        prefilled, prefill_seconds, source = 0, 0.0, None
        if self.enable_system_prompt_cache and self._prompt_cache is not None:
            full_tokens = self._tokenizer.encode(prompt_str)
            tools_key, static_lengths = self._static_prefixes(messages, tools, full_tokens)
            # The next prompt re-renders a user turn's reply: resume before it.
            turn_start = (
                self._turn_start(full_tokens)
                if messages and isinstance(messages[-1], HumanMessage) else 0
            )
            # Rolling the cache back, restoring snapshots and prefilling static
            # prefixes run Metal work, so they hold the generation lock too.
            with MLX_GEN_LOCK:
                common, prefilled, prefill_seconds, source = self._prepare_prompt_cache(
                    full_tokens, static_lengths, tools_key, turn_start,
                )
            prompt = full_tokens[common + prefilled:]
            boundary_hook = self._prompt_boundary_hook(full_tokens)
            cache_offset_before = common
        else:
            cache_offset_before = self._cache_offset()

        gen_kwargs: dict = {"max_tokens": self.max_tokens, **self._sampler_kwargs()}
        if self._prompt_cache is not None:
            gen_kwargs["prompt_cache"] = self._prompt_cache
        if boundary_hook is not None:
            gen_kwargs["prompt_progress_callback"] = boundary_hook
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
        self._write_ssd_snapshots()

        response_metadata = self._build_response_metadata(
            last_response, cache_offset_before, prefilled, prefill_seconds, source,
        )

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
