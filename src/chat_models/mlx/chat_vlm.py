"""Vision-language MLX chat model wrapper.

Wraps ``mlx_vlm`` for use as a LangChain ``BaseChatModel``.  Unlike the
original single-message implementation this version passes the full message
history to ``apply_chat_template`` so multi-turn conversations and
``MLXReActMiddleware`` scratchpads are formatted correctly.

Usage::

    from chat_models.mlx.chat_vlm import MLXVLChatModel

    llm = MLXVLChatModel(model_path="mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
"""

import asyncio
import base64
import io
import logging
import math
import threading
from typing import Any, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict

logger = logging.getLogger(__name__)


# ── Shared VLM registry ───────────────────────────────────────────────────────
#
# Mirrors the cache in :mod:`chat_models.mlx.chat_mlx_text` — VLMs are even
# heavier than text models (vision tower + projector + LM), so loading more
# than one copy per process is a fast way to OOM Metal memory.

_VLMTriple = Tuple[Any, Any, Any]  # (model, processor, config)
_LOADED_VLMS: dict[str, _VLMTriple] = {}
_VLM_WARMED_UP: set[str] = set()
_VLM_LOAD_LOCK = threading.Lock()


def _load_or_reuse_vlm(model_path: str) -> Tuple[_VLMTriple, bool]:
    """Return cached ``(model, processor, config)`` or load on miss.

    Returns ``(triple, freshly_loaded)`` so the caller can gate one-time
    side effects (logging, warmup) on the first load.
    """
    cached = _LOADED_VLMS.get(model_path)
    if cached is not None:
        return cached, False

    with _VLM_LOAD_LOCK:
        cached = _LOADED_VLMS.get(model_path)
        if cached is not None:
            return cached, False

        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        model, processor = load(model_path)
        config = load_config(model_path)
        triple = (model, processor, config)
        _LOADED_VLMS[model_path] = triple
        return triple, True


def loaded_mlx_vlms() -> list[str]:
    """Return the list of currently cached VLM model paths."""
    return list(_LOADED_VLMS.keys())


# ── Library compatibility patches ─────────────────────────────────────────────

def _patch_qwen3_vl_text_only() -> None:
    """Patch mlx_vlm ≤0.4.0 Qwen3-VL to handle text-only (no-image) input.

    ``Qwen3VLLanguageModel.__call__`` unconditionally slices ``visual_pos_masks``
    when ``n_to_process`` is set, even though ``visual_pos_masks`` is ``None``
    on text-only turns.  This raises::

        TypeError: 'NoneType' object is not subscriptable

    The fix adds the missing ``visual_pos_masks is not None`` guard.
    Applied once at import time; safe to call multiple times (idempotent).
    Remove when mlx_vlm ships a fix upstream.
    """
    try:
        import mlx_vlm.models.qwen3_vl.language as _qvl
    except ImportError:
        return

    cls = getattr(_qvl, "Qwen3VLLanguageModel", None) or getattr(_qvl, "LanguageModel", None)
    if cls is None:
        logger.warning("qwen3_vl text-only patch: could not find LanguageModel class, skipping")
        return

    if getattr(cls.__call__, "_patched_text_only", False):
        return

    _orig = cls.__call__

    def _patched(self, *args, visual_pos_masks=None, **kwargs):
        if visual_pos_masks is None:
            # _orig reads n_to_process from kwargs and slices visual_pos_masks
            # unconditionally — remove it so the slice is never attempted.
            kwargs.pop("n_to_process", None)
        return _orig(self, *args, visual_pos_masks=visual_pos_masks, **kwargs)

    _patched._patched_text_only = True
    cls.__call__ = _patched
    logger.debug("Applied qwen3_vl text-only patch (mlx_vlm ≤0.4.0 workaround)")


_patch_qwen3_vl_text_only()


def _to_role_content(msg: BaseMessage) -> dict[str, str]:
    """Convert a LangChain message to the ``{"role": ..., "content": ...}`` dict
    that ``mlx_vlm.prompt_utils.apply_chat_template`` expects.

    Multimodal content blocks (lists with ``image_url`` items) are collapsed
    to their text parts only — the actual images are extracted separately by
    :func:`_extract_images`.
    """
    raw = msg.content
    if isinstance(raw, list):
        text_parts = [
            block.get("text", "") for block in raw
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        content = "\n".join(text_parts) or str(raw)
    else:
        content = str(raw)
    if isinstance(msg, SystemMessage):
        return {"role": "system", "content": content}
    if isinstance(msg, AIMessage):
        return {"role": "assistant", "content": content}
    return {"role": "user", "content": content}


def _extract_images(messages: List[BaseMessage]) -> list:
    """Scan all messages for inline base64 images and return them as PIL Images.

    Looks for content blocks of type ``image_url`` with a
    ``data:image/...;base64,...`` URL. Returns a list of ``PIL.Image`` objects
    in the order they appear.
    """
    from PIL import Image

    images: list = []
    for msg in messages:
        if not isinstance(msg.content, list):
            continue
        for block in msg.content:
            if not isinstance(block, dict) or block.get("type") != "image_url":
                continue
            url = (block.get("image_url") or {}).get("url", "")
            if url.startswith("data:image/"):
                b64_data = url.split(",", 1)[-1]
                images.append(Image.open(io.BytesIO(base64.b64decode(b64_data))))
    return images


class MLXVLChatModel(BaseChatModel):
    """``BaseChatModel`` backed by a local ``mlx_vlm`` vision-language model.

    Passes the **full** message history to ``apply_chat_template`` so that
    multi-turn conversations — including the ``Thought / Action / Observation``
    scratchpad produced by ``MLXReActMiddleware`` — are formatted correctly.

    Images are supported via the ``images`` keyword argument in ``_generate``
    (or ``_agenerate``), which accepts a list of ``PIL.Image`` objects or URLs.

    Args:
        model_path:         HuggingFace model ID or local path
                            (e.g. ``"mlx-community/Qwen2.5-VL-7B-Instruct-4bit"``).
        max_tokens:         Maximum tokens to generate (default 8192).
        verbose:            Print token stream to stdout while generating (default False).
        kv_bits:            Quantize the KV cache to this many bits (4 or 8).
                            ``None`` keeps full-precision cache.  Equivalent to
                            ``mlx_lm``'s ``--cache-8bit`` / ``--cache-4bit``.
        kv_group_size:      Group size for KV cache quantization (default 64).
    """

    model_path: str
    max_tokens: int = 8192
    verbose: bool = False
    enable_prompt_cache: bool = False
    kv_bits: Optional[int] = None
    kv_group_size: int = 64
    image_max_side: Optional[int] = None
    """Cap the long side of input images to this many pixels before inference.

    Reduces vision-encoder patch count (and prefill cost) linearly with the
    ratio of the original to capped side.  ``None`` (default) auto-detects
    from the processor's ``max_pixels`` config; set explicitly to override.
    Use ``0`` to disable resizing entirely.
    """

    # mlx objects are not JSON-serialisable
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _model: Any = None
    _processor: Any = None
    _config: Any = None
    _prompt_cache: Any = None
    _image_max_side: Optional[int] = None  # resolved at init from field or processor config
    _last_prompt_tokens: Optional[List[int]] = None  # for text-only prefix reuse

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path=model_path, **kwargs)
        if self.model_path in _LOADED_VLMS:
            logger.info(
                "MLX VLM %s reused from process cache (no reload)",
                self.model_path,
            )
        else:
            logger.info("Loading MLX VLM: %s", self.model_path)

        triple, freshly_loaded = _load_or_reuse_vlm(self.model_path)
        self._model, self._processor, self._config = triple

        if freshly_loaded:
            try:
                import mlx.core as mx
                mem_gb = mx.get_active_memory() / (1024**3)
                logger.info(
                    "MLX VLM loaded: %s (active GPU memory: %.2f GB)",
                    self.model_path, mem_gb,
                )
            except Exception:
                logger.info("MLX VLM loaded: %s", self.model_path)

        if self.enable_prompt_cache:
            from mlx_vlm.models import cache as vlm_cache
            self._prompt_cache = vlm_cache.make_prompt_cache(self._model.language_model)

        self._image_max_side = self._resolve_image_max_side()
        if self._image_max_side:
            logger.info("VLM image resize cap: long side ≤ %d px", self._image_max_side)

        if self.model_path not in _VLM_WARMED_UP:
            self._warmup()
            _VLM_WARMED_UP.add(self.model_path)

    # ── Native tool-calling API ──────────────────────────────────────────────

    def supports_native_tools(self) -> bool:
        """Return False — VLMs in this codebase use the ReAct text shim.

        The ``MLXReActWrapper`` checks this method to decide whether to bypass
        the ReAct shim.  Returning ``False`` keeps VLMs on the well-tested
        ReAct path, which is sufficient for the vision workflows used here
        (screen description, web-voyager, computer-voyager).

        Native tool calling for VLMs would require routing tools through the
        underlying tokenizer's ``apply_chat_template(tools=...)`` rather than
        ``mlx_vlm.prompt_utils.apply_chat_template``, plus per-family
        message-dict round-tripping for tool results.  Leaving as a TODO
        until a vision use case actually needs it.
        """
        return False

    # ── Warmup ────────────────────────────────────────────────────────────────

    def _warmup(self) -> None:
        """Trigger MLX graph compilation with a short dummy generation.

        The first call to ``stream_generate`` traces and compiles the
        computation graph.  Running a short warmup at init time pays that
        cost once so that the first real user request is not penalised.
        """
        try:
            from mlx_vlm import stream_generate

            from chat_models.mlx._shared import MLX_GEN_LOCK

            dummy_prompt = "Hi"
            with MLX_GEN_LOCK:
                for _ in stream_generate(
                    self._model,
                    self._processor,
                    dummy_prompt,
                    None,
                    max_tokens=1,
                    verbose=False,
                ):
                    pass
                import mlx.core as mx
                mx.clear_cache()
            logger.debug("MLX VLM warmup complete (%s)", self.model_path)
        except Exception:
            logger.debug("MLX VLM warmup skipped", exc_info=True)

    # ── Image helpers ─────────────────────────────────────────────────────────

    def _resolve_image_max_side(self) -> Optional[int]:
        """Return the effective max long-side cap for input images.

        Priority: explicit ``image_max_side`` field → processor ``max_pixels``
        → processor ``size`` dict/int → ``None`` (no resize).

        ``image_max_side=0`` disables resizing even if the processor suggests one.
        """
        if self.image_max_side is not None:
            return self.image_max_side or None  # 0 → no resize

        try:
            ip = getattr(self._processor, "image_processor", None)
            if ip is None:
                return None
            if hasattr(ip, "max_pixels"):
                return int(math.isqrt(ip.max_pixels))
            if hasattr(ip, "size"):
                sz = ip.size
                if isinstance(sz, dict):
                    return max(sz.get("height", 0), sz.get("width", 0)) or None
                if isinstance(sz, int):
                    return sz or None
        except Exception:
            pass
        return None

    def _resize_images(self, images: list) -> list:
        """Proportionally resize images so neither side exceeds ``_image_max_side``.

        Only resizes when the image is actually larger than the cap — small
        images are returned unchanged.  No-op when ``_image_max_side`` is None.
        """
        cap = self._image_max_side
        if not cap or not images:
            return images

        resized = []
        for img in images:
            w, h = img.size  # PIL: (width, height)
            if max(w, h) > cap:
                ratio = cap / max(w, h)
                new_w, new_h = int(w * ratio), int(h * ratio)
                from PIL import Image
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                logger.debug("Resized image %dx%d → %dx%d (cap=%d)", w, h, new_w, new_h, cap)
            resized.append(img)
        return resized

    # ── Prompt formatting ─────────────────────────────────────────────────────

    def _to_prompt(
        self,
        messages: List[BaseMessage],
        images: list,
    ) -> str:
        """Convert the full LangChain message list to an mlx_vlm prompt string.

        Uses ``apply_chat_template`` with the complete message history so that
        multi-turn scratchpads are preserved.  Images are only tagged on the
        first user turn (``apply_chat_template`` handles ``skip_image_token``
        internally when a list of dicts is provided).
        """
        from mlx_vlm.prompt_utils import apply_chat_template

        role_content_msgs = [_to_role_content(m) for m in messages]
        return apply_chat_template(
            self._processor,
            self._config,
            role_content_msgs,
            num_images=len(images),
        )

    # ── Cache stats ───────────────────────────────────────────────────────────

    def _cache_offset(self) -> int:
        """Return the current KV cache offset (0 when caching is disabled or unsupported).

        Not all cache types expose an ``offset`` (e.g. ``ArraysCache`` used by
        GatedDeltaNet / state-space models).  Fall back to 0 gracefully.
        """
        if self._prompt_cache is None:
            return 0
        return getattr(self._prompt_cache[0], "offset", 0)

    def _get_vlm_tokenizer(self) -> Any:
        """Return the underlying text tokenizer from the VLM processor."""
        return getattr(self._processor, "tokenizer", self._processor)

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
        """Roll the VLM KV cache back to *target* tokens.

        Trims each layer individually — non-trimmable layers (e.g. ArraysCache
        used by state-space sub-layers) are skipped gracefully.
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
                    trimmed = n
        logger.debug(
            "VLM cache trimmed by %d tokens (%d → %d)",
            trimmed, current, self._cache_offset(),
        )

    def _build_response_metadata(self, last_response: Any, cache_offset_before: int, vision: bool) -> dict:
        """Build response metadata attached to ``AIMessage.response_metadata``.

        LangChain's standard location for model-level stats (token counts, TPS,
        memory, cache metrics).  Visible in LangSmith's Metadata panel and
        preserved through any number of wrapper layers.
        """
        tokens_prefilled = last_response.prompt_tokens
        tokens_from_cache = cache_offset_before
        total = tokens_from_cache + tokens_prefilled
        return {
            "tokens_from_cache": tokens_from_cache,
            "tokens_prefilled": tokens_prefilled,
            "cache_hit_ratio": round(tokens_from_cache / total, 3) if total else 0.0,
            "prompt_tps": round(last_response.prompt_tps, 1),
            "generation_tokens": last_response.generation_tokens,
            "generation_tps": round(last_response.generation_tps, 1),
            "cache_offset_after": self._cache_offset(),
            "peak_memory_gb": round(last_response.peak_memory, 3),
            "vision_turn": vision,
        }

    # ── Synchronous generation ────────────────────────────────────────────────

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        from mlx_vlm import stream_generate

        import mlx.core as mx

        images: list = kwargs.get("images", [])
        if not images:
            images = _extract_images(messages)
        images = self._resize_images(images)

        formatted_prompt = self._to_prompt(messages, images)

        gen_kwargs: dict = {"verbose": self.verbose, "max_tokens": self.max_tokens}
        if self._prompt_cache is not None:
            gen_kwargs["prompt_cache"] = self._prompt_cache
        if self.kv_bits is not None:
            gen_kwargs["kv_bits"] = self.kv_bits
            gen_kwargs["kv_group_size"] = self.kv_group_size

        from chat_models.mlx._shared import MLX_GEN_LOCK

        if images:
            # Image tokens shift KV positions — discard any text-only prefix state
            # and rebuild the cache so stale offsets are never reused.
            if self._prompt_cache is not None:
                from mlx_vlm.models import cache as vlm_cache
                self._prompt_cache = vlm_cache.make_prompt_cache(self._model.language_model)
                gen_kwargs["prompt_cache"] = self._prompt_cache
                logger.debug("Prompt cache reset for vision turn (%d image(s))", len(images))
            self._last_prompt_tokens = None
        elif self._prompt_cache is not None:
            # Text-only turn: attempt prefix reuse to avoid re-prefilling the
            # static system prompt and prior conversation history.
            tokenizer = self._get_vlm_tokenizer()
            full_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=True)
            common = self._find_common_prefix(full_tokens)
            new_tokens = len(full_tokens) - common
            if common > 0:
                self._trim_cache_to(common)
                if self._cache_offset() == common:
                    gen_kwargs["input_ids"] = mx.array(full_tokens[common:])
                else:
                    logger.warning(
                        "VLM KV prefix trim failed (wanted %d, got %d) — sending full prompt",
                        common, self._cache_offset(),
                    )
                    common = 0
                    new_tokens = len(full_tokens)
            logger.info(
                "VLM KV prefix cache: %d total tokens, %d reused, %d new (%.0f%% hit)",
                len(full_tokens), common, new_tokens,
                (common / len(full_tokens)) * 100 if full_tokens else 0,
            )
            self._last_prompt_tokens = full_tokens

        # Read cache offset after any trim/reset — this is what stream_generate
        # will use as its starting position, and is the correct "reused" count.
        cache_offset_before = self._cache_offset()

        text = ""
        last_response = None
        # Hold the shared MLX generation lock for the entire stream — see
        # ``MLX_GEN_LOCK`` in :mod:`chat_models.mlx.chat_mlx_text` for why a
        # process-wide lock is required to avoid Metal command-buffer aborts.
        with MLX_GEN_LOCK:
            for response in stream_generate(
                self._model,
                self._processor,
                formatted_prompt,
                images or None,
                **gen_kwargs,
            ):
                text += response.text
                last_response = response

        response_metadata = self._build_response_metadata(last_response, cache_offset_before, vision=bool(images))

        try:
            mx.clear_cache()
        except Exception:
            pass

        logger.debug(
            "vision=%s cache_hit_ratio=%.2f tokens_from_cache=%d tokens_prefilled=%d "
            "prompt_tps=%.1f gen_tps=%.1f cache_offset_after=%d peak_mem=%.3fGB",
            response_metadata["vision_turn"],
            response_metadata["cache_hit_ratio"],
            response_metadata["tokens_from_cache"],
            response_metadata["tokens_prefilled"],
            response_metadata["prompt_tps"],
            response_metadata["generation_tps"],
            response_metadata["cache_offset_after"],
            response_metadata["peak_memory_gb"],
        )
        return ChatResult(
            generations=[ChatGeneration(
                message=AIMessage(content=text, response_metadata=response_metadata),
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
        return "mlx-vlm-chat"
