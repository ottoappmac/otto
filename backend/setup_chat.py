"""Conversational response generator for the first-run setup chat.

Tries to use a small local Qwen3-1.7B-4bit MLX model when it is cached
on disk; falls back to pre-written template strings otherwise.  The
model is intentionally NOT auto-downloaded here — the frontend drives
the download via the existing ``POST /api/mlx/download`` route and
signals readiness through the ``model_ready`` field on the request.

The module is purely a text-generation layer.  All config mutations
(provider selection, API key storage, etc.) are done by the caller via
the existing ``PUT /api/settings`` and related endpoints.
"""

from __future__ import annotations

import asyncio
import logging
import platform
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SETUP_MODEL_ID = "mlx-community/Qwen3-1.7B-4bit"

# Template replies used when the LLM is unavailable or when the user's
# answer is unrecognised and we need a deterministic clarification prompt.
_TEMPLATES: dict[str, dict[str, str]] = {
    "provider": {
        "mlx": (
            "Running locally keeps everything on your Mac — no API keys needed. "
            "Let me check the models that fit your hardware."
        ),
        "anthropic": (
            "Claude is excellent for complex reasoning and writing. "
            "Which service would you like — Anthropic's API directly, or AWS Bedrock?"
        ),
        "openai": (
            "GPT models are versatile and widely supported. "
            "Which flavour — OpenAI directly, or Azure OpenAI?"
        ),
        "omlx": (
            "The oMLX server gives you optimised local inference with continuous batching. "
            "I'll guide you through the installation next."
        ),
        "unknown": (
            "I didn't quite catch that. "
            "Would you like to use a cloud AI service (Anthropic Claude / OpenAI) "
            "or run everything locally on your Mac?"
        ),
    },
    "cloud_sub": {
        "anthropic": (
            "Anthropic's API gives you direct access to Claude. "
            "Please paste your API key — you can find it at console.anthropic.com."
        ),
        "openai": (
            "OpenAI's API covers all GPT models. "
            "Please paste your API key from platform.openai.com."
        ),
        "azure": (
            "Azure OpenAI gives you enterprise-grade GPT access. "
            "I'll need your endpoint URL, deployment name, and API key."
        ),
        "bedrock": (
            "AWS Bedrock lets you run Claude through Amazon's infrastructure. "
            "I'll need your AWS region and credentials."
        ),
        "unknown": (
            "Which cloud service — Anthropic, OpenAI, Azure OpenAI, or AWS Bedrock?"
        ),
    },
    "cloud_key": {
        "success": (
            "Connected! Your {provider} account is working. "
            "Let's sort out two more quick settings."
        ),
        "failure": (
            "That key didn't connect — please check it and try again. "
            "Make sure there are no extra spaces."
        ),
    },
    "local_model": {
        "confirmed": (
            "Done. {model_id} is ready to go — "
            "it runs comfortably on your Mac and gives solid performance."
        ),
        "clarify": (
            "Which model would you like? "
            "I can use {recommended} (~{size_gb} GB) or you can pick from the full catalog."
        ),
    },
    "omlx": {
        "already_running": (
            "The oMLX server is already running. "
            "Which model would you like it to serve?"
        ),
        "needs_install": (
            "oMLX isn't installed yet. "
            "Opening the guided installer to set it up for you."
        ),
    },
    "memory": {
        "yes": (
            "Otto will remember things across your conversations — "
            "preferences, context, topics. You can fine-tune it in Settings later."
        ),
        "no": (
            "Memory stays off. You can enable it any time from Settings."
        ),
        "unknown": (
            "Should Otto remember things across conversations? "
            "For example, your preferences or context from past sessions. (Yes / No)"
        ),
    },
    "activity": {
        "yes": (
            "Activity tracking is on. "
            "macOS will ask for Accessibility permission — please grant it to complete setup."
        ),
        "no": (
            "No activity tracking. You can enable it in Settings whenever you like."
        ),
        "unknown": (
            "Should Otto track your Mac activity in the background "
            "to provide better context and suggestions? (Yes / No)"
        ),
    },
    "ambient": {
        "yes": (
            "Great — ambient suggestions are on. "
            "Otto will quietly watch your memory, sessions, and Mac activity "
            "and surface ideas when you're free. You can fine-tune the model "
            "and cadence any time in Settings → Agent Memory → Ambient."
        ),
        "no": (
            "No problem — ambient suggestions are off. "
            "You can turn them on from Settings → Agent Memory → Ambient whenever you like."
        ),
        "unknown": (
            "Would you like Otto to proactively surface suggestions based on your "
            "past sessions and Mac activity? It runs a small background model and "
            "only notifies you when you're idle. (Yes / No)"
        ),
    },
    "evaluation": {
        "yes": (
            "Auto-evaluation is on. "
            "Each completed run will be scored automatically — an LLM picks "
            "suitable metrics so you can track quality over time. "
            "You can tune the metrics and model in Settings → Observability."
        ),
        "no": (
            "No problem — runs won't be evaluated automatically. "
            "You can score any run with the Evaluate button, or turn this on "
            "later from Settings → Observability."
        ),
        "unknown": (
            "When a run finishes, should Otto automatically evaluate it? "
            "An LLM picks suitable metrics and scores the result so you can "
            "track quality over time. (Yes / No)"
        ),
    },
    "done": {
        "default": (
            "You're all set! Your settings have been saved. "
            "Click 'Open Otto' to start chatting."
        ),
    },
}

# ---------------------------------------------------------------------------
# MLX model cache (module-level, loaded lazily)
# ---------------------------------------------------------------------------

_IS_APPLE_SILICON: bool = (
    platform.system() == "Darwin" and platform.machine() == "arm64"
)

# Tuple of (model, tokenizer) once loaded; None until first successful load.
_setup_model: tuple[Any, Any] | None = None
_setup_model_load_attempted: bool = False


def _get_cached_model_path(model_id: str) -> str | None:
    """Return the local path of a cached HF repo, or None if not present."""
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_id, local_files_only=True)
    except Exception:  # noqa: BLE001
        return None


def _try_load_model(model_id: str) -> tuple[Any, Any] | None:
    """Attempt to load the model from the local HF cache.

    Never triggers a network download.  Returns None on any failure so
    callers can fall back to templates silently.
    """
    if not _IS_APPLE_SILICON:
        return None

    cached = _get_cached_model_path(model_id)
    if cached is None:
        return None

    try:
        from mlx_lm import load  # type: ignore[import-untyped]

        model, tokenizer = load(cached)
        logger.info("Setup chat model loaded: %s", model_id)
        return (model, tokenizer)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load setup model %s: %s", model_id, exc)
        return None


def _ensure_model(model_id: str) -> tuple[Any, Any] | None:
    """Return the cached (model, tokenizer) pair, loading on first call."""
    global _setup_model, _setup_model_load_attempted  # noqa: PLW0603
    if not _setup_model_load_attempted:
        _setup_model_load_attempted = True
        _setup_model = _try_load_model(model_id)
    return _setup_model


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are Otto's friendly setup assistant helping a user through first-time \
configuration. Keep your replies concise (1–3 sentences), warm, and clear. \
Never use bullet points or markdown. Confirm what you understood from their \
reply, then naturally lead into the next step if one is given.\
"""


def _build_prompt(
    step: str,
    user_message: str,
    extracted: str | None,
    context: dict[str, Any],
    tokenizer: Any,
) -> str:
    chip = context.get("chip", "your Mac")
    ram_gb = context.get("ram_gb", "?")

    user_turn = (
        f"Setup step: {step}\n"
        f"Hardware: {chip} with {ram_gb} GB RAM\n"
        f"User said: {user_message}\n"
        f"What was understood: {extracted or 'unclear — ask for clarification'}\n\n"
        "Reply:"
    )

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_turn},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # Fallback ChatML format (most small models use this)
    return (
        f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_turn}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


async def _generate(
    prompt: str,
    model: Any,
    tokenizer: Any,
    max_tokens: int,
    temp: float,
) -> str:
    """Run synchronous MLX generation in a thread pool."""

    def _sync() -> str:
        from mlx_lm import generate  # type: ignore[import-untyped]

        return generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temp,
            verbose=False,
        )

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _template(step: str, key: str, **kwargs: str) -> str | None:
    """Look up a template string, formatting any ``{placeholders}``."""
    bucket = _TEMPLATES.get(step, {})
    raw = bucket.get(key)
    if raw is None:
        return None
    try:
        return raw.format(**kwargs)
    except KeyError:
        return raw


async def generate_reply(
    *,
    step: str,
    user_message: str,
    extracted: str | None,
    context: dict[str, Any],
    model_ready: bool,
    model_id: str = SETUP_MODEL_ID,
    max_tokens: int = 150,
    temp: float = 0.4,
) -> str:
    """Generate a natural-language reply for the given setup step.

    Parameters
    ----------
    step:
        Current wizard step (``provider``, ``cloud_sub``, ``cloud_key``,
        ``local_model``, ``memory``, ``activity``, ``ambient``,
        ``evaluation``, ``done``).
    user_message:
        Raw text the user typed.
    extracted:
        The structured value the frontend extracted from ``user_message``
        (e.g. ``"anthropic"`` for the provider step).  ``None`` means
        the answer was ambiguous.
    context:
        Machine context dict with at least ``chip`` and ``ram_gb``.
    model_ready:
        ``True`` when the frontend has confirmed the setup model is in
        the local HF cache.  We skip loading when this is ``False``.
    model_id:
        HF repo id of the setup LLM (defaults to
        ``mlx-community/Qwen3-1.7B-4bit``).
    max_tokens / temp:
        Generation parameters.

    Returns
    -------
    str
        A short conversational reply.
    """
    # --- attempt LLM generation ---
    if model_ready:
        loaded = _ensure_model(model_id)
        if loaded is not None:
            model, tokenizer = loaded
            try:
                prompt = _build_prompt(
                    step, user_message, extracted, context, tokenizer
                )
                reply = await _generate(prompt, model, tokenizer, max_tokens, temp)
                if reply.strip():
                    return reply.strip()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Setup model generation failed: %s", exc)

    # --- template fallback ---
    key = extracted or "unknown"
    tmpl = _template(step, key)
    if tmpl:
        return tmpl

    # Generic fallback
    return _template(step, "unknown") or "Got it — let's continue with setup."
