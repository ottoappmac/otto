"""LLM and VLM construction from environment configuration."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from langchain_core.language_models import BaseChatModel

from utilities.environment import Environment

logger = logging.getLogger(__name__)


# ── Loop-recovery temperature bridge for OpenAI-compatible local servers ───────
#
# ChatOpenAI (used for oMLX and exo) fixes ``temperature`` at construction time
# and never changes it.  The ToolLoopGuard recovery path calls
# ``request_temperature_bump()`` which writes a one-shot override to the
# process-global ``_shared._recovery_temp``.  In the direct (in-process) MLX
# path that override is consumed by ``ChatMLXText._sampler_kwargs``; oMLX and
# exo, however, run as separate HTTP servers and would otherwise never see it.
#
# The mixin below bridges that gap for the HTTP providers: it overrides
# ``_get_request_payload`` (the single method that builds the dict sent to
# /v1/chat/completions) and injects the bumped temperature whenever a
# loop-guard recovery is pending — by calling the SAME
# ``consume_temperature_bump()`` accessor the in-process path uses, so all
# three providers honour a loop-recovery bump.  Both ``_OMLXChat`` and
# ``_ExoChat`` (defined lazily in the provider branches below) mix it in.
# Inheriting from ChatOpenAI means all other behaviour — tool calls, streaming,
# retries, auth — is unchanged.

class _LoopRecoveryChatOpenAI:
    """Mixin that injects a loop-recovery temperature bump into every request.

    Designed to be mixed with ``ChatOpenAI``:

        class _OMLXChat(_LoopRecoveryChatOpenAI, ChatOpenAI): ...

    ``_get_request_payload`` is called for both streaming and non-streaming
    paths in langchain_openai, so this hook covers both.
    """

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)  # type: ignore[misc]
        try:
            from chat_models.mlx._shared import consume_temperature_bump

            bump = consume_temperature_bump()
            if bump is not None:
                configured = payload.get("temperature", 0.0) or 0.0
                if bump > configured:
                    logger.warning(
                        "%s: loop-recovery temperature bump active — "
                        "sampling at temp=%.2f (configured=%.2f).",
                        self.__class__.__name__,
                        bump,
                        configured,
                    )
                    payload["temperature"] = bump
        except Exception:
            pass  # never let a bump failure affect the main generation path
        return payload


# ── oMLX per-request throughput stats ────────────────────────────────────────
#
# oMLX's OpenAI-compatible ``usage`` object carries server-specific extension
# fields beyond the standard token counts:
#
#     "usage": {
#         "prompt_tokens": 9752, "completion_tokens": 554,
#         "cached_tokens": 0,
#         "prompt_tokens_per_second": 164.93,
#         "generation_tokens_per_second": 8.22,
#         "time_to_first_token": 115.05, "generation_duration": 67.42
#     }
#
# We translate those into the SAME ``response_metadata`` keys that
# ``ChatMLXText._build_response_metadata`` emits for the in-process ``mlx``
# provider, so the existing pipeline (session accumulation → chat stats panel →
# Runs → Dashboard) lights up for oMLX with no further wiring.  ``langchain_openai``
# only surfaces the standard counts via ``usage_metadata`` and drops the extras,
# which is why we capture them ourselves in ``_OMLXChat._create_chat_result``.

def _map_omlx_usage(usage: dict) -> dict:
    """Translate an oMLX ``usage`` dict into ChatMLXText-style stats keys.

    Handles two generations of oMLX response format:
    - Newer: explicit ``prompt_tokens_per_second`` / ``generation_tokens_per_second``
      and top-level ``cached_tokens``.
    - Current: ``total_time`` (total request seconds) and
      ``prompt_tokens_details.cached_tokens`` (standard OpenAI format).

    Returns ``{}`` when none of the oMLX extras are present (plain OpenAI
    responses), so the caller safely skips the update.
    """
    if not isinstance(usage, dict):
        return {}

    # --- TPS fields (newer oMLX versions) ---
    prompt_tps_raw = usage.get("prompt_tokens_per_second")
    gen_tps_raw = usage.get("generation_tokens_per_second")

    # --- Cache hits ---
    # Newer: top-level "cached_tokens"; current: prompt_tokens_details.cached_tokens
    cached_top = usage.get("cached_tokens")
    details = usage.get("prompt_tokens_details") or {}
    cached_nested = details.get("cached_tokens")
    cached = cached_top if cached_top is not None else cached_nested

    # --- Timing (current oMLX versions) ---
    total_time = usage.get("total_time")

    # Bail if this looks like a plain OpenAI usage with none of the oMLX extras.
    # The standard shape has only prompt_tokens / completion_tokens / total_tokens.
    if (prompt_tps_raw is None and gen_tps_raw is None
            and cached is None and total_time is None):
        return {}

    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    cached_tokens = int(cached or 0)
    prefilled = max(prompt_tokens - cached_tokens, 0)

    stats: dict = {}

    # Cache stats
    if cached is not None:
        stats["tokens_from_cache"] = cached_tokens
        stats["tokens_prefilled"] = prefilled
        if prompt_tokens > 0:
            stats["cache_hit_ratio"] = round(cached_tokens / prompt_tokens, 3)

    # TIPS — only when explicitly provided (oMLX doesn't expose prefill time in
    # current releases, so total_time can't be split accurately).
    if prompt_tps_raw is not None:
        stats["prompt_tps"] = round(float(prompt_tps_raw), 1)

    # TOPS — prefer explicit field; fall back to completion_tokens / total_time
    # (a lower bound because total_time includes prefill, but useful when the
    # cache hit ratio is high and prefill is fast).
    if gen_tps_raw is not None:
        stats["generation_tps"] = round(float(gen_tps_raw), 1)
    elif total_time and total_time > 0 and completion_tokens > 0:
        stats["generation_tps"] = round(completion_tokens / total_time, 1)

    if completion_tokens:
        stats["generation_tokens"] = completion_tokens

    return stats


def _extract_usage_dict(response: Any) -> dict:
    """Pull the raw ``usage`` mapping off an OpenAI response (dict or pydantic),
    preserving server-specific extension fields."""
    usage: Any = None
    if isinstance(response, dict):
        usage = response.get("usage")
    else:
        usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    # OpenAI SDK pydantic model — model_dump keeps extras (extra="allow").
    if hasattr(usage, "model_dump"):
        try:
            return usage.model_dump()
        except Exception:  # noqa: BLE001
            pass
    try:
        return dict(usage)
    except Exception:  # noqa: BLE001
        return {}


# The concrete subclasses are defined lazily (inside the ``if provider ==``
# branches) so that ``langchain_openai`` is only imported when those providers
# are actually used.  The mixin above is pure Python and has no imports of its
# own, so it is safe to define unconditionally at module load time.

# Reasoning-distill models (DeepSeek-R1 and its Qwen/Llama distillations)
# simulate the whole task inside their ``<think>`` block and emit a prose
# "final answer" without ever producing structured ``tool_calls``.  They need
# the :class:`MLXReActWrapper` text shim (``force_action=True``) to be coerced
# into emitting an ``Action:`` block that we can parse into a real tool call.
_REASONING_MODEL_KEYWORDS = ("deepseek-r1", "deepseek_r1", "r1-distill")


def _is_reasoning_model_id(model_id: str) -> bool:
    """Whether *model_id* names a reasoning-distill model (DeepSeek-R1 family).

    Matched case-insensitively against the model id / repo name.  Used to
    decide whether to wrap a model in :class:`MLXReActWrapper` with
    ``force_action=True`` so it produces tool calls instead of a prose plan.
    """
    name = (model_id or "").lower()
    return any(kw in name for kw in _REASONING_MODEL_KEYWORDS)


# OpenAI / Azure models that share two quirks vs. the older chat models:
#   1. Only accept ``temperature=1`` (any other value is rejected).
#   2. Use ``max_completion_tokens`` instead of ``max_tokens`` — passing
#      ``max_tokens`` returns HTTP 400 ``unsupported_parameter``.
# This covers the GPT-5 family and the o-series reasoning models
# (o1, o2, o3, o4 and their dated/mini/preview variants).
_MODERN_OPENAI_PREFIXES = (
    "gpt-5",
    "o1", "o2", "o3", "o4",
)


def _is_modern_openai_model(model_name: str) -> bool:
    """Whether *model_name* belongs to the GPT-5 / o-series family.

    Match is on the dash-delimited prefix so e.g. ``gpt-5``, ``gpt-5-mini``,
    ``o1-preview``, ``o3-mini-2025-01-31`` all return True, while
    ``gpt-4o`` and ``gpt-4.1`` return False.
    """
    name = (model_name or "").lower().strip()
    for prefix in _MODERN_OPENAI_PREFIXES:
        if name == prefix or name.startswith(prefix + "-") or name.startswith(prefix + "."):
            return True
    return False


def _openai_temperature(model_name: str, configured: float) -> float:
    """Return the temperature to use for an OpenAI/Azure model.

    Modern OpenAI models (GPT-5, o-series reasoning) reject any temperature
    value other than the default (1).  For those we silently coerce to 1.
    """
    return 1.0 if _is_modern_openai_model(model_name) else configured


def _openai_token_kwargs(model_name: str, max_tokens: int) -> dict:
    """Return the correct token-limit kwargs for an OpenAI/Azure constructor.

    Modern OpenAI models (GPT-5, o-series) require ``max_completion_tokens``
    and reject ``max_tokens`` with HTTP 400.  Older models still expect
    ``max_tokens``.  We pass ``max_completion_tokens`` via ``model_kwargs``
    so this works regardless of the installed ``langchain_openai`` version
    (older releases don't expose ``max_completion_tokens`` as a constructor
    parameter, but ``model_kwargs`` is forwarded straight to the OpenAI
    SDK).
    """
    if max_tokens is None or max_tokens <= 0:
        return {}
    if _is_modern_openai_model(model_name):
        return {"model_kwargs": {"max_completion_tokens": max_tokens}}
    return {"max_tokens": max_tokens}


def _resolve_omlx_model_id(base_url: str, requested: str) -> str:
    """Map Otto's stored model name to oMLX's actual model id.

    Otto stores HuggingFace repo ids (``mlx-community/Qwen3.5-9B-4bit``)
    while oMLX can expose models under several id formats depending on how
    they were discovered:

    * Short directory name only (``Qwen3.5-9B-4bit``) — when the model dir
      sits directly inside ``model_dirs``.
    * Double-dash org--name form (``mlx-community--Qwen3.5-9B-4bit``) —
      when oMLX scans the HuggingFace hub cache and finds a directory like
      ``models--mlx-community--Foo-4bit``, it strips the ``models--`` prefix
      but preserves the ``org--name`` structure.

    Resolution order (first match wins):

    1. Exact match against ``GET /v1/models``.
    2. Basename after last ``/`` (drops org prefix).
    3. Double-dash form: ``org/name`` → ``org--name``.
    4. Case-insensitive comparison of all three forms.

    Falls back to the short basename if the server is unreachable — the
    session-start auto-load has already ensured the model is registered,
    so this is a safe last-ditch default.
    """
    import httpx
    short = requested.rsplit("/", 1)[-1]
    dashed = requested.replace("/", "--")
    try:
        with httpx.Client(timeout=4.0) as client:
            r = client.get(f"{base_url.rstrip('/')}/v1/models")
            if r.status_code != 200:
                return short
            data = r.json().get("data") or []
            ids = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]
    except Exception:  # noqa: BLE001
        return short
    if requested in ids:
        return requested
    if short in ids:
        return short
    if dashed in ids:
        return dashed
    lowered = {i.lower(): i for i in ids}
    for cand in (requested, short, dashed):
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return short


def _resolve_bedrock_creds() -> dict:
    """Return explicit AWS credential kwargs for ChatBedrockConverse.

    When ``ANTHROPIC_BEDROCK_AUTH_MODE`` is not ``keys``, returns an empty
    dict so boto3 uses the default credential chain (SSO, instance role, etc.).
    """
    if Environment.get_anthropic_bedrock_auth_mode() != "keys":
        return {}
    key_id = os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    if not (key_id and secret):
        raise RuntimeError(
            "No AWS credentials available for Bedrock. "
            "Set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY or configure access keys in Settings."
        )
    return {
        "aws_access_key_id": key_id,
        "aws_secret_access_key": secret,
    }


def _build_mlx_chat(
    *,
    model_id: str,
    draft_model_id: str,
    max_tokens: int,
    num_draft_tokens: int,
    thinking: bool,
    enable_prompt_cache: bool,
    enable_system_prompt_cache: bool,
    kv_bits: Any,
    kv_group_size: int,
) -> BaseChatModel:
    """Return a chat model for local MLX inference.

    Always uses the classic :class:`ChatMLXText` path. Turbo mode has
    been deprecated in favour of the oMLX server (which is a strictly
    superior reimplementation of the same ideas: paged KV allocator,
    continuous batching, prefix-tree sharing). Existing ``turbo_level``
    settings in ``config.json`` are now ignored; users that want
    oMLX-style optimisations should switch the provider to ``omlx``.
    """
    from chat_models.mlx import ChatMLXText

    prompt_cache_max_tokens = Environment.get_mlx_prompt_cache_max_tokens()

    turbo_level = Environment.get_mlx_turbo_level()
    if turbo_level != "off":
        logger.info(
            "MLX turbo mode (level=%s) is deprecated and ignored — using "
            "classic ChatMLXText. For oMLX-style optimisations, switch the "
            "LLM provider to 'omlx' in Settings.",
            turbo_level,
        )

    return ChatMLXText(
        model_path=model_id,
        draft_model_path=draft_model_id,
        num_draft_tokens=num_draft_tokens,
        max_tokens=max_tokens,
        temp=Environment.get_mlx_temp(),
        repetition_penalty=Environment.get_mlx_repetition_penalty(),
        repetition_context_size=Environment.get_mlx_repetition_context_size(),
        verbose=Environment.get_mlx_verbose(),
        thinking=thinking,
        enable_prompt_cache=enable_prompt_cache,
        enable_system_prompt_cache=enable_system_prompt_cache,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        prompt_cache_max_tokens=prompt_cache_max_tokens,
    )


def create_llm(provider: str) -> BaseChatModel:
    """Return a text-oriented ``BaseChatModel`` based on *provider*.

    Reads model IDs, tokens, regions, etc. from :class:`Environment`.

    When the privacy lock is engaged (Settings → Privacy & Security),
    cloud providers are refused before any client is constructed.  This
    is the single chokepoint -- there is no other path to an LLM
    inside Otto, so the guard here is sufficient to enforce the
    no-cloud promise.
    """
    logger.info("Creating main LLM (provider=%s)", provider)

    try:
        from backend.privacy_lock import enforce_provider_allowed
        enforce_provider_allowed(provider)
    except ImportError:
        # backend.privacy_lock is unavailable when src/ is imported
        # outside of the backend process (notebook examples, isolated
        # smoke tests).  The lock is still enforced wherever it
        # matters because the backend always imports it.
        pass

    if provider == "anthropic":
        model_provider = Environment.get_anthropic_model_provider()
        model_name = Environment.get_anthropic_model_name()
        logger.info(
            "Anthropic LLM: model=%s, transport=%s",
            model_name, model_provider,
        )

        if model_provider == "bedrock":
            from botocore.config import Config as BotoConfig
            from langchain_aws import ChatBedrockConverse

            additional_fields: dict = {}
            if Environment.get_anthropic_thinking_flag():
                additional_fields["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": Environment.get_anthropic_thinking_budget(),
                }

            region = Environment.get_anthropic_bedrock_region()
            boto_cfg = BotoConfig(
                read_timeout=600,
                max_pool_connections=50,
                retries={"max_attempts": 3, "mode": "adaptive"},
                request_min_compression_size_bytes=1048576,
            )

            client_kwargs = _resolve_bedrock_creds()

            return ChatBedrockConverse(
                model=model_name,
                region_name=region,
                additional_model_request_fields=additional_fields or None,
                config=boto_cfg,
                **client_kwargs,
            )

        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model_name, temperature=0.0)

    if provider == "cohere":
        from langchain_cohere import ChatCohere

        cohere_model = Environment.get_cohere_model()
        logger.info("Cohere LLM: model=%s", cohere_model)
        return ChatCohere(
            cohere_api_key=Environment.get_cohere_api_key(),
            model=cohere_model,
            temperature=0.0,
        )

    if provider == "mlx":
        from middleware.react_wrapper import MLXReActWrapper

        model_id = Environment.get_hf_llm_model_id()
        draft_id = Environment.get_hf_draft_llm_model_id()
        logger.info(
            "MLX LLM: model=%s%s",
            model_id,
            f", draft={draft_id}" if draft_id else "",
        )
        inner = _build_mlx_chat(
            model_id=model_id,
            draft_model_id=draft_id,
            num_draft_tokens=Environment.get_mlx_num_draft_tokens(),
            max_tokens=Environment.get_mlx_max_tokens(),
            thinking=Environment.get_mlx_thinking(),
            enable_prompt_cache=Environment.get_mlx_prompt_cache(),
            enable_system_prompt_cache=Environment.get_mlx_system_prompt_cache(),
            kv_bits=Environment.get_mlx_kv_bits(),
            kv_group_size=Environment.get_mlx_kv_group_size(),
        )
        return MLXReActWrapper(inner, force_action=_is_reasoning_model_id(model_id))

    if provider == "openai":
        model_provider = Environment.get_openai_model_provider()
        model_name = Environment.get_openai_model_name()
        max_tokens = Environment.get_openai_max_tokens()

        if model_provider == "azure":
            from langchain_openai import AzureChatOpenAI

            endpoint = Environment.get_openai_azure_endpoint()
            if not endpoint:
                raise ValueError(
                    "OPENAI_AZURE_ENDPOINT is empty. Set the Azure OpenAI endpoint "
                    "in Settings → LLM → OpenAI."
                )
            deployment = Environment.get_openai_azure_deployment()
            # On Azure the deployment name is the real model identifier — it
            # already falls back to model_name when blank (see Environment).
            # Use it for parameter routing so that a deployment named "gpt-5"
            # or "o4-mini" correctly gets max_completion_tokens / temperature=1.
            effective_name = deployment or model_name
            temperature = _openai_temperature(effective_name, Environment.get_openai_temperature())
            token_kwargs = _openai_token_kwargs(effective_name, max_tokens)
            logger.info(
                "OpenAI LLM (Azure): model=%s, deployment=%s, temperature=%s, token_param=%s",
                model_name, deployment, temperature,
                "max_completion_tokens" if _is_modern_openai_model(effective_name) else "max_tokens",
            )
            api_version = Environment.get_openai_azure_api_version()
            api_key = Environment.get_openai_api_key()
            return AzureChatOpenAI(
                azure_endpoint=endpoint,
                azure_deployment=deployment,
                api_version=api_version,
                api_key=api_key or "no-key",
                model=model_name,
                temperature=temperature,
                timeout=600.0,
                **token_kwargs,
            )

        temperature = _openai_temperature(model_name, Environment.get_openai_temperature())
        token_kwargs = _openai_token_kwargs(model_name, max_tokens)
        logger.info(
            "OpenAI LLM: model=%s, temperature=%s, token_param=%s",
            model_name, temperature,
            "max_completion_tokens" if _is_modern_openai_model(model_name) else "max_tokens",
        )
        from langchain_openai import ChatOpenAI

        api_key = Environment.get_openai_api_key()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is empty. Set the API key in Settings → LLM → OpenAI."
            )
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            timeout=600.0,
            **token_kwargs,
        )

    if provider == "omlx":
        # oMLX exposes an OpenAI-compatible ``/v1/chat/completions``
        # endpoint served by a local process (managed by
        # ``backend.omlx_provisioner``).  The server doesn't validate the
        # API key (``skip_api_key_verification`` is set true at provision
        # time), but ``langchain_openai`` requires *some* value, so we
        # supply a sentinel.  See ``model_factory.create_llm(provider="exo")``
        # for the analogous wiring.
        from langchain_openai import ChatOpenAI

        class _OMLXChat(_LoopRecoveryChatOpenAI, ChatOpenAI):  # type: ignore[misc]
            """ChatOpenAI for oMLX with loop-recovery temperature bumps and
            per-request throughput stats (TIPS/TOPS/KV cache).

            oMLX returns prefill/generation TPS and KV cache hits in the
            response ``usage`` object; we surface them on the AIMessage's
            ``response_metadata`` under the same keys the in-process MLX path
            uses, so the session stats pipeline treats both identically.
            """

            def _create_chat_result(self, response, generation_info=None):  # type: ignore[override]
                result = super()._create_chat_result(response, generation_info)
                try:
                    usage_raw = _extract_usage_dict(response)
                    stats = _map_omlx_usage(usage_raw)
                    logger.debug(
                        "oMLX _create_chat_result: usage_keys=%s stats=%s",
                        list(usage_raw.keys()) if usage_raw else [],
                        stats or "(none — no oMLX extras in response)",
                    )
                    if stats:
                        for gen in result.generations:
                            msg = getattr(gen, "message", None)
                            if msg is not None and hasattr(msg, "response_metadata"):
                                msg.response_metadata.update(stats)
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "oMLX stats extraction failed", exc_info=True
                    )
                return result

        base = Environment.get_omlx_base_url()
        if not base:
            raise ValueError(
                "OMLX_BASE_URL is empty. Enable oMLX and start the server "
                "before selecting it as the LLM provider."
            )
        model_name = Environment.get_omlx_model_name()
        if not model_name:
            raise ValueError(
                "OMLX_MODEL_NAME is empty. Pick a model in "
                "Settings → LLM → Models (provider = omlx)."
            )

        # oMLX exposes models under their short directory name
        # (e.g. ``Qwen3.5-9B-4bit``) while Otto stores HuggingFace repo
        # ids (e.g. ``mlx-community/Qwen3.5-9B-4bit``).  Translate the
        # latter so /v1/chat/completions doesn't 404 with "Model not
        # found".  We probe /v1/models synchronously to find the right
        # form; if the server is unreachable we fall back to the short
        # basename (the load step in session_manager has already ensured
        # the model is registered, so this is a safe default).
        resolved = _resolve_omlx_model_id(base, model_name)
        # oMLX has no global thinking toggle; control it per request via the
        # chat template kwarg the server forwards to the tokenizer.  Sending
        # it explicitly (true *or* false) overrides the model's template
        # default so the Settings toggle is always authoritative.
        thinking = Environment.get_omlx_thinking()
        extra_body = {"chat_template_kwargs": {"enable_thinking": thinking}}
        # Cap generation length.  Without max_tokens the server falls back
        # to sampling.max_tokens (32768 by default), letting a model that
        # fails to emit a stop token run away to the full cap (a single
        # ~10-minute completion).  Configurable via Settings → oMLX.
        omlx_max_tokens = Environment.get_omlx_max_tokens()
        is_reasoning_model = _is_reasoning_model_id(model_name)
        logger.info(
            "oMLX LLM: requested=%s, resolved=%s, base=%s/v1, thinking=%s, "
            "max_tokens=%s, reasoning_shim=%s",
            model_name, resolved, base, thinking, omlx_max_tokens,
            is_reasoning_model,
        )
        client = _OMLXChat(
            base_url=f"{base.rstrip('/')}/v1",
            model=resolved,
            api_key="omlx-no-auth",
            temperature=0.0,
            timeout=600.0,
            max_tokens=omlx_max_tokens,
            # Discourage degenerate repetition (no sampler repetition_penalty
            # on this OpenAI-compatible path); RepetitionGuard is the backstop.
            frequency_penalty=Environment.get_llm_frequency_penalty(),
            presence_penalty=Environment.get_llm_presence_penalty(),
            extra_body=extra_body,
            # Ask oMLX to append its usage (token counts + throughput +
            # cached_tokens) to streaming responses; on non-streaming calls
            # the usage is always present.  ``_create_chat_result`` maps the
            # oMLX-specific extras into MLX-style ``response_metadata`` stats.
            stream_usage=True,
        )
        # Reasoning-distill models (DeepSeek-R1 family) don't reliably emit
        # OpenAI-format ``tool_calls`` over oMLX's /v1/chat/completions — they
        # return the tool call as prose in ``content`` with
        # ``finish_reason="stop"``, so the agent loop exits having done nothing.
        # Wrap them in the ReAct text shim (force_action) so tool calls are
        # parsed out of the model's text, mirroring the in-process ``mlx`` path.
        # ``ChatOpenAI`` has no ``supports_native_tools()`` method, so the
        # wrapper automatically takes its text-based ReAct path.
        if is_reasoning_model:
            from middleware.react_wrapper import MLXReActWrapper

            return MLXReActWrapper(client, force_action=True)
        return client

    if provider == "exo":
        # exo exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint
        # served by the local cluster (see backend/exo_cli.py). Point a
        # generic ChatOpenAI at it; the exo daemon doesn't validate the
        # API key but ``langchain_openai`` requires *some* value, so we
        # supply a sentinel.
        from langchain_openai import ChatOpenAI

        class _ExoChat(_LoopRecoveryChatOpenAI, ChatOpenAI):  # type: ignore[misc]
            """ChatOpenAI for exo with loop-recovery temperature bumps."""

        base = Environment.get_exo_base_url()
        if not base:
            raise ValueError(
                "EXO_BASE_URL is empty. Enable exo and start the cluster "
                "before selecting it as the LLM provider."
            )
        model_name = Environment.get_exo_model_name()
        if not model_name:
            raise ValueError(
                "EXO_MODEL_NAME is empty. Pick a model in "
                "Settings → LLM → Models (provider = exo)."
            )
        # exo has no global thinking toggle; control it per request via the
        # OpenAI-compatible ``enable_thinking`` flag (exo maps it to
        # ``reasoning_effort`` server-side).  Sent explicitly so the Settings
        # toggle always overrides the model template default.
        thinking = Environment.get_exo_thinking()
        extra_body = {"enable_thinking": thinking}
        # Cap generation length.  Without max_tokens exo falls back to the
        # model's full context window, letting a model that fails to emit a
        # stop token run away to a multi-minute completion.
        exo_max_tokens = Environment.get_exo_max_tokens()
        logger.info(
            "exo LLM: model=%s, base=%s/v1, thinking=%s, max_tokens=%s",
            model_name, base, thinking, exo_max_tokens,
        )
        return _ExoChat(
            base_url=f"{base.rstrip('/')}/v1",
            model=model_name,
            api_key="exo-no-auth",  # cluster ignores it; lib requires non-empty
            temperature=0.0,
            timeout=600.0,
            max_tokens=exo_max_tokens,
            # Discourage degenerate repetition (no sampler repetition_penalty
            # on this OpenAI-compatible path); RepetitionGuard is the backstop.
            frequency_penalty=Environment.get_llm_frequency_penalty(),
            presence_penalty=Environment.get_llm_presence_penalty(),
            extra_body=extra_body,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER='{provider}'. "
        "Set LLM_PROVIDER to one of: anthropic, openai, cohere, mlx, exo, omlx."
    )


def create_deep_agent_llm(provider: str) -> BaseChatModel | None:
    """Return a dedicated orchestrator LLM when ``DEEP_AGENT_LLM_PROVIDER`` is set.

    Resolution logic:

    +--------------------------+--------------------------------------------+
    | DEEP_AGENT_LLM_PROVIDER  | Behaviour                                  |
    +--------------------------+--------------------------------------------+
    | (empty)                  | Return None → caller uses default LLM.     |
    | same as *provider*,      | Return None → same provider, same model,   |
    |   non-mlx                | no point creating a duplicate.             |
    | same as *provider*,      | Use DEEP_AGENT_MLX_MODEL_ID if set,        |
    |   mlx                    | else return None.                          |
    | different, non-mlx       | ``create_llm(da_provider)`` — e.g.         |
    |                          | anthropic orchestrator + mlx subagents.    |
    | different, mlx           | Use DEEP_AGENT_MLX_MODEL_ID if set,        |
    |                          | else HF_LLM_MODEL_ID.                     |
    +--------------------------+--------------------------------------------+

    Returns ``None`` when no separate model is needed.
    """
    da_provider = Environment.get_deep_agent_llm_provider()
    if not da_provider:
        return None

    same_provider = da_provider == provider

    if da_provider != "mlx":
        if same_provider:
            return None
        return create_llm(da_provider)

    # da_provider == "mlx"
    da_model_id = Environment.get_deep_agent_mlx_model_id()
    main_model_id = Environment.get_hf_llm_model_id()

    # Avoid loading a duplicate copy of the SAME MLX model — each ChatMLXText
    # pulls a full set of weights into Metal memory (~2 GB for a 4-bit 4B,
    # ~17 GB for a 4-bit 35B MoE).  When the orchestrator id matches the main
    # LLM we re-use the main model instead of paying that cost twice.
    if same_provider and (not da_model_id or da_model_id == main_model_id):
        if da_model_id and da_model_id == main_model_id:
            logger.info(
                "Deep Agent MLX model id matches main LLM (%s) — reusing main "
                "instance instead of loading a second copy.",
                main_model_id,
            )
        return None

    from middleware.react_wrapper import MLXReActWrapper

    model_id = da_model_id or main_model_id
    logger.info(
        "Creating Deep Agent orchestrator MLX LLM: model=%s (separate instance from main LLM)",
        model_id,
    )
    inner = _build_mlx_chat(
        model_id=model_id,
        draft_model_id=Environment.get_hf_draft_llm_model_id(),
        num_draft_tokens=Environment.get_mlx_num_draft_tokens(),
        max_tokens=Environment.get_mlx_max_tokens(),
        thinking=Environment.get_mlx_thinking(),
        enable_prompt_cache=Environment.get_mlx_prompt_cache(),
        enable_system_prompt_cache=Environment.get_mlx_system_prompt_cache(),
        kv_bits=Environment.get_mlx_kv_bits(),
        kv_group_size=Environment.get_mlx_kv_group_size(),
    )
    return MLXReActWrapper(inner, force_action=_is_reasoning_model_id(model_id))


def _mlx_is_vision_model(model_path: str) -> bool:
    """Return True if a local MLX model has a vision tower.

    Tries to read the downloaded snapshot's ``config.json`` and checks for
    the ``vision_config`` key that all VL model families (Qwen-VL, LLaVA,
    PaliGemma, InternVL, …) place there.  Falls back to a repo-name regex
    when the file is missing or unreadable (e.g. download in progress).
    """
    _VLM_NAME_RE = re.compile(
        r"\bvl\b|vision|llava|paligemma|moondream|internvl", re.IGNORECASE,
    )
    try:
        from mlx_vlm.utils import load_config

        cfg = load_config(model_path)
        # load_config returns a SimpleNamespace / dataclass-like object
        return (
            hasattr(cfg, "vision_config")
            or (isinstance(cfg, dict) and "vision_config" in cfg)
        )
    except Exception:
        return bool(_VLM_NAME_RE.search(model_path))


_CLOUD_VISION_PROVIDERS: frozenset[str] = frozenset({"anthropic", "openai"})
_VLM_NAME_RE = re.compile(
    r"\bvl\b|vision|llava|paligemma|moondream|internvl", re.IGNORECASE,
)


def _config_declares_vision(model_id: str) -> bool:
    """Return True if *model_id*'s cached ``config.json`` declares vision.

    Reads the config strictly from the local HF cache via
    :func:`huggingface_hub.try_to_load_from_cache` — it never triggers a
    download, so it is safe to call at session startup for oMLX/exo models
    whose weights may live only on the inference server.  Returns False when
    the config isn't cached or can't be parsed.
    """
    try:
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(model_id, "config.json")
        if not isinstance(path, str):
            return False
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        return "vision_config" in cfg or "image_token_id" in cfg
    except Exception:
        return False


def supports_vision(provider: str, model_id: str = "") -> bool:
    """Return True if *provider* + *model_id* can process image content blocks.

    Decision matrix:

    * **Cloud providers** (``anthropic``, ``openai``) — assume vision-capable;
      the relevant Anthropic and OpenAI models all accept base64 image blocks.
    * **MLX** — inspect the downloaded snapshot's ``config.json`` via
      :func:`_mlx_is_vision_model`; fall back to a name-based regex.
    * **oMLX / exo** — name regex first (cheap), then inspect the model's
      ``config.json`` in the local HF cache for a ``vision_config``.  Many
      multimodal models (e.g. the Qwen3.x MoE VLMs) carry no "vl"/"vision"
      token in their repo name but do declare ``vision_config``, so the
      name heuristic alone misclassifies them as text-only.
    * Everything else — conservatively returns ``False``.

    Args:
        provider: The provider string as stored in config (e.g. ``"mlx"``,
                  ``"anthropic"``, ``"omlx"``).
        model_id: The model identifier — HF repo id for MLX, model name for
                  oMLX/exo, ignored for cloud providers.
    """
    if provider in _CLOUD_VISION_PROVIDERS:
        return True
    if provider == "mlx":
        return _mlx_is_vision_model(model_id)
    if provider in ("omlx", "exo"):
        if _VLM_NAME_RE.search(model_id):
            return True
        return _config_declares_vision(model_id)
    return False


def create_mlx_vlm(provider: str, llm: BaseChatModel) -> BaseChatModel:
    """Return a vision-language model.

    For MLX this is a dedicated ``MLXVLChatModel``; for other providers
    the text LLM is returned as-is (most API models handle images natively).

    Returns *llm* unchanged when ``HF_VLM_MODEL_ID`` is not configured.
    """
    if provider == "mlx":
        vlm_model_id = Environment.get_hf_vlm_model_id()
        if not vlm_model_id:
            logger.info("No HF_VLM_MODEL_ID configured — using text LLM for vision tasks")
            return llm

        from chat_models.mlx.chat_vlm import MLXVLChatModel

        logger.info("MLX VLM: model=%s", vlm_model_id)
        return MLXVLChatModel(
            model_path=vlm_model_id,
            max_tokens=Environment.get_mlx_max_tokens(),
            enable_prompt_cache=Environment.get_mlx_prompt_cache(),
            kv_bits=Environment.get_mlx_kv_bits(),
            kv_group_size=Environment.get_mlx_kv_group_size(),
        )
    return llm
