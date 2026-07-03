"""Memory Layer 2 — per-turn topic relevance injection.

Runs a lightweight side-query on each LLM turn to pick the most
relevant memory topics, then injects their full content into the
system message.  The side-query uses the configured memory model
(falls back to the main provider model when not set).

This is implemented as deepagents ``AgentMiddleware`` so it
intercepts every model call transparently.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from backend.config import MemoryConfig

logger = logging.getLogger(__name__)

MAX_TOPICS = 5


def _memory_dir():
    from backend.config import get_app_data_dir
    return get_app_data_dir() / "memory"


def _append_hit(
    session_id: str,
    query: str,
    topics: list[str],
    cached: bool,
) -> None:
    """Append a memory hit record to the JSONL log."""
    path = _memory_dir() / "memory-hits.jsonl"
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "query": query[:200],
        "topics": topics,
        "cached": cached,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        logger.debug("Failed to write memory hit", exc_info=True)


def _scan_descriptions() -> list[dict[str, str]]:
    """Read frontmatter descriptions from topic files."""
    mem = _memory_dir()
    if not mem.exists():
        return []
    results: list[dict[str, str]] = []
    for f in mem.glob("*.md"):
        if f.name == "MEMORY.md":
            continue
        entry: dict[str, str] = {"file": f.name}
        try:
            text = f.read_text(encoding="utf-8")
            for line in text.split("\n")[:10]:
                s = line.strip()
                if s.startswith("name:"):
                    entry["name"] = (
                        s.split(":", 1)[1].strip().strip('"')
                    )
                elif s.startswith("description:"):
                    entry["description"] = (
                        s.split(":", 1)[1].strip().strip('"')
                    )
        except Exception:
            pass
        if "description" in entry:
            results.append(entry)
    return results


def _read_topic_file(filename: str) -> str:
    """Read full content of a topic file."""
    p = _memory_dir() / filename
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""


RANKING_PROMPT = """\
You are a relevance ranker. Given a user message and a list \
of memory topic descriptions, return the filenames of the \
{max_topics} most relevant topics as a JSON array.

## Topics
{topics}

## Rules
- Return ONLY a JSON array of filenames, e.g. ["a.md", "b.md"]
- If no topics are relevant, return []
- Max {max_topics} topics
"""


def _build_ranking_prompt(
    topics: list[dict[str, str]],
) -> str:
    lines = "\n".join(
        f"- {t['file']}: {t.get('description', '(no desc)')}"
        for t in topics
    )
    return RANKING_PROMPT.format(
        topics=lines,
        max_topics=MAX_TOPICS,
    )


_BEDROCK_HAIKU_FALLBACK = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_ANTHROPIC_HAIKU_DEFAULT = "claude-haiku-4-5-latest"

_resolved_bedrock_haiku: str | None = None


def _resolve_bedrock_haiku(region: str, boto_cfg: Any) -> str:
    """Discover the best Haiku model from Bedrock inference profiles.

    Queries active profiles, picks the newest Haiku, and caches the
    result for the lifetime of the process.  Falls back to a hardcoded
    ID if the API call fails or returns nothing.
    """
    global _resolved_bedrock_haiku  # noqa: PLW0603
    if _resolved_bedrock_haiku:
        return _resolved_bedrock_haiku

    try:
        import boto3

        from deep_agent.model_factory import _read_aws_creds_from_env

        env = _read_aws_creds_from_env()
        session = boto3.Session(
            aws_access_key_id=env["access_key"],
            aws_secret_access_key=env["secret_key"],
            aws_session_token=env.get("token"),
            region_name=region,
        )
        client = session.client("bedrock", config=boto_cfg)
        candidates: list[str] = []
        token = None
        while True:
            kwargs: dict = {
                "maxResults": 1000,
                "typeEquals": "SYSTEM_DEFINED",
            }
            if token:
                kwargs["nextToken"] = token
            resp = client.list_inference_profiles(**kwargs)
            for p in resp.get(
                "inferenceProfileSummaries", [],
            ):
                if p.get("status") != "ACTIVE":
                    continue
                pid = p["inferenceProfileId"]
                if "haiku" not in pid.lower():
                    continue
                models = p.get("models", [])
                if any(
                    "anthropic" in (m.get("modelArn") or "")
                    for m in models
                ):
                    candidates.append(pid)
            token = resp.get("nextToken")
            if not token:
                break

        if candidates:
            candidates.sort(reverse=True)
            _resolved_bedrock_haiku = candidates[0]
            logger.info(
                "[memory] auto-resolved Haiku model: %s",
                _resolved_bedrock_haiku,
            )
            return _resolved_bedrock_haiku
    except Exception:
        logger.debug(
            "[memory] Haiku auto-resolve failed, using fallback",
            exc_info=True,
        )

    _resolved_bedrock_haiku = _BEDROCK_HAIKU_FALLBACK
    return _resolved_bedrock_haiku


def _build_frontier_ranking_model(cfg: MemoryConfig):
    """Anthropic / Bedrock Haiku-style ranking model."""
    from utilities.environment import Environment

    provider_mode = Environment.get_anthropic_model_provider()

    if provider_mode == "bedrock":
        import os

        from botocore.config import Config as BotoConfig
        from langchain_aws import ChatBedrockConverse

        region = Environment.get_anthropic_bedrock_region()
        boto_cfg = BotoConfig(
            read_timeout=60,
            retries={"max_attempts": 2, "mode": "adaptive"},
            request_min_compression_size_bytes=1048576,
        )
        name = cfg.model_name or _resolve_bedrock_haiku(
            region, boto_cfg,
        )
        from deep_agent.model_factory import (
            _make_refreshable_bedrock_clients,
            _resolve_bedrock_creds,
        )

        auth_mode = os.environ.get(
            "ANTHROPIC_BEDROCK_AUTH_MODE", "sso",
        )
        if auth_mode == "sso" and os.environ.get(
            "AWS_SESSION_TOKEN",
        ):
            client_kwargs = _make_refreshable_bedrock_clients(
                region, boto_cfg,
            )
        else:
            client_kwargs = _resolve_bedrock_creds()

        return ChatBedrockConverse(
            model=name,
            region_name=region,
            config=boto_cfg,
            **client_kwargs,
        )

    name = cfg.model_name or _ANTHROPIC_HAIKU_DEFAULT
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        model=name, temperature=0.0,
    )


def _build_mlx_ranking_model(
    cfg: MemoryConfig,
    *,
    max_tokens: int | None = None,
):
    """Local MLX text model for ranking / consolidation.

    Honors ``cfg.mlx_model`` when set, otherwise falls back to the global
    ``HF_LLM_MODEL_ID``.  Mirrors the construction in
    :func:`deep_agent.model_factory.create_llm` so behavior is identical
    to a normal MLX chat session (prompt cache, KV bits, draft model).

    ``max_tokens`` overrides the global ``MLX_MAX_TOKENS`` cap; the
    consolidation pipeline uses this to avoid truncating large structured
    JSON responses.  When provided, the *larger* of (env, override) wins
    so an explicit user-set cap is never silently lowered.
    """
    from chat_models.mlx import ChatMLXText
    from middleware.react_wrapper import MLXReActWrapper
    from utilities.environment import Environment

    model_id = cfg.mlx_model or Environment.get_hf_llm_model_id()
    if not model_id:
        raise RuntimeError(
            "Memory llm_family='mlx' requires either MemoryConfig.mlx_model "
            "or HF_LLM_MODEL_ID to be set."
        )
    env_cap = Environment.get_mlx_max_tokens()
    effective_max = max(env_cap, max_tokens) if max_tokens else env_cap
    inner = ChatMLXText(
        model_path=model_id,
        draft_model_path=Environment.get_hf_draft_llm_model_id(),
        max_tokens=effective_max,
        thinking=Environment.get_mlx_thinking(),
        enable_prompt_cache=Environment.get_mlx_prompt_cache(),
        enable_system_prompt_cache=Environment.get_mlx_system_prompt_cache(),
        kv_bits=Environment.get_mlx_kv_bits(),
    )
    is_reasoning_model = any(
        kw in model_id.lower()
        for kw in ("deepseek-r1", "deepseek_r1", "r1-distill")
    )
    return MLXReActWrapper(inner, force_action=is_reasoning_model)


def _create_ranking_model(
    cfg: MemoryConfig,
    llm_provider: str,
    *,
    mlx_max_tokens: int | None = None,
):
    """Build a lightweight LLM for the ranking / consolidation side-query.

    Resolution order:

    * ``cfg.llm_family == "mlx"`` → always build a local MLX model.
    * ``cfg.llm_family == "frontier"`` → always build an Anthropic / Bedrock
      model (auto-resolves Haiku if ``cfg.model_name`` is blank).
    * ``cfg.llm_family == "follow_main"`` (default) → honor ``llm_provider``:
      Anthropic builds the Haiku ranking model; anything else (including
      ``mlx``) delegates to :func:`deep_agent.model_factory.create_llm`.
    """
    family = (getattr(cfg, "llm_family", "follow_main") or "follow_main").lower()

    # Don't build a second in-process MLX model for the ranking/consolidation
    # side-query while the main chat provider is a cloud/frontier one — loading
    # another full set of weights (plus warmup) into unified Metal memory on a
    # background schedule is what caused GPU OOM aborts.  Fall through to the
    # provider-based resolution (which uses the cloud model) instead.
    from utilities.environment import Environment
    if family == "mlx" and not Environment.is_oss_local_provider(llm_provider):
        logger.info(
            "memory llm_family='mlx' but main provider=%r is non-local "
            "— following main provider to avoid a second MLX load.",
            llm_provider,
        )
        family = "follow_main"

    if family == "mlx":
        return _build_mlx_ranking_model(cfg, max_tokens=mlx_max_tokens)

    if family == "frontier":
        return _build_frontier_ranking_model(cfg)

    if family == "exo":
        # exo cluster — share the same OpenAI-compatible endpoint /
        # model id the main chat path uses. The ranker is short-prompt
        # so we don't need a smaller "haiku" tier here.
        from deep_agent.model_factory import create_llm
        return create_llm("exo")

    if llm_provider == "anthropic":
        return _build_frontier_ranking_model(cfg)

    if llm_provider == "mlx":
        return _build_mlx_ranking_model(cfg, max_tokens=mlx_max_tokens)

    from deep_agent.model_factory import create_llm
    return create_llm(llm_provider)


async def _rank_topics(
    user_text: str,
    topics: list[dict[str, str]],
    cfg: MemoryConfig,
    llm_provider: str,
) -> list[str]:
    """Ask a fast model which topics are relevant."""
    from langchain_core.messages import (
        HumanMessage, SystemMessage,
    )

    model = _create_ranking_model(cfg, llm_provider)

    prompt = _build_ranking_prompt(topics)
    try:
        response = await model.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=user_text[:2000]),
        ])
    except Exception:
        logger.debug(
            "[memory-L2] ranking call failed",
            exc_info=True,
        )
        return []

    text = response.content
    if isinstance(text, list):
        text = " ".join(
            p.get("text", "") for p in text
            if isinstance(p, dict)
            and p.get("type") == "text"
        )
    text = str(text).strip()

    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        return []
    try:
        result = json.loads(text[start:end])
        if isinstance(result, list):
            return [
                str(f) for f in result[:MAX_TOPICS]
            ]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def _last_user_text(messages: list) -> str | None:
    """Extract text from the most recent user message."""
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "human":
            c = msg.content
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                parts = [
                    p.get("text", "")
                    for p in c
                    if isinstance(p, dict)
                    and p.get("type") == "text"
                ]
                return " ".join(parts)
    return None


class MemoryRelevanceMiddleware(
    AgentMiddleware[Any, ContextT, ResponseT],
):
    """Per-turn memory topic injection via relevance ranking.

    On each model call, scans topic file descriptions, runs a
    fast side-query to pick the most relevant ones, and injects
    their full content into the system message.
    """

    _RANK_CACHE_MAX = 32

    def __init__(
        self,
        *,
        memory_cfg: MemoryConfig,
        llm_provider: str,
        session_id: str = "",
    ) -> None:
        self._cfg = memory_cfg
        self._provider = llm_provider
        self._session_id = session_id
        self._rank_cache: dict[str, list[str]] = {}
        self._last_cached: bool = False

    async def _ranked(
        self, user_text: str, topics: list[dict[str, str]],
    ) -> list[str]:
        """Return ranked topics, using a cache to avoid duplicate Haiku calls."""
        key = user_text[:2000]
        if key in self._rank_cache:
            logger.info("[memory-L2] cache hit for query: %.80s", key)
            self._last_cached = True
            return self._rank_cache[key]

        selected = await _rank_topics(
            user_text, topics, self._cfg, self._provider,
        )
        if len(self._rank_cache) >= self._RANK_CACHE_MAX:
            self._rank_cache.pop(next(iter(self._rank_cache)))
        self._rank_cache[key] = selected
        self._last_cached = False
        return selected

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]],
            Awaitable[ModelResponse[ResponseT]],
        ],
    ) -> ModelResponse[ResponseT]:
        user_text = _last_user_text(request.messages)
        if not user_text:
            logger.info("[memory-L2] skipped — no user text in messages")
            return await handler(request)

        topics = await asyncio.to_thread(
            _scan_descriptions,
        )
        if not topics:
            logger.info("[memory-L2] no memory topics on disk")
            return await handler(request)

        logger.info(
            "[memory-L2] ranking %d topic(s) for query: %.80s",
            len(topics), user_text,
        )
        selected = await self._ranked(user_text, topics)
        if not selected:
            logger.info("[memory-L2] no topics matched")
            return await handler(request)

        parts: list[str] = []
        for fname in selected:
            content = await asyncio.to_thread(
                _read_topic_file, fname,
            )
            if content:
                parts.append(
                    f"### {fname}\n{content}"
                )

        if parts:
            from deepagents.middleware._utils import (
                append_to_system_message,
            )
            injection = (
                "<memory_context>\n"
                + "\n\n".join(parts)
                + "\n</memory_context>"
            )
            modified = request.override(
                system_message=append_to_system_message(
                    request.system_message, injection,
                ),
            )
            logger.info(
                "[memory-L2] injected %d topic(s): %s",
                len(selected), selected,
            )
            await asyncio.to_thread(
                _append_hit,
                self._session_id, user_text, selected, self._last_cached,
            )
            response = await handler(modified)
            for msg in response.result:
                if hasattr(msg, "response_metadata"):
                    msg.response_metadata["memory_topics"] = selected
                    break
            return response

        return await handler(request)

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]],
            ModelResponse[ResponseT],
        ],
    ) -> ModelResponse[ResponseT]:
        return handler(request)
