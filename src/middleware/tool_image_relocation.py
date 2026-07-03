"""Middleware that relocates tool-result images into user messages.

**Why this exists**

Tools such as ``view_image`` / ``load_image_from_url`` return visual content as
image blocks inside a ``ToolMessage``.  For Anthropic this is fine — the
Messages API renders images embedded in ``tool_result`` blocks.  The OpenAI
*Chat Completions* API (and the OpenAI-compatible servers we use for the
``openai``, ``omlx`` and ``exo`` providers) only render image content that
appears in a **user** message.  ``langchain_openai`` faithfully serialises a
tool-result image as::

    {"role": "tool", "content": [{"type": "image_url", "image_url": {...}}]}

which the server's chat template silently drops, so even a vision-capable model
(e.g. a Qwen3 VLM served by oMLX) never receives the pixels and ends up
hallucinating a generic description.

**What it does**

Before each model call it scans the message history, strips image blocks out of
every ``ToolMessage`` (leaving the textual portion and a short pointer note in
place so the tool-call/tool-result pairing stays intact), and re-inserts those
images as a ``HumanMessage`` immediately after the run of tool results they came
from.  Placing the user message *after* the complete set of tool responses keeps
the assistant→tool ordering the OpenAI API requires while delivering the images
in a role the server actually renders.

The transform is applied to a throwaway copy of the request (via
``request.override``) and is recomputed from canonical state on every call, so
it is idempotent and never mutates the persisted conversation.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.messages.content import create_text_block


def maybe_for_model(model: Any) -> "ToolImageRelocationMiddleware | None":
    """Return the middleware when *model* speaks the OpenAI Chat Completions API.

    Gating on ``BaseChatOpenAI`` (the common base of ``ChatOpenAI`` and
    ``AzureChatOpenAI``, which is what the ``openai``/``omlx``/``exo``
    providers instantiate) makes this a no-op for Anthropic, Bedrock and MLX
    models, all of which render tool-result images natively.
    """
    try:
        from langchain_openai.chat_models.base import BaseChatOpenAI
    except ImportError:
        return None
    if not isinstance(model, BaseChatOpenAI):
        return None
    return ToolImageRelocationMiddleware()


def _is_image_block(block: Any) -> bool:
    """Return ``True`` if *block* is an image content block.

    Handles both the LangChain v1 standard block (``{"type": "image", ...}``)
    and the legacy OpenAI-style block (``{"type": "image_url", ...}``).
    """
    return isinstance(block, dict) and block.get("type") in ("image", "image_url")


def _first_text(blocks: list[Any]) -> str:
    """Return the first text block's text from *blocks*, or an empty string."""
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            if text:
                return text.strip().splitlines()[0]
    return ""


class ToolImageRelocationMiddleware(AgentMiddleware):
    """Move tool-result images into user messages for OpenAI-compatible providers.

    Insert this for providers whose transport is the OpenAI Chat Completions
    API (``openai``, ``omlx``, ``exo``).  It is a no-op when no ``ToolMessage``
    in the history carries image content, so it is safe to include
    unconditionally for those providers.
    """

    _MOVED_NOTE = "(image content provided in the following user message)"

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._relocate_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._relocate_request(request))

    # ── Private helpers ─────────────────────────────────────────────────────

    def _relocate_request(self, request: ModelRequest) -> ModelRequest:
        relocated = self._relocate(list(request.messages))
        if relocated is None:
            return request
        return request.override(messages=relocated)

    def _relocate(self, messages: list[Any]) -> list[Any] | None:
        """Return a new message list with tool-result images moved to user
        messages, or ``None`` when there is nothing to relocate.
        """
        out: list[Any] = []
        pending: list[tuple[str, Any]] = []
        changed = False

        def flush() -> None:
            if not pending:
                return
            content: list[Any] = []
            for label, image in pending:
                content.append(create_text_block(text=label or "Image:"))
                content.append(image)
            out.append(HumanMessage(content=content))
            pending.clear()

        for i, msg in enumerate(messages):
            if isinstance(msg, ToolMessage) and isinstance(msg.content, list):
                kept = [b for b in msg.content if not _is_image_block(b)]
                images = [b for b in msg.content if _is_image_block(b)]
                if images:
                    changed = True
                    label = _first_text(kept)
                    pending.extend((label, img) for img in images)
                    note = create_text_block(text=self._MOVED_NOTE)
                    msg = msg.model_copy(update={"content": [*kept, note]})
                out.append(msg)
                nxt = messages[i + 1] if i + 1 < len(messages) else None
                if not isinstance(nxt, ToolMessage):
                    flush()
            else:
                flush()
                out.append(msg)
        flush()

        return out if changed else None
