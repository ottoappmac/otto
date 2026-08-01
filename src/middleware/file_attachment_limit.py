"""Middleware that caps the running total of ``read_file`` attachments.

**Why this exists**

Some OpenAI-compatible servers (e.g. the local oMLX server) count "file"
content blocks cumulatively across the *whole* conversation rather than per
request. Once a session has called ``read_file`` on more than a handful of
non-text attachments (PDF, PPTX, images embedded as files, …), the server
starts rejecting the request outright — even on turns that don't read any
new files — because the running total baked into the replayed message
history is already over its limit.

**What it does**

Before each model call it scans the message history for ``ToolMessage``
"file" content blocks and keeps only the most recent :data:`MAX_ATTACHMENTS`
of them. Older blocks are replaced with a short text note (the attachment
payload is dropped, not the whole tool message) so the tool-call/tool-result
pairing stays intact and the model still sees that a file was read, just not
its now-evicted contents.

The transform is applied to a throwaway copy of the request (via
``request.override``) and is recomputed from canonical state on every call,
so it is idempotent and never mutates the persisted conversation.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage
from langchain_core.messages.content import create_text_block

MAX_ATTACHMENTS = 8
"""Maximum number of ``read_file`` "file" blocks kept in the replayed history."""


def maybe_for_model(model: Any) -> "FileAttachmentLimitMiddleware | None":
    """Return the middleware when *model* speaks the OpenAI Chat Completions API.

    Gating on ``BaseChatOpenAI`` (the common base of ``ChatOpenAI`` and
    ``AzureChatOpenAI``, which is what the ``openai``/``omlx``/``exo``
    providers instantiate) makes this a no-op for Anthropic, Bedrock and MLX
    models — only servers speaking the OpenAI wire format are known to count
    attachments cumulatively.
    """
    try:
        from langchain_openai.chat_models.base import BaseChatOpenAI
    except ImportError:
        return None
    if not isinstance(model, BaseChatOpenAI):
        return None
    return FileAttachmentLimitMiddleware()


def _is_file_block(block: Any) -> bool:
    """Return ``True`` for a "file" content block carrying attachment data."""
    return isinstance(block, dict) and block.get("type") == "file"


class FileAttachmentLimitMiddleware(AgentMiddleware):
    """Cap the running total of ``read_file`` attachments for OpenAI-compatible providers.

    Insert this for providers whose transport is the OpenAI Chat Completions
    API (``openai``, ``omlx``, ``exo``). It is a no-op while the conversation
    holds at most :data:`MAX_ATTACHMENTS` file blocks, so it is safe to
    include unconditionally for those providers.
    """

    _EVICTED_NOTE = "(file attachment dropped to stay under the server's attachment limit)"

    def __init__(self, max_attachments: int = MAX_ATTACHMENTS) -> None:
        super().__init__()
        self._max_attachments = max_attachments

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._limit_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._limit_request(request))

    # ── Private helpers ─────────────────────────────────────────────────────

    def _limit_request(self, request: ModelRequest) -> ModelRequest:
        limited = self._limit_attachments(list(request.messages))
        if limited is None:
            return request
        return request.override(messages=limited)

    def _limit_attachments(self, messages: list[Any]) -> list[Any] | None:
        """Return a new message list with only the most recent
        :attr:`_max_attachments` file blocks kept, or ``None`` when nothing
        needs to be evicted.
        """
        total = sum(
            1
            for msg in messages
            if isinstance(msg, ToolMessage) and isinstance(msg.content, list)
            for block in msg.content
            if _is_file_block(block)
        )
        to_evict = total - self._max_attachments
        if to_evict <= 0:
            return None

        out: list[Any] = []
        for msg in messages:
            if not (isinstance(msg, ToolMessage) and isinstance(msg.content, list)):
                out.append(msg)
                continue

            if to_evict <= 0 or not any(_is_file_block(b) for b in msg.content):
                out.append(msg)
                continue

            new_content = []
            for block in msg.content:
                if _is_file_block(block) and to_evict > 0:
                    new_content.append(create_text_block(text=self._EVICTED_NOTE))
                    to_evict -= 1
                else:
                    new_content.append(block)
            out.append(msg.model_copy(update={"content": new_content}))

        return out
