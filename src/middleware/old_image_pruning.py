"""Middleware that keeps only the most recent tool-result images in the replayed history.

**Why this exists**

A vision agent that looks at many images (``view_image`` on a drawing, its tiles, renders and comparison
pictures) carries every one of them in the conversation until summarization triggers, and summarization
triggers on *context length* (85% of the window). A local server (oMLX) can run out of *memory* long
before that: one benchmark run died with "prefill would require ~104 GB" after 39 images, with the
context nowhere near full.

**What it does**

Before each model call it keeps the newest :data:`MAX_KEPT_IMAGES` images that sit inside tool results
and replaces each older one with a short text note saying it was removed and that the tool can be
called again to look at it. The tool-call / tool-result pairing is untouched (only the image block is
swapped). Images in user messages (attachments the user sent) are never touched.

Order: put it BEFORE ``ToolImageRelocationMiddleware`` in the list so it sees the images while they are
still inside the tool results.

The transform runs on a throwaway copy of the request (``request.override``) and is recomputed from the
saved conversation on every call, so it is idempotent and never changes what is persisted.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage
from langchain_core.messages.content import create_text_block

MAX_KEPT_IMAGES = 6
"""Most recent tool-result images kept in the replayed history."""

# Agents that look at many images and are known to need this. Others keep the default behaviour.
AGENTS = frozenset({"benchcad-agent"})


def maybe_for_agent(agent_name: str | None, max_images: int | None = None) -> "OldImagePruningMiddleware | None":
    """The middleware for agents that need it (see :data:`AGENTS`), else ``None``."""
    if (agent_name or "") not in AGENTS:
        return None
    return OldImagePruningMiddleware(max_images or MAX_KEPT_IMAGES)


def _is_image_block(block: Any) -> bool:
    return isinstance(block, dict) and block.get("type") in ("image", "image_url")


class OldImagePruningMiddleware(AgentMiddleware):
    """Keep only the newest ``max_images`` tool-result images; older ones become a text note."""

    _NOTE = "(older image removed to save memory; call the tool again to look at it)"

    def __init__(self, max_images: int = MAX_KEPT_IMAGES) -> None:
        super().__init__()
        self._max = max(1, int(max_images))

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._prune_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._prune_request(request))

    # ── Private helpers ─────────────────────────────────────────────────────

    def _prune_request(self, request: ModelRequest) -> ModelRequest:
        pruned = self._prune(list(request.messages))
        if pruned is None:
            return request
        return request.override(messages=pruned)

    def _prune(self, messages: list[Any]) -> list[Any] | None:
        """A new message list with all but the newest images replaced, or ``None`` if nothing to do."""
        total = sum(
            1
            for m in messages
            if isinstance(m, ToolMessage) and isinstance(m.content, list)
            for b in m.content
            if _is_image_block(b)
        )
        to_drop = total - self._max
        if to_drop <= 0:
            return None
        out: list[Any] = []
        for m in messages:
            if to_drop > 0 and isinstance(m, ToolMessage) and isinstance(m.content, list) \
                    and any(_is_image_block(b) for b in m.content):
                content: list[Any] = []
                for b in m.content:
                    if to_drop > 0 and _is_image_block(b):
                        content.append(create_text_block(text=self._NOTE))
                        to_drop -= 1
                    else:
                        content.append(b)
                m = m.model_copy(update={"content": content})
            out.append(m)
        return out
