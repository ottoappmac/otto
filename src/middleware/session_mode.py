"""Add the Plan or Debug instructions to the orchestrator's system prompt.

This is the advisory half of a session mode. The hard limits for Plan live in
the session backend and the action policy, so a model that ignores this text
still cannot change the project. The mode is read through *get_mode* on every
model call, so a switch between turns applies to the very next one.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage

logger = logging.getLogger(__name__)

_START = "<session_mode>"
_END = "</session_mode>"


class SessionModeMiddleware(AgentMiddleware):
    """Prompt addendum for Plan and Debug. Orchestrator only."""

    def __init__(self, get_mode: Callable[[], str]) -> None:
        super().__init__()
        self._get_mode = get_mode

    def _with_addendum(self, request: ModelRequest) -> ModelRequest:
        from backend.session_mode import addendum_for

        try:
            addendum = addendum_for(self._get_mode())
        except Exception:  # pragma: no cover — defensive
            logger.debug("Session mode: could not read mode", exc_info=True)
            return request
        if not addendum:
            return request
        existing = _system_text(request.system_message)
        if _START in existing and _END in existing:
            return request
        text = f"{existing}\n\n{addendum}".strip() if existing else addendum
        try:
            return request.override(system_message=SystemMessage(content=text))
        except Exception:  # pragma: no cover — defensive
            logger.debug("Session mode: could not override system message", exc_info=True)
            return request

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._with_addendum(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._with_addendum(request))


def _system_text(system_message: object) -> str:
    if system_message is None:
        return ""
    try:
        from middleware._react_core import content_to_text

        return content_to_text(system_message.content)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover — defensive
        content = getattr(system_message, "content", "")
        return content if isinstance(content, str) else str(content)
