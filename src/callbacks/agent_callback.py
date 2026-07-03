"""Abstract agent-level callback and emitter mixin.

:class:`AgentCallback` — subclass and override the hooks you need.  All methods
are async no-ops by default so consumers only implement what matters.

:class:`CallbackMixin` — add to any agent class to get ``_emit_info``,
``_emit_warning``, ``_emit_error``, and ``_emit_image`` helpers that
**both** log via Python :mod:`logging` **and** fire the callback (if set).

Example — forward events to a WebSocket::

    class WSAgentCallback(AgentCallback):
        def __init__(self, ws):
            self._ws = ws

        async def on_info(self, message: str, type: str = "") -> None:
            await self._ws.send_json({"level": "info", "type": type, "text": message})

        async def on_image(self, image_base64: str) -> None:
            await self._ws.send_json({"type": "image", "data": image_base64})
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.callbacks import AsyncCallbackHandler


class AgentCallback(AsyncCallbackHandler):
    """Base callback for agent-level lifecycle events.

    Extend this class and override any subset of the hooks below.
    Instances can also be passed into LangChain's ``config["callbacks"]``
    list because they inherit the full :class:`AsyncCallbackHandler` interface.

    The ``type`` argument on text-level hooks classifies the message
    (e.g. ``"thought"``, ``"tool"``, ``"status"``, ``"plan"``).
    """

    async def on_info(self, message: str, type: str = "") -> None:
        """Called for informational progress messages."""

    async def on_warning(self, message: str, type: str = "") -> None:
        """Called when the agent encounters a non-fatal issue."""

    async def on_error(self, message: str, type: str = "") -> None:
        """Called when the agent encounters an error."""

    async def on_image(self, image_base64: str) -> None:
        """Called when the agent captures a screenshot or other image.

        Args:
            image_base64: The image encoded as a base-64 string.
        """


class CallbackMixin:
    """Mixin that provides unified emit helpers for agent classes.

    Each ``_emit_*`` method logs via the host module's :class:`logging.Logger`
    **and** fires the corresponding :class:`AgentCallback` hook when a callback
    is present.  This eliminates scattered ``if self._callback:`` guards.

    Requirements for the host class:

    * Set ``self._callback: AgentCallback | None`` in ``__init__``.
    """

    _callback: Optional[AgentCallback]

    async def _emit_info(self, message: str, type: str = "") -> None:
        logging.getLogger(self.__class__.__module__).info(f"{type}: {message}")
        if self._callback:
            await self._callback.on_info(message, type=type)

    async def _emit_warning(self, message: str, type: str = "") -> None:
        logging.getLogger(self.__class__.__module__).warning(f"{type}: {message}")
        if self._callback:
            await self._callback.on_warning(message, type=type)

    async def _emit_error(self, message: str, type: str = "") -> None:
        logging.getLogger(self.__class__.__module__).error(f"{type}: {message}")
        if self._callback:
            await self._callback.on_error(message, type=type)

    async def _emit_image(self, image_base64: str) -> None:
        if self._callback:
            await self._callback.on_image(image_base64)
