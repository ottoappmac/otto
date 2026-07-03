"""Custom chat model wrappers for MLX.

Each concrete class is imported only on attribute access.  Eager imports
would force ``chat_mlx_no_stream`` and ``command_r`` to load on every
``from chat_models.mlx import ChatMLXText`` — both of those modules
depend on ``langchain_community`` at import time, so callers that only
need the streaming text model (the orchestrator's hot path) would pay
the cost (and risk the failure) of importing the other two.  Lazy
loading keeps each class independent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chat_models.mlx.chat_mlx_no_stream import ChatMLXNoStream
    from chat_models.mlx.chat_mlx_text import ChatMLXText
    from chat_models.mlx.command_r import CommandRMLXChat

__all__ = ["ChatMLXNoStream", "ChatMLXText", "CommandRMLXChat"]


def __getattr__(name: str) -> Any:
    if name == "ChatMLXText":
        from chat_models.mlx.chat_mlx_text import ChatMLXText as _Cls
        return _Cls
    if name == "ChatMLXNoStream":
        from chat_models.mlx.chat_mlx_no_stream import ChatMLXNoStream as _Cls
        return _Cls
    if name == "CommandRMLXChat":
        from chat_models.mlx.command_r import CommandRMLXChat as _Cls
        return _Cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
