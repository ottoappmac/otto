"""Custom chat model wrappers for MLX.

Concrete classes are exposed via a lazy ``__getattr__`` so that touching
this package doesn't force every wrapper (and its dependencies) to be
imported up-front.  Two of the three classes
(:class:`~chat_models.mlx.ChatMLXNoStream` and
:class:`~chat_models.mlx.CommandRMLXChat`) pull in ``langchain_community``
at module load, and :class:`~chat_models.mlx.ChatMLXText` lazily imports
``mlx_lm`` on first use.  Loading on demand keeps backend startup fast
and lets the bundle ship without optional deps installed for callers
that never touch them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chat_models.mlx import ChatMLXNoStream, ChatMLXText, CommandRMLXChat

__all__ = ["ChatMLXNoStream", "ChatMLXText", "CommandRMLXChat"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from chat_models import mlx as _mlx
        return getattr(_mlx, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
