"""Document loaders.

``WebLoader`` is exposed via a lazy ``__getattr__`` because it pulls in
both ``playwright`` and ``langchain_community`` at module load — neither
of which is guaranteed to be available everywhere the package is
imported (the bundled backend ships without ``playwright``, for
example).  Touching ``loaders`` itself stays cheap; only callers that
explicitly access ``WebLoader`` pay the heavy import cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from loaders.web_loader import WebLoader

__all__ = ["WebLoader"]


def __getattr__(name: str) -> Any:
    if name == "WebLoader":
        from loaders.web_loader import WebLoader as _Cls
        return _Cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
