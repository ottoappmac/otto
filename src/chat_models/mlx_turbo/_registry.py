"""Process-wide registry of :class:`TurboMLXChat` singletons.

At ``turbo_level == "cache"`` every session that targets the same
``(model_path, draft_path)`` pair is served by a single shared
TurboMLXChat so its running KV prompt cache (and the prefix-trim logic
it inherits from :class:`chat_models.mlx.ChatMLXText`) carries state
across sessions.  The first request in session B re-uses whatever the
last request in session A left in the cache — which for typical agent
workloads (same system prompt + tool list across sessions) means the
fixed prefix is prefilled exactly once for the life of the process.

We key singletons on ``(model_path, draft_path, turbo_level, *tuning)``
so user-visible settings changes (max tokens, kv_bits, thinking toggle,
…) correctly invalidate the cached instance — :func:`refresh_tools`
will pull a fresh TurboMLXChat on the next turn.  Stale singletons are
dropped on key change; their MLX weights remain alive in
:data:`chat_models.mlx._shared._LOADED_MODELS` so no weight reload
happens — only the per-instance prompt cache is reset.
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, Hashable, Tuple

_LOCK = threading.Lock()
# Forward-reference the runtime type so the registry has no import-time
# dependency on ``chat.py`` (which would otherwise be a cycle through
# ``mlx_turbo.__init__``).  flake8 F821 is silenced per-line because
# the name is genuinely unresolved at module load — only the factory
# supplied at call time actually instantiates the class.
_SINGLETONS: Dict[Tuple[Hashable, ...], "TurboMLXChat"] = {}  # noqa: F821


def get_or_create(
    key: Tuple[Hashable, ...],
    factory: Callable[[], "TurboMLXChat"],  # noqa: F821
) -> "TurboMLXChat":  # noqa: F821
    """Return the cached singleton for *key* or construct it via *factory*.

    Construction happens under the registry lock so two concurrent
    ``refresh_tools`` calls can't race into building two instances with
    the same key.
    """
    existing = _SINGLETONS.get(key)
    if existing is not None:
        return existing
    with _LOCK:
        existing = _SINGLETONS.get(key)
        if existing is not None:
            return existing
        instance = factory()
        _SINGLETONS[key] = instance
        return instance


def evict_all() -> None:
    """Drop every cached singleton.  Used by tests and by the future
    ``/api/mlx/turbo/reset`` helper."""
    with _LOCK:
        _SINGLETONS.clear()


def size() -> int:
    """Number of currently cached singletons (diagnostics)."""
    return len(_SINGLETONS)
