"""Single-threaded MLX executor — foundation for the turbo path.

Vendored and trimmed from ``omlx.engine_core.get_mlx_executor``.  All MLX
GPU work submitted by :class:`TurboMLXChat` runs on this one executor
thread, which gives us:

* Deterministic FIFO ordering on the Metal command stream — no command-
  encoder race, no per-call lock dance (the classic path still uses
  :data:`chat_models.mlx._shared.MLX_GEN_LOCK`; turbo gets the same
  safety via thread-serialisation instead).
* A stable hook for the next turbo levels to build on: paged KV-cache
  eviction, cross-session prefix lookup, and continuous batching all
  need a single thread that owns Metal.

Design note — why we deliberately do **not** replace
``mlx_lm.generate.generation_stream`` on the executor thread like omlx
does: the agents repo keeps the classic :class:`ChatMLXText` path and
:class:`MLXVLChatModel` alive by default, both of which still run on
whichever caller thread LangChain schedules them on.  If we reassigned
the module-level stream to one owned by the executor, any call from a
non-executor thread would fail with ``There is no Stream(gpu, 0) in
current thread``.  Keeping the original stream lets turbo and classic
coexist; they serialise against each other through ``MLX_GEN_LOCK``.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

_EXECUTOR: Optional[concurrent.futures.ThreadPoolExecutor] = None
_EXECUTOR_LOCK = threading.Lock()

T = TypeVar("T")


def _init_mlx_thread() -> None:
    """Executor-thread initialiser.

    Kept as an explicit hook so future turbo levels can perform one-off
    per-thread setup (e.g. pre-warming a paged-cache manager) without
    touching every call-site.  Intentionally a no-op today.
    """
    logger.debug("MLX turbo executor thread initialised")


def get_mlx_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Return the process-wide single-worker MLX executor, creating it
    lazily on first use under a double-checked lock."""
    global _EXECUTOR
    if _EXECUTOR is not None:
        return _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="mlx-turbo",
                initializer=_init_mlx_thread,
            )
    return _EXECUTOR


def run_on_mlx_thread(fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    r"""Submit *fn(\*args, \*\*kwargs)* to the MLX executor and block for
    its result.  Exceptions raised inside *fn* are re-raised on the
    caller's thread so LangChain error handling works unchanged."""
    return get_mlx_executor().submit(fn, *args, **kwargs).result()


def shutdown(wait: bool = False) -> None:
    """Tear the executor down.  Intended for tests and clean shutdown —
    the app never calls this in normal operation."""
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is not None:
            _EXECUTOR.shutdown(wait=wait)
            _EXECUTOR = None
