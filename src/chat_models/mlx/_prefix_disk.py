"""SSD tier of the static prompt-prefix store.

The RAM store (:mod:`chat_models.mlx._prefix_store`) dies with the process,
so after every backend restart each agent paid a full cold prefill of its
tool block and system turn again (~130 s each for OTTO's ~35k tokens on
Qwen3.8-9B).  Static snapshots that can be hit again are therefore also
written here with ``mlx_lm``'s ``save_prompt_cache`` — which serialises any
cache class through ``.state`` / ``.meta_state`` — and a RAM-store miss loads
the exact-length snapshot back with ``load_prompt_cache``.

Not every static snapshot can: OTTO's system turn names the session's files
directory and the current time, so no other session — and no session after a
restart — ever hits it.  Written are a prompt's first static prefix (the tool
block, shared by every session of its agent) and any other from its second
use in this process on (e.g. a subagent's unchanging system turn); see
:meth:`DiskPrefixStore.wants`.

A file is named by the sha256 of everything its contents depend on: the
loaded model weights (path, size and mtime of every weight file and of
``config.json``, taken when they were loaded), the ``mlx_lm`` version
(cache-class layout), the KV settings (``kv_bits``, ``kv_group_size``) and the
exact prefix token ids.  A file is written to a
``tmp/`` sibling and renamed into place, so a crash never leaves a partial
file under a real name; a file that still fails to load, or whose contents
don't match its name (token count, layer count, layer offsets), is deleted
and the caller prefills instead.

Writing costs one sequential write of the snapshot (~0.37 GB for a 4-bit
35k-token Qwen3.8-9B prefix: ~0.1-0.3 s), done once the reply is generated,
outside ``MLX_GEN_LOCK``; loading is a read of the same size.  The directory
is capped at ``MAX_BYTES``, least recently used (by mtime) first: every use
of a snapshot — loaded from here or served from the RAM store — touches its
file.

The directory is ``~/Library/Caches/otto/prefix``; the ``OTTO_MLX_PREFIX_CACHE_DIR``
environment variable moves it, and an empty value turns the tier off.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from array import array
from pathlib import Path
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)

CACHE_DIR_ENV = "OTTO_MLX_PREFIX_CACHE_DIR"
DEFAULT_DIR = Path.home() / "Library" / "Caches" / "otto" / "prefix"
MAX_BYTES = 4 * 1024**3
_FORMAT = "otto-prefix-1"
_WEIGHT_SUFFIXES = (".safetensors", ".npz")
_STALE_TMP_SECONDS = 3600


def cache_dir_from_env() -> Optional[Path]:
    """The snapshot directory: ``OTTO_MLX_PREFIX_CACHE_DIR`` if set (empty: off), else the default."""
    value = os.environ.get(CACHE_DIR_ENV)
    if value is None:
        return DEFAULT_DIR
    return Path(value).expanduser() if value.strip() else None


def model_fingerprint(local_dir: str) -> Optional[str]:
    """Identify the model files in *local_dir*, or ``None`` when it holds no weights.

    Taken when the model is loaded from there (see
    ``_shared.weights_fingerprint``): the files may change later — a catalog
    re-download — while those weights stay loaded.  Without a fingerprint
    nothing may be shared across restarts: another model could later load
    under the same name.
    """
    try:
        local = Path(local_dir).resolve()
        files = sorted(p for p in local.iterdir() if p.suffix in _WEIGHT_SUFFIXES or p.name == "config.json")
        if not any(p.suffix in _WEIGHT_SUFFIXES for p in files):
            return None
        digest = hashlib.sha256(str(local).encode())
        for path in files:
            stat = path.stat()
            digest.update(f"|{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    except OSError:  # not a local model directory
        return None
    return digest.hexdigest()


def snapshot_key(fingerprint: str, settings: Any, tokens: Sequence[int]) -> str:
    """The file name (sans suffix) of the snapshot of exactly *tokens*."""
    import mlx_lm

    digest = hashlib.sha256(f"{_FORMAT}|{mlx_lm.__version__}|{fingerprint}|{settings!r}|".encode())
    digest.update(array("q", tokens).tobytes())
    return digest.hexdigest()


class DiskPrefixStore:
    """Static-prefix snapshots as safetensors files in *directory* (``None``: disabled).

    ``load`` and ``wants`` are called under ``MLX_GEN_LOCK`` (loading
    evaluates arrays); ``save``, which only writes arrays already evaluated,
    outside it.
    """

    def __init__(self, directory: Optional[Path], max_bytes: int = MAX_BYTES) -> None:
        self.directory = directory
        self.max_bytes = max_bytes
        # Keys of the snapshots used in this process (see ``wants``).
        self._used: set = set()

    def wants(self, key: str, first_prefix: bool) -> bool:
        """Whether to write the snapshot *key*, just used (restored from RAM or prefilled).

        A snapshot already on SSD isn't: its file is touched instead — most
        recently used.  Written is a prompt's *first_prefix* — its tool
        block, shared by every session of the agent — and any other from its
        second use in this process on.  A system turn that names its session
        and the current time is never used again, not even after a restart:
        writing it (~0.37 GB) would only push reusable files out.
        """
        if self.touch(key):
            return False
        used_before = key in self._used
        self._used.add(key)
        return first_prefix or used_before

    def touch(self, key: str) -> bool:
        """Mark the snapshot *key* most recently used; ``False`` if it has no file."""
        if self.directory is None:
            return False
        try:
            os.utime(self.directory / f"{key}.safetensors")
        except OSError:  # not written, or evicted meanwhile (e.g. by another process)
            return False
        return True

    def load(self, key: str, n_tokens: int, n_layers: int) -> Optional[List[Any]]:
        """Return the evaluated snapshot stored under *key*, or ``None``.

        It must hold *n_layers* layers describing exactly *n_tokens* tokens;
        a file that doesn't load or doesn't match is deleted.
        """
        if self.directory is None:
            return None
        path = self.directory / f"{key}.safetensors"
        if not path.is_file():
            return None
        try:
            layers = self._read(path, key, n_tokens, n_layers)
        except Exception as exc:  # noqa: BLE001 — corrupt, partial or foreign file
            logger.warning(
                "KV prefix cache: SSD snapshot %s is unreadable (%s) — deleted, "
                "prefilling instead", path.name, exc,
            )
            path.unlink(missing_ok=True)
            return None
        self.touch(key)
        return layers

    @staticmethod
    def _read(path: Path, key: str, n_tokens: int, n_layers: int) -> List[Any]:
        import mlx.core as mx
        from mlx_lm.models.cache import load_prompt_cache

        layers, meta = load_prompt_cache(str(path), return_metadata=True)
        if (meta.get("otto_format"), meta.get("otto_key")) != (_FORMAT, key):
            raise ValueError("not this snapshot")
        if int(meta.get("otto_tokens", -1)) != n_tokens or len(layers) != n_layers:
            raise ValueError(f"{meta.get('otto_tokens')} tokens / {len(layers)} layers")
        mx.eval([layer.state for layer in layers])
        for layer in layers:
            offset = getattr(layer, "offset", None)
            if isinstance(offset, int) and offset != n_tokens:
                raise ValueError(f"a layer holds {offset} tokens")
        # Put back the empty slots save() had to drop (see _carrier).
        for i, (size, holes) in json.loads(meta.get("otto_holes", "{}")).items():
            filled = iter(layers[int(i)].state)
            layers[int(i)].state = [None if j in holes else next(filled) for j in range(size)]
        return layers

    def save(self, key: str, layers: List[Any], n_tokens: int) -> bool:
        """Write *layers*, the snapshot of *n_tokens* tokens, under *key*; ``True`` if written."""
        if self.directory is None:
            return False
        from mlx_lm.models.cache import save_prompt_cache

        try:
            holes: dict = {}
            carriers = [self._carrier(i, layer, holes) for i, layer in enumerate(layers)]
        except ValueError as exc:
            logger.debug("KV prefix cache: snapshot not saved to SSD (%s)", exc)
            return False
        path = self.directory / f"{key}.safetensors"
        tmp = self.directory / "tmp" / f"{key}.{os.getpid()}.{threading.get_ident()}.safetensors"
        try:
            tmp.parent.mkdir(parents=True, exist_ok=True)
            save_prompt_cache(str(tmp), carriers, {
                "otto_format": _FORMAT,
                "otto_key": key,
                "otto_tokens": str(n_tokens),
                "otto_holes": json.dumps(holes),
            })
            os.replace(tmp, path)
        except Exception as exc:  # noqa: BLE001 — disk full, permissions, ...
            logger.warning("KV prefix cache: couldn't write SSD snapshot %s (%s)", path.name, exc)
            tmp.unlink(missing_ok=True)
            return False
        self._evict(keep=path)
        return path.exists()

    @staticmethod
    def _carrier(i: int, layer: Any, holes: dict) -> Any:
        """*layer* in a form ``save_prompt_cache`` / ``load_prompt_cache`` round-trip.

        safetensors can't store ``None``: an ``ArraysCache`` slot that is
        still empty is dropped here and recorded in *holes*.  A layer class
        ``load_prompt_cache`` can't find, or a layer with no arrays at all
        (it would shift every later layer on load), can't be stored.
        """
        from mlx.utils import tree_flatten
        from mlx_lm.models import cache as cache_module

        if getattr(cache_module, type(layer).__name__, None) is not type(layer):
            raise ValueError(f"{type(layer).__name__} isn't an mlx_lm cache class")
        try:
            state = layer.state
        except Exception as exc:  # noqa: BLE001 — e.g. an empty KVCache
            raise ValueError(f"layer {i} has no state") from exc
        if isinstance(state, list) and any(x is None for x in state):
            holes[str(i)] = [len(state), [j for j, x in enumerate(state) if x is None]]
            layer = type(layer).from_state([x for x in state if x is not None], layer.meta_state)
        if not tree_flatten(layer.state):
            raise ValueError(f"layer {i} is empty")
        return layer

    def _evict(self, keep: Path) -> None:
        """Delete least recently used snapshots until the directory fits ``max_bytes``."""
        files = []
        for path in self.directory.glob("*.safetensors"):
            try:
                stat = path.stat()
            except OSError:
                continue
            files.append((stat.st_mtime, stat.st_size, path))
        total = sum(size for _, size, _ in files)
        for _, size, path in sorted(files):
            if total <= self.max_bytes:
                break
            if path != keep:
                path.unlink(missing_ok=True)
                total -= size
        if total > self.max_bytes:  # the new snapshot alone exceeds the cap
            keep.unlink(missing_ok=True)
        # Leftovers of writes interrupted by a crash.
        now = time.time()
        for path in (self.directory / "tmp").glob("*"):
            try:
                if now - path.stat().st_mtime > _STALE_TMP_SECONDS:
                    path.unlink(missing_ok=True)
            except OSError:
                pass


SSD_PREFIXES = DiskPrefixStore(cache_dir_from_env())
