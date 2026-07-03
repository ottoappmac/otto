"""SSD cold-tier KV prefix cache for ``turbo_level == "ssd"``.

This is the on-disk companion to the in-memory "cache" turbo level.  It
persists completed ``mlx_lm`` prompt caches to a safetensors file keyed
by the sha256 of the prompt token prefix, so a fresh process (or a
session that didn't pay the system-prompt prefill in memory) can pull
a warm KV state from disk instead of re-tracing it through the model.

Design choices
--------------

* **Per-model namespace, global LRU.**  Files live under
  ``<root>/<sanitised_model_path>/v=<fingerprint>/kv=<kv_sig>/<shard>/<hash>.safetensors``
  — a fingerprint mismatch (config.json / tokenizer.json changed) or a
  ``kv_bits`` change cannot accidentally deliver a stale cache.  The LRU
  index (``<root>/index.json``) is global across every model, so the
  ``turbo_ssd_max_gb`` budget covers the whole on-disk store rather than
  per-model shards that can't share space.

* **Flat save / load, not paged.**  TurboMLXChat at level ``ssd`` is
  still backed by the classic :func:`mlx_lm.models.cache.make_prompt_cache`
  — flat, not paged.  Saving the live cache with ``save_prompt_cache``
  and loading it back with ``load_prompt_cache`` is the simplest format
  that round-trips; the paged SSD cache from omlx would require the
  paged allocator plumbing which hasn't landed here yet.

* **Exact-prefix matching.**  Lookups return the longest previously
  saved prefix whose tokens are an exact prefix of the incoming prompt
  tokens (sha256 verified).  We deliberately don't fall back to partial
  matches — loading a mismatched cache would produce garbage completions
  with no clear error surface.

Thread-safety
-------------

All mutation of the in-memory index happens under ``self._lock``;
``save_prompt_cache`` / ``load_prompt_cache`` are wrapped by the caller
in :data:`chat_models.mlx._shared.MLX_GEN_LOCK` so they can't race with
a concurrent generation on the same Metal stream.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "SSDPrefixStore",
    "compute_model_fingerprint",
    "kv_signature",
    "sanitize_model_id",
    "resolve_root",
]


# ── Hashing helpers ──────────────────────────────────────────────────────


def _hash_tokens(tokens: List[int]) -> str:
    """Return the hex sha256 of a token sequence.

    We serialise as ``i4`` little-endian bytes rather than the tokenizer's
    decoded text to sidestep tokeniser non-determinism across processes
    (some tokenisers add whitespace defaults lazily).  The resulting hash
    is stable as long as the token id stream is.
    """
    h = hashlib.sha256()
    for t in tokens:
        # 4 bytes is enough for every real-world vocab size (<= 2**32).
        h.update(int(t).to_bytes(4, "little", signed=False))
    return h.hexdigest()


def _encode_tokens(tokens: List[int]) -> str:
    """Pack a token-id list into a compact base64 blob (``i4`` LE bytes).

    Stored in both the safetensors metadata and the JSON index so
    :meth:`SSDPrefixStore.find_best_common_prefix` can compute the
    longest-common-prefix between a new prompt and any saved entry
    without reading the safetensors file back.  Space cost is
    ~1.33 bytes per token (300-token system prompt ≈ 400 bytes),
    negligible against the KV tensors themselves (~50 MB per save).
    """
    raw = b"".join(int(t).to_bytes(4, "little", signed=False) for t in tokens)
    return base64.b64encode(raw).decode("ascii")


def _decode_tokens(blob: str) -> List[int]:
    """Inverse of :func:`_encode_tokens`; returns ``[]`` on any failure.

    Callers treat an empty return as "token stream unknown, skip this
    entry for prefix-match purposes" — legacy entries saved before the
    ``tokens_b64`` field existed always look like that.
    """
    if not blob:
        return []
    try:
        raw = base64.b64decode(blob.encode("ascii"))
    except (ValueError, UnicodeEncodeError):
        return []
    n = len(raw) // 4
    return [
        int.from_bytes(raw[i * 4:(i + 1) * 4], "little", signed=False)
        for i in range(n)
    ]


def sanitize_model_id(model_path: str) -> str:
    """Turn a HF repo id / local path into a safe directory name.

    We keep slashes readable by replacing them with ``__`` and strip any
    character that isn't alnum / ``._-``.  Collisions are theoretically
    possible but vanishingly unlikely for HF repo ids, and the
    ``v=<fingerprint>`` suffix rules out stale hits regardless.
    """
    slug = model_path.replace("/", "__")
    slug = re.sub(r"[^A-Za-z0-9._-]", "_", slug)
    # Cap length so we don't blow past filesystem name limits on
    # exotic HF repo ids.
    return slug[:200] or "unknown-model"


def kv_signature(kv_bits: Optional[int], kv_group_size: int) -> str:
    """Return a stable string describing the KV quantisation layout.

    Two caches produced with different ``kv_bits`` / group-size tensors
    aren't interchangeable — if we loaded a 4-bit cache into a model
    running with full-precision attention we'd get silently corrupt
    completions.  Stamping the signature into the directory name makes
    a mismatch a guaranteed miss.
    """
    if kv_bits is None:
        return "full"
    return f"b{int(kv_bits)}_gs{int(kv_group_size)}"


def compute_model_fingerprint(model_path: str) -> str:
    """Best-effort hash of the model's config + tokeniser contents.

    We walk the HF cache looking for ``config.json`` + ``tokenizer.json``
    (or ``tokenizer.model`` as a fallback) under the repo's snapshots
    folder and hash their bytes.  If we can't locate them (e.g. running
    against a local-only model path) we fall back to the mtime of the
    closest matching directory, so the fingerprint at least *changes*
    when the model is replaced.  The overall contract is: equal
    fingerprint ⇒ same weights/tokeniser, so a stored KV cache is
    replayable.
    """
    candidates = _find_model_files(model_path)
    if candidates:
        h = hashlib.sha256()
        for path in candidates:
            try:
                h.update(path.read_bytes())
            except OSError:
                continue
        digest = h.hexdigest()[:16]
        if digest:
            return digest

    # Degraded fingerprint: use the model id + current mtime bucket.  This
    # is stable within a run and flips to a new directory if the files
    # change, which is good enough to prevent hot-swap mismatches.
    try:
        mtime = int(os.path.getmtime(model_path))
    except OSError:
        mtime = 0
    return hashlib.sha256(f"{model_path}:{mtime}".encode()).hexdigest()[:16]


def _find_model_files(model_path: str) -> List[Path]:
    """Locate config.json + tokenizer files for *model_path* if present."""
    search_roots: List[Path] = []

    # Direct path (local model folder).
    p = Path(model_path)
    if p.is_dir():
        search_roots.append(p)

    # HF cache layout: ``<cache>/models--<org>--<name>/snapshots/<rev>/``.
    hub_cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
    if hub_cache:
        root = Path(hub_cache)
        slug = "models--" + model_path.replace("/", "--")
        snap_dir = root / slug / "snapshots"
        if snap_dir.is_dir():
            for child in snap_dir.iterdir():
                if child.is_dir():
                    search_roots.append(child)

    files: List[Path] = []
    for root in search_roots:
        for name in ("config.json", "tokenizer.json", "tokenizer.model"):
            f = root / name
            if f.is_file():
                files.append(f)
    return files


def resolve_root(override: str) -> Path:
    """Return the on-disk root for the SSD cache.

    Prefers the explicit ``turbo_ssd_dir`` override, otherwise falls
    back to ``<app_data>/kv_cache`` via the same
    :func:`backend.config.get_app_data_dir` helper used for
    ``config.json``.  Keeping the default under the app data dir means
    a user clearing the app's state cleans the cache too.
    """
    if override and override.strip():
        return Path(override).expanduser()

    try:
        # Imported lazily to avoid circular deps — ``chat_models`` is on
        # the import path of every MLX entry point, while ``backend`` is
        # only loaded inside the Tauri/FastAPI process.  Fall back to an
        # HOME-relative path when the backend module isn't available
        # (e.g. unit tests or CLI use).
        from backend.config import get_app_data_dir
        return get_app_data_dir() / "kv_cache"
    except Exception:
        return Path.home() / ".cache" / "otto" / "kv_cache"


# ── Index ────────────────────────────────────────────────────────────────


@dataclass
class _IndexEntry:
    """One safetensors blob on disk, scoped to a model namespace."""

    prefix_hash: str
    # Path is stored as a string for JSON round-tripping.  It's always
    # a subdirectory of the store root, under the per-model namespace.
    path: str
    model_dir: str
    prefix_len: int
    size: int
    created_at: float
    last_access: float
    # Base64-encoded ``i4`` LE token stream for the saved prefix.
    # Empty string for legacy entries written before this field
    # existed — those entries still work for exact-hash lookups but
    # can't participate in common-prefix discovery.  Declaring the
    # default here (rather than in ``__post_init__``) is what makes
    # the index JSON forward-compatible: dropping the key simply
    # falls through to ``""``.
    tokens_b64: str = field(default="")


class SSDPrefixStore:
    """Persistent, LRU-bounded store of prompt-cache snapshots."""

    INDEX_NAME = "index.json"
    INDEX_VERSION = 1

    def __init__(
        self,
        *,
        global_root: Path,
        model_path: str,
        model_fingerprint: str,
        kv_sig: str,
        max_bytes: int,
    ) -> None:
        self.global_root = Path(global_root)
        self.model_path = model_path
        # Namespacing (model, fingerprint, kv) is the *correctness*
        # barrier: every cached file only replays against a model whose
        # config + tokeniser bytes + quantisation match the directory
        # name.  Across-run invalidation therefore costs nothing more
        # than a few stale files waiting for the LRU to reap them.
        self.model_dir = (
            self.global_root
            / sanitize_model_id(model_path)
            / f"v={model_fingerprint}"
            / f"kv={kv_sig}"
        )
        self._index_path = self.global_root / self.INDEX_NAME
        self._max_bytes = max_bytes
        self._lock = threading.RLock()
        self._index: Dict[str, _IndexEntry] = {}

        self.global_root.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._load_index()
        self._scan_and_reconcile()

        logger.info(
            "SSDPrefixStore ready: root=%s model=%s (%d entries, "
            "%.2f / %.2f GB used)",
            self.global_root, model_path, len(self._index),
            self.total_bytes / (1024 ** 3), max_bytes / (1024 ** 3),
        )

    # ── Index persistence ────────────────────────────────────────────

    def _load_index(self) -> None:
        if not self._index_path.exists():
            return
        try:
            data = json.loads(self._index_path.read_text("utf-8"))
            if data.get("version") != self.INDEX_VERSION:
                logger.info(
                    "SSD cache index version mismatch (got %s, want %s) — "
                    "rebuilding from disk scan.",
                    data.get("version"), self.INDEX_VERSION,
                )
                return
            for raw in data.get("entries", []):
                try:
                    entry = _IndexEntry(**raw)
                    self._index[entry.prefix_hash] = entry
                except TypeError:
                    continue
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "SSD cache index at %s unreadable (%s) — rebuilding.",
                self._index_path, exc,
            )

    def _persist_index_locked(self) -> None:
        """Write the in-memory index to disk.  Caller must hold _lock."""
        try:
            data = {
                "version": self.INDEX_VERSION,
                "entries": [asdict(e) for e in self._index.values()],
            }
            tmp = self._index_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            os.replace(tmp, self._index_path)
        except OSError as exc:
            logger.warning("SSD cache index persist failed: %s", exc)

    def _scan_and_reconcile(self) -> None:
        """Bring the in-memory index in sync with the actual disk state.

        Drops index entries whose files were removed out-of-band and
        (optionally) re-indexes orphan safetensors files under the
        current model namespace by reading the ``prefix_hash`` /
        ``prefix_len`` metadata stamped in by :meth:`save`.  Running on
        every init keeps ``turbo_ssd_max_gb`` accurate after crashes.
        """
        with self._lock:
            removed: List[str] = []
            for h, entry in self._index.items():
                if not Path(entry.path).exists():
                    removed.append(h)
            for h in removed:
                self._index.pop(h, None)
            if removed:
                logger.info(
                    "SSD cache: dropped %d index entries with missing files",
                    len(removed),
                )

            # Re-index orphans only under the active model namespace.
            # Scanning every model's shards on every init would be
            # unnecessarily slow on machines with many cached repos;
            # the orphans belonging to other models are picked up
            # whenever those models are the active one.
            if self.model_dir.exists():
                added = 0
                known_paths = {e.path for e in self._index.values()}
                for path in self.model_dir.rglob("*.safetensors"):
                    if str(path) in known_paths:
                        continue
                    entry = self._read_entry_metadata(path)
                    if entry is None:
                        continue
                    self._index[entry.prefix_hash] = entry
                    added += 1
                if added:
                    logger.info(
                        "SSD cache: indexed %d orphan file(s) under %s",
                        added, self.model_dir,
                    )

            if removed or (self.model_dir.exists()):
                self._persist_index_locked()

    def _read_entry_metadata(self, path: Path) -> Optional[_IndexEntry]:
        """Recover an ``_IndexEntry`` from a safetensors file's metadata."""
        try:
            import mlx.core as mx
            _, meta = mx.load(str(path), return_metadata=True)
        except Exception:
            return None
        if not meta:
            return None
        prefix_hash = meta.get("prefix_hash", "")
        prefix_len_str = meta.get("prefix_len", "")
        if not prefix_hash or not prefix_len_str:
            return None
        try:
            prefix_len = int(prefix_len_str)
        except ValueError:
            return None
        try:
            stat = path.stat()
        except OSError:
            return None
        return _IndexEntry(
            prefix_hash=prefix_hash,
            path=str(path),
            model_dir=str(self.model_dir),
            prefix_len=prefix_len,
            size=stat.st_size,
            created_at=stat.st_ctime,
            last_access=stat.st_mtime,
            tokens_b64=meta.get("tokens_b64", ""),
        )

    # ── Query ────────────────────────────────────────────────────────

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return sum(e.size for e in self._index.values())

    def find_longest_match(
        self, tokens: List[int],
    ) -> Optional[Tuple[int, Path]]:
        """Return ``(prefix_len, file_path)`` for the longest saved prefix.

        Only entries under the current model namespace participate, so
        cross-model collisions are impossible.  Verification is done by
        rehashing ``tokens[:prefix_len]`` — collisions would require a
        sha256 preimage attack, which we accept as "not going to happen".
        """
        with self._lock:
            active = str(self.model_dir)
            candidates = sorted(
                (e for e in self._index.values() if e.model_dir == active),
                key=lambda e: e.prefix_len,
                reverse=True,
            )
        for entry in candidates:
            if entry.prefix_len > len(tokens):
                continue
            if _hash_tokens(tokens[:entry.prefix_len]) != entry.prefix_hash:
                continue
            return entry.prefix_len, Path(entry.path)
        return None

    def find_best_common_prefix(self, tokens: List[int]) -> int:
        """Return the longest token prefix *tokens* shares with any entry.

        Unlike :meth:`find_longest_match`, this is a *discovery* call: it
        doesn't require the saved entry to already be a prefix of the
        current tokens, it iterates every entry under the active model
        namespace and measures the common-prefix length between *tokens*
        and the entry's own stored tokens.  The answer is the "best
        system-prefix length we can prove is shared with something on
        disk" — which is exactly the save length that maximises future
        cross-session hit rate.

        Legacy entries with no ``tokens_b64`` stamp are skipped (we
        can't compute a prefix against tokens we don't have).  Returns
        0 if the store is empty or nothing overlaps.
        """
        if not tokens:
            return 0
        with self._lock:
            active = str(self.model_dir)
            snapshots = [
                e.tokens_b64 for e in self._index.values()
                if e.model_dir == active and e.tokens_b64
            ]
        best = 0
        for blob in snapshots:
            stored = _decode_tokens(blob)
            if not stored:
                continue
            limit = min(len(tokens), len(stored))
            if limit <= best:
                # Even a 100% match couldn't beat the current best;
                # skip the per-token compare entirely.
                continue
            lcp = 0
            for i in range(limit):
                if tokens[i] != stored[i]:
                    break
                lcp = i + 1
            if lcp > best:
                best = lcp
        return best

    def load(self, path: Path) -> Tuple[List[Any], Dict[str, str]]:
        """Load a cached safetensors blob back into a live prompt cache.

        Returns ``(cache, metadata)`` where *cache* is the list of
        ``mlx_lm`` cache layer objects and *metadata* is the string map
        that was stamped in at save time.  Touches the LRU on success.
        """
        from mlx_lm.models.cache import load_prompt_cache

        cache, metadata = load_prompt_cache(str(path), return_metadata=True)
        metadata = metadata or {}
        prefix_hash = metadata.get("prefix_hash")
        if prefix_hash:
            self._touch(prefix_hash)
        return cache, metadata

    def _touch(self, prefix_hash: str) -> None:
        """Update last-access time for *prefix_hash*."""
        with self._lock:
            entry = self._index.get(prefix_hash)
            if entry is None:
                return
            entry.last_access = time.time()
            self._persist_index_locked()

    # ── Save / evict ─────────────────────────────────────────────────

    def save(
        self,
        tokens: List[int],
        cache: List[Any],
        extra_meta: Optional[Dict[str, str]] = None,
    ) -> Optional[Path]:
        """Persist *cache* to disk keyed by the sha256 of *tokens*.

        Returns the file path on success, ``None`` when the write failed
        (e.g. disk full, mlx_lm refused to serialise this layer type).
        Re-saves of an already-stored prefix short-circuit after a
        touch-update.
        """
        if not tokens or not cache:
            return None
        prefix_hash = _hash_tokens(tokens)
        with self._lock:
            existing = self._index.get(prefix_hash)
        if existing is not None and Path(existing.path).exists():
            self._touch(prefix_hash)
            return Path(existing.path)

        # Shard under the first two hex chars to keep any single directory
        # well under the 10k-entry point where filesystems start to
        # slow down on enumeration.
        shard = prefix_hash[:2]
        target_dir = self.model_dir / shard
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / f"{prefix_hash}.safetensors"

        tokens_b64 = _encode_tokens(tokens)
        meta: Dict[str, str] = {
            "prefix_hash": prefix_hash,
            "prefix_len": str(len(tokens)),
            "model_path": self.model_path,
            "saved_at": str(int(time.time())),
            "tokens_b64": tokens_b64,
        }
        if extra_meta:
            meta.update(extra_meta)

        try:
            from mlx_lm.models.cache import save_prompt_cache
            save_prompt_cache(str(out_path), cache, meta)
        except Exception as exc:
            logger.warning(
                "SSD cache save failed for %s: %s", out_path.name, exc,
            )
            # Best-effort cleanup of a partially-written file so the next
            # save pass doesn't trip over a zero-byte shard.
            try:
                out_path.unlink()
            except OSError:
                pass
            return None

        try:
            size = out_path.stat().st_size
        except OSError:
            return None

        entry = _IndexEntry(
            prefix_hash=prefix_hash,
            path=str(out_path),
            model_dir=str(self.model_dir),
            prefix_len=len(tokens),
            size=size,
            created_at=time.time(),
            last_access=time.time(),
            tokens_b64=tokens_b64,
        )
        with self._lock:
            self._index[prefix_hash] = entry
            self._persist_index_locked()

        self._enforce_size_limit()

        logger.info(
            "SSD cache: saved prefix len=%d size=%.1f MB -> %s",
            len(tokens), size / (1024 * 1024), out_path,
        )
        return out_path

    def _enforce_size_limit(self) -> None:
        """Evict LRU entries until under the configured GB budget."""
        if self._max_bytes <= 0:
            return
        with self._lock:
            if sum(e.size for e in self._index.values()) <= self._max_bytes:
                return
            # Sort ascending by last_access: oldest first.
            by_lru = sorted(self._index.values(), key=lambda e: e.last_access)
            removed: List[_IndexEntry] = []
            remaining = sum(e.size for e in self._index.values())
            for entry in by_lru:
                if remaining <= self._max_bytes:
                    break
                self._index.pop(entry.prefix_hash, None)
                remaining -= entry.size
                removed.append(entry)
            self._persist_index_locked()
        # Unlink outside the lock — disk I/O can block.
        for entry in removed:
            try:
                Path(entry.path).unlink()
            except OSError:
                pass
        if removed:
            logger.info(
                "SSD cache: evicted %d LRU entries (%.1f MB freed)",
                len(removed),
                sum(e.size for e in removed) / (1024 * 1024),
            )

    # ── Admin / inspection ───────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return an overview usable by the settings UI."""
        with self._lock:
            entries = list(self._index.values())
        by_model: Dict[str, Dict[str, int]] = {}
        for e in entries:
            bucket = by_model.setdefault(
                e.model_dir,
                {"entries": 0, "size_bytes": 0},
            )
            bucket["entries"] += 1
            bucket["size_bytes"] += e.size
        total_size = sum(e.size for e in entries)
        return {
            "root": str(self.global_root),
            "active_model_dir": str(self.model_dir),
            "entries": len(entries),
            "total_bytes": total_size,
            "total_gb": round(total_size / (1024 ** 3), 3),
            "max_bytes": self._max_bytes,
            "max_gb": round(self._max_bytes / (1024 ** 3), 3),
            "per_model": [
                {
                    "model_dir": md,
                    "entries": v["entries"],
                    "size_bytes": v["size_bytes"],
                    "size_gb": round(v["size_bytes"] / (1024 ** 3), 3),
                }
                for md, v in sorted(by_model.items())
            ],
        }

    def clear_all(self) -> int:
        """Delete every file in the global cache and reset the index."""
        with self._lock:
            paths = [Path(e.path) for e in self._index.values()]
            self._index.clear()
            self._persist_index_locked()
        removed = 0
        for p in paths:
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
        # Best-effort removal of empty shard dirs so a cleared cache
        # actually looks empty on disk (the user can also delete the
        # root by hand; we don't touch non-empty dirs).
        _prune_empty_dirs(self.global_root)
        logger.info("SSD cache: cleared %d file(s)", removed)
        return removed

    def clear_model(self, model_path: str) -> int:
        """Delete only the entries under *model_path*'s namespace."""
        target_prefix = str(
            self.global_root / sanitize_model_id(model_path),
        )
        with self._lock:
            victims = [
                e for e in self._index.values()
                if e.path.startswith(target_prefix)
            ]
            for v in victims:
                self._index.pop(v.prefix_hash, None)
            self._persist_index_locked()
        for v in victims:
            try:
                Path(v.path).unlink()
            except OSError:
                pass
        _prune_empty_dirs(Path(target_prefix))
        logger.info(
            "SSD cache: cleared %d entries for %s",
            len(victims), model_path,
        )
        return len(victims)


def _prune_empty_dirs(root: Path) -> None:
    """Remove empty directories under *root*, bottom-up."""
    if not root.exists():
        return
    try:
        for dirpath, dirnames, filenames in os.walk(root, topdown=False):
            if not dirnames and not filenames and Path(dirpath) != root:
                try:
                    os.rmdir(dirpath)
                except OSError as exc:
                    logger.debug(
                        "SSD cache: could not rmdir empty shard %s: %s",
                        dirpath, exc,
                    )
    except OSError as exc:
        logger.debug("SSD cache: prune walk aborted at %s: %s", root, exc)
