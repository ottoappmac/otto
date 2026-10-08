"""Process-wide store of static prompt-prefix snapshots for ``ChatMLXText``.

A tool-calling agent's prompt opens with a block the conversation doesn't
change: the tool schemas and the system prompt.  OTTO's orchestrator block is
~34.6k tokens — ~130 s of prefill at ~260 tok/s — and every new session (a
new ``ChatMLXText`` instance, empty cache) used to prefill it again.  This
store keeps prompt-cache snapshots taken at exactly the end of such prefixes,
so any instance running the same loaded model can restore one instead (see
``ChatMLXText._prepare_prompt_cache``).  Snapshots worth it also go to SSD
(:mod:`chat_models.mlx._prefix_disk`) so they survive a restart; one loaded
back from there goes to its session only, not to this store.

Entries are keyed by the loaded model object, the cache-affecting settings
and the exact prefix tokens.  The dict lookup hashes the tokens and then
compares them in full, so a hit is always an exact match.  The model is held
by weak reference: an entry never keeps weights alive, and an entry whose
model is gone can't match a new model that happens to get the same ``id``.

Entries are grouped in families, one per bound tool set (an agent: the
orchestrator, a browser agent, ...).  Each family keeps its own
``max_entries`` — its tool block, shared by every session of that agent, and
its latest session's system turn — so agents running at the same time never
evict each other's snapshots.  Past ``max_families`` families, or
``max_bytes`` in total, whole families are evicted, least recently used
first; the family just stored into always stays.

The stored cache layers are shared, read-only data: callers clone them into a
live cache and never hand them to a model.

Memory: an entry holds the KV of every attention layer for the prefix plus
the state of every recurrent layer.  For Qwen3.8-9B (qwen3_5_text: 8
attention layers × 4 KV heads × 256 dims, 24 GatedDeltaNet layers) at
``kv_bits=4`` the KV is ~9.2 KB per token — ~0.32 GB for a 34.6k-token
prefix (bf16: ~1.1 GB) — and the recurrent state ~49 MB (a float32
32×128×128 SSM state per layer), ~0.37 GB per entry.  A family of two is
~0.6-0.7 GB: the orchestrator and a browser agent running at the same time
take ~1.3 GB, under the 1.5 GiB ``MAX_BYTES`` cap, which keeps a bf16 family
(~1.1 GB per entry) alone.  The entries are MLX buffers that stay resident
next to the weights during every generation, so the store stays small: an
evicted family's tool block comes back from SSD in ~0.1-0.3 s.
"""

from __future__ import annotations

import threading
import weakref
from typing import Any, Hashable, List, Optional, Sequence

# Per family: an agent's tool block plus one session's system turn; storing a
# third prefix evicts that family's least recently used one.
MAX_ENTRIES = 2
# Agents whose snapshots are kept at once, and the memory they may take.
MAX_FAMILIES = 2
MAX_BYTES = 3 * 1024**3 // 2


class PrefixSnapshotStore:
    """A thread-safe two-level LRU of prompt-cache snapshots, each taken at an exact token prefix."""

    def __init__(
        self,
        max_entries: int = MAX_ENTRIES,
        max_families: int = MAX_FAMILIES,
        max_bytes: int = MAX_BYTES,
    ) -> None:
        self.max_entries = max_entries
        self.max_families = max_families
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        # family → {key: (weak reference to the model, cache layers, bytes)};
        # least recently used first at both levels.
        self._families: dict = {}
        self._family_of: dict = {}

    def get(
        self, model: Any, settings: Hashable, tokens: Sequence[int],
    ) -> Optional[List[Any]]:
        """Return the snapshot of exactly *tokens* for *model*, or ``None``.

        A hit becomes the most recently used entry of the most recently used
        family.
        """
        key = (id(model), settings, tuple(tokens))
        with self._lock:
            if key not in self._family_of:
                return None
            family = self._family_of[key]
            entries = self._families.pop(family)
            self._families[family] = entries
            entry = entries.pop(key)
            if entry[0]() is not model:
                del self._family_of[key]
                if not entries:
                    del self._families[family]
                return None
            entries[key] = entry
            return entry[1]

    def put(
        self,
        model: Any,
        settings: Hashable,
        tokens: Sequence[int],
        layers: List[Any],
        family: Hashable = None,
        nbytes: int = 0,
    ) -> None:
        """Store *layers* (*nbytes* big) as the snapshot of exactly *tokens* in *family*."""
        key = (id(model), settings, tuple(tokens))
        with self._lock:
            self._discard(key)
            entries = self._families.pop(family, {})
            self._families[family] = entries
            entries[key] = (weakref.ref(model), layers, nbytes)
            self._family_of[key] = family
            while len(entries) > self.max_entries:
                self._discard(next(iter(entries)))
            while len(self._families) > 1 and (
                len(self._families) > self.max_families or self._nbytes() > self.max_bytes
            ):
                for stale in self._families.pop(next(iter(self._families))):
                    del self._family_of[stale]

    def _discard(self, key: Any) -> None:
        if key not in self._family_of:
            return
        family = self._family_of.pop(key)
        entries = self._families[family]
        del entries[key]
        if not entries:
            del self._families[family]

    def _nbytes(self) -> int:
        return sum(e[2] for entries in self._families.values() for e in entries.values())

    @property
    def family_count(self) -> int:
        return len(self._families)

    def clear(self) -> None:
        with self._lock:
            self._families.clear()
            self._family_of.clear()

    def __len__(self) -> int:
        return len(self._family_of)


STATIC_PREFIXES = PrefixSnapshotStore()
