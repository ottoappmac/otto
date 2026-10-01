"""Process-wide store of static prompt-prefix snapshots for ``ChatMLXText``.

A tool-calling agent's prompt opens with a block the conversation doesn't
change: the tool schemas and the system prompt.  OTTO's orchestrator block is
~34.6k tokens — ~130 s of prefill at ~260 tok/s — and every new session (a
new ``ChatMLXText`` instance, empty cache) used to prefill it again.  This
store keeps prompt-cache snapshots taken at exactly the end of such prefixes,
so any instance running the same loaded model can restore one instead (see
``ChatMLXText._prepare_prompt_cache``).

Entries are keyed by the loaded model object, the cache-affecting settings
and the exact prefix tokens.  The dict lookup hashes the tokens and then
compares them in full, so a hit is always an exact match.  The model is held
by weak reference: an entry never keeps weights alive, and an entry whose
model is gone can't match a new model that happens to get the same ``id``.

The stored cache layers are shared, read-only data: callers clone them into a
live cache and never hand them to a model.

Memory: an entry holds the KV of every attention layer for the prefix plus
the state of every recurrent layer.  For Qwen3.8-9B (qwen3_5_text: 8
attention layers × 4 KV heads × 256 dims, 24 GatedDeltaNet layers) at
``kv_bits=4`` the KV is ~9.2 KB per token — ~0.32 GB for a 34.6k-token
prefix (bf16: ~1.1 GB) — and the recurrent state ~49 MB (a float32
32×128×128 SSM state per layer), ~0.37 GB per entry.  OTTO keeps two: its
tool block, shared by every session, and the latest session's full system
turn — ~0.7 GB.
"""

from __future__ import annotations

import threading
import weakref
from typing import Any, Hashable, List, Optional, Sequence

# Room for an agent's tool block plus one session's system turn; storing a
# third prefix evicts the least recently used one.
MAX_ENTRIES = 2


class PrefixSnapshotStore:
    """A thread-safe LRU of prompt-cache snapshots, each taken at an exact token prefix."""

    def __init__(self, max_entries: int = MAX_ENTRIES) -> None:
        self.max_entries = max_entries
        self._lock = threading.Lock()
        # key → (weak reference to the model, cache layers); least recently
        # used first.
        self._entries: dict = {}

    def get(
        self, model: Any, settings: Hashable, tokens: Sequence[int],
    ) -> Optional[List[Any]]:
        """Return the snapshot of exactly *tokens* for *model*, or ``None``.

        A hit becomes the most recently used entry.
        """
        key = (id(model), settings, tuple(tokens))
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is None or entry[0]() is not model:
                return None
            self._entries[key] = entry
            return entry[1]

    def put(
        self, model: Any, settings: Hashable, tokens: Sequence[int], layers: List[Any],
    ) -> None:
        """Store *layers* as the snapshot of exactly *tokens*, evicting the least recently used."""
        key = (id(model), settings, tuple(tokens))
        with self._lock:
            self._entries.pop(key, None)
            self._entries[key] = (weakref.ref(model), layers)
            while len(self._entries) > self.max_entries:
                del self._entries[next(iter(self._entries))]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


STATIC_PREFIXES = PrefixSnapshotStore()
