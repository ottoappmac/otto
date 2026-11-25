"""Embedders for procedure items.

The default is the sentence-transformers model Otto already uses for memory (``EmbeddingConfig.model_name``,
``all-MiniLM-L6-v2``: 384-d, normalised). Intents are short single sentences, where small models do well, and no
published ablation shows the model choice matters much for agent memory; Phase 0 compares it with one stronger model
before anything changes. ``HashEmbedder`` is a dependency-free stand-in for tests and for machines without the model.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


class Embedder(Protocol):
    name: str

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceEmbedder:
    def __init__(self, model_name: str) -> None:
        from backend.embedding_index import _Embedder
        self.name = model_name
        self._inner = _Embedder(model_name)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await self._inner.embed(texts)


_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Bag of words and word pairs hashed into ``dim`` buckets, L2-normalised. Deterministic; no model."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        toks = _TOKEN_RE.findall((text or "").lower())
        grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for g in grams:
            h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little")
            vec[h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]


def default_embedder(model_name: str | None = None) -> Embedder:
    name = model_name
    if not name:
        try:
            from backend.config import AppConfig
            name = AppConfig.load().memory.embedding.model_name
        except Exception:  # noqa: BLE001 — config unreadable: use the shipped default
            name = "sentence-transformers/all-MiniLM-L6-v2"
    if name.startswith("hash-"):
        return HashEmbedder(int(name.split("-", 1)[1] or 256))
    return SentenceEmbedder(name)
