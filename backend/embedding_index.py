"""Persistent semantic embedding index backed by sqlite-vec.

Indexes three bounded, high-signal artifact sets:
  - Memory topic files (``memory/*.md``) — re-indexed post-consolidation.
  - Session transcripts (``transcripts/*.jsonl``) — incremental, post-session.
  - Session upload/symlink files — user-explicit, post-upload.
  - User-pinned directories — opt-in via the Settings UI.

The embedding model runs via ``sentence-transformers``, which works on any
platform (CPU/MPS/CUDA).  No HuggingFace token is required for public models.

The sqlite-vec index is a rebuildable side-effect; the source files are the
source of record.  Delete ``embeddings.db`` and re-trigger to rebuild.

Usage::

    from backend.embedding_index import get_embedding_index

    idx = get_embedding_index()
    await idx.index_memory()
    results = await idx.search("how do I authenticate?", k=5)
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.config import EmbeddingConfig, get_app_data_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source type constants
# ---------------------------------------------------------------------------

SOURCE_MEMORY = "memory"
SOURCE_TRANSCRIPT = "transcript"
SOURCE_FILE = "file"


# ---------------------------------------------------------------------------
# Lazy MLX embedder
# ---------------------------------------------------------------------------


class _Embedder:
    """Lazy-loading sentence-transformers embedder.

    Uses the ``sentence-transformers`` library which works on any platform
    (CPU, MPS on Apple Silicon, CUDA) and supports every public HuggingFace
    sentence-embedding model without requiring a HF token.

    The model is downloaded from HuggingFace Hub on first use.  Subsequent
    calls reuse the in-memory weights.  Embeddings are L2-normalised so
    cosine similarity == dot-product.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any = None
        self._dim: int = 384  # updated after first load

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install with: pip install sentence-transformers"
            ) from exc
        logger.info("[embedding] loading model %s …", self._model_name)
        self._model = SentenceTransformer(self._model_name)
        self._dim = self._model.get_sentence_embedding_dimension() or 384
        logger.info("[embedding] model ready, dim=%d", self._dim)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning L2-normalised float vectors."""
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return vecs.tolist()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        return await asyncio.to_thread(self._embed_sync, texts)

    @property
    def dim(self) -> int:
        return self._dim


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path TEXT    NOT NULL,
    source_type TEXT    NOT NULL,
    chunk_text  TEXT    NOT NULL,
    mtime       REAL    NOT NULL DEFAULT 0,
    indexed_at  REAL    NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_path);
CREATE INDEX IF NOT EXISTS idx_chunks_type   ON chunks(source_type);
"""

# The vec0 table is created dynamically once we know the embedding dimension.
_VEC_TABLE_TEMPLATE = """
CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding FLOAT[{dim}]
);
"""


def _vec_bytes(vec: list[float]) -> bytes:
    """Pack a float list into little-endian float32 bytes for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


# ---------------------------------------------------------------------------
# EmbeddingIndex
# ---------------------------------------------------------------------------


@dataclass
class IndexStatus:
    sources: list[dict[str, Any]] = field(default_factory=list)
    total_chunks: int = 0


class EmbeddingIndex:
    """Semantic index over OTTO's local artifacts.

    Stores vectors in SQLite.  Two backends are supported:

    * **sqlite-vec** (preferred) — uses the ``vec0`` virtual table for fast
      C-accelerated KNN search.  Requires a Python build with
      ``--enable-loadable-sqlite-extensions`` (Homebrew Python) and
      ``sqlite-vec`` installed.

    * **NumPy fallback** — vectors are stored as raw float32 blobs in a plain
      ``vectors`` table.  Search loads them into a NumPy matrix and computes
      cosine similarity.  Works with any Python build, including pyenv builds
      that lack loadable-extension support.  Fast enough for OTTO's scale
      (thousands of chunks).

    The backend is detected once at first use and stored on the instance.
    """

    def __init__(self, db_path: Path, cfg: EmbeddingConfig) -> None:
        self._db_path = db_path
        self._cfg = cfg
        self._embedder = _Embedder(cfg.model_name)
        self._db_lock = asyncio.Lock()
        self._ready = False
        self._use_vec_ext: bool = False  # resolved in _ensure_ready

    # ------------------------------------------------------------------
    # Connection + schema
    # ------------------------------------------------------------------

    @staticmethod
    def _probe_vec_ext() -> bool:
        """Return True if sqlite-vec can be loaded in this Python environment."""
        import sqlite3
        try:
            import sqlite_vec  # type: ignore[import]
        except ImportError:
            return False
        probe = sqlite3.connect(":memory:")
        try:
            if not hasattr(probe, "enable_load_extension"):
                return False
            probe.enable_load_extension(True)
            sqlite_vec.load(probe)
            return True
        except Exception:
            return False
        finally:
            probe.close()

    def _open(self):
        """Open a connection, loading sqlite-vec if the backend requires it."""
        import sqlite3
        db = sqlite3.connect(str(self._db_path), check_same_thread=False)
        db.row_factory = sqlite3.Row
        if self._use_vec_ext:
            import sqlite_vec  # type: ignore[import]
            db.enable_load_extension(True)
            sqlite_vec.load(db)
            db.enable_load_extension(False)
        return db

    def _init_schema_sync(self) -> None:
        """Detect backend, create tables if they don't exist yet."""
        self._use_vec_ext = self._probe_vec_ext()
        if self._use_vec_ext:
            logger.info("[embedding] backend: sqlite-vec KNN")
        else:
            logger.info("[embedding] backend: NumPy cosine fallback (pyenv Python detected)")

        db = self._open()
        try:
            db.executescript(_SCHEMA)
            if self._use_vec_ext:
                dim = self._embedder.dim
                try:
                    db.execute(_VEC_TABLE_TEMPLATE.format(dim=dim))
                    db.commit()
                except Exception:
                    logger.warning("[embedding] dimension mismatch, rebuilding vec table")
                    db.execute("DROP TABLE IF EXISTS vec_chunks")
                    db.execute("DELETE FROM chunks")
                    db.execute(_VEC_TABLE_TEMPLATE.format(dim=dim))
                    db.commit()
            else:
                # Plain blob table — no extension required.
                db.execute("""
                    CREATE TABLE IF NOT EXISTS vectors (
                        chunk_id INTEGER PRIMARY KEY
                                 REFERENCES chunks(id) ON DELETE CASCADE,
                        embedding BLOB NOT NULL
                    )
                """)
                db.commit()
        finally:
            db.close()

    async def _ensure_ready(self) -> None:
        if self._ready:
            return
        async with self._db_lock:
            if self._ready:
                return
            await asyncio.to_thread(self._embedder._ensure_loaded)
            await asyncio.to_thread(self._init_schema_sync)
            self._ready = True

    # ------------------------------------------------------------------
    # Internal upsert helpers (all run in a thread via asyncio.to_thread)
    # ------------------------------------------------------------------

    def _delete_source_sync(self, db, source_path: str) -> int:
        cur = db.execute("SELECT id FROM chunks WHERE source_path = ?", (source_path,))
        ids = [r[0] for r in cur.fetchall()]
        if ids:
            placeholders = ",".join("?" * len(ids))
            vec_table = "vec_chunks" if self._use_vec_ext else "vectors"
            db.execute(f"DELETE FROM {vec_table} WHERE chunk_id IN ({placeholders})", ids)
            db.execute("DELETE FROM chunks WHERE source_path = ?", (source_path,))
        return len(ids)

    def _upsert_chunks_sync(
        self,
        source_path: str,
        source_type: str,
        texts: list[str],
        vecs: list[list[float]],
        mtime: float,
    ) -> int:
        db = self._open()
        try:
            self._delete_source_sync(db, source_path)
            now = time.time()
            inserted = 0
            for text, vec in zip(texts, vecs):
                cur = db.execute(
                    "INSERT INTO chunks(source_path, source_type, chunk_text, mtime, indexed_at)"
                    " VALUES (?,?,?,?,?)",
                    (source_path, source_type, text, mtime, now),
                )
                chunk_id = cur.lastrowid
                blob = _vec_bytes(vec)
                if self._use_vec_ext:
                    db.execute(
                        "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?,?)",
                        (chunk_id, blob),
                    )
                else:
                    db.execute(
                        "INSERT INTO vectors(chunk_id, embedding) VALUES (?,?)",
                        (chunk_id, blob),
                    )
                inserted += 1
            db.commit()
            return inserted
        finally:
            db.close()

    def _needs_reindex_sync(self, source_path: str, current_mtime: float) -> bool:
        db = self._open()
        try:
            row = db.execute(
                "SELECT mtime FROM chunks WHERE source_path = ? LIMIT 1",
                (source_path,),
            ).fetchone()
            if row is None:
                return True
            return float(row["mtime"]) < current_mtime
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Chunking helper
    # ------------------------------------------------------------------

    def _chunk_text(self, text: str) -> list[str]:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._cfg.chunk_size,
            chunk_overlap=self._cfg.chunk_overlap,
        )
        return splitter.split_text(text)

    # ------------------------------------------------------------------
    # Core indexing — used by all public index_* methods
    # ------------------------------------------------------------------

    async def _index_text(
        self,
        source_path: str,
        source_type: str,
        text: str,
        mtime: float = 0.0,
    ) -> int:
        """Chunk, embed, and upsert a raw text string."""
        chunks = await asyncio.to_thread(self._chunk_text, text)
        if not chunks:
            return 0
        # Embed in batches of 32 to avoid OOM on large documents
        all_vecs: list[list[float]] = []
        batch_size = 32
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            all_vecs.extend(await self._embedder.embed(batch))

        return await asyncio.to_thread(
            self._upsert_chunks_sync,
            source_path, source_type, chunks, all_vecs, mtime,
        )

    # ------------------------------------------------------------------
    # Public indexing API
    # ------------------------------------------------------------------

    async def index_memory(self) -> int:
        """Re-index all memory topic files.

        Called automatically after each consolidation run.
        """
        await self._ensure_ready()
        mem_dir = get_app_data_dir() / "memory"
        if not mem_dir.exists():
            return 0

        total = 0
        for path in mem_dir.glob("*.md"):
            if path.name == "MEMORY.md":
                continue
            try:
                mtime = path.stat().st_mtime
                if await asyncio.to_thread(self._needs_reindex_sync, str(path), mtime):
                    text = await asyncio.to_thread(path.read_text, "utf-8")
                    n = await self._index_text(str(path), SOURCE_MEMORY, text, mtime)
                    total += n
                    logger.debug("[embedding] indexed memory %s (%d chunks)", path.name, n)
            except Exception:
                logger.debug("[embedding] failed to index memory %s", path.name, exc_info=True)

        logger.info("[embedding] index_memory complete — %d chunks upserted", total)
        return total

    async def index_transcript(self, session_id: str) -> int:
        """Index a single session transcript JSONL.

        Called incrementally after each session completes.
        """
        await self._ensure_ready()
        from backend.session_transcript import _transcript_path  # avoid circular at module level

        path = _transcript_path(session_id)
        if not path.exists():
            return 0
        try:
            mtime = path.stat().st_mtime
            if not await asyncio.to_thread(self._needs_reindex_sync, str(path), mtime):
                return 0
            text = await asyncio.to_thread(path.read_text, "utf-8")
            n = await self._index_text(str(path), SOURCE_TRANSCRIPT, text, mtime)
            logger.debug("[embedding] indexed transcript %s (%d chunks)", session_id[:8], n)
            return n
        except Exception:
            logger.debug("[embedding] failed to index transcript %s", session_id, exc_info=True)
            return 0

    async def index_file(self, path: Path, source_type: str = SOURCE_FILE) -> int:
        """Index a local file using the existing _loaders pipeline.

        Called after session file upload / symlink creation, or when a
        user pins a directory via the Settings UI.
        """
        await self._ensure_ready()
        try:
            mtime = path.stat().st_mtime if path.exists() else 0.0
            if not await asyncio.to_thread(self._needs_reindex_sync, str(path), mtime):
                return 0
            from tools.research._loaders import load_source, split_documents  # lazy import

            docs = await load_source(str(path))
            if not docs:
                return 0
            chunks_docs = await asyncio.to_thread(split_documents, docs)
            texts = [d.page_content for d in chunks_docs if d.page_content.strip()]
            if not texts:
                return 0
            all_vecs: list[list[float]] = []
            for i in range(0, len(texts), 32):
                all_vecs.extend(await self._embedder.embed(texts[i:i + 32]))
            n = await asyncio.to_thread(
                self._upsert_chunks_sync,
                str(path), source_type, texts, all_vecs, mtime,
            )
            logger.debug("[embedding] indexed file %s (%d chunks)", path.name, n)
            return n
        except Exception:
            logger.debug("[embedding] failed to index file %s", path, exc_info=True)
            return 0

    async def index_directory(self, directory: Path) -> int:
        """Index all supported files in a directory (user-pinned paths)."""
        _SUPPORTED = {".pdf", ".docx", ".pptx", ".xlsx", ".md", ".txt", ".rst", ".json", ".csv"}
        total = 0
        for p in directory.rglob("*"):
            if p.is_file() and p.suffix.lower() in _SUPPORTED:
                total += await self.index_file(p, source_type=SOURCE_FILE)
        return total

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _search_sync(
        self,
        query_vec: list[float],
        k: int,
        source_type: str | None,
    ) -> list[dict[str, Any]]:
        db = self._open()
        try:
            if self._use_vec_ext:
                # Fast C-accelerated KNN via sqlite-vec.
                vec_bytes = _vec_bytes(query_vec)
                if source_type:
                    sql = """
                        SELECT c.id, c.source_path, c.source_type, c.chunk_text,
                               v.distance
                        FROM vec_chunks v
                        JOIN chunks c ON c.id = v.chunk_id
                        WHERE v.embedding MATCH ?
                          AND v.k = ?
                          AND c.source_type = ?
                        ORDER BY v.distance
                    """
                    rows = db.execute(sql, (vec_bytes, k, source_type)).fetchall()
                else:
                    sql = """
                        SELECT c.id, c.source_path, c.source_type, c.chunk_text,
                               v.distance
                        FROM vec_chunks v
                        JOIN chunks c ON c.id = v.chunk_id
                        WHERE v.embedding MATCH ?
                          AND v.k = ?
                        ORDER BY v.distance
                    """
                    rows = db.execute(sql, (vec_bytes, k)).fetchall()
                return [
                    {
                        "text": row["chunk_text"],
                        "source_path": row["source_path"],
                        "source_type": row["source_type"],
                        "score": 1.0 - float(row["distance"]),
                    }
                    for row in rows
                ]
            else:
                # NumPy cosine similarity fallback.
                import numpy as np

                if source_type:
                    sql = """
                        SELECT v.embedding, c.source_path, c.source_type, c.chunk_text
                        FROM vectors v JOIN chunks c ON c.id = v.chunk_id
                        WHERE c.source_type = ?
                    """
                    rows = db.execute(sql, (source_type,)).fetchall()
                else:
                    sql = """
                        SELECT v.embedding, c.source_path, c.source_type, c.chunk_text
                        FROM vectors v JOIN chunks c ON c.id = v.chunk_id
                    """
                    rows = db.execute(sql).fetchall()

                if not rows:
                    return []

                # Stack into matrix and compute dot-product (== cosine sim for
                # L2-normalised vectors as produced by sentence-transformers).
                mat = np.array(
                    [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
                )
                q = np.array(query_vec, dtype=np.float32)
                scores: list[float] = (mat @ q).tolist()

                ranked = sorted(
                    zip(scores, rows), key=lambda t: t[0], reverse=True
                )[:k]
                return [
                    {
                        "text": r["chunk_text"],
                        "source_path": r["source_path"],
                        "source_type": r["source_type"],
                        "score": float(score),
                    }
                    for score, r in ranked
                    if score > 0.0
                ]
        finally:
            db.close()

    async def search(
        self,
        query: str,
        k: int = 10,
        source_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return up to *k* semantically relevant chunks.

        Args:
            query: Natural-language search query.
            k: Maximum results to return.
            source_type: Optional filter — ``"memory"``, ``"transcript"``,
                or ``"file"``.  ``None`` searches across all types.

        Returns:
            List of dicts with ``text``, ``source_path``, ``source_type``,
            ``score`` (0–1, higher is better).
        """
        await self._ensure_ready()
        vecs = await self._embedder.embed([query])
        query_vec = vecs[0]
        return await asyncio.to_thread(self._search_sync, query_vec, k, source_type)

    # ------------------------------------------------------------------
    # Status + management
    # ------------------------------------------------------------------

    def _get_status_sync(self) -> dict[str, Any]:
        db = self._open()
        try:
            total = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            rows = db.execute(
                "SELECT source_path, source_type, COUNT(*) as chunk_count,"
                "       MAX(indexed_at) as last_indexed"
                " FROM chunks GROUP BY source_path ORDER BY last_indexed DESC"
            ).fetchall()
            sources = [
                {
                    "source_path": r["source_path"],
                    "source_type": r["source_type"],
                    "chunk_count": r["chunk_count"],
                    "indexed_at": r["last_indexed"],
                }
                for r in rows
            ]
            return {"total_chunks": total, "sources": sources}
        finally:
            db.close()

    async def get_status(self) -> dict[str, Any]:
        if not self._ready:
            return {"total_chunks": 0, "sources": []}
        return await asyncio.to_thread(self._get_status_sync)

    def _remove_source_sync(self, source_path: str) -> int:
        db = self._open()
        try:
            n = self._delete_source_sync(db, source_path)
            db.commit()
            return n
        finally:
            db.close()

    async def remove_source(self, source_path: str) -> int:
        """Remove all chunks for a given source path from the index."""
        await self._ensure_ready()
        return await asyncio.to_thread(self._remove_source_sync, source_path)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_index: EmbeddingIndex | None = None
_index_lock = asyncio.Lock()


async def get_embedding_index() -> EmbeddingIndex:
    """Return the process-level singleton EmbeddingIndex.

    Lazily constructed on first call.  Callers should ``await`` this
    but the returned object itself is not async — call its async methods
    directly after obtaining it.
    """
    global _index
    if _index is not None:
        return _index
    async with _index_lock:
        if _index is not None:
            return _index
        from backend.config import AppConfig

        cfg = await AppConfig.aload()
        db_path = get_app_data_dir() / "embeddings.db"
        _index = EmbeddingIndex(db_path, cfg.memory.embedding)
        return _index
