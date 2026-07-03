#!/usr/bin/env python3
"""End-to-end smoke test for the semantic search / memory retrieval pipeline.

Tests:
  1. sqlite-vec can be imported and the extension loads (requires Homebrew Python)
  2. EmbeddingIndex schema creation (vec0 virtual table)
  3. Index text chunks + vector search returns ranked results
  4. EmbeddingIndex class with patched embedder
  5. Real memory files — provenance fields check (if any exist)
  6. mlx-embeddings can be imported (model NOT downloaded — import only)
  7. API endpoints respond: /api/embeddings/status, /api/memory/topics

Run with venv Python (will show sqlite-vec limitation):
    cd /Users/eugenetan/git/agents
    uv run python scripts/test_semantic_search.py

Run standalone (sqlite-vec + vector search only, no backend deps):
    /opt/homebrew/bin/python3.11 scripts/test_semantic_search.py --standalone
"""

from __future__ import annotations

import asyncio
import math
import random
import struct
import sys
import tempfile
import traceback
from pathlib import Path

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
SKIP = "\033[33m–\033[0m"
WARN = "\033[33m⚠\033[0m"

results: list[tuple[str, str, str]] = []


def ok(name: str, detail: str = "") -> None:
    results.append((PASS, name, detail))
    line = f"  {PASS}  {name}"
    if detail:
        line += f"  \033[2m{detail}\033[0m"
    print(line)


def fail(name: str, detail: str = "") -> None:
    results.append((FAIL, name, detail))
    print(f"  {FAIL}  {name}")
    if detail:
        for ln in detail.splitlines():
            print(f"         {ln}")


def skip(name: str, reason: str = "") -> None:
    results.append((SKIP, name, reason))
    print(f"  {SKIP}  {name}" + (f"  \033[2m({reason})\033[0m" if reason else ""))


def warn(name: str, detail: str = "") -> None:
    results.append((WARN, name, detail))
    print(f"  {WARN}  {name}")
    if detail:
        for ln in detail.splitlines():
            print(f"         {ln}")


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


# ---------------------------------------------------------------------------
# Deterministic fake embedder (no model download needed for logic tests)
# ---------------------------------------------------------------------------

DIM = 16


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic unit vectors derived from text hash — no model needed."""
    out = []
    for text in texts:
        seed = hash(text) % (2 ** 32)
        rng = random.Random(seed)
        vec = [rng.gauss(0, 1) for _ in range(DIM)]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out.append([v / norm for v in vec])
    return out


def _vec_bytes(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


# ---------------------------------------------------------------------------
# Helper: check if sqlite-vec extension loading is supported
# ---------------------------------------------------------------------------

def _sqlite_vec_available() -> tuple[bool, str]:
    """Return (available, reason) for sqlite-vec extension loading."""
    import sqlite3
    try:
        import sqlite_vec  # type: ignore[import]
    except ImportError:
        return False, "sqlite-vec package not installed"

    db = sqlite3.connect(":memory:")
    if not hasattr(db, "enable_load_extension"):
        db.close()
        return False, (
            "Python compiled without --enable-loadable-sqlite-extensions\n"
            "  Fix: recreate .venv with Homebrew Python:\n"
            "    /opt/homebrew/bin/python3.11 -m venv .venv --clear\n"
            "    source .venv/bin/activate && pip install -e ."
        )
    try:
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        version = db.execute("SELECT vec_version()").fetchone()[0]
        db.close()
        return True, f"sqlite-vec {version}"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Test 1: sqlite-vec import + extension load
# ---------------------------------------------------------------------------

def test_sqlite_vec_import() -> None:
    section("1. sqlite-vec extension")
    try:
        import sqlite_vec  # type: ignore[import]
        ok("import sqlite_vec", f"path: {sqlite_vec.__file__}")
    except ImportError as e:
        fail("import sqlite_vec", str(e))
        return

    available, detail = _sqlite_vec_available()
    if available:
        ok("load extension + vec_version()", detail)
    else:
        warn("enable_load_extension not available", detail)


# ---------------------------------------------------------------------------
# Test 2: EmbeddingIndex pipeline (raw SQL, fake vectors)
# ---------------------------------------------------------------------------

async def test_embedding_index_pipeline() -> None:
    section("2. EmbeddingIndex pipeline (raw SQL, fake embedder)")

    import sqlite3
    available, reason = _sqlite_vec_available()
    if not available:
        skip("raw pipeline", f"sqlite-vec unavailable: {reason.splitlines()[0]}")
        return

    import sqlite_vec  # type: ignore[import]

    def open_db(path: str):
        db = sqlite3.connect(path, check_same_thread=False)
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        db.row_factory = sqlite3.Row
        return db

    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.db")

        # Schema
        try:
            db = open_db(db_path)
            db.executescript("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_path TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    chunk_text  TEXT NOT NULL,
                    mtime       REAL NOT NULL DEFAULT 0,
                    indexed_at  REAL NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_src ON chunks(source_path);
            """)
            db.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                    chunk_id INTEGER PRIMARY KEY,
                    embedding FLOAT[{DIM}]
                )
            """)
            db.commit()
            db.close()
            ok("schema created (chunks + vec0 virtual table)")
        except Exception:
            fail("schema creation", traceback.format_exc(limit=3))
            return

        # Upsert synthetic docs
        docs = [
            ("authentication with Stripe API using API keys", "memory"),
            ("Q3 project goals: ship semantic search feature", "memory"),
            ("deployment error: missing DATABASE_URL env var", "transcript"),
            ("user prefers dark mode interface settings", "memory"),
            ("fixed critical bug in payment processing pipeline", "transcript"),
        ]
        import time as _time
        try:
            db = open_db(db_path)
            for i, (text, stype) in enumerate(docs):
                vec = _fake_embed([text])[0]
                cur = db.execute(
                    "INSERT INTO chunks(source_path, source_type, chunk_text, mtime, indexed_at)"
                    " VALUES (?,?,?,?,?)",
                    (f"/fake/{stype}-{i}.md", stype, text, _time.time(), _time.time()),
                )
                db.execute(
                    "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?,?)",
                    (cur.lastrowid, _vec_bytes(vec)),
                )
            db.commit()
            db.close()
            ok(f"upserted {len(docs)} chunks")
        except Exception:
            fail("upsert chunks", traceback.format_exc(limit=3))
            return

        # Unfiltered search
        try:
            db = open_db(db_path)
            q_vec = _fake_embed(["how do I authenticate with Stripe?"])[0]
            rows = db.execute(
                "SELECT c.chunk_text, c.source_type, v.distance"
                " FROM vec_chunks v JOIN chunks c ON c.id = v.chunk_id"
                " WHERE v.embedding MATCH ? AND v.k = 3 ORDER BY v.distance",
                (_vec_bytes(q_vec),),
            ).fetchall()
            db.close()
            if rows:
                top = rows[0]
                ok(
                    f"unfiltered search → {len(rows)} results",
                    f"top: '{top['chunk_text'][:55]}' dist={top['distance']:.4f}",
                )
            else:
                fail("unfiltered search returned 0 results")
        except Exception:
            fail("unfiltered vector search", traceback.format_exc(limit=3))
            return

        # Source-type filter
        try:
            db = open_db(db_path)
            q_vec = _fake_embed(["deployment error"])[0]
            rows = db.execute(
                "SELECT c.chunk_text, c.source_type, v.distance"
                " FROM vec_chunks v JOIN chunks c ON c.id = v.chunk_id"
                " WHERE v.embedding MATCH ? AND v.k = 5 AND c.source_type = 'transcript'"
                " ORDER BY v.distance",
                (_vec_bytes(q_vec),),
            ).fetchall()
            db.close()
            if rows and all(r["source_type"] == "transcript" for r in rows):
                ok(f"source_type filter → {len(rows)} transcript-only results")
            else:
                fail("source_type filter broken")
        except Exception:
            fail("source_type filter", traceback.format_exc(limit=3))

        # Delete source
        try:
            db = open_db(db_path)
            db.execute("DELETE FROM vec_chunks WHERE chunk_id IN (SELECT id FROM chunks WHERE source_path = ?)",
                       ("/fake/memory-0.md",))
            db.execute("DELETE FROM chunks WHERE source_path = ?", ("/fake/memory-0.md",))
            db.commit()
            count = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            db.close()
            ok(f"delete source → {count} chunks remain (was {len(docs)})")
        except Exception:
            fail("delete source", traceback.format_exc(limit=3))


# ---------------------------------------------------------------------------
# Test 3: EmbeddingIndex class (with monkey-patched embedder)
# ---------------------------------------------------------------------------

async def test_embedding_index_class() -> None:
    section("3. EmbeddingIndex class (fake embedder)")

    repo = Path(__file__).parent.parent
    for p in [str(repo / "src"), str(repo)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        from backend.config import EmbeddingConfig
        from backend.embedding_index import EmbeddingIndex
        ok("import EmbeddingIndex")
    except Exception as e:
        fail("import EmbeddingIndex", str(e))
        return

    available, reason = _sqlite_vec_available()
    if not available:
        skip("EmbeddingIndex class tests", reason.splitlines()[0])
        return

    with tempfile.TemporaryDirectory() as tmp:
        cfg = EmbeddingConfig(enabled=True, model_name="fake", chunk_size=200, chunk_overlap=20)
        idx = EmbeddingIndex(Path(tmp) / "test.db", cfg)

        class _FakeEmbedder:
            dim = DIM

            def _ensure_loaded(self) -> None:
                pass

            def _embed_sync(self, texts: list[str]) -> list[list[float]]:
                return _fake_embed(texts)

            async def embed(self, texts: list[str]) -> list[list[float]]:
                return _fake_embed(texts)

        idx._embedder = _FakeEmbedder()  # type: ignore[assignment]

        try:
            await idx._ensure_ready()
            ok("_ensure_ready() — schema initialised")
        except RuntimeError as e:
            warn("_ensure_ready() hit expected limitation", str(e).splitlines()[0])
            return
        except Exception:
            fail("_ensure_ready()", traceback.format_exc(limit=4))
            return

        try:
            n = await idx._index_text(
                "/fake/memory/auth.md", "memory",
                "User authenticates with Stripe using API keys stored in env vars. "
                "Prefers dark mode. Uses TypeScript for all projects.",
                mtime=1_234_567_890.0,
            )
            ok(f"_index_text() → {n} chunk(s) indexed")
        except Exception:
            fail("_index_text()", traceback.format_exc(limit=4))
            return

        try:
            results_list = await idx.search("Stripe authentication API keys", k=5)
            if results_list:
                top = results_list[0]
                ok(
                    f"search() → {len(results_list)} result(s)",
                    f"score={top['score']:.3f}  src={Path(top['source_path']).name}",
                )
            else:
                fail("search() returned empty results")
        except Exception:
            fail("search()", traceback.format_exc(limit=4))
            return

        try:
            status = await idx.get_status()
            ok(f"get_status() → {status['total_chunks']} chunks, {len(status['sources'])} source(s)")
        except Exception:
            fail("get_status()", traceback.format_exc(limit=3))

        try:
            removed = await idx.remove_source("/fake/memory/auth.md")
            status2 = await idx.get_status()
            ok(f"remove_source() → {removed} deleted, {status2['total_chunks']} remain")
        except Exception:
            fail("remove_source()", traceback.format_exc(limit=3))


# ---------------------------------------------------------------------------
# Test 4: Real memory files
# ---------------------------------------------------------------------------

async def test_real_memory_files() -> None:
    section("4. Real memory files")
    import platform
    import re

    if platform.system() == "Darwin":
        mem_dir = Path.home() / "Library" / "Application Support" / "Otto" / "memory"
    else:
        mem_dir = Path.home() / ".config" / "Otto" / "memory"

    md_files = [f for f in mem_dir.glob("*.md") if f.name != "MEMORY.md"] if mem_dir.exists() else []

    if not md_files:
        skip("provenance fields check", "no topic files yet — run a chat session + consolidation first")
        return

    ok(f"found {len(md_files)} topic file(s) in {mem_dir}")
    provenance_keys = {"source_sessions", "confidence", "created_at", "updated_at"}
    for f in md_files[:5]:
        text = f.read_text(encoding="utf-8")
        found = {k for k in provenance_keys if re.search(rf"^{k}:", text, re.MULTILINE)}
        missing = provenance_keys - found
        if missing:
            skip(f"{f.name}", f"missing provenance fields: {', '.join(sorted(missing))}")
        else:
            ok(f"{f.name}", f"has: {', '.join(sorted(found))}")


# ---------------------------------------------------------------------------
# Test 5: Live API endpoints
# ---------------------------------------------------------------------------

async def test_api_endpoints() -> None:
    section("5. Live API endpoints (backend must be running)")

    import json
    import urllib.request

    # Try 18082 first (clean test instance), fall back to 18081 (app backend)
    for port in (18082, 18081):
        try:
            import urllib.request as _u
            _u.urlopen(f"http://localhost:{port}/api/health", timeout=2).read()
            base = f"http://localhost:{port}"
            break
        except Exception:
            continue
    else:
        skip("API endpoints", "no backend reachable on 18081 or 18082")
        return

    base = base  # already set

    def get(path: str) -> dict:
        try:
            with urllib.request.urlopen(base + path, timeout=5) as resp:
                return json.loads(resp.read())
        except Exception as e:
            raise RuntimeError(str(e)) from e

    for path, label in [
        ("/api/embeddings/status", "/api/embeddings/status"),
        ("/api/memory/topics",     "/api/memory/topics"),
        ("/api/memory/stats",      "/api/memory/stats"),
    ]:
        try:
            data = get(path)
            if path == "/api/embeddings/status":
                ok(label, f"enabled={data.get('enabled')}  chunks={data.get('total_chunks')}")
            elif path == "/api/memory/topics":
                ok(label, f"{len(data.get('topics', []))} topic(s)")
            else:
                ok(label, f"files={data.get('memory_files')}  transcripts={data.get('total_transcripts')}")
        except RuntimeError as e:
            fail(label, str(e))


# ---------------------------------------------------------------------------
# Test 6: mlx-embeddings import (no model load)
# ---------------------------------------------------------------------------

def test_mlx_embeddings_import() -> None:
    section("6. mlx-embeddings (import-only check, no model download)")
    try:
        from mlx_embeddings.utils import load as _load  # type: ignore[import]  # noqa: F401
        ok("from mlx_embeddings.utils import load")
    except ImportError as e:
        fail("mlx-embeddings not installed", str(e))
    except Exception as e:
        fail("unexpected import error", str(e))


# ---------------------------------------------------------------------------
# Standalone mode: pure sqlite-vec test, no backend imports
# ---------------------------------------------------------------------------

async def standalone_test() -> None:
    """Run only sqlite-vec + vector search — works with any Python that has
    ``enable_load_extension``.  No backend imports required."""
    print("\n\033[1mStandalone sqlite-vec + vector search test\033[0m")
    print("=" * 50)

    section("Installing sqlite-vec if needed…")
    import subprocess
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "sqlite-vec", "-q"],
        check=True,
    )

    available, reason = _sqlite_vec_available()
    if not available:
        fail("sqlite-vec extension loading", reason)
        return

    ok("sqlite-vec extension loading", reason)
    await test_embedding_index_pipeline()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    standalone = "--standalone" in sys.argv

    if standalone:
        await standalone_test()
    else:
        print("\n\033[1mOTTO semantic search smoke test\033[0m")
        print("=" * 50)

        test_sqlite_vec_import()
        await test_embedding_index_pipeline()
        await test_embedding_index_class()
        await test_real_memory_files()
        await test_api_endpoints()
        test_mlx_embeddings_import()

    print("\n" + "=" * 50)
    passed = sum(1 for s, _, _ in results if s == PASS)
    failed = sum(1 for s, _, _ in results if s == FAIL)
    skipped = sum(1 for s, _, _ in results if s == SKIP)
    warned = sum(1 for s, _, _ in results if s == WARN)
    print(
        f"\033[1mResults:\033[0m  "
        f"{PASS} {passed} passed  "
        f"{FAIL} {failed} failed  "
        f"{WARN} {warned} warning  "
        f"{SKIP} {skipped} skipped\n"
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
