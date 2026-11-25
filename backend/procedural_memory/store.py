"""SQLite store for procedural memory.

One file holds a store. The app's lives at ``<app data>/procedures.db``; a benchmark run reads a frozen snapshot
(opened read-only) and writes its new episodes to a separate file. Records are the source of truth; vectors and the
FTS index are rebuildable from them.

Tables:
  episodes   one per delegated agent run (pending | admitted | retired | excluded), with its compacted data
  items      what retrieval returns: recipe (whole run or segment), fact, lesson; one vector each
  artifacts  scripts from the run, with whether they write or send
  labels     label history per episode
  usage      each time an item was returned, and how that run ended
  vectors    item embeddings (float32 blobs; brute-force cosine: stores hold thousands of items, not millions)
  fts_items  BM25 over intents and text (when SQLite has FTS5)
  meta       snapshot id, embedder name, fold, build info
"""

from __future__ import annotations

import json
import math
import sqlite3
import struct
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    subagent_tag TEXT NOT NULL,
    agent TEXT NOT NULL,
    family TEXT NOT NULL,
    track TEXT NOT NULL DEFAULT 'app',
    task_id TEXT,
    task_text TEXT NOT NULL DEFAULT '',
    intent TEXT NOT NULL DEFAULT '',
    apps TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'pending',
    label TEXT NOT NULL DEFAULT 'unknown',
    label_source TEXT,
    confidence REAL,
    self_report TEXT,
    model TEXT,
    otto_version TEXT,
    data TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(session_id, subagent_tag)
);
CREATE INDEX IF NOT EXISTS idx_ep_status ON episodes(status);
CREATE INDEX IF NOT EXISTS idx_ep_task ON episodes(task_id);
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    ord INTEGER NOT NULL DEFAULT 0,
    app TEXT,
    family TEXT NOT NULL,
    intent TEXT NOT NULL,
    body TEXT NOT NULL,
    label_source TEXT,
    from_outcome TEXT,
    token_count INTEGER NOT NULL DEFAULT 0,
    active INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_items_ep ON items(episode_id);
CREATE INDEX IF NOT EXISTS idx_items_app ON items(app);
CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    item_id TEXT,
    kind TEXT NOT NULL,
    body TEXT NOT NULL,
    writes_or_sends INTEGER NOT NULL DEFAULT 0,
    is_candidate INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS labels (
    episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence REAL,
    note TEXT,
    at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS usage (
    item_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    at REAL NOT NULL,
    score REAL,
    outcome TEXT
);
CREATE INDEX IF NOT EXISTS idx_usage_session ON usage(session_id);
CREATE INDEX IF NOT EXISTS idx_usage_item ON usage(item_id);
CREATE TABLE IF NOT EXISTS vectors (
    item_id TEXT PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL
);
"""

FTS_SCHEMA = "CREATE VIRTUAL TABLE IF NOT EXISTS fts_items USING fts5(item_id UNINDEXED, intent, text)"

STATUSES = ("pending", "admitted", "retired", "excluded")


def _now() -> float:
    return time.time()


def pack(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def unpack(blob: bytes, dim: int) -> list[float]:
    return list(struct.unpack(f"<{dim}f", blob))


def tokens(text: str) -> int:
    return max(1, math.ceil(len(text or "") / 4))


@dataclass
class ItemRow:
    id: str
    episode_id: str
    kind: str
    ord: int
    app: str | None
    family: str
    intent: str
    body: dict[str, Any]
    label_source: str | None
    from_outcome: str | None
    token_count: int
    task_id: str | None
    task_text: str
    created_at: float


class ProcedureStore:
    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        self._fts: bool | None = None
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.connect() as db:
                db.executescript(SCHEMA)
                try:
                    db.execute(FTS_SCHEMA)
                    self._fts = True
                except sqlite3.OperationalError:
                    self._fts = False

    # ── connections ─────────────────────────────────────────────────────────────────────────────────────

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        if self.read_only:
            db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        else:
            db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=10)
            db.execute("PRAGMA journal_mode=WAL")
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            if not self.read_only:
                db.commit()
        finally:
            db.close()

    @property
    def has_fts(self) -> bool:
        if self._fts is None:
            with self.connect() as db:
                row = db.execute("SELECT name FROM sqlite_master WHERE name='fts_items'").fetchone()
                self._fts = row is not None
        return bool(self._fts)

    # ── meta ────────────────────────────────────────────────────────────────────────────────────────────

    def get_meta(self) -> dict[str, str]:
        with self.connect() as db:
            return {r["key"]: r["value"] for r in db.execute("SELECT key, value FROM meta")}

    def set_meta(self, **values: Any) -> None:
        with self.connect() as db:
            for k, v in values.items():
                db.execute("INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                           (k, v if isinstance(v, str) else json.dumps(v)))

    # ── episodes ────────────────────────────────────────────────────────────────────────────────────────

    def upsert_episode(self, *, session_id: str, subagent_tag: str, agent: str, family: str, track: str,
                       task_id: str | None, task_text: str, apps: list[str], self_report: str,
                       data: dict[str, Any], model: str | None = None, otto_version: str | None = None) -> tuple[str, bool]:
        """Save or refresh a run's record. Returns (episode id, created). An admitted or excluded episode keeps its
        status; a pending one is replaced with the newer data (a chat that continued)."""
        now = _now()
        with self.connect() as db:
            row = db.execute("SELECT id, status FROM episodes WHERE session_id=? AND subagent_tag=?",
                             (session_id, subagent_tag)).fetchone()
            if row is not None:
                if row["status"] == "pending":
                    db.execute("UPDATE episodes SET task_text=?, apps=?, self_report=?, data=?, model=?, "
                               "updated_at=? WHERE id=?",
                               (task_text, json.dumps(apps), self_report, json.dumps(data), model, now, row["id"]))
                return row["id"], False
            eid = "ep_" + uuid.uuid4().hex[:12]
            db.execute("INSERT INTO episodes(id, session_id, subagent_tag, agent, family, track, task_id, task_text, "
                       "apps, status, self_report, model, otto_version, data, created_at, updated_at) "
                       "VALUES(?,?,?,?,?,?,?,?,?,'pending',?,?,?,?,?,?)",
                       (eid, session_id, subagent_tag, agent, family, track, task_id, task_text, json.dumps(apps),
                        self_report, model, otto_version, json.dumps(data), now, now))
            return eid, True

    def episode(self, episode_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM episodes WHERE id=?", (episode_id,)).fetchone()
            return _ep_dict(row) if row else None

    def episodes_for_session(self, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            return [_ep_dict(r) for r in db.execute("SELECT * FROM episodes WHERE session_id=? ORDER BY created_at",
                                                    (session_id,))]

    def list_episodes(self, *, status: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM episodes", []
        if status:
            q += " WHERE status=?"
            args.append(status)
        q += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self.connect() as db:
            return [_ep_dict(r) for r in db.execute(q, args)]

    def add_label(self, episode_id: str, source: str, value: str, confidence: float | None = None,
                  note: str | None = None) -> None:
        with self.connect() as db:
            db.execute("INSERT INTO labels(episode_id, source, value, confidence, note, at) VALUES(?,?,?,?,?,?)",
                       (episode_id, source, value, confidence, note, _now()))

    def labels(self, episode_id: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM labels WHERE episode_id=? ORDER BY at", (episode_id,))]

    def set_status(self, episode_id: str, status: str, *, label: str | None = None, label_source: str | None = None,
                   confidence: float | None = None, intent: str | None = None) -> None:
        assert status in STATUSES
        sets, args = ["status=?", "updated_at=?"], [status, _now()]
        for col, val in (("label", label), ("label_source", label_source), ("confidence", confidence),
                         ("intent", intent)):
            if val is not None:
                sets.append(f"{col}=?")
                args.append(val)
        args.append(episode_id)
        with self.connect() as db:
            db.execute(f"UPDATE episodes SET {', '.join(sets)} WHERE id=?", args)

    def delete_episode(self, episode_id: str) -> None:
        with self.connect() as db:
            self._drop_items(db, episode_id)
            db.execute("DELETE FROM episodes WHERE id=?", (episode_id,))

    def prune(self, max_runs: int, *, keep: tuple[str, ...] | list[str] = ()) -> list[str]:
        """Delete the oldest runs beyond ``max_runs`` (0 = no limit) and return their ids. Runs that teach the agent
        nothing go first (retired, excluded), then ones awaiting review, then admitted ones; oldest first within
        each. Ids in ``keep`` (the runs just saved) are never deleted."""
        if max_runs <= 0:
            return []
        with self.connect() as db:
            total = db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            extra = total - max_runs
            if extra <= 0:
                return []
            keep = tuple(keep)
            marks = ",".join("?" * len(keep))
            where = f"WHERE id NOT IN ({marks})" if keep else ""
            rows = db.execute(
                f"SELECT id FROM episodes {where} ORDER BY CASE status WHEN 'retired' THEN 0 WHEN 'excluded' THEN 1 "
                f"WHEN 'pending' THEN 2 ELSE 3 END, updated_at, created_at, rowid LIMIT ?", (*keep, extra)).fetchall()
            ids = [r["id"] for r in rows]
            for eid in ids:
                self._drop_items(db, eid)
                db.execute("DELETE FROM episodes WHERE id=?", (eid,))
            return ids

    # ── items ───────────────────────────────────────────────────────────────────────────────────────────

    def _drop_items(self, db: sqlite3.Connection, episode_id: str) -> None:
        ids = [r[0] for r in db.execute("SELECT id FROM items WHERE episode_id=?", (episode_id,))]
        if ids and self.has_fts:
            db.executemany("DELETE FROM fts_items WHERE item_id=?", [(i,) for i in ids])
        db.execute("DELETE FROM vectors WHERE item_id IN (SELECT id FROM items WHERE episode_id=?)", (episode_id,))
        db.execute("DELETE FROM artifacts WHERE episode_id=?", (episode_id,))
        db.execute("DELETE FROM items WHERE episode_id=?", (episode_id,))

    def replace_items(self, episode_id: str, items: list[dict[str, Any]], vectors: list[list[float]],
                      artifacts: list[dict[str, Any]]) -> None:
        """Write an admitted episode's retrievable pieces (replacing any earlier ones)."""
        assert len(items) == len(vectors)
        now = _now()
        with self.connect() as db:
            self._drop_items(db, episode_id)
            for it, vec in zip(items, vectors):
                body = json.dumps(it["body"])
                db.execute("INSERT INTO items(id, episode_id, kind, ord, app, family, intent, body, label_source, "
                           "from_outcome, token_count, active, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,1,?)",
                           (it["id"], episode_id, it["kind"], it.get("ord", 0), it.get("app"), it["family"],
                            it["intent"], body, it.get("label_source"), it.get("from_outcome"),
                            tokens(it["intent"] + body), now))
                db.execute("INSERT INTO vectors(item_id, dim, vec) VALUES(?,?,?)", (it["id"], len(vec), pack(vec)))
                if self.has_fts:
                    db.execute("INSERT INTO fts_items(item_id, intent, text) VALUES(?,?,?)",
                               (it["id"], it["intent"], it.get("search_text", "")))
            for a in artifacts:
                db.execute("INSERT INTO artifacts(id, episode_id, item_id, kind, body, writes_or_sends, is_candidate) "
                           "VALUES(?,?,?,?,?,?,0)",
                           (a["id"], episode_id, a.get("item_id"), a["kind"], a["body"], int(a["writes_or_sends"])))

    def drop_items(self, episode_id: str) -> None:
        with self.connect() as db:
            self._drop_items(db, episode_id)

    def candidate_items(self, *, family: str, apps: list[str] | None, kinds: tuple[str, ...],
                        exclude_task_ids: set[str] = frozenset(), exclude_episode_ids: set[str] = frozenset()
                        ) -> list[tuple[ItemRow, list[float]]]:
        """Active items of admitted episodes that pass the hard filters, with their vectors."""
        q = ("SELECT i.*, e.task_id, e.task_text, v.dim, v.vec FROM items i JOIN episodes e ON e.id=i.episode_id "
             "JOIN vectors v ON v.item_id=i.id WHERE e.status='admitted' AND i.active=1 AND i.family=? "
             f"AND i.kind IN ({','.join('?' * len(kinds))})")
        args: list[Any] = [family, *kinds]
        if apps:
            low = [a.lower() for a in apps]
            q += f" AND lower(i.app) IN ({','.join('?' * len(low))})"
            args.extend(low)
        out = []
        with self.connect() as db:
            for r in db.execute(q, args):
                if r["task_id"] and r["task_id"] in exclude_task_ids:
                    continue
                if r["episode_id"] in exclude_episode_ids:
                    continue
                out.append((_item(r), unpack(r["vec"], r["dim"])))
        return out

    def bm25(self, query: str, limit: int = 40) -> dict[str, float]:
        """item id -> BM25 relevance (higher is better), or {} without FTS5."""
        if not self.has_fts or not query.strip():
            return {}
        terms = [t for t in "".join(c if c.isalnum() else " " for c in query.lower()).split() if len(t) > 2]
        if not terms:
            return {}
        match = " OR ".join(f'"{t}"' for t in terms[:16])
        with self.connect() as db:
            try:
                rows = db.execute("SELECT item_id, bm25(fts_items) AS s FROM fts_items WHERE fts_items MATCH ? "
                                  "ORDER BY s LIMIT ?", (match, limit)).fetchall()
            except sqlite3.OperationalError:
                return {}
        return {r["item_id"]: -float(r["s"]) for r in rows}

    def item(self, item_id: str) -> ItemRow | None:
        with self.connect() as db:
            r = db.execute("SELECT i.*, e.task_id, e.task_text FROM items i JOIN episodes e ON e.id=i.episode_id "
                           "WHERE i.id=?", (item_id,)).fetchone()
            return _item(r) if r else None

    def items_for_episode(self, episode_id: str) -> list[ItemRow]:
        with self.connect() as db:
            return [_item(r) for r in db.execute(
                "SELECT i.*, e.task_id, e.task_text FROM items i JOIN episodes e ON e.id=i.episode_id "
                "WHERE i.episode_id=? ORDER BY i.kind, i.ord", (episode_id,))]

    def artifacts_for(self, item_id: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM artifacts WHERE item_id=?", (item_id,))]

    def set_item_active(self, item_id: str, active: bool) -> None:
        with self.connect() as db:
            db.execute("UPDATE items SET active=? WHERE id=?", (int(active), item_id))

    # ── usage ───────────────────────────────────────────────────────────────────────────────────────────

    def record_usage(self, item_ids: list[str], session_id: str) -> None:
        now = _now()
        with self.connect() as db:
            db.executemany("INSERT INTO usage(item_id, session_id, at) VALUES(?,?,?)",
                           [(i, session_id, now) for i in item_ids])

    def record_outcome(self, session_id: str, *, score: float | None, outcome: str) -> int:
        with self.connect() as db:
            cur = db.execute("UPDATE usage SET score=?, outcome=? WHERE session_id=? AND outcome IS NULL",
                             (score, outcome, session_id))
            return cur.rowcount

    def usage_stats(self) -> dict[str, dict[str, float]]:
        """item id -> {"uses", "known", "passes"}."""
        with self.connect() as db:
            rows = db.execute("SELECT item_id, COUNT(*) AS uses, SUM(outcome IS NOT NULL) AS known, "
                              "SUM(outcome='success') AS passes FROM usage GROUP BY item_id").fetchall()
        return {r["item_id"]: {"uses": r["uses"], "known": r["known"] or 0, "passes": r["passes"] or 0} for r in rows}

    def usage_for_session(self, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM usage WHERE session_id=? ORDER BY at", (session_id,))]

    def apps(self, family: str) -> list[str]:
        """Apps (or sites) that have admitted, active items for an agent family."""
        with self.connect() as db:
            return [r[0] for r in db.execute(
                "SELECT DISTINCT i.app FROM items i JOIN episodes e ON e.id=i.episode_id "
                "WHERE e.status='admitted' AND i.active=1 AND i.family=? AND i.app IS NOT NULL", (family,))]

    def counts(self) -> dict[str, int]:
        with self.connect() as db:
            out = {f"episodes_{r['status']}": r["n"] for r in
                   db.execute("SELECT status, COUNT(*) AS n FROM episodes GROUP BY status")}
            for r in db.execute("SELECT kind, COUNT(*) AS n FROM items WHERE active=1 GROUP BY kind"):
                out[f"items_{r['kind']}"] = r["n"]
        return out


def _ep_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["apps"] = json.loads(d.get("apps") or "[]")
    d["data"] = json.loads(d.get("data") or "{}")
    return d


def _item(r: sqlite3.Row) -> ItemRow:
    return ItemRow(id=r["id"], episode_id=r["episode_id"], kind=r["kind"], ord=r["ord"], app=r["app"],
                   family=r["family"], intent=r["intent"], body=json.loads(r["body"]),
                   label_source=r["label_source"], from_outcome=r["from_outcome"], token_count=r["token_count"],
                   task_id=r["task_id"], task_text=r["task_text"] or "", created_at=r["created_at"])
