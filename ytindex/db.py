"""SQLite úložiště: metadata videí, segmenty, chunky, FTS5 fulltext a sqlite-vec vektory."""

from __future__ import annotations

import json
import sqlite3
import struct
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

from .chunking import Chunk, Segment
from .query import build_fts_query

SCHEMA = """
CREATE TABLE IF NOT EXISTS channels (
    id TEXT PRIMARY KEY,
    name TEXT,
    url TEXT,
    input TEXT,
    handle TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    channel_id TEXT NOT NULL REFERENCES channels(id),
    title TEXT NOT NULL,
    description TEXT DEFAULT '',
    published_at TEXT,
    duration_s INTEGER,
    view_count INTEGER,
    thumbnail_url TEXT,
    transcript_status TEXT NOT NULL DEFAULT 'pending',  -- pending | done | missing | failed
    transcript_source TEXT,                             -- captions | ytdlp | gemini
    transcript_lang TEXT,
    transcript_error TEXT,
    position INTEGER,
    updated_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_videos_channel ON videos(channel_id);
CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(transcript_status);

CREATE TABLE IF NOT EXISTS segments (
    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    start REAL NOT NULL,
    duration REAL NOT NULL,
    text TEXT NOT NULL,
    PRIMARY KEY (video_id, idx)
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    start REAL NOT NULL,
    end REAL NOT NULL,
    text TEXT NOT NULL,
    embedded INTEGER NOT NULL DEFAULT 0,
    channel_id TEXT                      -- denormalizováno z videos kvůli rychlému filtru kanálu
);
CREATE INDEX IF NOT EXISTS idx_chunks_video ON chunks(video_id);
CREATE INDEX IF NOT EXISTS idx_chunks_channel ON chunks(channel_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedded ON chunks(embedded);

-- Fulltext: unicode61 + remove_diacritics 2 => 'svoboda' najde i 'SVOBODA', 'prístav' najde 'přístav'
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id',
    tokenize = "unicode61 remove_diacritics 2"
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def handle_from_input(value: str | None) -> str | None:
    """'https://youtube.com/@kanal?si=…' / '@kanal' → '@kanal'; jinak None."""
    import re as _re
    if not value:
        return None
    m = _re.search(r"(?:^|/)(@[\w.\-]+)", value)
    return m.group(1) if m else None


def channel_dict(row) -> dict | None:
    """Jednotná podoba kanálu v API: id, name, url (handle má přednost), handle."""
    if row is None:
        return None
    handle = row["handle"] if "handle" in row.keys() else None
    url = f"https://www.youtube.com/{handle}" if handle else row["url"]
    return {"id": row["id"], "name": row["name"], "url": url, "handle": handle}


def serialize_f32(vec: Iterable[float]) -> bytes:
    v = list(vec)
    return struct.pack(f"{len(v)}f", *v)


class Database:
    def __init__(self, path: str | Path, embedding_dim: int = 768, reset_on_mismatch: bool = False):
        """reset_on_mismatch: True jen při `ytindex embed` – jiná dimenze pak zahodí vektory a přepočítá je.
        Jinak (serve, search) se převezme dimenze uložená v DB, aby se index nikdy tiše nezničil."""
        self.path = str(path)
        self.embedding_dim = embedding_dim
        self.reset_on_mismatch = reset_on_mismatch
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.vec_available = self._load_vec()
        self._init_schema()

    def _load_vec(self) -> bool:
        try:
            import sqlite_vec  # type: ignore
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            return True
        except Exception:
            return False

    def _init_schema(self) -> None:
        self.conn.executescript(SCHEMA)
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(videos)")}
        if "position" not in cols:
            self.conn.execute("ALTER TABLE videos ADD COLUMN position INTEGER")
        ccols = {r[1] for r in self.conn.execute("PRAGMA table_info(channels)")}
        if "handle" not in ccols:
            self.conn.execute("ALTER TABLE channels ADD COLUMN handle TEXT")
        chcols = {r[1] for r in self.conn.execute("PRAGMA table_info(chunks)")}
        if "channel_id" not in chcols:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN channel_id TEXT")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_channel ON chunks(channel_id)")
        if self.conn.execute("SELECT 1 FROM chunks WHERE channel_id IS NULL LIMIT 1").fetchone():
            self.conn.execute("UPDATE chunks SET channel_id = (SELECT channel_id FROM videos WHERE videos.id = chunks.video_id) "
                              "WHERE channel_id IS NULL")
        if self.vec_available:
            stored = self.get_meta("embedding_dim")
            if stored and int(stored) != self.embedding_dim:
                if self.reset_on_mismatch:
                    # jiná dimenze než uložené vektory – zahodit a re-embedovat
                    self.conn.execute("DROP TABLE IF EXISTS chunk_vec")
                    self.conn.execute("UPDATE chunks SET embedded = 0")
                else:
                    self.embedding_dim = int(stored)
            self.conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vec USING vec0("
                f"chunk_id INTEGER PRIMARY KEY, embedding float[{self.embedding_dim}] distance_metric=cosine)"
            )
            self.set_meta("embedding_dim", str(self.embedding_dim))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def tx(self):
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ---------- meta ----------
    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))

    # ---------- channels / videos ----------
    def upsert_channel(self, id: str, name: str, url: str, input: str = "", handle: str | None = None) -> None:
        handle = handle or handle_from_input(input)
        with self.tx() as c:
            c.execute(
                "INSERT INTO channels(id, name, url, input, handle) VALUES (?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET name=excluded.name, url=excluded.url, input=excluded.input, "
                "handle=COALESCE(excluded.handle, channels.handle)",
                (id, name, url, input, handle),
            )

    def get_channel(self, id: str | None = None) -> sqlite3.Row | None:
        if id:
            return self.conn.execute("SELECT * FROM channels WHERE id=?", (id,)).fetchone()
        return self.conn.execute("SELECT * FROM channels ORDER BY created_at LIMIT 1").fetchone()

    def list_channels(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM channels ORDER BY created_at").fetchall()

    def channel_stats(self) -> list[dict]:
        """Souhrn per kanál: počet videí, hotových, hodin přepisu."""
        rows = self.conn.execute(
            "SELECT c.id, c.name, c.url, c.handle, COUNT(v.id) AS videos, "
            "SUM(CASE WHEN v.transcript_status='done' THEN 1 ELSE 0 END) AS done, "
            "COALESCE(SUM(CASE WHEN v.transcript_status='done' THEN v.duration_s ELSE 0 END),0)/3600.0 AS hours "
            "FROM channels c LEFT JOIN videos v ON v.channel_id = c.id GROUP BY c.id ORDER BY c.created_at").fetchall()
        return [{**channel_dict(r), "videos": r["videos"], "done": r["done"], "hours": round(r["hours"], 1)} for r in rows]

    def upsert_video(self, v: dict) -> None:
        """v: id, channel_id, title, description, published_at, duration_s, view_count, thumbnail_url"""
        self.conn.execute(
            """INSERT INTO videos(id, channel_id, title, description, published_at, duration_s, view_count, thumbnail_url, position)
               VALUES (:id, :channel_id, :title, :description, :published_at, :duration_s, :view_count, :thumbnail_url, :position)
               ON CONFLICT(id) DO UPDATE SET
                 title=excluded.title,
                 position=COALESCE(excluded.position, videos.position),
                 description=COALESCE(NULLIF(excluded.description,''), videos.description),
                 published_at=COALESCE(excluded.published_at, videos.published_at),
                 duration_s=COALESCE(excluded.duration_s, videos.duration_s),
                 view_count=COALESCE(excluded.view_count, videos.view_count),
                 thumbnail_url=COALESCE(excluded.thumbnail_url, videos.thumbnail_url),
                 updated_at=datetime('now')""",
            {"description": "", "published_at": None, "duration_s": None, "view_count": None,
             "thumbnail_url": None, "position": None, **v},
        )

    def get_video(self, video_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()

    def list_videos(self, channel_id: str | None = None, status: str | None = None,
                    limit: int = 50, offset: int = 0, order: str = "published_at DESC") -> list[sqlite3.Row]:
        where, params = [], []
        if channel_id:
            where.append("channel_id=?"); params.append(channel_id)
        if status:
            where.append("transcript_status=?"); params.append(status)
        allowed = {"published_at DESC", "published_at ASC", "title ASC", "duration_s DESC", "view_count DESC", "position ASC"}
        order = order if order in allowed else "published_at DESC"
        if order == "published_at DESC":
            order = "published_at DESC, position ASC"
        sql = "SELECT * FROM videos" + (" WHERE " + " AND ".join(where) if where else "")
        sql += f" ORDER BY {order} NULLS LAST LIMIT ? OFFSET ?"
        return self.conn.execute(sql, (*params, limit, offset)).fetchall()

    def count_videos(self, channel_id: str | None = None, status: str | None = None) -> int:
        where, params = [], []
        if channel_id:
            where.append("channel_id=?"); params.append(channel_id)
        if status:
            where.append("transcript_status=?"); params.append(status)
        sql = "SELECT COUNT(*) FROM videos" + (" WHERE " + " AND ".join(where) if where else "")
        return self.conn.execute(sql, params).fetchone()[0]

    def set_transcript_status(self, video_id: str, status: str, source: str | None = None,
                              lang: str | None = None, error: str | None = None) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE videos SET transcript_status=?, transcript_source=?, transcript_lang=?, "
                "transcript_error=?, updated_at=datetime('now') WHERE id=?",
                (status, source, lang, (error if (error or "").startswith("gaps:") else (error or "")[:500]) or None, video_id),
            )

    # ---------- transcripts ----------
    def store_transcript(self, video_id: str, segments: list[Segment], chunks: list[Chunk],
                         source: str, lang: str | None) -> None:
        with self.tx() as c:
            if self.vec_available:   # vektory mazat PŘED chunky (poddotaz by jinak nic nenašel → sirotci)
                c.execute("DELETE FROM chunk_vec WHERE chunk_id IN "
                          "(SELECT id FROM chunks WHERE video_id=?)", (video_id,))
            c.execute("DELETE FROM segments WHERE video_id=?", (video_id,))
            c.execute("DELETE FROM chunks WHERE video_id=?", (video_id,))
            c.executemany(
                "INSERT INTO segments(video_id, idx, start, duration, text) VALUES (?,?,?,?,?)",
                [(video_id, i, s.start, s.duration, s.text) for i, s in enumerate(segments)],
            )
            row = c.execute("SELECT channel_id FROM videos WHERE id=?", (video_id,)).fetchone()
            channel_id = row["channel_id"] if row else None
            c.executemany(
                "INSERT INTO chunks(video_id, idx, start, end, text, channel_id) VALUES (?,?,?,?,?,?)",
                [(video_id, i, ch.start, ch.end, ch.text, channel_id) for i, ch in enumerate(chunks)],
            )
            c.execute(
                "UPDATE videos SET transcript_status='done', transcript_source=?, transcript_lang=?, "
                "transcript_error=NULL, updated_at=datetime('now') WHERE id=?",
                (source, lang, video_id),
            )

    def get_segments(self, video_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT idx, start, duration, text FROM segments WHERE video_id=? ORDER BY idx", (video_id,)
        ).fetchall()

    def get_segments_range(self, video_id: str, start: float, end: float) -> list[sqlite3.Row]:
        """Segmenty, které zasahují do intervalu <start, end> (pro rozšíření kontextu kolem shody)."""
        return self.conn.execute(
            "SELECT idx, start, duration, text FROM segments WHERE video_id=? AND start + duration >= ? AND start <= ? "
            "ORDER BY idx", (video_id, start, end),
        ).fetchall()

    # ---------- embeddings ----------
    def pending_embeddings(self, limit: int = 100) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, video_id, text FROM chunks WHERE embedded=0 ORDER BY id LIMIT ?", (limit,)
        ).fetchall()

    def store_embeddings(self, items: list[tuple[int, list[float]]]) -> None:
        if not self.vec_available:
            raise RuntimeError("sqlite-vec není k dispozici")
        with self.tx() as c:
            for chunk_id, vec in items:
                c.execute("DELETE FROM chunk_vec WHERE chunk_id=?", (chunk_id,))
                c.execute("INSERT INTO chunk_vec(chunk_id, embedding) VALUES (?, ?)",
                          (chunk_id, serialize_f32(vec)))
                c.execute("UPDATE chunks SET embedded=1 WHERE id=?", (chunk_id,))

    def count_embedded(self) -> tuple[int, int]:
        row = self.conn.execute("SELECT SUM(embedded), COUNT(*) FROM chunks").fetchone()
        return int(row[0] or 0), int(row[1] or 0)

    def delete_orphan_vectors(self) -> int:
        """Smaže vektory bez chunku (po přepsání videa). Vrací počet smazaných."""
        if not self.vec_available:
            return 0
        ids = [r[0] for r in self.conn.execute(
            "SELECT v.chunk_id FROM chunk_vec v LEFT JOIN chunks c ON c.id = v.chunk_id WHERE c.id IS NULL")]
        with self.tx() as c:
            for cid in ids:
                c.execute("DELETE FROM chunk_vec WHERE chunk_id=?", (cid,))
        return len(ids)

    def chunk_ids(self, video_id: str) -> dict[int, int]:
        """{idx: chunk_id} pro video."""
        return {r["idx"]: r["id"] for r in self.conn.execute("SELECT id, idx FROM chunks WHERE video_id=?", (video_id,))}

    def get_embeddings(self, video_id: str) -> dict[int, list[float]]:
        """{idx: vektor} – pro přenos indexu do jiné databáze (ytindex copy)."""
        if not self.vec_available:
            return {}
        rows = self.conn.execute(
            "SELECT c.idx, v.embedding FROM chunk_vec v JOIN chunks c ON c.id = v.chunk_id WHERE c.video_id=?",
            (video_id,)).fetchall()
        return {r["idx"]: list(struct.unpack(f"{len(r['embedding']) // 4}f", r["embedding"])) for r in rows}

    # ---------- hromadný přístup (export, /api/chunks, /api/changes) ----------
    def iter_embeddings(self):
        """Generátor (chunk_id, vektor) pro všechny embedované chunky, v pořadí chunk_id."""
        if not self.vec_available:
            return iter(())
        return ((r["chunk_id"], list(struct.unpack(f"{len(r['embedding']) // 4}f", r["embedding"])))
                for r in self.conn.execute("SELECT chunk_id, embedding FROM chunk_vec ORDER BY chunk_id"))

    def list_chunks(self, video_id: str | None = None, after: int = 0, limit: int = 500,
                    with_embeddings: bool = False, channel_id: str | None = None) -> list[dict]:
        where, params = ["c.id > ?"], [after]
        if video_id:
            where.append("c.video_id = ?"); params.append(video_id)
        if channel_id:
            where.append("vid.channel_id = ?"); params.append(channel_id)
        sql = ('SELECT c.id, c.video_id, c.idx, c.start, c.end, c.text, c.embedded, vid.channel_id'
               + (', v.embedding' if with_embeddings and self.vec_available else ', NULL AS embedding')
               + ' FROM chunks c JOIN videos vid ON vid.id = c.video_id'
               + (' LEFT JOIN chunk_vec v ON v.chunk_id = c.id' if with_embeddings and self.vec_available else '')
               + ' WHERE ' + ' AND '.join(where) + ' ORDER BY c.id LIMIT ?')
        out = []
        for r in self.conn.execute(sql, (*params, limit)):
            d = {"chunk_id": r["id"], "video_id": r["video_id"], "channel_id": r["channel_id"], "idx": r["idx"],
                 "start": r["start"], "end": r["end"], "text": r["text"], "embedded": bool(r["embedded"])}
            if with_embeddings:
                d["embedding"] = list(struct.unpack(f"{len(r['embedding']) // 4}f", r["embedding"])) if r["embedding"] else None
            out.append(d)
        return out

    def videos_changed_since(self, since: str, limit: int = 1000, channel_id: str | None = None) -> list[sqlite3.Row]:
        """Videa s updated_at > since (ISO/`YYYY-MM-DD HH:MM:SS`), nejstarší změna první."""
        if channel_id:
            return self.conn.execute(
                "SELECT * FROM videos WHERE updated_at > ? AND channel_id = ? ORDER BY updated_at, id LIMIT ?",
                (since, channel_id, limit)).fetchall()
        return self.conn.execute(
            "SELECT * FROM videos WHERE updated_at > ? ORDER BY updated_at, id LIMIT ?", (since, limit)).fetchall()

    # ---------- vyhledávací primitiva (stejné rozhraní má PostgresDatabase) ----------
    backend = "sqlite"

    def _only_channel(self) -> str | None:
        """ID kanálu, pokud je v indexu právě jeden (filtr je pak zbytečný)."""
        rows = self.conn.execute("SELECT id FROM channels LIMIT 2").fetchall()
        return rows[0]["id"] if len(rows) == 1 else None

    def _filter_channel(self, ids: list[int], channel_id: str) -> set[int]:
        if not ids:
            return set()
        out: set[int] = set()
        for i in range(0, len(ids), 500):
            part = ids[i:i + 500]
            out.update(r[0] for r in self.conn.execute(
                f"SELECT id FROM chunks WHERE channel_id = ? AND id IN ({','.join('?' * len(part))})", (channel_id, *part)))
        return out

    def fts_hits(self, q: str, mode: str, limit: int, channel_id: str | None = None) -> list[tuple[int, float]]:
        """[(chunk_id, rank)] seřazené od nejlepšího. mode: and | or. channel_id omezí na jeden kanál.

        Filtr kanálu se dělá až nad kandidáty (FTS5 s `rowid IN (subdotaz)` trvá minuty); při nedostatku
        výsledků se kandidáti rozšíří.
        """
        fts_query = build_fts_query(q, mode)
        if not fts_query:
            return []
        if channel_id and channel_id == self._only_channel():
            channel_id = None
        for factor in ((1,) if not channel_id else (4, 25)):
            try:
                rows = self.conn.execute(
                    "SELECT rowid, bm25(chunks_fts) AS r FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY r LIMIT ?",
                    (fts_query, limit * factor),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
            if not channel_id:
                return [(r["rowid"], r["r"]) for r in rows]
            keep = self._filter_channel([r["rowid"] for r in rows], channel_id)
            hits = [(r["rowid"], r["r"]) for r in rows if r["rowid"] in keep]
            if len(hits) >= limit or len(rows) < limit * factor:
                return hits[:limit]
        return hits[:limit]

    def vec_hits(self, qvec: list[float], limit: int, channel_id: str | None = None) -> list[tuple[int, float]]:
        """[(chunk_id, cosine_distance)] seřazené od nejbližšího; filtr kanálu post-filtrem jako u FTS."""
        if not self.vec_available:
            return []
        if channel_id and channel_id == self._only_channel():
            channel_id = None
        for factor in ((1,) if not channel_id else (4, 25)):
            rows = self.conn.execute(
                "SELECT chunk_id, distance FROM chunk_vec WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (serialize_f32(qvec), limit * factor),
            ).fetchall()
            if not channel_id:
                return [(r["chunk_id"], r["distance"]) for r in rows]
            keep = self._filter_channel([r["chunk_id"] for r in rows], channel_id)
            hits = [(r["chunk_id"], r["distance"]) for r in rows if r["chunk_id"] in keep]
            if len(hits) >= limit or len(rows) < limit * factor:
                return hits[:limit]
        return hits[:limit]

    def snippet(self, chunk_id: int, q: str, mode: str) -> str | None:
        """Úryvek se <mark> značkami, nebo None pokud chunk dotazu lexikálně neodpovídá."""
        fts_query = build_fts_query(q, mode)
        if not fts_query:
            return None
        try:
            row = self.conn.execute(
                "SELECT snippet(chunks_fts, 0, '<mark>', '</mark>', '…', 40) AS s FROM chunks_fts "
                "WHERE rowid = ? AND chunks_fts MATCH ?", (chunk_id, fts_query),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        return row["s"] if row and row["s"] else None

    def fetch_chunks(self, ids: list[int]) -> list[dict]:
        if not ids:
            return []
        rows = self.conn.execute(
            f"SELECT c.id, c.video_id, c.start, c.end, c.text, v.title, v.published_at, v.duration_s, v.thumbnail_url, "
            f"v.channel_id, ch.name AS channel_name, ch.url AS channel_url, ch.handle AS channel_handle "
            f"FROM chunks c JOIN videos v ON v.id = c.video_id LEFT JOIN channels ch ON ch.id = v.channel_id "
            f"WHERE c.id IN ({','.join('?' * len(ids))})", ids,
        ).fetchall()
        return [dict(r) for r in rows]

    # ---------- stats ----------
    def stats(self) -> dict:
        st = {r["transcript_status"]: r["c"] for r in self.conn.execute(
            "SELECT transcript_status, COUNT(*) c FROM videos GROUP BY transcript_status")}
        embedded, chunks = self.count_embedded()
        hours = self.conn.execute("SELECT COALESCE(SUM(duration_s),0)/3600.0 FROM videos WHERE transcript_status='done'").fetchone()[0]
        return {
            "backend": self.backend,
            "videos": self.count_videos(),
            "videos_by_status": st,
            "chunks": chunks,
            "chunks_embedded": embedded,
            "total_hours": round(hours, 1),   # hodiny zaindexovaných videí
            "vector_search": self.vec_available and embedded > 0,
        }


class ReloadableDatabase:
    """Obal nad Database, který umí za běhu vyměnit podkladovou DB (nový index stažený z GCS).

    Všechna volání se delegují na aktuální instanci, takže API/MCP/search nic nepoznají.
    """

    def __init__(self, db):
        self._db = db

    def __getattr__(self, name):
        return getattr(self._db, name)

    def swap(self, new_db) -> None:
        old, self._db = self._db, new_db
        try:
            old.close()
        except Exception:
            pass


def open_database(cfg, url: str | None = None, reset_on_mismatch: bool = False):
    """Factory: DATABASE_URL (postgresql://…) → Postgres + pgvector, jinak lokální SQLite."""
    import os
    url = url or os.environ.get("DATABASE_URL", "")
    if url.startswith(("postgres://", "postgresql://")):
        from .db_postgres import PostgresDatabase
        return PostgresDatabase(url, cfg.gemini.embedding_dim, reset_on_mismatch)
    return Database(url or cfg.db_path, cfg.gemini.embedding_dim, reset_on_mismatch)
