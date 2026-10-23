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
    embedded INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chunks_video ON chunks(video_id);
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


def serialize_f32(vec: Iterable[float]) -> bytes:
    v = list(vec)
    return struct.pack(f"{len(v)}f", *v)


class Database:
    def __init__(self, path: str | Path, embedding_dim: int = 768):
        self.path = str(path)
        self.embedding_dim = embedding_dim
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
        if self.vec_available:
            stored = self.get_meta("embedding_dim")
            if stored and int(stored) != self.embedding_dim:
                # jiná dimenze než uložené vektory – nutno re-embedovat
                self.conn.execute("DROP TABLE IF EXISTS chunk_vec")
                self.conn.execute("UPDATE chunks SET embedded = 0")
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
    def upsert_channel(self, id: str, name: str, url: str, input: str = "") -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO channels(id, name, url, input) VALUES (?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET name=excluded.name, url=excluded.url, input=excluded.input",
                (id, name, url, input),
            )

    def get_channel(self, id: str | None = None) -> sqlite3.Row | None:
        if id:
            return self.conn.execute("SELECT * FROM channels WHERE id=?", (id,)).fetchone()
        return self.conn.execute("SELECT * FROM channels ORDER BY created_at LIMIT 1").fetchone()

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
                (status, source, lang, (error or "")[:500] or None, video_id),
            )

    # ---------- transcripts ----------
    def store_transcript(self, video_id: str, segments: list[Segment], chunks: list[Chunk],
                         source: str, lang: str | None) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM segments WHERE video_id=?", (video_id,))
            c.execute("DELETE FROM chunks WHERE video_id=?", (video_id,))
            if self.vec_available:
                c.execute("DELETE FROM chunk_vec WHERE chunk_id IN "
                          "(SELECT id FROM chunks WHERE video_id=?)", (video_id,))
            c.executemany(
                "INSERT INTO segments(video_id, idx, start, duration, text) VALUES (?,?,?,?,?)",
                [(video_id, i, s.start, s.duration, s.text) for i, s in enumerate(segments)],
            )
            c.executemany(
                "INSERT INTO chunks(video_id, idx, start, end, text) VALUES (?,?,?,?,?)",
                [(video_id, i, ch.start, ch.end, ch.text) for i, ch in enumerate(chunks)],
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

    # ---------- vyhledávací primitiva (stejné rozhraní má PostgresDatabase) ----------
    backend = "sqlite"

    def fts_hits(self, q: str, mode: str, limit: int) -> list[tuple[int, float]]:
        """[(chunk_id, rank)] seřazené od nejlepšího. mode: and | or."""
        fts_query = build_fts_query(q, mode)
        if not fts_query:
            return []
        try:
            rows = self.conn.execute(
                "SELECT rowid, bm25(chunks_fts) AS r FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY r LIMIT ?",
                (fts_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(r["rowid"], r["r"]) for r in rows]

    def vec_hits(self, qvec: list[float], limit: int) -> list[tuple[int, float]]:
        """[(chunk_id, cosine_distance)] seřazené od nejbližšího."""
        if not self.vec_available:
            return []
        rows = self.conn.execute(
            "SELECT chunk_id, distance FROM chunk_vec WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (serialize_f32(qvec), limit),
        ).fetchall()
        return [(r["chunk_id"], r["distance"]) for r in rows]

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
            f"SELECT c.id, c.video_id, c.start, c.end, c.text, v.title, v.published_at, v.duration_s, v.thumbnail_url "
            f"FROM chunks c JOIN videos v ON v.id = c.video_id WHERE c.id IN ({','.join('?' * len(ids))})", ids,
        ).fetchall()
        return [dict(r) for r in rows]

    # ---------- stats ----------
    def stats(self) -> dict:
        st = {r["transcript_status"]: r["c"] for r in self.conn.execute(
            "SELECT transcript_status, COUNT(*) c FROM videos GROUP BY transcript_status")}
        embedded, chunks = self.count_embedded()
        hours = self.conn.execute("SELECT COALESCE(SUM(duration_s),0)/3600.0 FROM videos").fetchone()[0]
        return {
            "backend": self.backend,
            "videos": self.count_videos(),
            "videos_by_status": st,
            "chunks": chunks,
            "chunks_embedded": embedded,
            "total_hours": round(hours, 1),
            "vector_search": self.vec_available and embedded > 0,
        }


def open_database(cfg, url: str | None = None):
    """Factory: DATABASE_URL (postgresql://…) → Cloud SQL / Postgres + pgvector, jinak lokální SQLite."""
    import os
    url = url or os.environ.get("DATABASE_URL", "")
    if url.startswith(("postgres://", "postgresql://")):
        from .db_postgres import PostgresDatabase
        return PostgresDatabase(url, cfg.gemini.embedding_dim)
    return Database(url or cfg.db_path, cfg.gemini.embedding_dim)
