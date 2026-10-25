"""PostgreSQL backend (Cloud SQL / AlloyDB / libovolný Postgres ≥ 15) s rozšířeními pgvector + unaccent.

Stejné rozhraní jako ytindex.db.Database. Fulltext = tsvector s konfigurací `cs_simple`
(simple + unaccent → bez ohledu na diakritiku), vektory = pgvector HNSW (cosine).
"""

from __future__ import annotations

from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from .chunking import Chunk, Segment
from .query import build_tsquery

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_ts_config WHERE cfgname = 'cs_simple') THEN
    CREATE TEXT SEARCH CONFIGURATION cs_simple (COPY = simple);
    ALTER TEXT SEARCH CONFIGURATION cs_simple
      ALTER MAPPING FOR hword, hword_part, word, asciiword, asciihword, hword_asciipart
      WITH unaccent, simple;
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS channels (
    id TEXT PRIMARY KEY,
    name TEXT,
    url TEXT,
    input TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    channel_id TEXT NOT NULL REFERENCES channels(id),
    title TEXT NOT NULL,
    description TEXT DEFAULT '',
    published_at TEXT,
    duration_s INTEGER,
    view_count BIGINT,
    thumbnail_url TEXT,
    transcript_status TEXT NOT NULL DEFAULT 'pending',
    transcript_source TEXT,
    transcript_lang TEXT,
    transcript_error TEXT,
    position INTEGER,
    updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_videos_channel ON videos(channel_id);
CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(transcript_status);

CREATE TABLE IF NOT EXISTS segments (
    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    start DOUBLE PRECISION NOT NULL,
    duration DOUBLE PRECISION NOT NULL,
    text TEXT NOT NULL,
    PRIMARY KEY (video_id, idx)
);

CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    start DOUBLE PRECISION NOT NULL,
    "end" DOUBLE PRECISION NOT NULL,
    text TEXT NOT NULL,
    tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('cs_simple'::regconfig, text)) STORED
);
CREATE INDEX IF NOT EXISTS idx_chunks_video ON chunks(video_id);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (tsv);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


class PostgresDatabase:
    backend = "postgres"
    vec_available = True

    def __init__(self, url: str, embedding_dim: int = 768, reset_on_mismatch: bool = False):
        self.url = url
        self.embedding_dim = embedding_dim
        self.reset_on_mismatch = reset_on_mismatch
        self.conn = psycopg.connect(url, row_factory=dict_row, autocommit=False)
        self._init_schema()
        register_vector(self.conn)

    def _init_schema(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(SCHEMA)
            cur.execute("SELECT value FROM meta WHERE key='embedding_dim'")
            row = cur.fetchone()
            stored = int(row["value"]) if row else None
            cur.execute("SELECT 1 FROM information_schema.columns WHERE table_name='chunks' AND column_name='embedding'")
            has_col = cur.fetchone() is not None
            if has_col and stored and stored != self.embedding_dim:
                if self.reset_on_mismatch:
                    cur.execute("DROP INDEX IF EXISTS idx_chunks_embedding")
                    cur.execute("ALTER TABLE chunks DROP COLUMN embedding")
                    has_col = False
                else:
                    self.embedding_dim = stored
            if not has_col:
                cur.execute(f"ALTER TABLE chunks ADD COLUMN embedding vector({self.embedding_dim})")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks "
                        "USING hnsw (embedding vector_cosine_ops)")
            cur.execute("INSERT INTO meta(key, value) VALUES ('embedding_dim', %s) "
                        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value", (str(self.embedding_dim),))
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

    def _one(self, sql: str, params=()):
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()

    def _all(self, sql: str, params=()):
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    # ---------- meta ----------
    def get_meta(self, key: str) -> str | None:
        row = self._one("SELECT value FROM meta WHERE key=%s", (key,))
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO meta(key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value",
                      (key, value))

    # ---------- channels / videos ----------
    def upsert_channel(self, id: str, name: str, url: str, input: str = "") -> None:
        with self.tx() as c:
            c.execute("INSERT INTO channels(id, name, url, input) VALUES (%s,%s,%s,%s) "
                      "ON CONFLICT (id) DO UPDATE SET name=EXCLUDED.name, url=EXCLUDED.url, input=EXCLUDED.input",
                      (id, name, url, input))

    def get_channel(self, id: str | None = None):
        if id:
            return self._one("SELECT * FROM channels WHERE id=%s", (id,))
        return self._one("SELECT * FROM channels ORDER BY created_at LIMIT 1")

    def upsert_video(self, v: dict) -> None:
        params = {"description": "", "published_at": None, "duration_s": None, "view_count": None,
                  "thumbnail_url": None, "position": None, **v}
        self.conn.execute(
            """INSERT INTO videos(id, channel_id, title, description, published_at, duration_s, view_count, thumbnail_url, position)
               VALUES (%(id)s, %(channel_id)s, %(title)s, %(description)s, %(published_at)s, %(duration_s)s,
                       %(view_count)s, %(thumbnail_url)s, %(position)s)
               ON CONFLICT (id) DO UPDATE SET
                 title=EXCLUDED.title,
                 position=COALESCE(EXCLUDED.position, videos.position),
                 description=COALESCE(NULLIF(EXCLUDED.description,''), videos.description),
                 published_at=COALESCE(EXCLUDED.published_at, videos.published_at),
                 duration_s=COALESCE(EXCLUDED.duration_s, videos.duration_s),
                 view_count=COALESCE(EXCLUDED.view_count, videos.view_count),
                 thumbnail_url=COALESCE(EXCLUDED.thumbnail_url, videos.thumbnail_url),
                 updated_at=now()""", params)

    def get_video(self, video_id: str):
        return self._one("SELECT * FROM videos WHERE id=%s", (video_id,))

    _ORDERS = {"published_at DESC": "published_at DESC NULLS LAST, position ASC NULLS LAST",
               "published_at ASC": "published_at ASC NULLS LAST", "title ASC": "title ASC",
               "duration_s DESC": "duration_s DESC NULLS LAST", "view_count DESC": "view_count DESC NULLS LAST",
               "position ASC": "position ASC NULLS LAST"}

    def list_videos(self, channel_id: str | None = None, status: str | None = None,
                    limit: int = 50, offset: int = 0, order: str = "published_at DESC"):
        where, params = [], []
        if channel_id:
            where.append("channel_id=%s"); params.append(channel_id)
        if status:
            where.append("transcript_status=%s"); params.append(status)
        sql = "SELECT * FROM videos" + (" WHERE " + " AND ".join(where) if where else "")
        sql += f" ORDER BY {self._ORDERS.get(order, self._ORDERS['published_at DESC'])} LIMIT %s OFFSET %s"
        return self._all(sql, (*params, limit, offset))

    def count_videos(self, channel_id: str | None = None, status: str | None = None) -> int:
        where, params = [], []
        if channel_id:
            where.append("channel_id=%s"); params.append(channel_id)
        if status:
            where.append("transcript_status=%s"); params.append(status)
        sql = "SELECT COUNT(*) AS c FROM videos" + (" WHERE " + " AND ".join(where) if where else "")
        return self._one(sql, params)["c"]

    def set_transcript_status(self, video_id: str, status: str, source: str | None = None,
                              lang: str | None = None, error: str | None = None) -> None:
        with self.tx() as c:
            c.execute("UPDATE videos SET transcript_status=%s, transcript_source=%s, transcript_lang=%s, "
                      "transcript_error=%s, updated_at=now() WHERE id=%s",
                      (status, source, lang, (error if (error or "").startswith("gaps:") else (error or "")[:500]) or None, video_id))

    # ---------- transcripts ----------
    def store_transcript(self, video_id: str, segments: list[Segment], chunks: list[Chunk],
                         source: str, lang: str | None) -> None:
        with self.tx() as c, c.cursor() as cur:
            cur.execute("DELETE FROM segments WHERE video_id=%s", (video_id,))
            cur.execute("DELETE FROM chunks WHERE video_id=%s", (video_id,))
            cur.executemany("INSERT INTO segments(video_id, idx, start, duration, text) VALUES (%s,%s,%s,%s,%s)",
                            [(video_id, i, s.start, s.duration, s.text) for i, s in enumerate(segments)])
            cur.executemany('INSERT INTO chunks(video_id, idx, start, "end", text) VALUES (%s,%s,%s,%s,%s)',
                            [(video_id, i, ch.start, ch.end, ch.text) for i, ch in enumerate(chunks)])
            cur.execute("UPDATE videos SET transcript_status='done', transcript_source=%s, transcript_lang=%s, "
                        "transcript_error=NULL, updated_at=now() WHERE id=%s", (source, lang, video_id))

    def get_segments(self, video_id: str):
        return self._all("SELECT idx, start, duration, text FROM segments WHERE video_id=%s ORDER BY idx", (video_id,))

    def get_segments_range(self, video_id: str, start: float, end: float):
        return self._all("SELECT idx, start, duration, text FROM segments WHERE video_id=%s AND start + duration >= %s "
                         "AND start <= %s ORDER BY idx", (video_id, start, end))

    # ---------- embeddings ----------
    def pending_embeddings(self, limit: int = 100):
        return self._all("SELECT id, video_id, text FROM chunks WHERE embedding IS NULL ORDER BY id LIMIT %s", (limit,))

    def store_embeddings(self, items: list[tuple[int, list[float]]]) -> None:
        import numpy as np
        with self.tx() as c, c.cursor() as cur:
            cur.executemany("UPDATE chunks SET embedding=%s WHERE id=%s",
                            [(np.asarray(vec, dtype=np.float32), cid) for cid, vec in items])

    def count_embedded(self) -> tuple[int, int]:
        row = self._one("SELECT COUNT(embedding) AS e, COUNT(*) AS c FROM chunks")
        return int(row["e"]), int(row["c"])

    def chunk_ids(self, video_id: str) -> dict[int, int]:
        return {r["idx"]: r["id"] for r in self._all("SELECT id, idx FROM chunks WHERE video_id=%s", (video_id,))}

    def get_embeddings(self, video_id: str) -> dict[int, list[float]]:
        rows = self._all("SELECT idx, embedding FROM chunks WHERE video_id=%s AND embedding IS NOT NULL", (video_id,))
        return {r["idx"]: [float(x) for x in r["embedding"]] for r in rows}

    # ---------- vyhledávací primitiva ----------
    def fts_hits(self, q: str, mode: str, limit: int) -> list[tuple[int, float]]:
        tsq = build_tsquery(q, mode)
        if not tsq:
            return []
        try:
            rows = self._all(
                "SELECT id, ts_rank_cd(tsv, query) AS r FROM chunks, to_tsquery('cs_simple', %s) query "
                "WHERE tsv @@ query ORDER BY r DESC LIMIT %s", (tsq, limit))
        except psycopg.Error:
            self.conn.rollback()
            return []
        return [(r["id"], -float(r["r"])) for r in rows]  # záporně, aby 'nižší = lepší' jako u bm25

    def vec_hits(self, qvec: list[float], limit: int) -> list[tuple[int, float]]:
        import numpy as np
        rows = self._all(
            "SELECT id, embedding <=> %s AS d FROM chunks WHERE embedding IS NOT NULL ORDER BY d LIMIT %s",
            (np.asarray(qvec, dtype=np.float32), limit))
        return [(r["id"], float(r["d"])) for r in rows]

    def snippet(self, chunk_id: int, q: str, mode: str) -> str | None:
        tsq = build_tsquery(q, mode)
        if not tsq:
            return None
        try:
            row = self._one(
                "SELECT ts_headline('cs_simple', text, query, "
                "'StartSel=<mark>, StopSel=</mark>, MaxWords=45, MinWords=25, MaxFragments=2, FragmentDelimiter=\" … \"') AS s "
                "FROM chunks, to_tsquery('cs_simple', %s) query WHERE id=%s AND tsv @@ query", (tsq, chunk_id))
        except psycopg.Error:
            self.conn.rollback()
            return None
        return row["s"] if row and row["s"] else None

    def fetch_chunks(self, ids: list[int]) -> list[dict]:
        if not ids:
            return []
        return self._all(
            'SELECT c.id, c.video_id, c.start, c."end", c.text, v.title, v.published_at, v.duration_s, v.thumbnail_url '
            "FROM chunks c JOIN videos v ON v.id = c.video_id WHERE c.id = ANY(%s)", (ids,))

    # ---------- stats ----------
    def stats(self) -> dict:
        st = {r["transcript_status"]: r["c"] for r in self._all(
            "SELECT transcript_status, COUNT(*) AS c FROM videos GROUP BY transcript_status")}
        embedded, chunks = self.count_embedded()
        hours = self._one("SELECT COALESCE(SUM(duration_s),0)/3600.0 AS h FROM videos WHERE transcript_status='done'")["h"]
        return {"backend": self.backend, "videos": self.count_videos(), "videos_by_status": st, "chunks": chunks,
                "chunks_embedded": embedded, "total_hours": round(float(hours), 1), "vector_search": embedded > 0}
