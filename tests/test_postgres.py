"""Testy Postgres backendu – běží jen s TEST_DATABASE_URL (lokální Postgres s pgvector + unaccent)."""
import math
import os

import pytest

from ytindex.chunking import chunk_segments, merge_segments
from ytindex.search import search, strip_diacritics
from tests.conftest import TRANSCRIPTS

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL není nastaven")


@pytest.fixture
def pgdb():
    from ytindex.db_postgres import PostgresDatabase
    import psycopg
    with psycopg.connect(URL, autocommit=True) as c:
        c.execute("DROP TABLE IF EXISTS chunks, segments, videos, channels, meta CASCADE")
    d = PostgresDatabase(URL, embedding_dim=8)
    d.upsert_channel("UCtest", "Test kanál", "https://www.youtube.com/channel/UCtest")
    with d.tx():
        for i, (vid, (title, _)) in enumerate(TRANSCRIPTS.items()):
            d.upsert_video({"id": vid, "channel_id": "UCtest", "title": title, "position": i,
                            "published_at": f"2024-0{i+1}-01T00:00:00Z", "duration_s": 100 * (i + 1)})
    for vid, (_, segs) in TRANSCRIPTS.items():
        segs = merge_segments(segs)
        d.store_transcript(vid, segs, chunk_segments(segs, target_chars=120), "captions", "cs")
    yield d
    d.close()


def test_pg_fts_diacritics_and_highlight(pgdb):
    hits = search(pgdb, "dane kradez", mode="fts")
    assert len(hits) == 1 and hits[0].video_id == "vid00000001"
    assert "<mark>Daně</mark>" in hits[0].match.highlighted and "<mark>krádež</mark>" in hits[0].match.highlighted
    assert search(pgdb, "svobod", mode="fts")[0].video_id == "vid00000001"   # prefix
    assert len(search(pgdb, "inflace anarchokapitalismus", mode="fts")) == 2  # AND → OR fallback
    assert search(pgdb, '"centrální banka"', mode="fts")[0].video_id == "vid00000002"


def test_pg_vectors_and_hybrid(pgdb):
    vocab = ["bitcoin", "inflace", "vzdelavani", "stat", "deti"]

    def emb(text):
        t = strip_diacritics(text).lower()
        v = [1.0 if w in t else 0.0 for w in vocab] + [0.05, 0.0, 0.0]
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v]

    rows = pgdb.pending_embeddings(limit=100)
    pgdb.store_embeddings([(r["id"], emb(r["text"])) for r in rows])
    assert pgdb.count_embedded() == (len(rows), len(rows))
    assert search(pgdb, "děti školáci", mode="hybrid", embed_query=emb)[0].video_id == "vid00000003"
    assert search(pgdb, "vzdělávání", mode="vector", embed_query=emb)[0].video_id == "vid00000003"


def test_pg_videos_and_stats(pgdb):
    assert pgdb.count_videos() == 3
    assert [v["id"] for v in pgdb.list_videos(order="position ASC")] == list(TRANSCRIPTS)
    assert pgdb.get_video("vid00000002")["transcript_status"] == "done"
    assert len(pgdb.get_segments("vid00000002")) == 3
    s = pgdb.stats()
    assert s["backend"] == "postgres" and s["videos"] == 3 and s["chunks"] > 0
    pgdb.set_transcript_status("vid00000003", "failed", error="x")
    assert pgdb.count_videos(status="failed") == 1
