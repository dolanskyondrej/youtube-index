import base64
import json
import struct

from fastapi.testclient import TestClient

from ytindex.api import create_app
from ytindex.export import parse_gaps, write_export


def test_parse_gaps():
    assert parse_gaps('gaps:[[600, 1200, "RECITATION"]]') == [{"start": 600, "end": 1200, "reason": "RECITATION"}]
    assert parse_gaps("něco jiného") == [] and parse_gaps(None) == []


def test_export_files_and_manifest(cfg, db, tmp_path):
    import numpy as np
    rows = db.pending_embeddings(limit=100)
    db.store_embeddings([(r["id"], [0.1] * db.embedding_dim) for r in rows[:2]])
    m = write_export(db, tmp_path / "exp")
    assert m["counts"] == {"videos": 3, "segments": 10, "chunks": len(rows), "embeddings": 2}
    assert set(m["files"]) == {"videos.jsonl.gz", "segments.jsonl.gz", "chunks.jsonl.gz", "embeddings.npy", "chunk_ids.npy"}
    emb = np.load(tmp_path / "exp" / "embeddings.npy"); ids = np.load(tmp_path / "exp" / "chunk_ids.npy")
    assert emb.shape == (2, db.embedding_dim) and emb.dtype == np.float32 and list(ids) == [rows[0]["id"], rows[1]["id"]]
    assert json.loads((tmp_path / "exp" / "manifest.json").read_text())["embedding"]["dim"] == db.embedding_dim


def test_bulk_endpoints(cfg, db, tmp_path, monkeypatch):
    monkeypatch.setenv("YTINDEX_EXPORT_DIR", str(tmp_path / "exp"))
    monkeypatch.setenv("YTINDEX_API_KEY", "tajne")
    write_export(db, tmp_path / "exp")
    c = TestClient(create_app(cfg, db, rate_limit=False))
    # API klíč chrání hromadné cesty
    assert c.get("/api/chunks").status_code == 401
    h = {"X-Api-Key": "tajne"}
    r = c.get("/api/chunks", params={"limit": 2, "with_embeddings": 1}, headers=h)
    assert r.status_code == 200 and r.headers["X-Index-Version"]
    d = r.json(); assert len(d["items"]) == 2 and d["next_after"] == d["items"][-1]["chunk_id"]
    r2 = c.get("/api/chunks", params={"limit": 100, "after": d["next_after"]}, headers=h).json()
    assert r2["next_after"] is None and r2["items"][0]["chunk_id"] > d["next_after"]
    # embedding jako base64 float32
    rows = db.pending_embeddings(limit=1)
    db.store_embeddings([(rows[0]["id"], [0.5] * db.embedding_dim)])
    e = c.get("/api/chunks", params={"video_id": rows[0]["video_id"], "with_embeddings": 1}, headers=h).json()
    vec = next(i["embedding"] for i in e["items"] if i["chunk_id"] == rows[0]["id"])
    assert struct.unpack("f", base64.b64decode(vec)[:4])[0] == 0.5
    # changes od minulosti vrátí všechna videa, od budoucnosti nic
    ch = c.get("/api/changes", params={"since": "2000-01-01T00:00:00Z"}, headers=h).json()
    assert len(ch["items"]) == 3 and ch["next_since"] is None and ch["items"][0]["gaps"] == []
    assert c.get("/api/changes", params={"since": "2999-01-01 00:00:00"}, headers=h).json()["items"] == []
    # segments alias a export manifest (lokální soubory)
    assert len(c.get("/api/videos/vid00000002/segments").json()) == 3
    ex = c.get("/api/export").json()
    assert ex["counts"]["videos"] == 3 and ex["files"]["chunks.jsonl.gz"]["url"].endswith("/api/export/files/chunks.jsonl.gz")
    assert c.get("/api/export/files/manifest.json").status_code == 401        # soubory chrání API klíč
    assert c.get("/api/export/files/manifest.json", headers=h).status_code == 200
