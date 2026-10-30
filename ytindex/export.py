"""Hromadný export indexu pro jiné systémy: JSONL (videa, segmenty, chunky), NumPy embeddingy, manifest.

  ytindex export [--out data/export] [--gcs gs://bucket/export]

Manifest (export/manifest.json) popisuje verzi indexu a soubory; API ho vrací na /api/export.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import struct
import time
from datetime import datetime, timezone
from pathlib import Path

from . import __version__

FILES = ("videos.jsonl.gz", "segments.jsonl.gz", "chunks.jsonl.gz", "embeddings.npy", "chunk_ids.npy")


def parse_gaps(transcript_error: str | None) -> list[dict]:
    """'gaps:[[start,end,reason],…]' → [{start,end,reason}], jinak []."""
    if not transcript_error or not transcript_error.startswith("gaps:"):
        return []
    try:
        return [{"start": a, "end": b, "reason": r} for a, b, r in json.loads(transcript_error[5:])]
    except (json.JSONDecodeError, ValueError):
        return []


def video_record(r) -> dict:
    return {
        "video_id": r["id"], "title": r["title"], "description": r["description"] or "",
        "url": f"https://www.youtube.com/watch?v={r['id']}", "thumbnail": r["thumbnail_url"],
        "published_at": r["published_at"], "duration_s": r["duration_s"], "view_count": r["view_count"],
        "position": r["position"], "transcript_status": r["transcript_status"],
        "transcript_source": r["transcript_source"], "transcript_lang": r["transcript_lang"],
        "gaps": parse_gaps(r["transcript_error"]), "updated_at": r["updated_at"],
    }


def index_version(db) -> str:
    """Verze indexu: uložená (z GCS generation / exportu), jinak otisk obsahu."""
    v = db.get_meta("index_version")
    if v:
        return v
    row = db.conn.execute("SELECT COUNT(*), MAX(updated_at) FROM videos").fetchone()
    n_chunks = db.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    return hashlib.sha1(f"{row[0]}|{row[1]}|{n_chunks}".encode()).hexdigest()[:12]


def write_export(db, out_dir: str | os.PathLike, version: str | None = None) -> dict:
    """Zapíše soubory + manifest.json do out_dir. Vrací manifest."""
    import numpy as np

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    counts = {}

    with gzip.open(out / "videos.jsonl.gz", "wt", encoding="utf-8") as f:
        n = 0
        for r in db.list_videos(limit=10**7, order="position ASC"):
            f.write(json.dumps(video_record(r), ensure_ascii=False) + "\n"); n += 1
        counts["videos"] = n

    with gzip.open(out / "segments.jsonl.gz", "wt", encoding="utf-8") as f:
        n = 0
        for r in db.conn.execute("SELECT video_id, idx, start, duration, text FROM segments ORDER BY video_id, idx"):
            f.write(json.dumps({"video_id": r[0], "idx": r[1], "start": r[2], "duration": r[3], "text": r[4]},
                               ensure_ascii=False) + "\n"); n += 1
        counts["segments"] = n

    ids: list[int] = []
    with gzip.open(out / "chunks.jsonl.gz", "wt", encoding="utf-8") as f:
        n = 0
        for r in db.conn.execute('SELECT id, video_id, idx, start, "end", text, embedded FROM chunks ORDER BY id'):
            f.write(json.dumps({"chunk_id": r[0], "video_id": r[1], "idx": r[2], "start": r[3], "end": r[4],
                                "text": r[5], "embedded": bool(r[6])}, ensure_ascii=False) + "\n")
            n += 1
        counts["chunks"] = n

    # embeddingy: float32 [N, dim] + chunk_ids int64 [N] ve stejném pořadí
    dim = db.embedding_dim
    vec_rows = db.iter_embeddings()
    vecs, ids = [], []
    for chunk_id, vec in vec_rows:
        ids.append(chunk_id); vecs.append(vec)
    emb = np.asarray(vecs, dtype=np.float32).reshape(len(ids), dim) if ids else np.zeros((0, dim), dtype=np.float32)
    np.save(out / "embeddings.npy", emb)
    np.save(out / "chunk_ids.npy", np.asarray(ids, dtype=np.int64))
    counts["embeddings"] = len(ids)

    ch = db.get_channel()
    manifest = {
        "schema": 1, "ytindex": __version__,
        "version": version or index_version(db),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "channel": {"id": ch["id"], "name": ch["name"], "url": ch["url"]} if ch else None,
        "counts": counts,
        "embedding": {"model": db.get_meta("embedding_model") or "gemini-embedding-001", "dim": dim,
                      "normalized": True, "metric": "cosine", "task_type_documents": "RETRIEVAL_DOCUMENT",
                      "task_type_query": "RETRIEVAL_QUERY"},
        "files": {},
        "notes": "segments = přepis (zdroj pravdy); chunks = úseky pro hledání (překryv 1 segment); "
                 "embeddings.npy[i] patří k chunk_ids.npy[i]; odkaz na čas: https://www.youtube.com/watch?v={video_id}&t={int(start)}s",
    }
    for name in FILES:
        p = out / name
        manifest["files"][name] = {"size": p.stat().st_size, "sha256": _sha256(p)}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    manifest["elapsed_s"] = round(time.time() - t0, 1)
    return manifest


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def upload_export(out_dir: str | os.PathLike, gs_prefix: str) -> dict[str, str]:
    """Nahraje soubory + manifest do gs://bucket/prefix/; vrací {name: veřejná https URL}."""
    from google.cloud import storage

    assert gs_prefix.startswith("gs://")
    bucket_name, _, prefix = gs_prefix[5:].partition("/")
    prefix = prefix.strip("/")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    urls = {}
    for name in (*FILES, "manifest.json"):
        blob = bucket.blob(f"{prefix}/{name}" if prefix else name)
        blob.cache_control = "public, max-age=300"
        blob.upload_from_filename(str(Path(out_dir) / name))
        urls[name] = f"https://storage.googleapis.com/{bucket_name}/{blob.name}"
    return urls


def read_manifest_gcs(gs_prefix: str) -> dict | None:
    from google.cloud import storage

    bucket_name, _, prefix = gs_prefix[5:].partition("/")
    blob = storage.Client().bucket(bucket_name).blob(f"{prefix.strip('/')}/manifest.json" if prefix.strip("/") else "manifest.json")
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())


def open_gcs_file(gs_prefix: str, name: str):
    """(blob, velikost) souboru exportu v bucketu – pro proxy přes API, když bucket není veřejný."""
    from google.cloud import storage

    bucket_name, _, prefix = gs_prefix[5:].partition("/")
    blob = storage.Client().bucket(bucket_name).get_blob(f"{prefix.strip('/')}/{name}" if prefix.strip("/") else name)
    if blob is None:
        return None, 0
    return blob, blob.size


def iter_blob_ranges(blob, start: int, end: int, block: int = 8 << 20):
    """Čte objekt po blocích samostatnými (opakovatelnými) range requesty – odolné proti přerušení spojení."""
    from google.cloud.storage.retry import DEFAULT_RETRY

    pos = start
    while pos <= end:
        last = min(pos + block - 1, end)
        yield blob.download_as_bytes(start=pos, end=last, retry=DEFAULT_RETRY)
        pos = last + 1


def public_urls(gs_prefix: str) -> dict[str, str]:
    bucket_name, _, prefix = gs_prefix[5:].partition("/")
    prefix = prefix.strip("/")
    return {name: f"https://storage.googleapis.com/{bucket_name}/{(prefix + '/') if prefix else ''}{name}"
            for name in (*FILES, "manifest.json")}


def embedding_b64(vec: list[float]) -> str:
    import base64
    return base64.b64encode(struct.pack(f"{len(vec)}f", *vec)).decode()
