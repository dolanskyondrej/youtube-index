"""Hybridní vyhledávání: FTS5 (BM25, bez diakritiky) + vektory (sqlite-vec, cosine) spojené přes Reciprocal Rank Fusion."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, asdict

from .chunking import format_timestamp
from .db import Database, serialize_f32

RRF_K = 60
# vektorové kandidáty ořízneme na ty, které jsou "blízko" nejlepšího zásahu – k-NN vrací vždy k výsledků,
# i když jsou nesmyslné, a ty by jinak falešně zvyšovaly match_count
VEC_SIM_MARGIN = 0.15
VEC_SIM_FLOOR = 0.2
_WORD = re.compile(r"[\w]+", re.UNICODE)


def strip_diacritics(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def build_fts_query(q: str, mode: str = "and") -> str:
    """Z uživatelského dotazu vyrobí bezpečný FTS5 dotaz.

    - fráze v uvozovkách zůstávají frází
    - jednotlivá slova dostanou prefix (*), aby 'svobod' našlo 'svoboda' i 'svobodný' (čeština skloňuje)
    """
    phrases = re.findall(r'"([^"]+)"', q)
    rest = re.sub(r'"[^"]*"', " ", q)
    terms = []
    for p in phrases:
        words = _WORD.findall(p)
        if words:
            terms.append('"' + " ".join(words) + '"')
    for w in _WORD.findall(rest):
        if len(w) >= 2:
            terms.append(f'"{w}"*')
        else:
            terms.append(f'"{w}"')
    joiner = " AND " if mode == "and" else " OR "
    return joiner.join(terms)


@dataclass
class Match:
    text: str          # čistý text chunku (zkrácený)
    highlighted: str   # text se <mark> značkami
    start: float
    end: float
    timestamp: str
    url: str           # odkaz na video s časem


@dataclass
class VideoHit:
    video_id: str
    title: str
    url: str
    thumbnail: str
    published_at: str | None
    duration_s: int | None
    score: float
    match_count: int
    match: Match
    matches: list[Match]


def _video_url(video_id: str, start: float | None = None) -> str:
    base = f"https://www.youtube.com/watch?v={video_id}"
    return f"{base}&t={int(start)}s" if start is not None else base


def _fts_hits(db: Database, q: str, limit: int) -> list[tuple[int, float]]:
    """Vrátí [(chunk_id, bm25)] – nižší bm25 = lepší."""
    for mode in ("and", "or"):
        fq = build_fts_query(q, mode)
        if not fq:
            return []
        try:
            rows = db.conn.execute(
                "SELECT rowid, bm25(chunks_fts) AS r FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY r LIMIT ?",
                (fq, limit),
            ).fetchall()
        except Exception:
            rows = []
        if rows:
            return [(r["rowid"], r["r"]) for r in rows]
    return []


def _vec_hits(db: Database, qvec: list[float], limit: int) -> list[tuple[int, float]]:
    rows = db.conn.execute(
        "SELECT chunk_id, distance FROM chunk_vec WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (serialize_f32(qvec), limit),
    ).fetchall()
    if not rows:
        return []
    best_sim = 1.0 - rows[0]["distance"]
    threshold = max(best_sim - VEC_SIM_MARGIN, VEC_SIM_FLOOR)
    return [(r["chunk_id"], r["distance"]) for r in rows if (1.0 - r["distance"]) >= threshold]


def _snippet(db: Database, chunk_id: int, q: str, text: str, max_chars: int = 320) -> tuple[str, str]:
    """(plain, highlighted). Zvýraznění dělá FTS5 snippet(); pro čistě sémantické zásahy ruční fallback."""
    for mode in ("and", "or"):
        fq = build_fts_query(q, mode)
        if not fq:
            break
        try:
            row = db.conn.execute(
                "SELECT snippet(chunks_fts, 0, '<mark>', '</mark>', '…', 40) AS s FROM chunks_fts "
                "WHERE rowid = ? AND chunks_fts MATCH ?",
                (chunk_id, fq),
            ).fetchone()
        except Exception:
            row = None
        if row and row["s"]:
            hl = row["s"]
            return re.sub(r"</?mark>", "", hl), hl
    plain = text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + "…"
    return plain, _manual_highlight(plain, q)


def _manual_highlight(text: str, q: str) -> str:
    words = [w for w in _WORD.findall(strip_diacritics(q).lower()) if len(w) >= 3]
    if not words:
        return text
    folded = strip_diacritics(text).lower()
    if len(folded) != len(text):  # NFD mapování nesedí 1:1 – raději nezvýrazňovat
        return text
    spans = []
    for w in words:
        for m in re.finditer(r"\b" + re.escape(w) + r"\w*", folded):
            spans.append((m.start(), m.end()))
    if not spans:
        return text
    spans.sort()
    out, pos = [], 0
    for s, e in spans:
        if s < pos:
            continue
        out.append(text[pos:s]); out.append("<mark>" + text[s:e] + "</mark>"); pos = e
    out.append(text[pos:])
    return "".join(out)


def search(db: Database, q: str, limit: int = 20, mode: str = "hybrid",
           embed_query=None, per_video_matches: int = 3, candidates: int = 200) -> list[VideoHit]:
    """mode: hybrid | fts | vector. embed_query: callable(str) -> list[float] (None = jen fulltext)."""
    q = q.strip()
    if not q:
        return []
    use_vec = mode in ("hybrid", "vector") and embed_query is not None and db.vec_available
    rrf: dict[int, float] = {}
    if mode in ("hybrid", "fts") or not use_vec:
        for rank, (cid, _) in enumerate(_fts_hits(db, q, candidates)):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    if use_vec:
        try:
            qvec = embed_query(q)
            for rank, (cid, _) in enumerate(_vec_hits(db, qvec, candidates)):
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        except Exception:
            if mode == "vector":
                raise
    if not rrf:
        return []

    ids = list(rrf)
    rows = db.conn.execute(
        f"SELECT c.id, c.video_id, c.start, c.end, c.text, v.title, v.published_at, v.duration_s, v.thumbnail_url "
        f"FROM chunks c JOIN videos v ON v.id = c.video_id WHERE c.id IN ({','.join('?' * len(ids))})",
        ids,
    ).fetchall()

    by_video: dict[str, list] = {}
    for r in rows:
        by_video.setdefault(r["video_id"], []).append(r)

    hits: list[VideoHit] = []
    for vid, chunks in by_video.items():
        chunks.sort(key=lambda r: -rrf[r["id"]])
        best = max(rrf[r["id"]] for r in chunks)
        # skóre videa: nejlepší chunk + malý bonus za počet dalších zásahů
        score = best * (1 + 0.1 * math.log1p(len(chunks) - 1))
        matches = []
        for r in chunks[:per_video_matches]:
            plain, hl = _snippet(db, r["id"], q, r["text"])
            matches.append(Match(plain, hl, r["start"], r["end"], format_timestamp(r["start"]),
                                 _video_url(vid, r["start"])))
        v = chunks[0]
        hits.append(VideoHit(
            video_id=vid, title=v["title"], url=_video_url(vid),
            thumbnail=v["thumbnail_url"] or f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg",
            published_at=v["published_at"], duration_s=v["duration_s"],
            score=round(score, 6), match_count=len(chunks), match=matches[0], matches=matches,
        ))
    hits.sort(key=lambda h: -h.score)
    return hits[:limit]


def hit_to_dict(h: VideoHit) -> dict:
    return asdict(h)
