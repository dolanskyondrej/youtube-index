"""Hybridní vyhledávání: FTS5 (BM25, bez diakritiky) + vektory (sqlite-vec, cosine) spojené přes Reciprocal Rank Fusion."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, asdict

from .chunking import format_timestamp
from .query import build_fts_query  # noqa: F401  (re-export pro zpětnou kompatibilitu testů)

RRF_K = 60
# Gemini embeddingy mají plochou škálu podobností (relevantní ~0.70–0.75, nesmysl ~0.60–0.65), proto:
# - do fúze jde vektorový top-N podle pořadí,
# - "shoda" (match_count) se počítá jen z chunků blízko nejlepšího (margin) nebo s lexikální shodou.
VEC_SIM_MARGIN = 0.03
VEC_SIM_FLOOR = 0.5
VEC_CANDIDATES = 80
# váhy ve fúzi: fulltext AND (všechna slova) je silný signál → vyšší váha; OR fallback je šum,
# v hybridním režimu se nepoužívá (sémantiku pokryjí vektory), jen v čistém fts
W_FTS_AND, W_FTS_OR, W_VEC = 1.0, 0.4, 1.0
# chunk, který najde fulltext i vektory zároveň, je skoro jistě správně → bonus (potlačí lexikální falešné shody,
# např. "stát" jako sloveso v dotazu "proč je stát zbytečný")
BOTH_BONUS = 1.3
_WORD = re.compile(r"[\w]+", re.UNICODE)


def strip_diacritics(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


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


def _fts_hits(db, q: str, limit: int, allow_or: bool = True) -> tuple[list[tuple[int, float]], float]:
    """Nejprve AND, při nule výsledků (volitelně) OR. Vrací (zásahy, váha)."""
    for mode, weight in (("and", W_FTS_AND), ("or", W_FTS_OR)):
        if mode == "or" and not allow_or:
            break
        rows = db.fts_hits(q, mode, limit)
        if rows:
            return rows, weight
    return [], 0.0


def _vec_hits(db, qvec: list[float], limit: int) -> tuple[list[tuple[int, float]], set[int]]:
    """(top-N podle vzdálenosti, množina id "blízko nejlepšího" pro počítání shod)"""
    rows = db.vec_hits(qvec, limit)
    if not rows:
        return [], set()
    best_sim = 1.0 - rows[0][1]
    threshold = max(best_sim - VEC_SIM_MARGIN, VEC_SIM_FLOOR)
    close = {cid for cid, d in rows if (1.0 - d) >= threshold}
    return rows, close


def _snippet(db, chunk_id: int, q: str, text: str, max_chars: int = 320) -> tuple[str, str]:
    """(plain, highlighted). Zvýraznění dělá databáze; pro čistě sémantické zásahy ruční fallback."""
    for mode in ("and", "or"):
        hl = db.snippet(chunk_id, q, mode)
        if hl:
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


def search(db, q: str, limit: int = 20, mode: str = "hybrid",
           embed_query=None, per_video_matches: int = 3, candidates: int = 200) -> list[VideoHit]:
    """mode: hybrid | fts | vector. embed_query: callable(str) -> list[float] (None = jen fulltext)."""
    q = q.strip()
    if not q:
        return []
    use_vec = mode in ("hybrid", "vector") and embed_query is not None and db.vec_available
    rrf: dict[int, float] = {}
    counted: set[int] = set()   # chunky, které se počítají jako "shoda"
    in_fts: set[int] = set()
    if mode in ("hybrid", "fts") or not use_vec:
        hits, weight = _fts_hits(db, q, candidates, allow_or=not use_vec)
        for rank, (cid, _) in enumerate(hits):
            rrf[cid] = rrf.get(cid, 0.0) + weight / (RRF_K + rank)
            counted.add(cid)
            in_fts.add(cid)
    if use_vec:
        try:
            qvec = embed_query(q)
            hits, close = _vec_hits(db, qvec, VEC_CANDIDATES)
            for rank, (cid, _) in enumerate(hits):
                rrf[cid] = rrf.get(cid, 0.0) + W_VEC / (RRF_K + rank)
                if cid in in_fts:
                    rrf[cid] *= BOTH_BONUS
            counted |= close
        except Exception:
            if mode == "vector":
                raise
    if not rrf:
        return []

    rows = db.fetch_chunks(list(rrf))

    by_video: dict[str, list] = {}
    for r in rows:
        by_video.setdefault(r["video_id"], []).append(r)

    hits: list[VideoHit] = []
    for vid, chunks in by_video.items():
        chunks.sort(key=lambda r: -rrf[r["id"]])
        best = max(rrf[r["id"]] for r in chunks)
        n_matches = sum(1 for r in chunks if r["id"] in counted) or 1
        # skóre videa: nejlepší chunk + malý bonus za počet dalších skutečných zásahů
        score = best * (1 + 0.1 * math.log1p(n_matches - 1))
        matches = []
        taken: list[tuple[float, float]] = []
        for r in chunks:
            if len(matches) >= per_video_matches:
                break
            # sousední chunky se překrývají (overlap) – nezobrazovat dvakrát stejné místo
            if any(r["start"] < e and r["end"] > s_ for s_, e in taken):
                continue
            taken.append((r["start"], r["end"]))
            plain, hl = _snippet(db, r["id"], q, r["text"])
            matches.append(Match(plain, hl, r["start"], r["end"], format_timestamp(r["start"]),
                                 _video_url(vid, r["start"])))
        v = chunks[0]
        hits.append(VideoHit(
            video_id=vid, title=v["title"], url=_video_url(vid),
            thumbnail=v["thumbnail_url"] or f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg",
            published_at=v["published_at"], duration_s=v["duration_s"],
            score=round(score, 6), match_count=n_matches, match=matches[0], matches=matches,
        ))
    hits.sort(key=lambda h: -h.score)
    return hits[:limit]


def hit_to_dict(h: VideoHit) -> dict:
    return asdict(h)
