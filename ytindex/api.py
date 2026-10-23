"""REST API (FastAPI) – univerzální backend pro libovolný frontend. OpenAPI na /docs, JSON schéma na /openapi.json."""

from __future__ import annotations

import contextlib
import logging
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from . import __version__
from .chunking import format_timestamp
from .config import Config, load_config
from .db import open_database
from .search import hit_to_dict, search

log = logging.getLogger(__name__)
WEB_DIR = Path(__file__).parent / "web"


# ---------- schémata odpovědí (= kontrakt pro frontendy) ----------

class MatchOut(BaseModel):
    text: str = Field(description="Úryvek transkriptu se shodou (čistý text)")
    highlighted: str = Field(description="Stejný úryvek s <mark>…</mark> kolem nalezených slov")
    start: float = Field(description="Začátek úryvku ve videu (s)")
    end: float
    timestamp: str = Field(description="Čas začátku jako h:mm:ss")
    url: str = Field(description="Odkaz na video otevřené v čase shody")


class VideoHitOut(BaseModel):
    video_id: str
    title: str
    url: str
    thumbnail: str
    published_at: str | None
    duration_s: int | None
    score: float
    match_count: int = Field(description="Kolik úseků videa odpovídá dotazu")
    match: MatchOut = Field(description="Nejlepší shoda")
    matches: list[MatchOut] = Field(description="Několik nejlepších shod v tomto videu")


class SearchOut(BaseModel):
    query: str
    mode: str
    total: int
    results: list[VideoHitOut]


class VideoOut(BaseModel):
    video_id: str
    title: str
    url: str
    thumbnail: str | None
    published_at: str | None
    duration_s: int | None
    view_count: int | None
    transcript_status: str
    transcript_source: str | None
    transcript_lang: str | None


class SegmentOut(BaseModel):
    start: float
    duration: float
    timestamp: str
    text: str
    url: str


class VideoDetailOut(VideoOut):
    description: str
    segments: list[SegmentOut]


class VideoListOut(BaseModel):
    total: int
    page: int
    per_page: int
    items: list[VideoOut]


class ChannelOut(BaseModel):
    id: str
    name: str
    url: str
    stats: dict


def _video_out(r) -> dict:
    return {
        "video_id": r["id"], "title": r["title"], "url": f"https://www.youtube.com/watch?v={r['id']}",
        "thumbnail": r["thumbnail_url"], "published_at": r["published_at"], "duration_s": r["duration_s"],
        "view_count": r["view_count"], "transcript_status": r["transcript_status"],
        "transcript_source": r["transcript_source"], "transcript_lang": r["transcript_lang"],
    }


def create_app(cfg: Config | None = None, db: Database | None = None) -> FastAPI:
    cfg = cfg or load_config()
    db = db or open_database(cfg)

    from .mcp_server import build_mcp, http_app
    mcp = build_mcp(cfg, db)
    mcp_app = http_app(mcp)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        async with mcp.session_manager.run():  # lifespan sub-aplikace FastAPI nespouští, proto ručně
            yield

    app = FastAPI(
        title="ytindex API",
        version=__version__,
        description="Vyhledávání v transkriptech všech videí YouTube kanálu. "
                    "Hybridní (fulltext + sémantické) hledání, výsledky po videích s nejlepší shodou a odkazem na čas. "
                    "MCP server (Streamable HTTP) běží na /mcp.",
        lifespan=lifespan,
    )
    app.mount("/mcp", mcp_app)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "DELETE"],
                       allow_headers=["*"], expose_headers=["Mcp-Session-Id"])

    @lru_cache(maxsize=1)
    def embedder():
        if not cfg.gemini_api_key:
            return None
        try:
            from .embeddings import Embedder
            return Embedder(cfg)
        except Exception as e:
            log.warning("Embedder nedostupný: %s", e)
            return None

    @lru_cache(maxsize=512)
    def cached_query_embedding(q: str):
        e = embedder()
        return e.embed_query(q) if e else None

    def embed_query_fn():
        embedded, _ = db.count_embedded()
        if embedded == 0 or embedder() is None:
            return None
        return cached_query_embedding

    @app.get("/health")
    def health():
        return {"ok": True, "version": __version__}

    @app.get("/api/channel", response_model=ChannelOut)
    def channel():
        ch = db.get_channel(cfg.channel.id or None)
        if not ch:
            raise HTTPException(404, "Kanál ještě nebyl inicializován (spusť `ytindex init`)")
        return {"id": ch["id"], "name": ch["name"], "url": ch["url"], "stats": db.stats()}

    @app.get("/api/search", response_model=SearchOut)
    def api_search(
        q: str = Query(..., min_length=1, max_length=300, description="Hledaný výraz; fráze v uvozovkách"),
        limit: int = Query(20, ge=1, le=100),
        mode: str = Query("hybrid", pattern="^(hybrid|fts|vector)$"),
        per_video: int = Query(3, ge=1, le=10, description="Kolik shod vrátit pro každé video"),
    ):
        hits = search(db, q, limit=limit, mode=mode, embed_query=embed_query_fn(), per_video_matches=per_video)
        effective = mode if (mode == "fts" or embed_query_fn() is not None) else "fts"
        return {"query": q, "mode": effective, "total": len(hits), "results": [hit_to_dict(h) for h in hits]}

    @app.get("/api/videos", response_model=VideoListOut)
    def videos(page: int = Query(1, ge=1), per_page: int = Query(50, ge=1, le=200),
               status: str | None = Query(None, pattern="^(pending|done|missing|failed)$"),
               order: str = Query("published_at DESC")):
        rows = db.list_videos(status=status, limit=per_page, offset=(page - 1) * per_page, order=order)
        return {"total": db.count_videos(status=status), "page": page, "per_page": per_page,
                "items": [_video_out(r) for r in rows]}

    @app.get("/api/videos/{video_id}", response_model=VideoDetailOut)
    def video_detail(video_id: str):
        r = db.get_video(video_id)
        if not r:
            raise HTTPException(404, "Video nenalezeno")
        segs = [{"start": s["start"], "duration": s["duration"], "timestamp": format_timestamp(s["start"]),
                 "text": s["text"], "url": f"https://www.youtube.com/watch?v={video_id}&t={int(s['start'])}s"}
                for s in db.get_segments(video_id)]
        return {**_video_out(r), "description": r["description"] or "", "segments": segs}

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(status_code=204)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")

    return app


def app() -> FastAPI:  # pro `uvicorn ytindex.api:app --factory`
    return create_app()
