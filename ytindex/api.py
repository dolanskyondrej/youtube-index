"""REST API (FastAPI) – univerzální backend pro libovolný frontend. OpenAPI na /docs, JSON schéma na /openapi.json."""

from __future__ import annotations

import contextlib
import logging
import os
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .chunking import format_timestamp
from .config import Config, load_config
from .db import open_database
from .export import FILES, embedding_b64, index_version, open_gcs_file, parse_gaps, public_urls, read_manifest_gcs
from .ratelimit import rate_limit_middleware
from .research import answer as rag_answer, gather_context
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


class PassageOut(BaseModel):
    n: int = Field(description="Číslo citace")
    video_id: str
    title: str
    published_at: str | None
    start: float
    end: float
    timestamp: str
    url: str = Field(description="Odkaz na video v čase začátku pasáže")
    text: str = Field(description="Souvislý text pasáže (shoda + okolní řeč)")
    score: float


class ContextOut(BaseModel):
    question: str
    indexed_videos: int
    passages: list[PassageOut]


class AskOut(BaseModel):
    question: str
    answer: str | None
    model: str | None = None
    error: str | None = None
    citations: list[PassageOut]


class GapOut(BaseModel):
    start: int
    end: int
    reason: str


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
    gaps: list[GapOut] = Field(default_factory=list, description="Okna přepisu, která chybí (odmítnutá modelem apod.)")
    updated_at: str | None = None


class ChunkOut(BaseModel):
    chunk_id: int
    video_id: str
    idx: int
    start: float
    end: float
    text: str
    embedded: bool
    embedding: str | None = Field(None, description="base64 float32[dim] little-endian (jen s with_embeddings=1)")


class ChunkListOut(BaseModel):
    index_version: str
    embedding_dim: int
    next_after: int | None = Field(description="Kurzor pro další stránku (chunk_id), null = konec")
    items: list[ChunkOut]


class ChangesOut(BaseModel):
    index_version: str
    since: str
    next_since: str | None = Field(description="updated_at posledního vráceného videa; pošli jako since v dalším volání")
    items: list[VideoOut]


class ExportFileOut(BaseModel):
    url: str
    size: int | None = None
    sha256: str | None = None


class ExportOut(BaseModel):
    index_version: str
    generated_at: str | None
    counts: dict | None
    embedding: dict | None
    files: dict[str, ExportFileOut] = Field(description="videos.jsonl.gz, segments.jsonl.gz, chunks.jsonl.gz, embeddings.npy, chunk_ids.npy, manifest.json")
    notes: str | None = None


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
        "gaps": parse_gaps(r["transcript_error"]), "updated_at": str(r["updated_at"]) if r["updated_at"] else None,
    }


def create_app(cfg: Config | None = None, db: Database | None = None, rate_limit: bool = True) -> FastAPI:
    cfg = cfg or load_config()
    db = db or open_database(cfg)

    from .mcp_server import build_mcp, http_app
    mcp = build_mcp(cfg, db)
    mcp_app = http_app(mcp)   # Starlette app s jedinou routou "/mcp"

    class _McpEndpoint:  # raw ASGI (třída, ne funkce – Starlette by funkci obalil jako request handler)
        async def __call__(self, scope, receive, send):
            # /mcp i /mcp/ bez přesměrování (za proxy by redirect vedl na http://)
            if scope["type"] == "http" and scope["path"].rstrip("/") == "/mcp":
                scope = {**scope, "path": "/mcp", "raw_path": b"/mcp"}
            await mcp_app(scope, receive, send)

    mcp_endpoint = _McpEndpoint()

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
    from starlette.routing import Route
    app.router.routes.append(Route("/mcp", endpoint=mcp_endpoint, methods=["GET", "POST", "DELETE"]))
    app.router.routes.append(Route("/mcp/", endpoint=mcp_endpoint, methods=["GET", "POST", "DELETE"]))
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "DELETE"],
                       allow_headers=["*"], expose_headers=["Mcp-Session-Id", "Retry-After", "X-Index-Version"])
    if rate_limit:
        app.middleware("http")(rate_limit_middleware)

    @app.middleware("http")
    async def add_index_version(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            try:
                response.headers["X-Index-Version"] = index_version(db)
            except Exception:
                pass
        return response

    api_key = os.environ.get("YTINDEX_API_KEY", "")

    def require_api_key(x_api_key: str | None = Header(None, alias="X-Api-Key")):
        """Hromadné endpointy: pokud je nastaven YTINDEX_API_KEY, vyžadují hlavičku X-Api-Key."""
        if api_key and x_api_key != api_key:
            raise HTTPException(401, "Chybí nebo nesouhlasí X-Api-Key")
        return True

    @lru_cache(maxsize=1)
    def embedder():
        if not cfg.gemini_api_key:
            return None
        try:
            from .embeddings import Embedder
            return Embedder(cfg, db)
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

    @app.get("/api/context", response_model=ContextOut)
    def api_context(q: str = Query(..., min_length=1, max_length=500, description="Otázka nebo téma"),
                    max_passages: int = Query(8, ge=1, le=20), context_seconds: int = Query(90, ge=20, le=600),
                    per_video: int = Query(3, ge=1, le=10)):
        """Rešerše: souvislé pasáže k tématu (pro psaní, RAG v jiném frontendu)."""
        ps = gather_context(db, q, embed_query=embed_query_fn(), max_passages=max_passages,
                            context_s=float(context_seconds), per_video=per_video)
        return {"question": q, "indexed_videos": db.stats()["videos_by_status"].get("done", 0),
                "passages": [p.__dict__ for p in ps]}

    @app.get("/api/ask", response_model=AskOut)
    def api_ask(q: str = Query(..., min_length=1, max_length=500), max_passages: int = Query(8, ge=1, le=20),
                context_seconds: int = Query(90, ge=20, le=600)):
        """Odpověď Gemini sestavená jen z přepisů, s citacemi [n] a odkazy na čas."""
        return rag_answer(cfg, db, q, embed_query=embed_query_fn(), max_passages=max_passages,
                          context_s=float(context_seconds))

    @app.get("/api/videos", response_model=VideoListOut)
    def videos(page: int = Query(1, ge=1), per_page: int = Query(50, ge=1, le=200),
               status: str | None = Query("done", pattern="^(pending|queued|done|missing|failed|all)$",
                                          description="Výchozí jen videa s transkriptem (done); all = všechna"),
               order: str = Query("published_at DESC")):
        st = None if status == "all" else status
        rows = db.list_videos(status=st, limit=per_page, offset=(page - 1) * per_page, order=order)
        return {"total": db.count_videos(status=st), "page": page, "per_page": per_page,
                "items": [_video_out(r) for r in rows]}

    @app.get("/api/videos/{video_id}/segments", response_model=list[SegmentOut])
    def video_segments(video_id: str):
        """Celý přepis videa jako segmenty s časem (zdroj pravdy)."""
        if not db.get_video(video_id):
            raise HTTPException(404, "Video nenalezeno")
        return [{"start": s["start"], "duration": s["duration"], "timestamp": format_timestamp(s["start"]),
                 "text": s["text"], "url": f"https://www.youtube.com/watch?v={video_id}&t={int(s['start'])}s"}
                for s in db.get_segments(video_id)]

    @app.get("/api/chunks", response_model=ChunkListOut, dependencies=[Depends(require_api_key)])
    def chunks(video_id: str | None = Query(None), after: int = Query(0, ge=0, description="kurzor: poslední chunk_id"),
               limit: int = Query(500, ge=1, le=2000),
               with_embeddings: bool = Query(False, description="přidat embedding (base64 float32)")):
        """Hromadné čtení úseků (stránkování kurzorem). Pro celý index je levnější /api/export."""
        rows = db.list_chunks(video_id=video_id, after=after, limit=limit, with_embeddings=with_embeddings)
        items = []
        for r in rows:
            if with_embeddings:
                r["embedding"] = embedding_b64(r["embedding"]) if r.get("embedding") else None
            else:
                r.pop("embedding", None)
            items.append(r)
        return {"index_version": index_version(db), "embedding_dim": db.embedding_dim,
                "next_after": rows[-1]["chunk_id"] if len(rows) == limit else None, "items": items}

    @app.get("/api/changes", response_model=ChangesOut, dependencies=[Depends(require_api_key)])
    def changes(since: str = Query(..., description="updated_at, od kterého vracet změny (ISO nebo 'YYYY-MM-DD HH:MM:SS', UTC)"),
                limit: int = Query(500, ge=1, le=2000)):
        """Videa změněná od zadaného času (nový/opravený přepis, metadata). Inkrementální synchronizace."""
        rows = db.videos_changed_since(since.replace("T", " ").replace("Z", ""), limit=limit)
        items = [_video_out(r) for r in rows]
        return {"index_version": index_version(db), "since": since,
                "next_since": items[-1]["updated_at"] if len(items) == limit else None, "items": items}

    export_gcs = os.environ.get("YTINDEX_EXPORT_GCS", "")          # gs://bucket/prefix s exportem
    export_public = os.environ.get("YTINDEX_EXPORT_PUBLIC", "") == "1"   # bucket je veřejný → přímé URL
    export_dir = Path(os.environ.get("YTINDEX_EXPORT_DIR", "data/export"))
    export_names = (*FILES, "manifest.json")

    @lru_cache(maxsize=1)
    def _manifest_cached(version_key: str):
        if export_gcs:
            return read_manifest_gcs(export_gcs)
        p = export_dir / "manifest.json"
        return __import__("json").loads(p.read_text(encoding="utf-8")) if p.exists() else None

    @app.get("/api/export", response_model=ExportOut)
    def export_manifest():
        """Manifest hromadného exportu: verze indexu a odkazy na soubory (JSONL + NumPy + SQLite).

        Soubory se stahují z /api/export/files/{name} (proxy z bucketu), nebo přímo z bucketu, je-li veřejný.
        """
        manifest = _manifest_cached(index_version(db))
        if manifest is None:
            raise HTTPException(404, "Export zatím neexistuje (spusť `ytindex export`)")
        if export_gcs and export_public:
            urls = public_urls(export_gcs)
        else:
            urls = {name: f"/api/export/files/{name}" for name in export_names}
        files = {name: {"url": url, "size": manifest.get("files", {}).get(name, {}).get("size"),
                        "sha256": manifest.get("files", {}).get(name, {}).get("sha256")} for name, url in urls.items()}
        gcs_db = os.environ.get("YTINDEX_GCS_DB", "")
        if gcs_db:
            files["index.sqlite"] = {"url": f"https://storage.googleapis.com/{gcs_db[5:]}" if export_public
                                     else "/api/export/files/index.sqlite"}
        return {"index_version": manifest.get("version") or index_version(db), "generated_at": manifest.get("generated_at"),
                "counts": manifest.get("counts"), "embedding": manifest.get("embedding"), "files": files,
                "notes": manifest.get("notes")}

    def _stream_blob(stream, size: int, name: str, media_type: str, range_header: str | None):
        """Streamování po 1 MB blocích; podporuje jednoduchý Range (bytes=a-b) pro navazování stahování."""
        start, end = 0, size - 1
        status = 200
        if range_header and range_header.startswith("bytes="):
            a, _, b = range_header[6:].partition("-")
            try:
                start = int(a) if a else max(0, size - int(b))
                end = int(b) if (b and a) else size - 1
            except ValueError:
                start, end = 0, size - 1
            if start >= size or end < start:
                stream.close()
                raise HTTPException(416, "Range mimo soubor", headers={"Content-Range": f"bytes */{size}"})
            end = min(end, size - 1)
            status = 206
        if start:
            stream.seek(start)
        remaining = end - start + 1

        def gen():
            left = remaining
            try:
                while left > 0:
                    block = stream.read(min(1 << 20, left))
                    if not block:
                        break
                    left -= len(block)
                    yield block
            finally:
                stream.close()

        headers = {"Content-Length": str(remaining), "Accept-Ranges": "bytes",
                   "Content-Disposition": f"attachment; filename={name}"}
        if status == 206:
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        return StreamingResponse(gen(), status_code=status, media_type=media_type, headers=headers)

    @app.get("/api/export/files/{name}", include_in_schema=False, dependencies=[Depends(require_api_key)])
    def export_file(name: str, request: Request):
        name = Path(name).name
        rng = request.headers.get("range")
        if name == "index.sqlite":
            gcs_db = os.environ.get("YTINDEX_GCS_DB", "")
            if gcs_db:
                bucket, _, path = gcs_db[5:].partition("/")
                stream, size = open_gcs_file(f"gs://{bucket}", path)
                if stream is None:
                    raise HTTPException(404, "Index v bucketu neexistuje")
                return _stream_blob(stream, size, name, "application/vnd.sqlite3", rng)
            return FileResponse(db.path, filename="index.sqlite") if getattr(db, "path", None) else Response(status_code=404)
        if name not in export_names:
            raise HTTPException(404, "Soubor neexistuje")
        if export_gcs:
            stream, size = open_gcs_file(export_gcs, name)
            if stream is None:
                raise HTTPException(404, "Soubor v bucketu neexistuje")
            return _stream_blob(stream, size, name, "application/octet-stream", rng)
        p = export_dir / name
        if not p.exists():
            raise HTTPException(404, "Soubor neexistuje")
        return FileResponse(p, filename=name)

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
        ico = WEB_DIR / "favicon.ico"
        return FileResponse(ico, media_type="image/x-icon") if ico.exists() else Response(status_code=204)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")

    return app


def app() -> FastAPI:  # pro `uvicorn ytindex.api:app --factory`
    return create_app()
