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
from .db import channel_dict, open_database
from .export import FILES, embedding_b64, index_version, iter_blob_ranges, open_gcs_file, parse_gaps, public_urls, read_manifest_gcs
from .ratelimit import rate_limit_middleware
from .research import answer as rag_answer, gather_context
from .search import hit_to_dict, search

log = logging.getLogger(__name__)
WEB_DIR = Path(__file__).parent / "web"

API_DESCRIPTION = """
Vyhledávání v přepisech všech videí YouTube kanálu a hromadný přístup k datům (přepisy, úseky, embeddingy).
Podrobný návod: `docs/API.md` v repozitáři. MCP server pro AI agenty běží na `/mcp`.

**Kanály:** index může obsahovat více YouTube kanálů. Každé video, výsledek hledání, pasáž i chunk nese
`channel` / `channel_id`; seznam kanálů dává `/api/channels`, parametr `channel=<UC id|@handle|primary>` omezí
libovolný endpoint na jeden kanál (web používá `primary`). Bez parametru se pracuje nad všemi kanály.

**Datový model:** `video` (metadata, `transcript_status`, `gaps`, `updated_at`) → `segments` (přepis s časy, zdroj
pravdy) → `chunks` (úseky ~450 znaků pro hledání, každý s embeddingem `gemini-embedding-001`, 768 d, normalizovaný,
cosine; dotazy embeduj s `task_type=RETRIEVAL_QUERY`). Odkaz na místo ve videu:
`https://www.youtube.com/watch?v={video_id}&t={int(start)}s`.

**Verze indexu:** každá odpověď `/api/*` nese hlavičku `X-Index-Version`. Změna hodnoty = nová data.

**Autentizace:** `/api/export/files/*`, `/api/chunks`, `/api/changes` vyžadují hlavičku `X-Api-Key`.

**Limity:** per IP, při překročení `429` + `Retry-After` (s).

**Stažení celého obsahu:** `GET /api/export` → při změně `index_version` stáhnout soubory z manifestu
(`videos/segments/chunks.jsonl.gz`, `embeddings.npy` float32 [N,768] + `chunk_ids.npy`, volitelně `index.sqlite`);
soubory se streamují a podporují `Range`.

**Inkrementálně:** `GET /api/changes?since=<updated_at>` → pro každé změněné video `GET /api/videos/{id}`
(přepis) a `GET /api/chunks?video_id=…&with_embeddings=1`; staré chunky videa nahraď, ulož max(`updated_at`).
"""


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
    channel: ChannelRefOut | None = Field(None, description="Kanál, ze kterého video pochází")
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
    channel: str | None = Field(None, description="ID kanálu, na který bylo hledání omezeno (null = všechny)")
    total: int
    results: list[VideoHitOut]


class PassageOut(BaseModel):
    n: int = Field(description="Číslo citace")
    video_id: str
    channel: ChannelRefOut | None = None
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
    channel: str | None = None
    indexed_videos: int
    passages: list[PassageOut]


class AskOut(BaseModel):
    question: str
    answer: str | None
    model: str | None = None
    error: str | None = None
    citations: list[PassageOut]


class ChannelRefOut(BaseModel):
    id: str = Field(description="YouTube channel id (UC…)")
    name: str | None
    url: str | None = Field(description="URL kanálu (handle, např. https://www.youtube.com/@kanal)")
    handle: str | None


class ChannelStatsOut(ChannelRefOut):
    videos: int
    done: int
    hours: float


class ChannelsOut(BaseModel):
    primary: str | None = Field(description="ID kanálu, který je výchozí pro web (konfigurace instance)")
    index_version: str
    channels: list[ChannelStatsOut]


class GapOut(BaseModel):
    start: int
    end: int
    reason: str


class VideoOut(BaseModel):
    video_id: str
    channel_id: str | None = None
    channel: ChannelRefOut | None = Field(None, description="Kanál, ze kterého video pochází")
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
    channel_id: str | None = None
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
    segments: list[SegmentOut] = Field(description="Celý přepis s časy (zdroj pravdy)")
    chunks: list[ChunkOut] | None = Field(None, description="Úseky pro hledání; jen s `with_chunks=1` (embeddingy s `with_embeddings=1` + X-Api-Key)")
    embedding_dim: int | None = None
    index_version: str | None = None


class VideoListOut(BaseModel):
    total: int
    page: int
    per_page: int
    items: list[VideoOut]


class ChannelOut(BaseModel):
    id: str
    name: str
    url: str
    handle: str | None = None
    stats: dict
    channels: list[ChannelStatsOut] = Field(default_factory=list, description="Všechny kanály v indexu")


def _video_out(r, channels: dict | None = None) -> dict:
    return {
        "video_id": r["id"], "channel_id": r["channel_id"],
        "channel": (channels or {}).get(r["channel_id"]),
        "title": r["title"], "url": f"https://www.youtube.com/watch?v={r['id']}",
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
        description=API_DESCRIPTION,
        openapi_tags=[
            {"name": "hledání", "description": "Hybridní hledání v přepisech (fulltext + význam), výsledky po videích s odkazem na čas."},
            {"name": "videa", "description": "Metadata videí a celé přepisy (segmenty s časem)."},
            {"name": "hromadný přístup", "description": "Stažení celého obsahu a inkrementální synchronizace. Vyžaduje `X-Api-Key`."},
            {"name": "rešerše", "description": "Souvislé pasáže k tématu a odpovědi s citacemi (RAG)."},
        ],
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

    def channels_map() -> dict:
        return {c["id"]: channel_dict(c) for c in db.list_channels()}

    def resolve_channel(value: str | None) -> str | None:
        """Parametr channel: UC id, @handle nebo 'primary'; None = bez filtru. Neznámý → 404."""
        if not value:
            return None
        if value == "primary":
            return cfg.channel.id or (db.get_channel() or {"id": None})["id"]
        for c in db.list_channels():
            if value in (c["id"], c["handle"], (c["handle"] or "").lstrip("@")):
                return c["id"]
        raise HTTPException(404, f"Kanál {value!r} není v indexu")

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

    @app.get("/api/channel", response_model=ChannelOut, tags=["videa"], summary="Primární kanál a stav indexu")
    def channel():
        """Primární kanál instance (ten, který web filtruje) + celkové statistiky indexu + seznam všech kanálů."""
        ch = db.get_channel(cfg.channel.id or None)
        if not ch:
            raise HTTPException(404, "Kanál ještě nebyl inicializován (spusť `ytindex init`)")
        return {**channel_dict(ch), "stats": db.stats(), "channels": db.channel_stats()}

    @app.get("/api/channels", response_model=ChannelsOut, tags=["videa"], summary="Všechny kanály v indexu")
    def channels_list():
        """Kanály v indexu s počty videí; `primary` je kanál, na který se omezuje web. Hodnoty `id` (nebo `handle`)
        se používají v parametru `channel` ostatních endpointů."""
        return {"primary": cfg.channel.id or None, "index_version": index_version(db), "channels": db.channel_stats()}

    @app.get("/api/search", response_model=SearchOut, tags=["hledání"], summary="Hybridní hledání")
    def api_search(
        q: str = Query(..., min_length=1, max_length=300, description="Hledaný výraz; fráze v uvozovkách"),
        limit: int = Query(20, ge=1, le=100),
        mode: str = Query("hybrid", pattern="^(hybrid|fts|vector)$"),
        per_video: int = Query(3, ge=1, le=10, description="Kolik shod vrátit pro každé video"),
        channel: str | None = Query(None, description="Omezit na kanál: UC id, @handle nebo 'primary'. Bez parametru všechny kanály."),
    ):
        cid = resolve_channel(channel)
        hits = search(db, q, limit=limit, mode=mode, embed_query=embed_query_fn(), per_video_matches=per_video, channel_id=cid)
        effective = mode if (mode == "fts" or embed_query_fn() is not None) else "fts"
        return {"query": q, "mode": effective, "channel": cid, "total": len(hits), "results": [hit_to_dict(h) for h in hits]}

    @app.get("/api/context", response_model=ContextOut, tags=["rešerše"], summary="Pasáže k tématu")
    def api_context(q: str = Query(..., min_length=1, max_length=500, description="Otázka nebo téma"),
                    max_passages: int = Query(8, ge=1, le=20), context_seconds: int = Query(90, ge=20, le=600),
                    per_video: int = Query(3, ge=1, le=10),
                    channel: str | None = Query(None, description="Omezit na kanál (UC id, @handle, 'primary')")):
        """Rešerše: souvislé pasáže k tématu (pro psaní, RAG v jiném frontendu)."""
        cid = resolve_channel(channel)
        ps = gather_context(db, q, embed_query=embed_query_fn(), max_passages=max_passages,
                            context_s=float(context_seconds), per_video=per_video, channel_id=cid)
        return {"question": q, "channel": cid, "indexed_videos": db.count_videos(channel_id=cid, status="done"),
                "passages": [p.__dict__ for p in ps]}

    @app.get("/api/ask", response_model=AskOut, tags=["rešerše"], summary="Odpověď s citacemi (Gemini)")
    def api_ask(q: str = Query(..., min_length=1, max_length=500), max_passages: int = Query(8, ge=1, le=20),
                context_seconds: int = Query(90, ge=20, le=600),
                channel: str | None = Query(None, description="Omezit na kanál (UC id, @handle, 'primary')")):
        """Odpověď Gemini sestavená jen z přepisů, s citacemi [n] a odkazy na čas."""
        return rag_answer(cfg, db, q, embed_query=embed_query_fn(), max_passages=max_passages,
                          context_s=float(context_seconds), channel_id=resolve_channel(channel))

    @app.get("/api/videos", response_model=VideoListOut, tags=["videa"], summary="Seznam videí")
    def videos(page: int = Query(1, ge=1), per_page: int = Query(50, ge=1, le=200),
               status: str | None = Query("done", pattern="^(pending|queued|done|missing|failed|all)$",
                                          description="Výchozí jen videa s transkriptem (done); all = všechna"),
               order: str = Query("published_at DESC"),
               channel: str | None = Query(None, description="Omezit na kanál (UC id, @handle, 'primary')")):
        st = None if status == "all" else status
        cid = resolve_channel(channel)
        rows = db.list_videos(channel_id=cid, status=st, limit=per_page, offset=(page - 1) * per_page, order=order)
        chans = channels_map()
        return {"total": db.count_videos(channel_id=cid, status=st), "page": page, "per_page": per_page,
                "items": [_video_out(r, chans) for r in rows]}

    @app.get("/api/videos/{video_id}/segments", response_model=list[SegmentOut], tags=["videa"], summary="Přepis videa")
    def video_segments(video_id: str):
        """Celý přepis videa jako segmenty s časem (zdroj pravdy)."""
        if not db.get_video(video_id):
            raise HTTPException(404, "Video nenalezeno")
        return [{"start": s["start"], "duration": s["duration"], "timestamp": format_timestamp(s["start"]),
                 "text": s["text"], "url": f"https://www.youtube.com/watch?v={video_id}&t={int(s['start'])}s"}
                for s in db.get_segments(video_id)]

    @app.get("/api/chunks", response_model=ChunkListOut, dependencies=[Depends(require_api_key)],
             tags=["hromadný přístup"], summary="Úseky s embeddingy (kurzor)")
    def chunks(video_id: str | None = Query(None), after: int = Query(0, ge=0, description="kurzor: poslední chunk_id"),
               limit: int = Query(500, ge=1, le=2000),
               with_embeddings: bool = Query(False, description="přidat embedding (base64 float32)"),
               channel: str | None = Query(None, description="Omezit na kanál (UC id, @handle, 'primary')")):
        """Úseky seřazené podle `chunk_id`, stránkování kurzorem: další stránka = `after=next_after`, konec = `next_after` null.
        `with_embeddings=1` přidá `embedding` jako base64 z 768 × float32 little‑endian (null, pokud ještě nebyl spočítán).
        Pro celý index použij `/api/export` (NumPy je ~5× menší než JSON)."""
        rows = db.list_chunks(video_id=video_id, after=after, limit=limit, with_embeddings=with_embeddings,
                              channel_id=resolve_channel(channel))
        items = []
        for r in rows:
            if with_embeddings:
                r["embedding"] = embedding_b64(r["embedding"]) if r.get("embedding") else None
            else:
                r.pop("embedding", None)
            items.append(r)
        return {"index_version": index_version(db), "embedding_dim": db.embedding_dim,
                "next_after": rows[-1]["chunk_id"] if len(rows) == limit else None, "items": items}

    @app.get("/api/changes", response_model=ChangesOut, dependencies=[Depends(require_api_key)],
             tags=["hromadný přístup"], summary="Změněná videa od času")
    def changes(since: str = Query(..., description="updated_at, od kterého vracet změny (ISO nebo 'YYYY-MM-DD HH:MM:SS', UTC)"),
                limit: int = Query(500, ge=1, le=2000),
                channel: str | None = Query(None, description="Omezit na kanál (UC id, @handle, 'primary')")):
        """Videa s `updated_at` > `since`, nejstarší první. Opakuj se `since=next_since`, dokud není null; pak ulož
        poslední `updated_at`. Změna přepisu = nové `chunk_id` pro celé video (staré úseky videa smaž a nahraď).
        Smazaná videa se tu neobjeví, porovnej občas `/api/videos?status=all`."""
        rows = db.videos_changed_since(since.replace("T", " ").replace("Z", ""), limit=limit,
                                       channel_id=resolve_channel(channel))
        chans = channels_map()
        items = [_video_out(r, chans) for r in rows]
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

    @app.get("/api/export", response_model=ExportOut, tags=["hromadný přístup"], summary="Manifest exportu")
    def export_manifest():
        """Verze indexu, počty, parametry embeddingů a soubory exportu (`url`, `size`, `sha256`): `videos.jsonl.gz`,
        `segments.jsonl.gz`, `chunks.jsonl.gz`, `embeddings.npy` (float32 [N,768]), `chunk_ids.npy` (int64 [N], stejné
        pořadí), `manifest.json`, `index.sqlite`. Soubory stahuj z `url` s hlavičkou `X-Api-Key`; streamují se, velikost
        je v `X-Content-Size`, podporují `Range` (206). Postup: ulož `index_version`, stahuj jen při změně."""
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

    def _stream_blob(blob, size: int, name: str, media_type: str, range_header: str | None):
        """Streamování po blocích (samostatné range requesty do GCS); podporuje HTTP Range pro navazování."""
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
                raise HTTPException(416, "Range mimo soubor", headers={"Content-Range": f"bytes */{size}"})
            end = min(end, size - 1)
            status = 206
        # bez Content-Length: Cloud Run omezuje nestreamované odpovědi na 32 MB, chunked streaming limit nemá
        headers = {"Accept-Ranges": "bytes", "Content-Disposition": f"attachment; filename={name}",
                   "X-Content-Size": str(end - start + 1)}
        if status == 206:
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        return StreamingResponse(iter_blob_ranges(blob, start, end), status_code=status, media_type=media_type, headers=headers)

    @app.get("/api/export/files/{name}", dependencies=[Depends(require_api_key)], tags=["hromadný přístup"],
             summary="Stažení souboru exportu", response_class=StreamingResponse,
             responses={200: {"content": {"application/octet-stream": {}}, "description": "Soubor (chunked stream)"},
                        206: {"description": "Část souboru podle hlavičky Range"}, 401: {"description": "Chybí X-Api-Key"},
                        404: {"description": "Soubor neexistuje"}, 416: {"description": "Range mimo soubor"}})
    def export_file(name: str, request: Request):
        """`name` je jeden ze souborů z manifestu (`/api/export`) nebo `index.sqlite`. Odpověď je chunked stream bez
        `Content-Length`; velikost v `X-Content-Size`. `Range: bytes=a-b` vrací 206 s `Content-Range` (navazování)."""
        name = Path(name).name
        rng = request.headers.get("range")
        if name == "index.sqlite":
            gcs_db = os.environ.get("YTINDEX_GCS_DB", "")
            if gcs_db:
                bucket, _, path = gcs_db[5:].partition("/")
                blob, size = open_gcs_file(f"gs://{bucket}", path)
                if blob is None:
                    raise HTTPException(404, "Index v bucketu neexistuje")
                return _stream_blob(blob, size, name, "application/vnd.sqlite3", rng)
            return FileResponse(db.path, filename="index.sqlite") if getattr(db, "path", None) else Response(status_code=404)
        if name not in export_names:
            raise HTTPException(404, "Soubor neexistuje")
        if export_gcs:
            blob, size = open_gcs_file(export_gcs, name)
            if blob is None:
                raise HTTPException(404, "Soubor v bucketu neexistuje")
            return _stream_blob(blob, size, name, "application/octet-stream", rng)
        p = export_dir / name
        if not p.exists():
            raise HTTPException(404, "Soubor neexistuje")
        return FileResponse(p, filename=name)

    @app.get("/api/videos/{video_id}", response_model=VideoDetailOut, tags=["videa"], summary="Vše o jednom videu")
    def video_detail(video_id: str,
                     with_chunks: bool = Query(False, description="přidat úseky pro hledání (chunks)"),
                     with_embeddings: bool = Query(False, description="k úsekům přidat embedding (base64 float32); vyžaduje X-Api-Key"),
                     x_api_key: str | None = Header(None, alias="X-Api-Key")):
        """Kompletní informace k videu v jednom volání: metadata, kanál, mezery v přepisu (`gaps`), celý přepis
        (`segments`) a volitelně úseky (`chunks`) včetně embeddingů. Klient, který si zrcadlí jedno video, nepotřebuje
        nic dalšího; `index_version` říká, ke které verzi indexu data patří."""
        r = db.get_video(video_id)
        if not r:
            raise HTTPException(404, "Video nenalezeno")
        if with_embeddings and api_key and x_api_key != api_key:
            raise HTTPException(401, "Embeddingy vyžadují X-Api-Key")
        segs = [{"start": s["start"], "duration": s["duration"], "timestamp": format_timestamp(s["start"]),
                 "text": s["text"], "url": f"https://www.youtube.com/watch?v={video_id}&t={int(s['start'])}s"}
                for s in db.get_segments(video_id)]
        out = {**_video_out(r, channels_map()), "description": r["description"] or "", "segments": segs,
               "index_version": index_version(db)}
        if with_chunks or with_embeddings:
            rows = db.list_chunks(video_id=video_id, limit=100_000, with_embeddings=with_embeddings)
            for c in rows:
                if with_embeddings:
                    c["embedding"] = embedding_b64(c["embedding"]) if c.get("embedding") else None
                else:
                    c.pop("embedding", None)
            out["chunks"] = rows
            out["embedding_dim"] = db.embedding_dim
        return out

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
