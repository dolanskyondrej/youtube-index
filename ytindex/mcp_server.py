"""MCP server – dává AI asistentům (Claude Desktop / Claude Code / claude.ai) přístup k hledání v kanálu.

Spuštění:
  ytindex mcp                      # stdio (lokální konfigurace v Claude Desktop / Claude Code)
  ytindex serve                    # stejné nástroje jako Streamable HTTP na /mcp (vzdálené MCP)
"""

from __future__ import annotations

from functools import lru_cache

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer
    from mcp.server.transport_security import TransportSecuritySettings
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer  # type: ignore
    TransportSecuritySettings = None  # type: ignore

from dataclasses import asdict

from .chunking import format_timestamp
from .config import Config, load_config
from .research import answer as rag_answer, gather_context
from .search import hit_to_dict, search


def build_mcp(cfg: Config | None = None, db=None) -> MCPServer:
    cfg = cfg or load_config()
    from .db import open_database
    db = db or open_database(cfg)
    name = cfg.channel.name or "YouTube kanál"

    mcp = MCPServer(
        "ytindex",
        instructions=(
            f"Vyhledávání v transkriptech všech videí YouTube kanálu „{name}“. "
            "search_videos = klasické hledání (videa, úryvky se shodou, odkazy s časem). "
            "research_topic = pro psaní a rešerši: k otázce vrátí delší souvislé pasáže toho, co k tématu zaznělo; "
            "z nich sestav odpověď, cituj doslovně a vždy uváděj odkaz s časem (pole url). "
            "ask_channel = hotová odpověď s citacemi sestavená Gemini (když nechceš syntetizovat sám). "
            "get_transcript = celý přepis videa nebo jeho úsek, list_videos = přehled kanálu."
        ),
    )

    @lru_cache(maxsize=1)
    def embed_fn():
        if not cfg.gemini_api_key or db.count_embedded()[0] == 0:
            return None
        try:
            from .embeddings import Embedder
            return Embedder(cfg, db).embed_query
        except Exception:
            return None

    @mcp.tool()
    def search_videos(query: str, limit: int = 10, mode: str = "hybrid", per_video: int = 3) -> dict:
        """Prohledá transkripty všech videí kanálu. Vrací videa seřazená podle relevance; u každého
        název, URL, náhledovku a nejlepší úryvky se zvýrazněnou shodou (<mark>) a odkazem na přesný čas.
        mode: hybrid (fulltext + sémantické, výchozí) | fts (jen fulltext, přesné fráze v uvozovkách) | vector.
        """
        hits = search(db, query, limit=max(1, min(limit, 50)), mode=mode, embed_query=embed_fn(),
                      per_video_matches=max(1, min(per_video, 10)))
        return {"query": query, "total": len(hits), "results": [hit_to_dict(h) for h in hits]}

    @mcp.tool()
    def research_topic(question: str, max_passages: int = 8, context_seconds: int = 90, per_video: int = 3) -> dict:
        """Rešerše pro psaní: k otázce/tématu vrátí nejrelevantnější SOUVISLÉ pasáže z přepisů (každá shoda rozšířená
        o okolní řeč, výchozí ±45 s), střídavě z různých videí, s číslem citace, názvem videa, časem a odkazem.
        Použij, když chceš vědět, co k tématu zaznělo, jak dotyčný uvažuje a formuluje – a pak to citovat.
        Mluvčí není rozlišen (ve studiu bývají hosté), posuzuj z kontextu.
        """
        ps = gather_context(db, question, embed_query=embed_fn(), max_passages=max(1, min(max_passages, 20)),
                            context_s=float(max(20, min(context_seconds, 600))), per_video=max(1, min(per_video, 10)))
        stats = db.stats()
        return {"question": question, "indexed_videos": stats["videos_by_status"].get("done", 0),
                "passages": [asdict(p) for p in ps],
                "hint": "Cituj doslovně z pole text a ke každé citaci uveď url (odkaz na čas ve videu)."}

    @mcp.tool()
    def ask_channel(question: str, max_passages: int = 8, context_seconds: int = 90) -> dict:
        """Odpověď na otázku sestavená Gemini výhradně z přepisů (RAG) s citacemi [n]; pole citations obsahuje
        pasáže s odkazy na čas. Hodí se pro rychlé „co k tomu Urza říkal“; pro vlastní syntézu použij research_topic.
        """
        return rag_answer(cfg, db, question, embed_query=embed_fn(), max_passages=max(1, min(max_passages, 20)),
                          context_s=float(max(20, min(context_seconds, 600))))

    @mcp.tool()
    def get_transcript(video_id: str, start_s: float | None = None, end_s: float | None = None) -> dict:
        """Vrátí přepis videa (volitelně jen úsek start_s–end_s v sekundách) jako segmenty s časem a odkazem."""
        v = db.get_video(video_id)
        if not v:
            return {"error": f"Video {video_id} není v indexu"}
        segs = []
        for s in db.get_segments(video_id):
            if start_s is not None and s["start"] + s["duration"] < start_s:
                continue
            if end_s is not None and s["start"] > end_s:
                break
            segs.append({"timestamp": format_timestamp(s["start"]), "start": s["start"], "text": s["text"],
                         "url": f"https://www.youtube.com/watch?v={video_id}&t={int(s['start'])}s"})
        return {"video_id": video_id, "title": v["title"], "url": f"https://www.youtube.com/watch?v={video_id}",
                "published_at": v["published_at"], "duration_s": v["duration_s"],
                "transcript_status": v["transcript_status"], "segments": segs}

    @mcp.tool()
    def list_videos(page: int = 1, per_page: int = 50, status: str = "done") -> dict:
        """Seznam videí kanálu (nejnovější první). status: done (výchozí, jen videa s přepisem) | pending | queued | missing | failed | all."""
        per_page = max(1, min(per_page, 200))
        st = None if status == "all" else status
        rows = db.list_videos(status=st, limit=per_page, offset=(max(page, 1) - 1) * per_page)
        return {"total": db.count_videos(status=st), "page": page, "items": [
            {"video_id": r["id"], "title": r["title"], "url": f"https://www.youtube.com/watch?v={r['id']}",
             "published_at": r["published_at"], "duration_s": r["duration_s"],
             "transcript_status": r["transcript_status"]} for r in rows]}

    @mcp.tool()
    def channel_info() -> dict:
        """Název a URL indexovaného kanálu + statistiky indexu (počet videí, transkriptů, hodin)."""
        ch = db.get_channel(cfg.channel.id or None)
        return {"id": ch["id"] if ch else None, "name": ch["name"] if ch else name,
                "url": ch["url"] if ch else cfg.channel.url, "stats": db.stats()}

    return mcp


def http_app(mcp: MCPServer):
    """Starlette aplikace Streamable HTTP (bez session state – vhodné pro Cloud Run s více instancemi)."""
    kwargs = dict(streamable_http_path="/mcp", stateless_http=True, json_response=True)
    if TransportSecuritySettings is not None:
        # veřejné API za reverzní proxy (Cloud Run) – Host hlavička není localhost
        kwargs["transport_security"] = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    return mcp.streamable_http_app(**kwargs)


def main() -> None:
    build_mcp().run(transport="stdio")


if __name__ == "__main__":
    main()
