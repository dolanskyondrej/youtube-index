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

from .chunking import format_timestamp
from .config import Config, load_config
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
            "Použij search_videos pro dotaz uživatele (vrací videa, úryvky se shodou a odkazy s časem), "
            "get_transcript pro celý přepis konkrétního videa a list_videos pro přehled kanálu. "
            "Při odpovědi uváděj odkazy na čas ve videu (pole url u shody)."
        ),
    )

    @lru_cache(maxsize=1)
    def embed_fn():
        if not cfg.gemini_api_key or db.count_embedded()[0] == 0:
            return None
        try:
            from .embeddings import Embedder
            return Embedder(cfg).embed_query
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
    def list_videos(page: int = 1, per_page: int = 50, status: str | None = None) -> dict:
        """Seznam videí kanálu (nejnovější první). status: done | pending | missing | failed."""
        per_page = max(1, min(per_page, 200))
        rows = db.list_videos(status=status, limit=per_page, offset=(max(page, 1) - 1) * per_page)
        return {"total": db.count_videos(status=status), "page": page, "items": [
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
    kwargs = dict(streamable_http_path="/", stateless_http=True, json_response=True)
    if TransportSecuritySettings is not None:
        # veřejné API za reverzní proxy (Cloud Run) – Host hlavička není localhost
        kwargs["transport_security"] = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    return mcp.streamable_http_app(**kwargs)


def main() -> None:
    build_mcp().run(transport="stdio")


if __name__ == "__main__":
    main()
