"""CLI: ytindex init | ingest | embed | search | serve | stats"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from .config import CONFIG_FILE, Config, load_config, save_config

log = logging.getLogger("ytindex")


def _load_dotenv() -> None:
    """Minimalistické načtení .env (bez závislosti)."""
    if not os.path.exists(".env"):
        return
    for line in open(".env", encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def cmd_init(args) -> int:
    from .db import Database
    from .youtube import resolve_channel, resolve_channel_data_api

    cfg = load_config()
    value = args.channel
    if not value:
        print("Zadej YouTube kanál (handle @nazev, ID UC..., nebo URL kanálu):")
        value = input("> ").strip()
    if not value:
        print("Nebyl zadán kanál.", file=sys.stderr)
        return 2
    print(f"Zjišťuji kanál pro: {value} …")
    info = None
    if cfg.youtube_api_key:
        try:
            info = resolve_channel_data_api(value, cfg.youtube_api_key)
        except Exception as e:
            log.warning("Data API selhalo (%s), zkouším yt-dlp", e)
    if info is None:
        info = resolve_channel(value, proxy=cfg.proxy_url or None, cookies_file=cfg.transcripts.cookies_file or None)
    cfg.channel.input, cfg.channel.id, cfg.channel.name, cfg.channel.url = value, info.id, info.name, info.url
    if args.db:
        cfg.db_path = args.db
    path = save_config(cfg)
    db = Database(cfg.db_path, cfg.gemini.embedding_dim)
    db.upsert_channel(info.id, info.name, info.url, value)
    db.close()
    print(f"✔ Kanál: {info.name} ({info.id})\n✔ Konfigurace uložena do {path}\n✔ Databáze: {cfg.db_path}")
    print("Další krok: `ytindex ingest` (stažení seznamu videí a transkriptů), potom `ytindex serve`.")
    return 0


def _require_channel(cfg: Config) -> None:
    if not cfg.channel.id:
        print(f"Chybí {CONFIG_FILE} s kanálem – spusť nejdřív `ytindex init`.", file=sys.stderr)
        sys.exit(2)


def cmd_ingest(args) -> int:
    from .chunking import chunk_segments, merge_segments
    from .db import Database
    from .transcripts import get_transcript
    from .youtube import enrich_with_data_api, list_channel_videos

    cfg = load_config()
    _require_channel(cfg)
    db = Database(cfg.db_path, cfg.gemini.embedding_dim)
    db.upsert_channel(cfg.channel.id, cfg.channel.name, cfg.channel.url, cfg.channel.input)

    if not args.skip_listing:
        print(f"Stahuji seznam videí kanálu {cfg.channel.name} …")
        videos = list(list_channel_videos(cfg.channel.id, proxy=cfg.proxy_url or None,
                                          cookies_file=cfg.transcripts.cookies_file or None))
        print(f"  nalezeno {len(videos)} videí")
        if cfg.youtube_api_key:
            print("  doplňuji metadata z YouTube Data API …")
            try:
                enrich_with_data_api(videos, cfg.youtube_api_key)
            except Exception as e:
                log.warning("Data API: %s", e)
        with db.tx():
            for v in videos:
                db.upsert_video({"id": v.id, "channel_id": cfg.channel.id, "title": v.title,
                                 "description": v.description, "published_at": v.published_at,
                                 "duration_s": v.duration_s, "view_count": v.view_count,
                                 "thumbnail_url": v.thumbnail_url})

    statuses = ["pending"] + (["failed", "missing"] if args.retry else [])
    todo = []
    for st in statuses:
        todo += db.list_videos(status=st, limit=10**6, order="published_at DESC")
    if args.limit:
        todo = todo[:args.limit]
    providers = args.providers.split(",") if args.providers else None
    print(f"Transkripty: {len(todo)} videí ke zpracování (providery: {providers or cfg.transcripts.providers})")
    disabled: set[str] = set() if cfg.gemini_api_key else {"gemini"}
    ok = miss = fail = 0
    for i, v in enumerate(todo, 1):
        if disabled >= set(providers or cfg.transcripts.providers):
            print("Všechny providery jsou blokované – končím. Spusť ingest z jiné IP / s proxy / s GEMINI_API_KEY.")
            break
        try:
            res = get_transcript(v["id"], cfg, v["duration_s"], providers, disabled)
        except Exception as e:
            fail += 1
            db.set_transcript_status(v["id"], "failed", error=str(e))
            print(f"[{i}/{len(todo)}] ✖ {v['id']} {v['title'][:60]} – {str(e)[:80]}")
            continue
        if not res:
            miss += 1
            db.set_transcript_status(v["id"], "missing")
            print(f"[{i}/{len(todo)}] – {v['id']} {v['title'][:60]} – bez transkriptu")
            continue
        segs = merge_segments(res.segments)
        chunks = chunk_segments(segs, cfg.chunking.target_chars, cfg.chunking.max_seconds,
                                cfg.chunking.overlap_segments)
        db.store_transcript(v["id"], segs, chunks, res.source, res.lang)
        ok += 1
        print(f"[{i}/{len(todo)}] ✔ {v['id']} {v['title'][:60]} – {res.source}/{res.lang} {len(segs)} seg → {len(chunks)} chunků")
        time.sleep(cfg.transcripts.request_delay)
    print(f"Hotovo: {ok} ok, {miss} bez transkriptu, {fail} chyb.")
    if not args.no_embed and cfg.gemini_api_key:
        return cmd_embed(args, db=db)
    elif not cfg.gemini_api_key:
        print("Pozn.: GEMINI_API_KEY není nastaven – sémantické hledání (embeddingy) přeskočeno, funguje fulltext.")
    db.close()
    return 0


def cmd_embed(args, db=None) -> int:
    from .db import Database
    from .embeddings import Embedder

    cfg = load_config()
    db = db or Database(cfg.db_path, cfg.gemini.embedding_dim)
    if not db.vec_available:
        print("sqlite-vec není k dispozici – vektorové hledání vypnuto.", file=sys.stderr)
        return 1
    emb = Embedder(cfg)
    done, total = db.count_embedded()
    print(f"Embeddingy: {done}/{total} chunků hotovo, model {cfg.gemini.embedding_model} ({cfg.gemini.embedding_dim}d)")
    while True:
        rows = db.pending_embeddings(limit=256)
        if not rows:
            break
        vecs = emb.embed_documents([r["text"] for r in rows])
        db.store_embeddings([(r["id"], v) for r, v in zip(rows, vecs)])
        done += len(rows)
        print(f"  {done}/{total}", end="\r", flush=True)
    print(f"\n✔ Embeddingy kompletní ({done}/{total}).")
    return 0


def cmd_search(args) -> int:
    from .db import Database
    from .search import search

    cfg = load_config()
    db = Database(cfg.db_path, cfg.gemini.embedding_dim)
    embed = None
    if args.mode != "fts" and cfg.gemini_api_key and db.count_embedded()[0]:
        from .embeddings import Embedder
        embed = Embedder(cfg).embed_query
    hits = search(db, args.query, limit=args.limit, mode=args.mode, embed_query=embed)
    for h in hits:
        print(f"{h.title}\n  {h.match.url}  [{h.match.timestamp}]  (shod: {h.match_count}, skóre {h.score:.4f})")
        print(f"  {h.match.text[:200]}\n")
    if not hits:
        print("Nic nenalezeno.")
    return 0


def cmd_serve(args) -> int:
    import uvicorn
    from .api import create_app

    cfg = load_config()
    _require_channel(cfg)
    uvicorn.run(create_app(cfg), host=args.host, port=args.port, log_level="info")
    return 0


def cmd_stats(args) -> int:
    import json
    from .db import Database

    cfg = load_config()
    db = Database(cfg.db_path, cfg.gemini.embedding_dim)
    print(json.dumps({"channel": cfg.channel.__dict__, **db.stats()}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ytindex", description="Prohledávatelný index transkriptů YouTube kanálu")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="zeptá se na kanál, uloží ytindex.toml a založí DB")
    s.add_argument("--channel", help="handle/@nazev, UC id nebo URL kanálu (jinak interaktivně)")
    s.add_argument("--db", help="cesta k SQLite souboru (výchozí data/index.sqlite)")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("ingest", help="stáhne seznam videí + transkripty, naindexuje, (volitelně) embeduje")
    s.add_argument("--limit", type=int, help="zpracovat jen N videí (testování)")
    s.add_argument("--retry", action="store_true", help="znovu zkusit i videa se stavem failed/missing")
    s.add_argument("--skip-listing", action="store_true", help="neaktualizovat seznam videí")
    s.add_argument("--providers", help="pořadí providerů, např. captions,gemini")
    s.add_argument("--no-embed", action="store_true", help="nepočítat embeddingy")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("embed", help="dopočítá chybějící embeddingy (Gemini)")
    s.set_defaults(fn=cmd_embed)

    s = sub.add_parser("search", help="vyhledá z příkazové řádky")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--mode", choices=["hybrid", "fts", "vector"], default="hybrid")
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("serve", help="spustí REST API + webové rozhraní")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("stats", help="stav indexu")
    s.set_defaults(fn=cmd_stats)
    return p


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
