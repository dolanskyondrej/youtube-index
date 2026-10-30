"""CLI: ytindex init | ingest | embed | search | serve | stats"""

from __future__ import annotations

import argparse
import json
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
    from .db import open_database
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
    db = open_database(cfg)
    db.upsert_channel(info.id, info.name, info.url, value)
    db.close()
    print(f"✔ Kanál: {info.name} ({info.id})\n✔ Konfigurace uložena do {path}\n✔ Databáze: {db.backend} ({os.environ.get('DATABASE_URL', cfg.db_path).split('@')[-1]})")
    print("Další krok: `ytindex ingest` (stažení seznamu videí a transkriptů), potom `ytindex serve`.")
    return 0


def _require_channel(cfg: Config) -> None:
    if not cfg.channel.id:
        print(f"Chybí {CONFIG_FILE} s kanálem – spusť nejdřív `ytindex init`.", file=sys.stderr)
        sys.exit(2)


def _ensure_channel(cfg: Config, db) -> None:
    """V cloudu není ytindex.toml: kanál se vezme z DB, nebo z env YTINDEX_CHANNEL (a uloží do DB)."""
    from .youtube import resolve_channel

    if not cfg.channel.id:
        ch = db.get_channel()
        if ch:
            cfg.channel.id, cfg.channel.name, cfg.channel.url, cfg.channel.input = ch["id"], ch["name"], ch["url"], ch["input"] or ""
        elif os.environ.get("YTINDEX_CHANNEL"):
            value = os.environ["YTINDEX_CHANNEL"]
            info = resolve_channel(value, proxy=cfg.proxy_url or None)
            cfg.channel.input, cfg.channel.id, cfg.channel.name, cfg.channel.url = value, info.id, info.name, info.url
    if not cfg.channel.id:
        print(f"Chybí kanál: spusť `ytindex init`, nebo nastav YTINDEX_CHANNEL.", file=sys.stderr)
        sys.exit(2)
    db.upsert_channel(cfg.channel.id, cfg.channel.name, cfg.channel.url, cfg.channel.input)


def run_ingest(cfg: Config, db, limit: int | None = None, retry: bool = False, skip_listing: bool = False,
               providers: list[str] | None = None, shard: tuple[int, int] | None = None, embed: bool = True,
               ids: list[str] | None = None, checkpoint=None, checkpoint_every: int = 5) -> int:
    """checkpoint: callable() volaný po každých `checkpoint_every` zpracovaných videích (např. upload do GCS)."""
    from .chunking import chunk_segments, merge_segments
    from .transcripts import get_transcript
    from .youtube import enrich_with_data_api, list_channel_videos

    if not skip_listing:
        print(f"Stahuji seznam videí kanálu {cfg.channel.name} …")
        videos = list(list_channel_videos(cfg.channel.id, proxy=cfg.proxy_url or None,
                                          cookies_file=cfg.transcripts.cookies_file or None))
        print(f"  nalezeno {len(videos)} videí")
        if cfg.youtube_api_key:
            print("  doplňuji metadata z YouTube Data API …")
            try:
                enrich_with_data_api(videos, cfg.youtube_api_key)
            except Exception as e:
                log.warning("Data API: %s", str(e)[:200])
        with db.tx():
            for pos, v in enumerate(videos):
                db.upsert_video({"id": v.id, "channel_id": cfg.channel.id, "title": v.title, "position": pos,
                                 "description": v.description, "published_at": v.published_at,
                                 "duration_s": v.duration_s, "view_count": v.view_count,
                                 "thumbnail_url": v.thumbnail_url})
        if cfg.youtube_api_key:
            # výpis z cloudu bývá neúplný (YouTube blokuje další stránky) – doplň metadata všem videím bez data
            from .youtube import VideoInfo
            missing = [VideoInfo(id=r["id"], title=r["title"]) for r in db.list_videos(limit=10**6)
                       if not r["published_at"]]
            if missing:
                print(f"  doplňuji datum publikace u {len(missing)} videí bez metadat …")
                try:
                    enrich_with_data_api(missing, cfg.youtube_api_key)
                    with db.tx():
                        for v in missing:
                            if v.published_at:
                                db.upsert_video({"id": v.id, "channel_id": cfg.channel.id, "title": v.title,
                                                 "description": v.description, "published_at": v.published_at,
                                                 "duration_s": v.duration_s, "view_count": v.view_count,
                                                 "thumbnail_url": v.thumbnail_url})
                except Exception as e:
                    log.warning("Data API: %s", str(e)[:200])

    statuses = ["pending"] + (["failed", "missing"] if retry else [])
    todo = []
    for st in statuses:
        todo += db.list_videos(channel_id=cfg.channel.id, status=st, limit=10**6, order="position ASC")
    if ids:
        wanted = set(ids)
        todo = [v for v in todo if v["id"] in wanted]
    if limit:
        todo = todo[:limit]
    if shard:
        idx, count = shard
        todo = [v for i, v in enumerate(todo) if i % count == idx]
        print(f"Shard {idx + 1}/{count}")
    providers = providers or cfg.transcripts.providers
    print(f"Transkripty: {len(todo)} videí ke zpracování (providery: {providers})")
    disabled: set[str] = set() if cfg.gemini_api_key else {"gemini"}
    ok = miss = fail = 0
    for i, v in enumerate(todo, 1):
        if disabled >= set(providers):
            print("Všechny providery jsou blokované – končím. Spusť ingest z jiné IP / s proxy / s GEMINI_API_KEY.")
            break
        try:
            res = get_transcript(v["id"], cfg, v["duration_s"], providers, disabled)
        except Exception as e:
            fail += 1
            db.set_transcript_status(v["id"], "failed", error=str(e))
            print(f"[{i}/{len(todo)}] ✖ {v['id']} {v['title'][:60]} – {str(e)[:80]}", flush=True)
            continue
        if not res:
            miss += 1
            db.set_transcript_status(v["id"], "missing")
            print(f"[{i}/{len(todo)}] – {v['id']} {v['title'][:60]} – bez transkriptu", flush=True)
            continue
        segs = merge_segments(res.segments)
        chunks = chunk_segments(segs, cfg.chunking.target_chars, cfg.chunking.max_seconds,
                                cfg.chunking.overlap_segments)
        db.store_transcript(v["id"], segs, chunks, res.source, res.lang)
        ok += 1
        print(f"[{i}/{len(todo)}] ✔ {v['id']} {v['title'][:60]} – {res.source}/{res.lang} {len(segs)} seg → {len(chunks)} chunků", flush=True)
        if checkpoint and ok % checkpoint_every == 0:
            if embed and cfg.gemini_api_key:
                run_embed(cfg, db, quiet=True)
            checkpoint()
        time.sleep(cfg.transcripts.request_delay)
    print(f"Hotovo: {ok} ok, {miss} bez transkriptu, {fail} chyb.")
    if embed and cfg.gemini_api_key:
        return run_embed(cfg, db)
    if not cfg.gemini_api_key:
        print("Pozn.: GEMINI_API_KEY není nastaven – sémantické hledání (embeddingy) přeskočeno, funguje fulltext.")
    return 0


def cmd_ingest(args) -> int:
    from .db import open_database

    cfg = load_config()
    if args.channel:
        # další kanál do stejného indexu: vyřešit a zapsat, zpracovávat jen jeho videa
        from .youtube import resolve_channel
        info = resolve_channel(args.channel, proxy=cfg.proxy_url or None, cookies_file=cfg.transcripts.cookies_file or None)
        cfg.channel.input, cfg.channel.id, cfg.channel.name, cfg.channel.url = args.channel, info.id, info.name, info.url
    _require_channel(cfg)
    db = open_database(cfg)
    db.upsert_channel(cfg.channel.id, cfg.channel.name, cfg.channel.url, cfg.channel.input)
    providers = args.providers.split(",") if args.providers else None
    rc = run_ingest(cfg, db, limit=args.limit, retry=args.retry, skip_listing=args.skip_listing,
                    providers=providers, embed=not args.no_embed, ids=args.ids.split(",") if args.ids else None)
    db.close()
    return rc


def _gcs_pull(cfg: Config) -> str | None:
    """Pokud je YTINDEX_GCS_DB, stáhne SQLite index z Cloud Storage do cfg.db_path a vrátí gs URL."""
    gs = os.environ.get("YTINDEX_GCS_DB")
    if not gs:
        return None
    from .gcs import download

    cfg.db_path = os.environ.get("YTINDEX_LOCAL_DB", "/tmp/ytindex/index.sqlite")
    download(gs, cfg.db_path)
    return gs


def _stamp_version(db, gs: str | None) -> None:
    """Uloží verzi indexu (GCS generation) do meta, aby ji API vracelo v X-Index-Version."""
    if not gs:
        return
    try:
        from .gcs import generation
        gen = generation(gs)
        if gen:
            db.set_meta("index_version", str(gen))
            db.conn.commit()
    except Exception as e:
        log.warning("verze indexu: %s", e)


def cmd_cloud_ingest(args) -> int:
    """Cloud Run Job: konfigurace z env (YTINDEX_CHANNEL, YTINDEX_GCS_DB nebo DATABASE_URL, GEMINI_API_KEY)."""
    from .db import open_database

    cfg = load_config()
    gs = _gcs_pull(cfg)
    db = open_database(cfg)
    _ensure_channel(cfg, db)
    providers = (args.providers or os.environ.get("YTINDEX_PROVIDERS") or "gemini").split(",")
    shard = None
    if args.shard:
        idx, count = (int(x) for x in args.shard.split("/"))
        shard = (idx, count)
    elif os.environ.get("CLOUD_RUN_TASK_COUNT", "1") != "1":
        shard = (int(os.environ["CLOUD_RUN_TASK_INDEX"]), int(os.environ["CLOUD_RUN_TASK_COUNT"]))
    skip_listing = shard is not None and shard[0] != 0 and not args.limit  # seznam aktualizuje jen task 0
    limit = args.limit if args.limit is not None else (int(os.environ["YTINDEX_LIMIT"]) if os.environ.get("YTINDEX_LIMIT") else None)
    checkpoint = None
    if gs:
        from .gcs import upload

        def checkpoint():
            upload(cfg.db_path, gs)
            print(f"  ✔ checkpoint: index nahrán do {gs}", flush=True)
    rc = run_ingest(cfg, db, limit=limit, retry=args.retry, skip_listing=skip_listing,
                    providers=providers, shard=shard, embed=not args.no_embed, checkpoint=checkpoint)
    db.close()
    if gs:
        checkpoint()
    return rc


def cmd_cloud_serve(args) -> int:
    import threading
    import uvicorn
    from .api import create_app
    from .db import ReloadableDatabase, open_database

    cfg = load_config()
    gs = _gcs_pull(cfg)
    db = ReloadableDatabase(open_database(cfg))
    _ensure_channel(cfg, db)
    _stamp_version(db, gs)

    if gs:
        # index se v bucketu mění během indexace – každých N minut zkontrolovat verzi a případně vyměnit
        from .gcs import download, generation

        interval = int(os.environ.get("YTINDEX_REFRESH_SECONDS", "300"))
        state = {"gen": generation(gs)}

        def refresher():
            while True:
                time.sleep(interval)
                try:
                    gen = generation(gs)
                    if gen and gen != state["gen"]:
                        new_path = f"{cfg.db_path}.{gen}"
                        download(gs, new_path)
                        new_db = open_database(cfg, url=new_path)
                        new_db.set_meta("index_version", str(gen)); new_db.conn.commit()
                        old_path = db.path
                        db.swap(new_db)
                        state["gen"] = gen
                        for suffix in ("", "-wal", "-shm"):
                            try:
                                os.remove(old_path + suffix)
                            except OSError:
                                pass
                        print(f"✔ Index obnoven z {gs} (verze {gen}): {db.stats()['videos_by_status']}", flush=True)
                except Exception as e:  # síť apod. – zkusit příště
                    print(f"refresh indexu selhal: {str(e)[:200]}", flush=True)

        threading.Thread(target=refresher, daemon=True, name="index-refresh").start()

    uvicorn.run(create_app(cfg, db), host="0.0.0.0", port=int(os.environ.get("PORT", args.port)), log_level="info",
                proxy_headers=True, forwarded_allow_ips="*")
    return 0


def cmd_embed(args) -> int:
    from .db import open_database

    cfg = load_config()
    return run_embed(cfg, open_database(cfg, reset_on_mismatch=True))


def run_embed(cfg: Config, db, quiet: bool = False) -> int:
    from .embeddings import Embedder

    if not db.vec_available:
        print("sqlite-vec není k dispozici – vektorové hledání vypnuto.", file=sys.stderr)
        return 1
    stored_model = db.get_meta("embedding_model")
    if stored_model and stored_model != cfg.gemini.embedding_model and db.count_embedded()[0]:
        print(f"Index má embeddingy z {stored_model}, config chce {cfg.gemini.embedding_model} – přepočítávám vše.")
        db.set_meta("embedding_model", cfg.gemini.embedding_model)
        with db.tx() as c:
            c.execute("UPDATE chunks SET embedded = 0") if db.backend == "sqlite" else c.execute("UPDATE chunks SET embedding = NULL")
    db.set_meta("embedding_model", cfg.gemini.embedding_model)
    db.conn.commit()
    orphans = db.delete_orphan_vectors()
    if orphans:
        print(f"  smazáno {orphans} osiřelých vektorů")
    emb = Embedder(cfg)
    done, total = db.count_embedded()
    if done == total:
        return 0
    if not quiet:
        print(f"Embeddingy: {done}/{total} chunků hotovo, model {cfg.gemini.embedding_model} ({cfg.gemini.embedding_dim}d)")
    while True:
        rows = db.pending_embeddings(limit=256)
        if not rows:
            break
        vecs = emb.embed_documents([r["text"] for r in rows])
        db.store_embeddings([(r["id"], v) for r, v in zip(rows, vecs)])
        done += len(rows)
        if not quiet:
            print(f"  {done}/{total}", end="\r", flush=True)
    print(f"✔ Embeddingy kompletní ({done}/{total}).", flush=True)
    return 0


def cmd_search(args) -> int:
    from .db import open_database
    from .search import search

    cfg = load_config()
    db = open_database(cfg)
    embed = None
    if args.mode != "fts" and cfg.gemini_api_key and db.count_embedded()[0]:
        from .embeddings import Embedder
        embed = Embedder(cfg, db).embed_query
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


def cmd_batch(args) -> int:
    """Přepis přes Gemini Batch API (poloviční cena): submit → (čekání) → collect."""
    from . import batch
    from .db import open_database

    cfg = load_config()
    if args.model:
        cfg.gemini.model = args.model
    gs = _gcs_pull(cfg) if args.gcs else None
    db = open_database(cfg)
    _ensure_channel(cfg, db)
    if args.action == "status":
        for j in batch.status(db):
            print(f"{j['state']:10s} {j['name']}  {j['model']}  {j['windows']} oken  {j['created_at']}  {j['error'] or ''}")
        print(json.dumps(db.stats()["videos_by_status"], ensure_ascii=False))
        return 0
    if args.action == "fill-gaps":
        summary = batch.fill_gaps(cfg, db, only_reason=args.reason)
        print("Batch fill-gaps:", json.dumps(summary, ensure_ascii=False))
        if summary["filled"] and not args.no_embed and cfg.gemini_api_key:
            run_embed(cfg, db)
        if gs and summary["filled"]:
            from .gcs import upload
            upload(cfg.db_path, gs)
        return 0
    if args.action == "submit":
        if not cfg.gemini_api_key:
            print("Chybí GEMINI_API_KEY", file=sys.stderr); return 2
        names = batch.submit(cfg, db, limit=args.limit, ids=args.ids.split(",") if args.ids else None)
        print(f"✔ Odesláno {len(names)} dávek. Vyzvednutí: ytindex batch collect [--wait]")
        if gs:
            from .gcs import upload
            upload(cfg.db_path, gs)
        return 0
    # collect
    checkpoint = None
    if gs:
        from .gcs import upload

        def checkpoint():
            upload(cfg.db_path, gs)
            print(f"  ✔ index nahrán do {gs}", flush=True)
    summary = batch.collect(cfg, db, wait=args.wait, poll_s=args.poll)
    print("Batch collect:", json.dumps(summary, ensure_ascii=False))
    if summary["videos_done"] and not args.no_embed and cfg.gemini_api_key:
        run_embed(cfg, db)
    if checkpoint and (summary["videos_done"] or summary["videos_failed"]):
        checkpoint()
    db.close()
    return 0


def cmd_export(args) -> int:
    """Hromadný export (JSONL + NumPy + manifest) pro jiné systémy; volitelně nahrání do veřejného bucketu."""
    from .db import open_database
    from .export import upload_export, write_export

    cfg = load_config()
    db = open_database(cfg)
    gs = args.gcs or os.environ.get("YTINDEX_EXPORT_GCS", "")
    version = args.version or db.get_meta("index_version")
    m = write_export(db, args.out, version=version)
    print(f"✔ Export {m['version']}: {json.dumps(m['counts'])} za {m['elapsed_s']} s → {args.out}")
    if gs:
        urls = upload_export(args.out, gs)
        print("✔ Nahráno:", urls["manifest.json"])
    return 0


def cmd_quality(args) -> int:
    """Najde zacyklené přepisy (stejné slovo/fráze opakovaně) a volitelně je označí k novému přepisu."""
    from . import quality
    from .db import open_database

    cfg = load_config()
    db = open_database(cfg)
    res = quality.scan(db, min_word_repeats=args.min_word, min_phrase_repeats=args.min_phrase)
    print(f"Prohledáno {res['videos_scanned']} videí: zasaženo {res['videos_hit']} videí, {res['windows']} oken "
          f"({res['hours']} h), {res['loops']} smyček; podle zdroje: {json.dumps(res['by_source'], ensure_ascii=False)}")
    for d in sorted(res["details"], key=lambda d: -len(d["windows"]))[: args.show]:
        print(f"  {d['video_id']} [{d['source']}] {len(d['windows'])} oken, {d['loops']} smyček, např. {d['example']} | {d['title'][:50]}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f"detail zapsán do {args.json}")
    if args.mark:
        n = quality.mark(db, res["details"])
        print(f"✔ Označeno {n} oken jako mezery → `ytindex batch fill-gaps` je znovu přepíše.")
    return 0


def cmd_copy(args) -> int:
    """Přenese index (kanál, videa, přepisy, chunky, embeddingy) do jiné DB, např. z lokálního SQLite do Cloud SQL."""
    from .chunking import Chunk, Segment
    from .db import open_database

    cfg = load_config()
    src = open_database(cfg, url=args.source) if args.source else open_database(cfg)
    dst = open_database(cfg, url=args.to)
    if src.embedding_dim != dst.embedding_dim:
        print("Různá dimenze embeddingů – vektory se nepřenesou, spusť `ytindex embed` nad cílem.", file=sys.stderr)
    ch = src.get_channel()
    if not ch:
        print("Zdrojová DB nemá kanál.", file=sys.stderr)
        return 2
    dst.upsert_channel(ch["id"], ch["name"], ch["url"], ch["input"] or "")
    videos = src.list_videos(limit=10**6, order="position ASC")
    with dst.tx():
        for v in videos:
            dst.upsert_video({k: v[k] for k in ("id", "channel_id", "title", "description", "published_at",
                                                "duration_s", "view_count", "thumbnail_url", "position")})
    done = [v for v in videos if v["transcript_status"] == "done"]
    for i, v in enumerate(done, 1):
        segs = [Segment(s["start"], s["duration"], s["text"]) for s in src.get_segments(v["id"])]
        rows = src.fetch_chunks(list(src.chunk_ids(v["id"]).values()))
        rows.sort(key=lambda r: r["start"])
        chunks = [Chunk(r["start"], r["end"], r["text"]) for r in rows]
        dst.store_transcript(v["id"], segs, chunks, v["transcript_source"], v["transcript_lang"])
        if src.embedding_dim == dst.embedding_dim:
            embs = src.get_embeddings(v["id"])
            if embs:
                ids = dst.chunk_ids(v["id"])
                dst.store_embeddings([(ids[idx], vec) for idx, vec in embs.items() if idx in ids])
        print(f"[{i}/{len(done)}] {v['id']} {len(segs)} seg, {len(chunks)} chunků", flush=True)
    for v in videos:
        if v["transcript_status"] in ("failed", "missing"):
            dst.set_transcript_status(v["id"], v["transcript_status"], error=v["transcript_error"])
    print("✔ Přeneseno:", json.dumps(dst.stats(), ensure_ascii=False))
    return 0


def cmd_mcp(args) -> int:
    from .mcp_server import build_mcp

    cfg = load_config()
    _require_channel(cfg)
    build_mcp(cfg).run(transport="stdio")
    return 0


def cmd_stats(args) -> int:
    from .db import open_database

    cfg = load_config()
    db = open_database(cfg)
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
    s.add_argument("--ids", help="zpracovat jen tato video ID (oddělená čárkou)")
    s.add_argument("--channel", help="jiný kanál (handle/ID/URL) do stejného indexu; web dál filtruje primární kanál z ytindex.toml")
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

    s = sub.add_parser("cloud-ingest", help="Cloud Run Job: ingest s konfigurací z env (YTINDEX_CHANNEL, DATABASE_URL)")
    s.add_argument("--limit", type=int)
    s.add_argument("--retry", action="store_true")
    s.add_argument("--providers", help="výchozí gemini (z cloudu YouTube titulky blokuje)")
    s.add_argument("--shard", help="N/M – zpracovat jen N-tý díl z M (automaticky z CLOUD_RUN_TASK_INDEX/COUNT)")
    s.add_argument("--no-embed", action="store_true")
    s.set_defaults(fn=cmd_cloud_ingest)

    s = sub.add_parser("cloud-serve", help="Cloud Run service: web + API + MCP s konfigurací z env")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_cloud_serve)

    s = sub.add_parser("batch", help="přepis přes Gemini Batch API za poloviční cenu: submit | collect | status")
    s.add_argument("action", choices=["submit", "collect", "fill-gaps", "status"])
    s.add_argument("--model", help="např. gemini-3.1-flash-lite (výchozí z konfigurace)")
    s.add_argument("--limit", type=int, help="submit: jen N videí (test)")
    s.add_argument("--ids", help="submit: jen tato video ID")
    s.add_argument("--wait", action="store_true", help="collect: čekat, dokud nejsou všechny dávky hotové")
    s.add_argument("--poll", type=int, default=120, help="collect --wait: interval kontroly v sekundách")
    s.add_argument("--no-embed", action="store_true")
    s.add_argument("--gcs", action="store_true", help="pracovat s indexem v YTINDEX_GCS_DB (stáhnout před, nahrát po)")
    s.add_argument("--reason", help="fill-gaps: doplnit jen mezery s tímto důvodem (např. loop)")
    s.set_defaults(fn=cmd_batch)

    s = sub.add_parser("export", help="hromadný export indexu (JSONL, NumPy embeddingy, manifest) pro jiné systémy")
    s.add_argument("--out", default="data/export")
    s.add_argument("--gcs", help="gs://bucket/prefix – nahrát export (veřejný bucket); výchozí env YTINDEX_EXPORT_GCS")
    s.add_argument("--version", help="verze indexu do manifestu (výchozí meta index_version nebo otisk obsahu)")
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("quality", help="kontrola přepisů: zacyklená slova/fráze; --mark označí okna k novému přepisu")
    s.add_argument("action", choices=["scan"])
    s.add_argument("--min-word", type=int, default=8, help="min. počet stejných slov za sebou (výchozí 8; 3–6 bývá běžná řeč „jo jo jo“)")
    s.add_argument("--min-phrase", type=int, default=8, help="min. počet opakování fráze 2–8 slov (výchozí 8)")
    s.add_argument("--show", type=int, default=15, help="kolik nejhorších videí vypsat")
    s.add_argument("--json", help="zapsat detail do souboru")
    s.add_argument("--mark", action="store_true", help="označit zasažená okna jako mezery (gaps) k novému přepisu")
    s.set_defaults(fn=cmd_quality)

    s = sub.add_parser("copy", help="přenese index do jiné DB (např. lokální SQLite → Cloud SQL)")
    s.add_argument("--to", required=True, help="cílové DATABASE_URL (postgresql://…) nebo cesta k SQLite")
    s.add_argument("--source", help="zdrojové DATABASE_URL / cesta (výchozí aktuální konfigurace)")
    s.set_defaults(fn=cmd_copy)

    s = sub.add_parser("mcp", help="spustí MCP server (stdio) pro Claude Desktop / Claude Code")
    s.set_defaults(fn=cmd_mcp)

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
