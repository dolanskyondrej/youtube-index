# ytindex – prohledávatelný index všech videí YouTube kanálu

Nástroj, který pro zadaný kanál (handle, ID nebo URL) stáhne seznam všech videí a jejich transkripty, uloží je do
databáze a nad nimi poskytne **REST API + jednoduchý web**. Výsledkem hledání je seznam videí a ke každému:
**název, úryvek se shodou (zvýrazněný), odkaz na přesný čas ve videu, náhledovka**.

Testovací kanál: https://www.youtube.com/@kanalsvobodnehopristavu (955 videí vč. streamů, ~1 500 h).

## Jak to funguje

```
ytindex init  ──►  ytindex ingest  ──►  ytindex embed (volitelné)  ──►  ytindex serve
 (zeptá se na      1. seznam videí      Gemini embeddingy           FastAPI: /api/search,
  kanál, uloží     (yt-dlp, taby        → sqlite-vec                /api/videos, /api/channel,
  ytindex.toml)    videos + streams)                                web UI na /
                   2. transkripty
                      captions → ytdlp → gemini
                   3. chunking + FTS5
```

| Vrstva | Volba | Proč |
|---|---|---|
| Seznam videí | **yt-dlp** (flat playlist, bez klíče), volitelně **YouTube Data API v3** pro datum publikace, přesnou délku a popis | Data API neumí stáhnout cizí titulky, ale listování je levné (50 videí = 1 quota jednotka) |
| Transkripty | 1) `youtube-transcript-api` 2) `yt-dlp` titulky (json3/vtt) 3) **Gemini** (přepis přímo z YouTube URL) | Titulky jsou zdarma a mají přesné časy; Gemini řeší videa bez titulků a nevadí mu blokace IP |
| Databáze | **SQLite** + **FTS5** (fulltext, `remove_diacritics 2`) + **sqlite-vec** (vektory, cosine) | Jeden soubor, nulový provoz, pro ~100 k chunků bohatě stačí; migrace na Postgres+pgvector je přímočará |
| Hledání | **hybridní**: BM25 + sémantické embeddingy (Gemini) spojené přes Reciprocal Rank Fusion, seskupení po videích | Fulltext = přesné fráze a jména, vektory = význam („školáci" najde „děti ve škole") |
| API | **FastAPI**, OpenAPI na `/docs`, CORS `*`, čistě JSON | Použitelné z libovolného frontendu (web, mobil, Slack bot, LLM agent) |

### Transkript: co umí a neumí Gemini (odpověď na otázku k NotebookLM)

- **Gemini API neumí „stáhnout index/titulky" videa.** Žádný endpoint typu „dej mi transkript videa X" neexistuje.
  Umí ale přijmout **veřejnou YouTube URL jako vstup** (`file_data.file_uri`) a video na své straně zpracovat,
  takže ho necháme **přepsat řeč s časovými značkami**. To je to, co dělá NotebookLM (interně používá stejné
  video understanding); navenek to NotebookLM neexportuje a nemá API.
- Limity: free tier **8 h YouTube videa / den**, placený bez limitu délky; jen veřejná videa. Ověřeno
  na `gemini-3.5-flash-lite`: 10 min videa při `fps = 0.1` ≈ **19 k vstupních tokenů, ~13 s**, tj. ~115 k tokenů/h
  ≈ 0,05 USD/h videa → **~75 USD za celý kanál (1 500 h)**. Videa posíláme po oknech (`transcribe_window_s`,
  výchozí 10 min) přes `video_metadata.start/end_offset`.
- **Časové značky chtějte ve formátu `MM:SS`** – v číselných sekundách model driftuje (~1,6×), v `MM:SS` sedí
  (ověřeno na okně 20:00–30:00). Model vrací časy absolutní od začátku videa i při výřezu.
- Proto je Gemini až **třetí v pořadí**: titulky z YouTube jsou zdarma, přesné a má je naprostá většina videí
  (auto‑captions cs). Gemini nastupuje jen pro videa bez titulků (`providers = ["captions","ytdlp","gemini"]`).

### Důležité: odkud spouštět ingest

YouTube **blokuje stahování titulků z IP adres cloudových providerů** (AWS, GCP, Hetzner…): `youtube-transcript-api`
vrací `IpBlocked`, yt-dlp „Sign in to confirm you're not a bot". Listování videí z cloudu funguje. Možnosti:

1. **Ingest pouštět z domácí/kancelářské IP** (notebook) a výsledný `data/index.sqlite` nahrát na server. Nejjednodušší,
   ingest je jednorázový + občasný doběh nových videí.
2. Rezidenční proxy (`WEBSHARE_PROXY_USERNAME/PASSWORD` nebo `YTINDEX_PROXY_URL`).
3. `cookies_file` z přihlášeného prohlížeče pro yt-dlp (riziko blokace účtu, nedoporučeno).
4. **Gemini jako jediný provider** (`--providers gemini`): běží kdekoli, stojí peníze a čas, ale je to „čisté API".

## Rychlý start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # doplň GEMINI_API_KEY (volitelně YOUTUBE_API_KEY)

ytindex init                    # zeptá se na kanál; nebo: ytindex init --channel @kanalsvobodnehopristavu
ytindex ingest --limit 20       # zkouška na 20 videích (bez --limit všechno; běh je resumable)
ytindex stats
ytindex search "daně jsou krádež"
ytindex serve                   # http://localhost:8000  (API docs: /docs)
```

Opakované `ytindex ingest` stáhne jen nová videa (stav `pending`); `--retry` zkusí znovu `failed`/`missing`;
`--ids a,b,c` zpracuje jen vybraná videa (hodí se na paralelní běh ve více procesech).
`ytindex copy --to postgresql://…` přenese hotový index (včetně embeddingů) z lokálního SQLite do Cloud SQL,
takže levná cesta je: indexovat doma z titulků zdarma → nahrát do cloudu.
Embeddingy se počítají automaticky, pokud je `GEMINI_API_KEY` (`--no-embed` vypne; `ytindex embed` dopočítá).

Docker: `docker compose up -d` (servíruje hotový `data/index.sqlite` + `ytindex.toml`).

## API (kontrakt pro libovolný frontend)

Vše `GET`, JSON, CORS povolen, schéma v `/openapi.json` (lze z něj generovat TypeScript/Swift klienty).

| Endpoint | Popis |
|---|---|
| `/api/search?q=…&limit=20&mode=hybrid\|fts\|vector&per_video=3` | hlavní hledání; výsledky seskupené po videích |
| `/api/videos?page=1&per_page=50&status=done` | seznam videí kanálu |
| `/api/videos/{id}` | detail videa včetně celého transkriptu (segmenty s časy a odkazy) |
| `/api/channel` | název, URL a statistiky indexu |
| `/health` | liveness |

Příklad odpovědi `/api/search?q=bitcoin`:

```json
{
  "query": "bitcoin", "mode": "hybrid", "total": 2,
  "results": [{
    "video_id": "SvgW2XWesQ0",
    "title": "Studio Svobodného přístavu: Nejen o ekonomii se Sarou Polak",
    "url": "https://www.youtube.com/watch?v=SvgW2XWesQ0",
    "thumbnail": "https://i.ytimg.com/vi/SvgW2XWesQ0/hqdefault.jpg",
    "published_at": "2025-09-01T12:00:00Z", "duration_s": 8820,
    "score": 0.0171, "match_count": 4,
    "match": {
      "text": "… Bitcoin je podle mě nejdůležitější vynález …",
      "highlighted": "… <mark>Bitcoin</mark> je podle mě nejdůležitější vynález …",
      "start": 5489.2, "end": 5531.0, "timestamp": "1:31:29",
      "url": "https://www.youtube.com/watch?v=SvgW2XWesQ0&t=5489s"
    },
    "matches": [ "… až per_video nejlepších shod …" ]
  }]
}
```

`highlighted` obsahuje **pouze** značky `<mark>`; frontend zbytek escapuje (viz `ytindex/web/index.html`).
Víc kanálů = víc instancí (každý kanál má vlastní `ytindex.toml` + DB), API je na kanálu nezávislé.

## Google Cloud (produkční nasazení)

Kompletní návod a architektura v [deploy/README.md](deploy/README.md). Zkráceně:

```bash
gcloud auth login && gcloud config set project youtube-index-512512
GEMINI_API_KEY=… CHANNEL="https://youtube.com/@kanalsvobodnehopristavu" ./deploy/deploy.sh
gcloud run jobs execute ytindex-ingest --region europe-west1 --args cloud-ingest,--limit,10 --wait   # test 10 videí
```

* **Cloud Run service** `ytindex cloud-serve`: web + REST API + MCP (`/mcp`) v jednom kontejneru.
* **Cloud Run Job** `ytindex cloud-ingest`: indexace přes Gemini (z cloudu jsou YouTube titulky blokované), resumable,
  paralelizovatelná přes `--tasks N` (shardování podle `CLOUD_RUN_TASK_INDEX`).
* **Cloud SQL PostgreSQL 16 + pgvector + unaccent**: nastavením `DATABASE_URL=postgresql://…` se místo SQLite použije
  backend `ytindex/db_postgres.py` (tsvector s konfigurací `cs_simple` = fulltext bez diakritiky, HNSW index pro
  cosine vzdálenost). Stejné API, stejné hledání.
* Klíče jdou výhradně přes Secret Manager / env (`GEMINI_API_KEY`, `DATABASE_URL`); konfigurace kanálu přes
  `YTINDEX_CHANNEL` (bez `ytindex.toml`).

## MCP server (hledání z Claude / jiných AI)

Nástroje: `search_videos`, `get_transcript`, `list_videos`, `channel_info`.

| Kde | Jak |
|---|---|
| claude.ai / Claude Desktop | Settings → Connectors → Add custom connector → `https://<cloud-run-url>/mcp` |
| Claude Code (vzdáleně) | `claude mcp add --transport http ytindex https://<cloud-run-url>/mcp` |
| Claude Code / Desktop (lokálně, stdio) | `claude mcp add ytindex -- ytindex mcp` (spouštět v adresáři s `ytindex.toml`) |

## Kde to provozovat (doporučení)

- **Web + API:** jeden malý VPS/kontejner stačí (DB je SQLite soubor, dotaz trvá jednotky až desítky ms).
  Nejjednodušší: **Fly.io** (`fly launch` z Dockerfile, volume na `/app/data`, ~$3–5/měs.) nebo **Hetzner CX22**
  s `docker compose`. Railway/Render fungují stejně. Sémantické hledání potřebuje backend (embedding dotazu),
  čistě statické nasazení by umělo jen fulltext.
- **Ingest:** z notebooku / domácí IP (viz výše), případně GitHub Actions self‑hosted runner. Výsledný SQLite
  nahrát na server (`scp`/`rsync`), nebo DB držet v Litestream/S3.
- **Klíče:** `GEMINI_API_KEY` (AI Studio; free tier stačí na embeddingy, ~10 M tokenů za celý kanál ≈ $2 placeně),
  `YOUTUBE_API_KEY` (Google Cloud Console, YouTube Data API v3, zdarma do 10 000 jednotek/den; celý kanál = ~20 jednotek).

## Konfigurace (`ytindex.toml`)

Vytvoří ji `ytindex init`. Klíčové volby: `transcripts.languages` (priorita jazyků titulků), `transcripts.providers`,
`gemini.model` / `embedding_model` / `embedding_dim` (změna dimenze = automatický re‑embed), `chunking.*`
(délka úseků ≈ 30–60 s řeči, překryv 1 segment, aby fráze na hranici nezmizela).

## Vývoj

```bash
pytest                 # chunking, FTS/hybridní hledání (s falešným embedderem), API, MCP, parsery titulků
TEST_DATABASE_URL=postgresql://user:pass@localhost:5432/ytindex pytest tests/test_postgres.py   # Postgres backend
uvicorn ytindex.api:app --factory --reload
```

Struktura: `ytindex/youtube.py` (kanál + seznam videí), `transcripts.py` (providery), `chunking.py`, `db.py`
(SQLite: FTS5 + sqlite-vec), `db_postgres.py` (Postgres: tsvector + pgvector), `query.py` (převod dotazu),
`embeddings.py`, `search.py` (RRF), `api.py`, `mcp_server.py`, `cli.py`, `web/index.html`, `deploy/`.

## Roadmap / nápady

- Lokální Whisper (`faster-whisper`) jako čtvrtý provider pro offline přepis.
- Reranker (Gemini / cross-encoder) nad top‑50 kandidáty pro přesnější pořadí.
- „Odpověz z kanálu" (RAG): `/api/ask`, Gemini odpoví s citacemi na časy ve videích.
- MCP server nad stejnou DB, aby v kanálu mohl hledat Claude/ChatGPT.
