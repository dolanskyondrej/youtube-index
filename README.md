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

## Batch přepis a kontrola kvality

```bash
ytindex batch submit              # videa bez přepisu → Gemini Batch API (poloviční cena), dávky po ≤400 oknech
ytindex batch collect --wait      # vyzvedne hotové dávky, uloží přepisy (i částečné – chybějící okna jako „gaps“)
ytindex batch fill-gaps [--reason loop]   # chybějící/odmítnutá okna online: posunuté okno → poloviny → náhradní model
ytindex batch status
ytindex quality scan [--mark]     # zacyklené přepisy (stejné slovo/fráze 8+ za sebou); --mark je označí k novému přepisu
```

Poznatky z celého kanálu: Gemini účtuje vstupní audio i u oken, která pak odmítne (RECITATION), a při
`temperature=0` umí v okně zacyklit (jedno slovo tisíckrát) – proto kontrola kvality a opakování s jinými hranicemi.
Nikdy nespouštět dva ingest/batch běhy nad stejným indexem souběžně.

## API (kontrakt pro libovolný frontend)

Vše `GET`, JSON, CORS povolen, schéma v `/openapi.json` (lze z něj generovat TypeScript/Swift klienty).

| Endpoint | Popis |
|---|---|
| `/api/search?q=…&limit=20&mode=hybrid\|fts\|vector&per_video=3` | hlavní hledání; výsledky seskupené po videích |
| `/api/context?q=…&max_passages=8&context_seconds=90` | rešerše pro psaní: souvislé pasáže k tématu (shoda + okolní řeč, střídavě z videí), s čísly citací a odkazy na čas |
| `/api/ask?q=…` | odpověď Gemini sestavená jen z přepisů, s citacemi [n] a seznamem pasáží |
| `/api/videos?page=1&per_page=50&status=done` | seznam videí kanálu (status=all pro vše; pole `gaps`, `updated_at`) |
| `/api/videos/{id}/segments` | celý přepis videa jako segmenty s časem |
| `/api/export` | manifest hromadného exportu: verze indexu, počty, parametry embeddingů, odkazy na soubory |
| `/api/export/files/{name}` | stažení exportu: `videos.jsonl.gz`, `segments.jsonl.gz`, `chunks.jsonl.gz`, `embeddings.npy` (float32 [N,768]), `chunk_ids.npy`, `manifest.json`, `index.sqlite` |
| `/api/chunks?video_id=&after=&limit=500&with_embeddings=1` | hromadné čtení úseků kurzorem, embedding jako base64 float32 |
| `/api/changes?since=<ISO>` | videa změněná od času (inkrementální synchronizace; `next_since` pro další stránku) |
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

**Hromadné stahování celého obsahu.** Každá odpověď `/api/*` nese hlavičku `X-Index-Version` (verze indexu =
generation souboru v bucketu). Doporučený postup klienta: číst `/api/export`, při změně `index_version` stáhnout
soubory z manifestu (JSONL + NumPy; `embeddings.npy[i]` patří k `chunk_ids.npy[i]`), nebo jen `/api/changes?since=`
a pro změněná videa `/api/videos/{id}/segments` a `/api/chunks?video_id=…`. Export vyrábí `ytindex export`
(po každé indexaci; `--gcs gs://bucket/export` ho nahraje). Hromadné cesty (`/api/chunks`, `/api/changes`,
`/api/export/files/*`) lze chránit klíčem: env `YTINDEX_API_KEY`, hlavička `X-Api-Key`. Když je bucket veřejný
(`YTINDEX_EXPORT_PUBLIC=1`), manifest vrací přímé URL do Cloud Storage místo proxy přes službu.
Víc kanálů = víc instancí (každý kanál má vlastní `ytindex.toml` + DB), API je na kanálu nezávislé.

## Google Cloud (produkční nasazení, < 1 USD/měsíc)

Kompletní návod v [deploy/README.md](deploy/README.md). Zkráceně:

```bash
gcloud auth login && gcloud config set project youtube-index-512512
GEMINI_API_KEY=… CHANNEL="https://youtube.com/@kanalsvobodnehopristavu" YTINDEX_LIMIT=10 ./deploy/deploy.sh
```

* **Cloud Run service** `ytindex cloud-serve`: web + REST API + MCP (`/mcp`) v jednom kontejneru, škáluje na 0
  (při malém provozu v rámci free tieru).
* **SQLite index v Cloud Storage** (`YTINDEX_GCS_DB=gs://…/index.sqlite`): žádný databázový server. Instance si index
  stáhne při startu; po nové indexaci se nahraje zpět (`ytindex cloud-ingest`) nebo ručně
  (`gcloud storage cp data/index.sqlite gs://…`). Cena úložiště: centy.
* **Cloud Run Job** `ytindex cloud-ingest`: indexace přes Gemini (z cloudu jsou YouTube titulky blokované).
  Okna videa se přepisují souběžně (`gemini.concurrency`, výchozí 4), každých 5 videí se index i s embeddingy
  nahraje do bucketu (checkpoint), takže pád nebo limit jobu (24 h) nic neztratí – job se prostě spustí znovu
  a pokračuje. Bucket má zapnuté verzování objektů, starší verze indexu jdou obnovit.
  `YTINDEX_LIMIT=10` omezí test na prvních 10 videí. Levnější cesta: indexovat doma z titulků a index jen nahrát.
* **Přenositelnost:** přepisy (segmenty s časy), chunky i embeddingy jsou v indexu; `ytindex copy --to …` je
  přenese do Postgresu nebo jiného SQLite beze ztráty, přepis se nikdy nemusí platit znovu.
* Volitelně **PostgreSQL + pgvector** (`DATABASE_URL=postgresql://…`, backend `ytindex/db_postgres.py`) pro větší
  provoz nebo více zapisujících procesů; stejné API i hledání.
* Klíče jdou výhradně přes Secret Manager / env (`GEMINI_API_KEY`); kanál přes `YTINDEX_CHANNEL` (bez `ytindex.toml`).

## Kvalita hledání (měřeno na 10 videích)

Eval: 30 náhodných úseků přepisu → Gemini k nim vygeneroval otázku *jinými slovy* (parafráze) a zvlášť dotaz
ze dvou vzácných slov úseku (klíčová slova). Měří se, zda je správné video první / v top 5 a zda je správné
místo (±150 s) první / v top 5. Skript: `scratchpad/eval.py` v session, data v `tests/` nejsou (potřebuje API klíč).

| dotazy | režim | video@1 | video@5 | místo@1 | místo@5 |
|---|---|---|---|---|---|
| parafráze | fts | 0.23 | 0.63 | 0.07 | 0.17 |
| parafráze | vector | 0.90 | 1.00 | 0.80 | 0.90 |
| parafráze | **hybrid** | **0.90** | **1.00** | **0.80** | **0.90** |
| klíčová slova | fts | 1.00 | 1.00 | 1.00 | 1.00 |
| klíčová slova | vector | 0.57 | 0.90 | 0.50 | 0.77 |
| klíčová slova | **hybrid** | **0.83** | **1.00** | **0.83** | **1.00** |

Hybrid tedy drží kvalitu vektorů na parafrázích a téměř plnou kvalitu fulltextu na klíčových slovech (pro čistě
lexikální dotazy je k dispozici `mode=fts`). Klíčové nastavení: dokumenty `RETRIEVAL_DOCUMENT`, dotazy
`RETRIEVAL_QUERY`, normalizované vektory, cosine; stopslova se z fulltextu vyhazují; ve fúzi (RRF) má fulltext se
všemi slovy váhu 1,0, vektory 1,0, chunk nalezený oběma cestami dostane bonus 1,3 a OR‑fallback se v hybridu nepoužívá
(vyšší váha fulltextu zlepší klíčová slova na 1,00, ale protlačí lexikální falešné shody jako „stát“ = sloveso). Gemini podobnosti jsou „ploché“
(relevantní ≈ 0,70–0,75, nesmysl ≈ 0,60–0,65), proto se počet shod počítá jen z chunků do 0,03 od nejlepšího.
Stejný eval s 256‑d vektory: parafráze 0,83 / 0,93 (místo 0,77 / 0,83), index 2× menší.

## MCP server (hledání z Claude / jiných AI)

Nástroje a endpoint `https://<url>/mcp` (Streamable HTTP, bez autentizace):

| nástroj | k čemu |
|---|---|
| `search_videos` | klasické hledání: videa, úryvky se shodou, odkazy s časem |
| `research_topic` | **pro psaní**: k otázce vrátí delší souvislé pasáže toho, co k tématu zaznělo (±45 s kolem shody, střídavě z videí); Claude z nich syntetizuje a cituje s odkazy |
| `ask_channel` | hotová odpověď Gemini jen z přepisů, s citacemi [n] a pasážemi |
| `get_transcript` | celý přepis videa nebo úsek `start_s`–`end_s` |
| `list_videos`, `channel_info` | přehled kanálu a stav indexu |

Tip pro rešerši: otázky typu „Co si Urza myslí o X?“ se před hledáním ořežou na téma X (jméno mluvčího a tázací
rámec jinak táhnou k úvodům videí). Mluvčí v přepisech není rozlišen, ve studiu bývají hosté.

| Kde | Jak |
|---|---|
| claude.ai (web) | Settings → Connectors → **Add custom connector** → Name `ytindex`, URL `https://<cloud-run-url>/mcp` → Add. Pak v chatu zapnout v menu nástrojů. |
| Claude Desktop | stejné nastavení Connectors (sdílené s claude.ai), nebo lokální stdio (níže) |
| Claude Code (tento repozitář) | hotovo v `.mcp.json` (project scope) – po otevření repa se zeptá na schválení; jinde `claude mcp add --transport http ytindex https://<cloud-run-url>/mcp` |
| Lokálně (stdio) | `claude mcp add ytindex -- ytindex mcp` (v adresáři s `ytindex.toml`) |

## Limity API

Token bucket per IP (z `X-Forwarded-For`) + globální strop na instanci; při překročení `429` s `Retry-After`.
Nastavení v `ytindex/ratelimit.py`: `/api/ask` 6/min na IP (burst 3, globálně 30/min), `/api/context` 30/min,
`/api/search` a `/mcp` 60/min (burst 20), ostatní `/api/*` 120/min. Smysl: hledání volá Gemini embedding dotazu
a `/api/ask` generaci, limiter drží náklady pod kontrolou i při zneužití.

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
