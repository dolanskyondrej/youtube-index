# Nasazení na Google Cloud

```
┌──────────────┐    ┌────────────────────────┐    ┌──────────────────────────┐
│ Cloud Run Job│    │ Cloud Run service      │    │ Cloud SQL PostgreSQL 16  │
│ ytindex-     │───►│ ytindex (web + REST    │◄──►│ + pgvector (HNSW, cosine)│
│ ingest       │    │ API + MCP na /mcp)     │    │ + tsvector/unaccent FTS  │
└──────┬───────┘    └────────────────────────┘    └──────────────────────────┘
       │ transkripty (Gemini video understanding z YouTube URL) + embeddingy (gemini-embedding-001)
       ▼
   Gemini API
```

* **Cloud Run service** – bezstavový kontejner (`ytindex cloud-serve`), škáluje 0→N, cena prakticky 0 při nízkém provozu.
* **Cloud Run Job** – `ytindex cloud-ingest [--limit N]`; stejný image. Z cloudu YouTube blokuje stahování titulků,
  proto job používá provider **gemini** (Gemini si video stáhne sám). Spouštět ručně nebo přes Cloud Scheduler.
* **Cloud SQL** – „speciální DB“: jeden Postgres s rozšířeními `vector` (sémantické hledání) a `unaccent`
  (fulltext bez diakritiky). Nejmenší tier `db-f1-micro` stačí; pro větší provoz `db-custom-1-3840`.
* **Secret Manager** – `GEMINI_API_KEY`, `DATABASE_URL`. Klíče nikdy necommitovat.

## Jednorázové nasazení

```bash
gcloud auth login
gcloud config set project youtube-index-512512
GEMINI_API_KEY=... CHANNEL="https://youtube.com/@kanalsvobodnehopristavu" ./deploy/deploy.sh
```

Skript zapne API, založí Artifact Registry, Cloud SQL, secrets, sestaví image přes Cloud Build a nasadí service + job.
Na konci vypíše URL aplikace a příkazy pro indexaci.

## Test na 10 videích → celý kanál

```bash
gcloud run jobs execute ytindex-ingest --region europe-west1 --args cloud-ingest,--limit,10 --wait
# zkontrolovat https://<url>/?q=svoboda a https://<url>/api/channel, pak:
gcloud run jobs execute ytindex-ingest --region europe-west1 --wait
```

Celý kanál (~1 500 h) = cca 11 000 požadavků po 10 min videa; job běží sekvenčně ~2 dny (limit tasku 24 h → spustit
opakovaně, ingest je resumable), nebo `--tasks 4` s paralelizací (viz `cloud-ingest --shard`).
Odhad ceny Gemini 3.5 Flash‑Lite: ~115 k tokenů/h videa ≈ 0,05 USD/h → **~75 USD za celý kanál**, embeddingy ~2 USD.

## MCP v Claude

Vzdálený server (claude.ai / Claude Desktop → Settings → Connectors → Add custom connector):
`https://<url>/mcp`

Lokálně (Claude Code): `claude mcp add --transport http ytindex https://<url>/mcp`
nebo stdio: `claude mcp add ytindex -- ytindex mcp` (v adresáři s `ytindex.toml`).
