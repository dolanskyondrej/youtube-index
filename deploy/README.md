# Nasazení na Google Cloud (levná varianta)

```
┌────────────────────┐   stáhne při startu   ┌──────────────────────────┐   nahraje po doběhu   ┌────────────────────┐
│ Cloud Run service  │◄──────────────────────│ Cloud Storage            │◄──────────────────────│ Cloud Run Job      │
│ ytindex            │                       │ gs://<projekt>-ytindex/  │                       │ ytindex-ingest     │
│ web + API + /mcp   │                       │ index.sqlite             │                       │ (Gemini přepis +   │
│ škáluje 0 → 2      │                       │ (FTS5 + sqlite-vec)      │                       │  embeddingy)       │
└────────────────────┘                       └──────────────────────────┘                       └────────────────────┘
```

| Položka | Cena (odhad) |
|---|---|
| Cloud Run service, min-instances 0, pár tisíc dotazů/měsíc | 0 USD (free tier 2 M požadavků, 180 k vCPU‑s) |
| Cloud Storage, index 17 MB (10 videí) / ~400 MB (celý kanál) | < 0,01 USD |
| Artifact Registry, image ~300 MB | ~0,03 USD |
| Secret Manager, 1 secret | 0 USD (free tier) |
| Cloud Run Job, indexace celého kanálu (~4 h vCPU jednorázově) | ~0,40 USD |
| Gemini (přepis 1 500 h + embeddingy, jednorázově) | ~75 USD + ~2 USD |

Běžný provoz tedy pod 1 USD/měsíc. Alternativa e2-micro VM je ve free tieru jen v USA, vyžaduje údržbu (TLS, aktualizace)
a není levnější.

## Nasazení

```bash
gcloud auth login                       # nebo: gcloud auth activate-service-account --key-file klic.json
gcloud config set project youtube-index-512512
GEMINI_API_KEY=… CHANNEL="https://youtube.com/@kanalsvobodnehopristavu" YTINDEX_LIMIT=10 ./deploy/deploy.sh
```

Skript zapne API, založí bucket, Artifact Registry a secret, sestaví image přes Cloud Build a nasadí service + job.
Je idempotentní: další spuštění jen nasadí novou verzi.

## Index: lokálně nebo v cloudu

* **Lokálně (levně, z titulků):** `ytindex ingest` doma → `gcloud storage cp data/index.sqlite gs://<projekt>-ytindex/index.sqlite`.
* **V cloudu (Gemini):** `gcloud run jobs execute ytindex-ingest --region europe-west1 --wait`
  (`YTINDEX_LIMIT` v env jobu omezuje počet videí; bez něj celý kanál, job je resumable, limit 24 h na běh).

Služba po nahrání nového indexu začne nový index používat, jakmile se vytvoří nová instance (škáluje na 0, tj.
po ~15 min klidu). Vynutit ihned: `gcloud run services update ytindex --region europe-west1 --update-env-vars INDEX_VERSION=$(date +%s)`.

Paměť: pro 10 videí stačí 512 Mi–1 Gi; celý kanál (~100 k chunků × 768 d) nastav `MEMORY=2Gi` (index se drží v `/tmp`, což je RAM).

## MCP v Claude

* claude.ai / Claude Desktop → Settings → Connectors → Add custom connector → `https://<url>/mcp`
* Claude Code: `claude mcp add --transport http ytindex https://<url>/mcp`
