# ytindex API – napojení a postupné stahování

Base URL: `https://ytindex-128410263808.europe-west1.run.app` · OpenAPI: `/openapi.json`, Swagger UI: `/docs` · MCP: `/mcp`

Všechny odpovědi jsou JSON (UTF‑8), všechny endpointy `GET`, CORS povolen. Časy jsou v sekundách od začátku videa,
odkaz na místo ve videu je vždy `https://www.youtube.com/watch?v={video_id}&t={int(start)}s`.

## 0. Kanály

Index může obsahovat více YouTube kanálů. Každé video, výsledek hledání, pasáž i chunk nese odkud pochází:
`channel_id` (UC…) a u videí/pasáží objekt `channel {id, name, url, handle}` (`url` je adresa kanálu, např.
`https://www.youtube.com/@kanalsvobodnehopristavu`). Seznam kanálů s počty: `GET /api/channels` (pole `primary`
= kanál, na který se omezuje web). Všechny endpointy přijímají `channel=<UC id | @handle | primary>`; bez parametru
pracují nad všemi kanály. Export: `videos.jsonl.gz` má `channel_id` + `channel`, `chunks.jsonl.gz` má `channel_id`,
manifest má `channels`.

## 1. Datový model

```
channel (id, name, url, handle)
  └─ video   (video_id, channel_id, title, published_at, duration_s, transcript_status, transcript_source, gaps, updated_at)
  └─ segments  přepis = zdroj pravdy: (idx, start, duration, text), seřazeno podle času
  └─ chunks    úseky ~450 znaků pro hledání, překryv 1 segment: (chunk_id, idx, start, end, text, embedding)
```

* `transcript_status`: `done` (má přepis) · `pending` · `queued` · `missing` · `failed`. Veřejné výpisy vrací
  jen `done`, pokud nepošleš `status=all`.
* `transcript_source`: `captions` / `ytdlp` (YouTube titulky), `gemini` / `gemini-batch` (přepis modelem).
* `gaps`: seznam `{start, end, reason}` oken (s), která v přepisu chybí (model je odmítl). Prázdný = kompletní.
* `updated_at`: čas poslední změny videa (metadata nebo přepis), UTC `YYYY-MM-DD HH:MM:SS`. Základ inkrementální synchronizace.
* Embeddingy: `gemini-embedding-001`, 768 dimenzí, L2‑normalizované, metrika cosine. Dokumenty embedované s
  `task_type=RETRIEVAL_DOCUMENT`, dotazy embeduj s `RETRIEVAL_QUERY` stejným modelem, jinak výsledky nesedí.

## 2. Verze indexu

Index se mění jen při indexaci (nová videa, opravy). Každá odpověď `/api/*` nese hlavičku `X-Index-Version`
(řetězec, roste s časem). Stejná hodnota = stejná data; změna = je co stáhnout. Hodnotu si ukládej.

## 3. Autentizace

Klíč je jen pro hromadné stahování: `/api/export/files/*`, `/api/chunks` **bez** `video_id` a `/api/changes`
vyžadují hlavičku `X-Api-Key: <klíč>`, jinak `401`. Všechno ostatní je bez klíče: hledání, manifest, seznam videí a
**vše k jednomu videu** (`/api/videos/{id}` včetně `with_chunks=1&with_embeddings=1`, `/api/chunks?video_id=…`).

## 4. Limity

Token bucket na IP (`429` + `Retry-After` v sekundách): hledání 60/min, `/api/chunks` 120/min, `/api/changes`
60/min, `/api/export` 30/min, `/api/ask` 6/min. Při `429` počkej `Retry-After` a opakuj.

## 5. Stažení celého obsahu (doporučeno pro první načtení a pravidelné zrcadlení)

```
1. GET /api/export                       → manifest (index_version, counts, embedding, files{name: url,size,sha256})
2. pokud index_version != uložená:
     GET /api/export/files/videos.jsonl.gz     (X-Api-Key)   ~0,3 MB   1 řádek = 1 video (pole jako v /api/videos)
     GET /api/export/files/segments.jsonl.gz   (X-Api-Key)   ~46 MB    {video_id, idx, start, duration, text}
     GET /api/export/files/chunks.jsonl.gz     (X-Api-Key)   ~38 MB    {chunk_id, video_id, idx, start, end, text, embedded}
     GET /api/export/files/embeddings.npy      (X-Api-Key)   ~837 MB   NumPy float32 [N,768]
     GET /api/export/files/chunk_ids.npy       (X-Api-Key)   ~2 MB     NumPy int64 [N]; embeddings[i] patří k chunk_ids[i]
     (volitelně) /api/export/files/index.sqlite              ~1,3 GB   celý index včetně FTS a vektorů (SQLite + sqlite-vec)
3. ověř sha256 z manifestu, ulož index_version
```

* Soubory se streamují (chunked), velikost je v hlavičce `X-Content-Size`. Podporují `Range: bytes=a-b`
  (`206`, `Content-Range`), takže přerušené stahování navážeš. Přenos 837 MB trvá ~1 minutu.
* Všechny soubory jednoho manifestu pocházejí z téže verze indexu. Když se během stahování změní
  `X-Index-Version`, stáhni manifest znovu.
* `embeddings.npy` načteš `numpy.load(path, mmap_mode="r")`; pro jiné jazyky je to standardní `.npy` (hlavička + raw float32 little‑endian).

## 6. Inkrementální synchronizace (jen změny)

```
1. GET /api/changes?since=<poslední updated_at>&limit=500        (X-Api-Key)
     → items = videa s updated_at > since (nejstarší první), next_since = kurzor
     opakuj se since=next_since, dokud next_since != null
2. pro každé změněné video jedno volání (bez klíče):
     GET /api/videos/{id}?with_chunks=1&with_embeddings=1
       → metadata, channel, gaps, segments (přepis), chunks s embeddingy, index_version
     (embedding = base64 ze 768 × float32 little‑endian; null, pokud ještě nebyl spočítán;
      `/api/chunks?video_id=…&with_embeddings=1` je alternativa se stránkováním, také bez klíče)
3. ulož max(updated_at) jako nový since
```

Změna přepisu znamená nové `chunk_id` pro celé video: staré chunky videa smaž a nahraď. Video, které zmizelo
z kanálu, v `changes` nepřijde; porovnej občas seznam z `/api/videos?status=all` (nebo z exportu).

Pro první načtení bez exportu lze projít `/api/chunks?after=<poslední chunk_id>&limit=2000` kurzorem přes celý index
(`next_after` = null značí konec); JSON s embeddingy je ale ~5× větší než `.npy`, proto preferuj export.

## 7. Hledání (pro aplikace, které nechtějí vlastní index)

* `GET /api/search?q=…&mode=hybrid|fts|vector&limit=20&per_video=3` → videa s nejlepšími úryvky
  (`highlighted` obsahuje jen značky `<mark>`, zbytek escapuj), `match.url` vede na čas ve videu.
* `GET /api/context?q=…&max_passages=8&context_seconds=90` → souvislé pasáže k tématu (pro RAG a psaní).
* `GET /api/ask?q=…` → odpověď sestavená Gemini jen z přepisů, s citacemi `[n]` do pole `citations`.

## 8. Chyby

`401` chybí/špatný klíč · `404` neexistující video/soubor, nebo export ještě nevznikl · `416` Range mimo soubor ·
`422` neplatný parametr · `429` limit, viz `Retry-After` · `5xx` zkus znovu s odstupem (data v bucketu jsou vždy konzistentní).

## 9. Minimální klient (Python)

```python
import httpx, json, gzip, numpy as np
BASE, KEY = "https://ytindex-128410263808.europe-west1.run.app", "..."
h = {"X-Api-Key": KEY}
m = httpx.get(f"{BASE}/api/export").json()
if m["index_version"] != stored_version:
    for name in ("videos.jsonl.gz", "segments.jsonl.gz", "chunks.jsonl.gz", "chunk_ids.npy", "embeddings.npy"):
        with httpx.stream("GET", f"{BASE}{m['files'][name]['url']}", headers=h, timeout=None) as r, open(name, "wb") as f:
            for block in r.iter_bytes(): f.write(block)
    emb, ids = np.load("embeddings.npy", mmap_mode="r"), np.load("chunk_ids.npy")
    stored_version = m["index_version"]
# později jen změny:
r = httpx.get(f"{BASE}/api/changes", params={"since": last_sync}, headers=h).json()
for v in r["items"]:
    chunks = httpx.get(f"{BASE}/api/chunks", params={"video_id": v["video_id"], "with_embeddings": 1, "limit": 2000}, headers=h).json()
```
