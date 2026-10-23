# Vytěžení titulků z vlastního počítače (zdarma) a nahrání indexu do cloudu

Z cloudu YouTube stahování titulků blokuje, z domácí sítě funguje. Tento postup doplní do existujícího indexu
(v Cloud Storage) všechna zbývající videa z YouTube titulků, dopočítá embeddingy a nahraje index zpět.
Web a MCP si nový index načtou do 5 minut samy.

Cena: titulky 0 Kč, embeddingy přes Gemini cca 100 Kč za celý kanál. Čas: cca 1–2 h (větší část jsou embeddingy).

## Předpoklady

- Python 3.11+ (`python3 --version`), git
- Google Cloud SDK (`gcloud`) přihlášené účtem, který vlastní projekt `youtube-index-512512`
  (`gcloud auth login`, pak `gcloud config set project youtube-index-512512`)
- Gemini API klíč (ten, který je v Secret Manageru jako `ytindex-gemini-key`; stačí ho vložit do `.env`)

## Postup

```bash
git clone https://github.com/dolanskyondrej/youtube-index
cd youtube-index
git checkout claude/youtube-channel-search-tool-ucsb45
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .

cp .env.example .env
# do .env doplň: GEMINI_API_KEY=...   (YOUTUBE_API_KEY není nutný, data publikace už v indexu jsou)

# 1) převzít aktuální index z cloudu (obsahuje už hotová videa)
mkdir -p data
gcloud storage cp gs://youtube-index-512512-ytindex/index.sqlite data/index.sqlite

# 2) konfigurace kanálu (vytvoří ytindex.toml, nic nestahuje)
ytindex init --channel https://www.youtube.com/@kanalsvobodnehopristavu

# 3) titulky pro všechna nehotová videa (pending/failed/missing), bez Gemini; běh je resumable
ytindex ingest --providers captions,ytdlp --retry

# 4) kontrola
ytindex stats
ytindex search "daně jsou krádež"

# 5) nahrát zpět do cloudu (web + MCP si to do 5 minut načtou)
gcloud storage cp data/index.sqlite gs://youtube-index-512512-ytindex/index.sqlite
```

Poznámky:

- `ingest` nejdřív aktualizuje seznam videí kanálu (z domácí IP projde celý), pak stahuje titulky
  s pauzou 1 s mezi videi. Pokud YouTube začne vracet `IpBlocked`, běh se zastaví; počkej hodinu a spusť
  stejný příkaz znovu, naváže.
- Videa bez titulků skončí jako `missing` (`ytindex stats` je spočítá). Ta lze později dopřepsat přes Gemini
  v cloudu za cca 5 Kč/h videa:
  `gcloud run jobs execute ytindex-ingest --region europe-west1 --args cloud-ingest,--retry --wait`
  (job spouštět vždy jen jeden).
- Embeddingy se počítají na konci automaticky (`GEMINI_API_KEY` v `.env`). Při chybě 429 (kvóta) se čeká
  a pokračuje; kdyby běh spadl, `ytindex embed` dopočítá zbytek.
- Index pro celý kanál bude mít cca 1,3 GB; nahrání trvá podle linky několik minut.
- Nic z toho se nemusí posílat do chatu; výsledek jde rovnou do bucketu.
