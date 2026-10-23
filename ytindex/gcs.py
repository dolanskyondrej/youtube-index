"""SQLite index v Cloud Storage: levný provoz bez databázového serveru (Cloud Run škáluje na nulu).

YTINDEX_GCS_DB=gs://bucket/index.sqlite
  cloud-serve  → stáhne index při startu instance (nové instance = nejnovější index)
  cloud-ingest → stáhne, doplní, nahraje zpět
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


def parse_gs(url: str) -> tuple[str, str]:
    assert url.startswith("gs://"), url
    bucket, _, name = url[5:].partition("/")
    return bucket, name


def generation(gs_url: str) -> int | None:
    """Číslo verze objektu v GCS (mění se při každém nahrání) – levná kontrola, zda je nový index."""
    from google.cloud import storage

    bucket, name = parse_gs(gs_url)
    blob = storage.Client().bucket(bucket).get_blob(name)
    return int(blob.generation) if blob else None


def download(gs_url: str, local: str | os.PathLike) -> bool:
    """Stáhne objekt do local; vrací False, když v bucketu ještě není."""
    from google.cloud import storage

    bucket, name = parse_gs(gs_url)
    blob = storage.Client().bucket(bucket).blob(name)
    if not blob.exists():
        log.info("GCS: %s zatím neexistuje", gs_url)
        return False
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(local) + ".download"
    blob.download_to_filename(tmp)
    for suffix in ("-wal", "-shm"):   # zahodit cizí WAL soubory, stažený soubor je kompletní
        Path(str(local) + suffix).unlink(missing_ok=True)
    shutil.move(tmp, local)
    log.info("GCS: staženo %s (%.1f MB)", gs_url, Path(local).stat().st_size / 1e6)
    return True


def upload(local: str | os.PathLike, gs_url: str) -> None:
    """Nahraje konzistentní kopii SQLite (přes VACUUM INTO, aby se nepřenášel WAL)."""
    import sqlite3

    from google.cloud import storage

    snapshot = str(local) + ".snapshot"
    Path(snapshot).unlink(missing_ok=True)
    conn = sqlite3.connect(local)
    conn.execute("VACUUM INTO ?", (snapshot,))
    conn.close()
    bucket, name = parse_gs(gs_url)
    storage.Client().bucket(bucket).blob(name).upload_from_filename(snapshot)
    Path(snapshot).unlink(missing_ok=True)
    log.info("GCS: nahráno %s", gs_url)
