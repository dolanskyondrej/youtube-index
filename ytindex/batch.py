"""Přepis přes Gemini Batch API (poloviční cena, výsledky typicky do hodin, garance 24 h).

Dvě fáze, obě idempotentní:
  ytindex batch submit   – videa bez přepisu rozdělí na okna, odešle dávky, uloží jména jobů do tabulky batch_jobs
  ytindex batch collect  – stáhne hotové dávky, složí přepisy, uloží je, (volitelně čeká na všechny)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

from .chunking import Segment
from .config import Config
from .transcripts import _TRANSCRIBE_PROMPT, parse_timestamp

log = logging.getLogger(__name__)

MAX_REQUESTS_PER_JOB = 400     # menší dávky = menší dopad jedné selhané
BATCH_SCHEMA = """
CREATE TABLE IF NOT EXISTS batch_jobs (
    name TEXT PRIMARY KEY,          -- batches/...
    model TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'submitted',   -- submitted | collected | failed
    windows TEXT NOT NULL,          -- JSON: [[video_id, start, end], ...] v pořadí požadavků
    created_at TEXT DEFAULT (datetime('now')),
    collected_at TEXT,
    error TEXT
);
"""


@dataclass
class Window:
    video_id: str
    start: int
    end: int


def ensure_schema(db) -> None:
    db.conn.execute(BATCH_SCHEMA)
    db.conn.commit()


def build_windows(video_id: str, duration_s: int, window_s: int) -> list[Window]:
    out, offset = [], 0
    while offset < duration_s:
        out.append(Window(video_id, offset, min(offset + window_s, duration_s)))
        offset = out[-1].end
    return out


def parse_window_items(items: list[dict], offset: int, end: int) -> list[tuple[float, str]]:
    """Stejná logika jako online přepis: časy MM:SS jsou absolutní; relativní k oknu by se posunuly o offset."""
    parsed = [(parse_timestamp(it.get("start")), str(it.get("text", "")).strip()) for it in items]
    parsed = [(t, x) for t, x in parsed if t is not None and x]
    rel = offset > 0 and all(t <= (end - offset) + 5 for t, _ in parsed)
    base = offset if rel else 0
    return [(t + base, x) for t, x in parsed]


def assemble_segments(parsed: list[tuple[float, str]]) -> list[Segment]:
    segments: list[Segment] = []
    prev = None
    for k, (t, text) in enumerate(parsed):
        start = max(t, prev) if prev is not None else t
        nxt = parsed[k + 1][0] if k + 1 < len(parsed) else start + 10
        segments.append(Segment(start, max(nxt - start, 1.0), text))
        prev = start
    return segments


def _client(cfg: Config):
    from google import genai
    return genai.Client(api_key=cfg.gemini_api_key)


def _request(cfg: Config, w: Window):
    from google.genai import types
    schema = types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(type=types.Type.OBJECT,
                           properties={"start": types.Schema(type=types.Type.STRING, description="čas začátku MM:SS nebo H:MM:SS"),
                                       "text": types.Schema(type=types.Type.STRING)},
                           required=["start", "text"]))
    part = types.Part(
        file_data=types.FileData(file_uri=f"https://www.youtube.com/watch?v={w.video_id}"),
        video_metadata=types.VideoMetadata(start_offset=f"{w.start}s", end_offset=f"{w.end}s", fps=cfg.gemini.transcribe_fps),
    )
    return types.InlinedRequest(
        contents=[types.Content(role="user", parts=[part, types.Part(text=_TRANSCRIBE_PROMPT)])],
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema, temperature=0.0),
    )


def pending_video_ids(db) -> set[str]:
    """Videa, která už jsou v nějaké odeslané (nevyzvednuté) dávce."""
    ids: set[str] = set()
    for row in db.conn.execute("SELECT windows FROM batch_jobs WHERE state='submitted'"):
        ids.update(v for v, _, _ in json.loads(row["windows"]))
    return ids


def submit(cfg: Config, db, limit: int | None = None, ids: list[str] | None = None,
           statuses: tuple[str, ...] = ("pending", "missing", "failed")) -> list[str]:
    """Odešle dávky pro videa bez přepisu. Vrací jména vytvořených jobů."""
    ensure_schema(db)
    client = _client(cfg)
    already = pending_video_ids(db)
    todo = []
    for st in statuses:
        todo += [v for v in db.list_videos(status=st, limit=10**6, order="position ASC") if v["id"] not in already]
    if ids:
        wanted = set(ids)
        todo = [v for v in todo if v["id"] in wanted]
    if limit:
        todo = todo[:limit]
    windows: list[Window] = []
    for v in todo:
        if not v["duration_s"]:
            log.warning("%s nemá délku – přeskočeno", v["id"])
            continue
        windows += build_windows(v["id"], int(v["duration_s"]), cfg.gemini.transcribe_window_s)
    print(f"Batch submit: {len(todo)} videí, {len(windows)} oken po {cfg.gemini.transcribe_window_s} s, model {cfg.gemini.model}")
    names: list[str] = []
    # okna jednoho videa drž v jedné dávce (kvůli jednoduchému skládání) – řez jen mezi videi
    chunk: list[Window] = []
    def flush():
        if not chunk:
            return
        job = client.batches.create(model=cfg.gemini.model, src=[_request(cfg, w) for w in chunk],
                                    config={"display_name": f"ytindex-{len(names) + 1}"})
        with db.tx() as c:
            c.execute("INSERT INTO batch_jobs(name, model, windows) VALUES (?,?,?)",
                      (job.name, cfg.gemini.model, json.dumps([[w.video_id, w.start, w.end] for w in chunk])))
        for vid in {w.video_id for w in chunk}:
            db.set_transcript_status(vid, "queued")
        names.append(job.name)
        print(f"  ✔ {job.name}: {len(chunk)} oken, {len({w.video_id for w in chunk})} videí", flush=True)
        chunk.clear()
    i = 0
    while i < len(windows):
        vid = windows[i].video_id
        j = i
        while j < len(windows) and windows[j].video_id == vid:
            j += 1
        if chunk and len(chunk) + (j - i) > MAX_REQUESTS_PER_JOB:
            flush()
        chunk.extend(windows[i:j])
        i = j
    flush()
    return names


def collect(cfg: Config, db, wait: bool = False, poll_s: int = 120, on_video=None) -> dict:
    """Vyzvedne hotové dávky. wait=True čeká, dokud nejsou všechny odeslané dávky vyřízené."""
    from .chunking import chunk_segments, merge_segments

    ensure_schema(db)
    client = _client(cfg)
    done_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
    summary = {"videos_done": 0, "videos_failed": 0, "jobs_collected": 0, "jobs_failed": 0, "jobs_waiting": 0}
    while True:
        rows = db.conn.execute("SELECT name, windows FROM batch_jobs WHERE state='submitted'").fetchall()
        if not rows:
            break
        waiting = 0
        for row in rows:
            job = client.batches.get(name=row["name"])
            state = job.state.name
            if state not in done_states:
                waiting += 1
                continue
            windows = [Window(*w) for w in json.loads(row["windows"])]
            if state != "JOB_STATE_SUCCEEDED":
                err = f"{state}: {job.error}"
                with db.tx() as c:
                    c.execute("UPDATE batch_jobs SET state='failed', error=?, collected_at=datetime('now') WHERE name=?",
                              (err[:500], row["name"]))
                for vid in {w.video_id for w in windows}:
                    db.set_transcript_status(vid, "failed", error=err)
                summary["jobs_failed"] += 1
                print(f"  ✖ {row['name']} {err[:120]}", flush=True)
                continue
            responses = list(job.dest.inlined_responses or [])
            per_video: dict[str, list[tuple[float, str]]] = {}
            failed: dict[str, str] = {}
            for w, r in zip(windows, responses):
                if r.response is None or not r.response.text:
                    failed[w.video_id] = str(r.error)[:200] if r.error else "prázdná odpověď"
                    continue
                try:
                    items = json.loads(r.response.text)
                except json.JSONDecodeError:
                    items = []
                per_video.setdefault(w.video_id, []).extend(parse_window_items(items, w.start, w.end))
            for vid in {w.video_id for w in windows}:
                if vid in failed:
                    db.set_transcript_status(vid, "failed", error=f"batch okno: {failed[vid]}")
                    summary["videos_failed"] += 1
                    continue
                segs = merge_segments(assemble_segments(per_video.get(vid, [])))
                if not segs:
                    db.set_transcript_status(vid, "missing")
                    summary["videos_failed"] += 1
                    continue
                chunks = chunk_segments(segs, cfg.chunking.target_chars, cfg.chunking.max_seconds, cfg.chunking.overlap_segments)
                db.store_transcript(vid, segs, chunks, "gemini-batch", None)
                summary["videos_done"] += 1
                if on_video:
                    on_video(vid, len(segs), len(chunks))
            with db.tx() as c:
                c.execute("UPDATE batch_jobs SET state='collected', collected_at=datetime('now') WHERE name=?", (row["name"],))
            summary["jobs_collected"] += 1
            print(f"  ✔ {row['name']}: {len({w.video_id for w in windows}) - len(failed)} videí uloženo, {len(failed)} selhalo", flush=True)
        summary["jobs_waiting"] = waiting
        if not waiting or not wait:
            break
        print(f"  … čeká {waiting} dávek, další kontrola za {poll_s} s", flush=True)
        time.sleep(poll_s)
    return summary


def status(db) -> list[dict]:
    ensure_schema(db)
    return [dict(r) for r in db.conn.execute(
        "SELECT name, model, state, created_at, collected_at, error, json_array_length(windows) AS windows FROM batch_jobs ORDER BY created_at")]
