"""Zdroje transkriptů. Pořadí: captions (youtube-transcript-api) → ytdlp (titulky json3) → gemini (video understanding).

Každý provider vrátí (segments, lang) nebo None (= pro toto video nemá data). Výjimky = tvrdá chyba (blokace IP, síť).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass

import httpx

from .chunking import Segment
from .config import Config

log = logging.getLogger(__name__)


class TranscriptBlocked(Exception):
    """YouTube blokuje naši IP – nemá smysl dál zkoušet stejný provider."""


@dataclass
class TranscriptResult:
    segments: list[Segment]
    source: str
    lang: str | None


# ---------------- captions: youtube-transcript-api ----------------

def _proxy_config(cfg: Config):
    import os
    from youtube_transcript_api.proxies import GenericProxyConfig, WebshareProxyConfig

    user, pwd = os.environ.get("WEBSHARE_PROXY_USERNAME"), os.environ.get("WEBSHARE_PROXY_PASSWORD")
    if user and pwd:
        return WebshareProxyConfig(proxy_username=user, proxy_password=pwd)
    if cfg.proxy_url:
        return GenericProxyConfig(http_url=cfg.proxy_url, https_url=cfg.proxy_url)
    return None


def fetch_captions(video_id: str, cfg: Config) -> TranscriptResult | None:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        IpBlocked, RequestBlocked, NoTranscriptFound, TranscriptsDisabled, VideoUnavailable,
    )

    api = YouTubeTranscriptApi(proxy_config=_proxy_config(cfg))
    try:
        tlist = api.list(video_id)
    except (TranscriptsDisabled, VideoUnavailable, NoTranscriptFound):
        return None
    except (IpBlocked, RequestBlocked) as e:
        raise TranscriptBlocked(str(e).splitlines()[0]) from e

    langs = cfg.transcripts.languages
    transcript = None
    for finder in (tlist.find_manually_created_transcript, tlist.find_generated_transcript):
        try:
            transcript = finder(langs)
            break
        except NoTranscriptFound:
            continue
    if transcript is None:
        # cokoliv, co existuje (např. jen auto-titulky v jiném jazyce)
        available = list(tlist)
        if not available:
            return None
        transcript = available[0]
    fetched = transcript.fetch()
    segs = [Segment(s.start, s.duration, s.text) for s in fetched]
    return TranscriptResult(segs, "captions", transcript.language_code) if segs else None


# ---------------- ytdlp: titulky/auto-titulky ve formátu json3 ----------------

def parse_json3(data: dict) -> list[Segment]:
    segs: list[Segment] = []
    for ev in data.get("events", []):
        if not ev.get("segs"):
            continue
        text = "".join(s.get("utf8", "") for s in ev["segs"]).replace("\n", " ").strip()
        if not text:
            continue
        start = ev.get("tStartMs", 0) / 1000.0
        dur = ev.get("dDurationMs", 0) / 1000.0
        segs.append(Segment(start, dur, text))
    return segs


def parse_vtt(text: str) -> list[Segment]:
    """Jednoduchý parser WebVTT (fallback, když není json3)."""
    segs: list[Segment] = []
    ts = re.compile(r"(\d+):(\d\d):(\d\d)\.(\d{3})\s*-->\s*(\d+):(\d\d):(\d\d)\.(\d{3})")
    cur_start = cur_end = None
    buf: list[str] = []

    def flush():
        if cur_start is not None and buf:
            t = re.sub(r"<[^>]+>", "", " ".join(buf)).strip()
            if t and (not segs or segs[-1].text != t):
                segs.append(Segment(cur_start, cur_end - cur_start, t))

    for line in text.splitlines():
        m = ts.search(line)
        if m:
            flush()
            g = [int(x) for x in m.groups()]
            cur_start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
            cur_end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
            buf = []
        elif line.strip() and not line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            buf.append(line.strip())
    flush()
    return segs


def _pick_sub(subs: dict, langs: list[str]) -> tuple[str, dict] | None:
    """Vybere titulky podle priority jazyků (cs > cs-CZ > ... ), preferuje json3, jinak vtt."""
    for lang in langs:
        for key in sorted(subs):
            if key == lang or key.startswith(lang + "-") or key.startswith(lang + "_"):
                fmts = subs[key]
                for ext in ("json3", "vtt"):
                    f = next((f for f in fmts if f.get("ext") == ext), None)
                    if f:
                        return key, f
    return None


def fetch_ytdlp_subs(video_id: str, cfg: Config) -> TranscriptResult | None:
    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noprogress": True}
    if cfg.proxy_url:
        opts["proxy"] = cfg.proxy_url
    if cfg.transcripts.cookies_file:
        opts["cookiefile"] = cfg.transcripts.cookies_file
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    except yt_dlp.utils.DownloadError as e:
        msg = str(e)
        if "Sign in to confirm" in msg or "429" in msg or "bot" in msg:
            raise TranscriptBlocked(msg[:200]) from e
        return None
    if not info:
        return None
    langs = cfg.transcripts.languages
    chosen = _pick_sub(info.get("subtitles") or {}, langs) or _pick_sub(info.get("automatic_captions") or {}, langs)
    if not chosen:
        return None
    lang, fmt = chosen
    with httpx.Client(timeout=60, proxy=cfg.proxy_url or None) as client:
        r = client.get(fmt["url"])
        r.raise_for_status()
        segs = parse_json3(r.json()) if fmt.get("ext") == "json3" else parse_vtt(r.text)
    return TranscriptResult(segs, "ytdlp", lang) if segs else None


# ---------------- gemini: transkripce přímo z YouTube URL ----------------

_TRANSCRIBE_PROMPT = (
    "Přepiš doslovně veškerou mluvenou řeč v tomto úseku videa v původním jazyce. "
    "Rozděl přepis na krátké úseky po cca 10–20 sekundách. Pro každý úsek uveď časovou značku začátku "
    "ve formátu MM:SS nebo H:MM:SS (pole start) a text (pole text). "
    "Vynech hudbu a zvukové efekty. Nic nepřidávej ani neshrnuj."
)

# Pozn.: Gemini má správné časové značky jen ve formátu MM:SS (nativní formát z tréninku);
# číselné sekundy driftují (~1,6×). Ověřeno na gemini-3.5-flash-lite.


def parse_timestamp(value) -> float | None:
    """'1:02:03' / '02:03' / '63' → sekundy."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    parts = s.split(":")
    try:
        nums = [float(p.replace(",", ".")) for p in parts]
    except ValueError:
        return None
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    return nums[-3] * 3600 + nums[-2] * 60 + nums[-1]


def fetch_gemini(video_id: str, cfg: Config, duration_s: int | None) -> TranscriptResult | None:
    """Gemini si veřejné YouTube video stáhne samo (bez problému s blokací IP).

    Dlouhá videa se posílají po oknech (transcribe_window_s) pomocí video_metadata start/end offset.
    """
    if not cfg.gemini_api_key:
        log.warning("GEMINI_API_KEY není nastaven – provider gemini přeskočen")
        return None
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=cfg.gemini_api_key)
    url = f"https://www.youtube.com/watch?v={video_id}"
    window = cfg.gemini.transcribe_window_s
    total = duration_s or window
    schema = types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(
            type=types.Type.OBJECT,
            properties={"start": types.Schema(type=types.Type.STRING, description="čas začátku MM:SS nebo H:MM:SS"),
                        "text": types.Schema(type=types.Type.STRING)},
            required=["start", "text"],
        ),
    )
    windows = []
    offset = 0
    while offset < total:
        windows.append((offset, min(offset + window, total)))
        offset = windows[-1][1]

    def transcribe_window(bounds: tuple[int, int]) -> list[tuple[float, str]]:
        offset, end = bounds
        part = types.Part(
            file_data=types.FileData(file_uri=url),
            video_metadata=types.VideoMetadata(start_offset=f"{offset}s", end_offset=f"{end}s",
                                               fps=cfg.gemini.transcribe_fps),
        )
        resp = _with_retry(lambda: client.models.generate_content(
            model=cfg.gemini.model,
            contents=[types.Content(role="user", parts=[part, types.Part(text=_TRANSCRIBE_PROMPT)])],
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema,
                                               temperature=0.0),
        ))
        try:
            items = json.loads(resp.text or "[]")
        except json.JSONDecodeError:
            items = []
        parsed = [(parse_timestamp(it.get("start")), str(it.get("text", "")).strip()) for it in items]
        parsed = [(t, x) for t, x in parsed if t is not None and x]
        # model vrací časy absolutní (od začátku videa); pokud by byly relativní k oknu, přičteme offset
        rel = offset > 0 and all(t <= (end - offset) + 5 for t, _ in parsed)
        base = offset if rel else 0
        return [(t + base, x) for t, x in parsed]

    # okna se přepisují souběžně (Gemini je I/O bound), výsledky se skládají v pořadí
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, cfg.gemini.concurrency)) as pool:
        results = list(pool.map(transcribe_window, windows))

    parsed = [item for chunk in results for item in chunk]
    segments: list[Segment] = []
    prev = None
    for k, (t, text) in enumerate(parsed):
        start = t
        if prev is not None and start < prev:   # monotónnost (model občas vrátí zpětný skok)
            start = prev
        nxt = parsed[k + 1][0] if k + 1 < len(parsed) else start + 10
        segments.append(Segment(start, max(nxt - start, 1.0), text))
        prev = start
    return TranscriptResult(segments, "gemini", None) if segments else None


def _with_retry(fn, attempts: int = 5, base: float = 5.0):
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # 429 / 503
            msg = str(e)
            if i == attempts - 1 or not any(x in msg for x in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE")):
                raise
            wait = base * (2 ** i)
            log.warning("Gemini: %s – čekám %.0fs", msg[:120], wait)
            time.sleep(wait)


# ---------------- orchestrace ----------------

PROVIDERS = {
    "captions": lambda vid, cfg, dur: fetch_captions(vid, cfg),
    "ytdlp": lambda vid, cfg, dur: fetch_ytdlp_subs(vid, cfg),
    "gemini": fetch_gemini,
}


def get_transcript(video_id: str, cfg: Config, duration_s: int | None = None,
                   providers: list[str] | None = None,
                   disabled: set[str] | None = None) -> TranscriptResult | None:
    """Zkouší providery v pořadí. `disabled` je sdílená množina providerů, které jsou v tomto běhu blokované."""
    disabled = disabled if disabled is not None else set()
    if not cfg.gemini_api_key:
        disabled.add("gemini")  # bez klíče nemá smysl zkoušet; nesmí to vypadat jako "video bez transkriptu"
    last_err: Exception | None = None
    for name in providers or cfg.transcripts.providers:
        if name in disabled or name not in PROVIDERS:
            continue
        try:
            res = PROVIDERS[name](video_id, cfg, duration_s)
        except TranscriptBlocked as e:
            log.warning("Provider %s blokován (%s) – pro zbytek běhu vypnut", name, e)
            disabled.add(name)
            last_err = e
            continue
        except Exception as e:
            log.warning("Provider %s selhal pro %s: %s", name, video_id, str(e)[:200])
            last_err = e
            continue
        if res and res.segments:
            return res
    if last_err:
        raise last_err
    return None
