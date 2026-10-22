"""Získání seznamu videí kanálu (yt-dlp, bez API klíče) a volitelné doplnění metadat z YouTube Data API v3."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterator

import httpx

log = logging.getLogger(__name__)

CHANNEL_ID_RE = re.compile(r"^UC[\w-]{22}$")
VIDEO_ID_RE = re.compile(r"^[\w-]{11}$")


@dataclass
class ChannelInfo:
    id: str
    name: str
    url: str


@dataclass
class VideoInfo:
    id: str
    title: str
    duration_s: int | None = None
    view_count: int | None = None
    published_at: str | None = None
    description: str = ""
    thumbnail_url: str | None = None

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.id}"


def thumbnail_for(video_id: str, quality: str = "hqdefault") -> str:
    return f"https://i.ytimg.com/vi/{video_id}/{quality}.jpg"


def normalize_channel_input(value: str) -> str:
    """Z handle / ID / URL (vč. ?si=...) vyrobí kanonickou URL kanálu."""
    v = value.strip()
    if CHANNEL_ID_RE.match(v):
        return f"https://www.youtube.com/channel/{v}"
    if v.startswith("@"):
        return f"https://www.youtube.com/{v}"
    if not v.startswith("http"):
        if "/" not in v and " " not in v:
            return f"https://www.youtube.com/@{v}"
        v = "https://" + v
    v = v.split("?")[0].split("#")[0].rstrip("/")
    v = v.replace("youtube.com/", "www.youtube.com/").replace("www.www.", "www.")
    v = v.replace("m.youtube.com", "www.youtube.com")
    # odstraň případný tab (/videos, /streams, /about ...)
    v = re.sub(r"/(videos|streams|shorts|featured|about|playlists|community|live)$", "", v)
    return v


def _ydl(extra: dict | None = None, proxy: str | None = None, cookies_file: str | None = None):
    import yt_dlp  # lazy import (pomalý)

    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": "in_playlist",
        "ignoreerrors": True,
        "noprogress": True,
    }
    if proxy:
        opts["proxy"] = proxy
    if cookies_file:
        opts["cookiefile"] = cookies_file
    if extra:
        opts.update(extra)
    return yt_dlp.YoutubeDL(opts)


def resolve_channel(value: str, proxy: str | None = None, cookies_file: str | None = None) -> ChannelInfo:
    """Z libovolného zadání (handle, UC id, URL) zjistí ID a název kanálu."""
    url = normalize_channel_input(value)
    with _ydl({"playlistend": 1}, proxy, cookies_file) as ydl:
        info = ydl.extract_info(url + "/videos", download=False)
    if not info:
        raise ValueError(f"Kanál nenalezen: {value}")
    cid = info.get("channel_id") or info.get("uploader_id") or info.get("playlist_channel_id")
    name = info.get("channel") or info.get("uploader") or info.get("playlist_uploader") or ""
    if not name:
        title = info.get("title") or info.get("playlist_title") or ""
        name = re.sub(r" - (Videos|Videa)$", "", title)
    if not cid and info.get("entries"):
        e = next((e for e in info["entries"] if e), None)
        if e:
            cid = e.get("channel_id")
            name = name or e.get("channel") or e.get("uploader") or ""
    if not cid or not CHANNEL_ID_RE.match(cid):
        raise ValueError(f"Nepodařilo se zjistit ID kanálu pro: {value}")
    canonical = info.get("channel_url") or info.get("uploader_url") or f"https://www.youtube.com/channel/{cid}"
    return ChannelInfo(id=cid, name=name.strip() or cid, url=canonical)


def list_channel_videos(channel_id: str, tabs: tuple[str, ...] = ("videos", "streams"),
                        proxy: str | None = None, cookies_file: str | None = None) -> Iterator[VideoInfo]:
    """Projde taby kanálu (videa + živé přenosy) a vrátí všechna videa (flat, bez per-video requestů)."""
    seen: set[str] = set()
    for tab in tabs:
        url = f"https://www.youtube.com/channel/{channel_id}/{tab}"
        try:
            with _ydl(None, proxy, cookies_file) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as e:  # tab nemusí existovat
            log.info("Tab %s přeskočen: %s", tab, e)
            continue
        for e in (info or {}).get("entries") or []:
            if not e or not e.get("id") or e["id"] in seen:
                continue
            vid = e["id"]
            if not VIDEO_ID_RE.match(vid):
                continue
            seen.add(vid)
            dur = e.get("duration")
            yield VideoInfo(
                id=vid,
                title=e.get("title") or vid,
                duration_s=int(dur) if dur else None,
                view_count=e.get("view_count"),
                thumbnail_url=thumbnail_for(vid),
            )


# ---------- YouTube Data API v3 (volitelné doplnění metadat) ----------

_ISO_DUR = re.compile(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")


def parse_iso_duration(s: str) -> int | None:
    m = _ISO_DUR.fullmatch(s or "")
    if not m:
        return None
    d, h, mi, se = (int(x) if x else 0 for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + se


def enrich_with_data_api(videos: list[VideoInfo], api_key: str, client: httpx.Client | None = None) -> None:
    """videos.list – 50 videí na 1 quota jednotku: datum publikace, přesná délka, popis, zhlédnutí."""
    own = client is None
    client = client or httpx.Client(timeout=30)
    try:
        by_id = {v.id: v for v in videos}
        ids = list(by_id)
        for i in range(0, len(ids), 50):
            batch = ids[i:i + 50]
            r = client.get(
                "https://www.googleapis.com/youtube/v3/videos",
                params={"part": "snippet,contentDetails,statistics", "id": ",".join(batch),
                        "maxResults": 50, "key": api_key},
            )
            r.raise_for_status()
            for item in r.json().get("items", []):
                v = by_id.get(item["id"])
                if not v:
                    continue
                sn, cd, st = item.get("snippet", {}), item.get("contentDetails", {}), item.get("statistics", {})
                v.published_at = sn.get("publishedAt") or v.published_at
                v.description = sn.get("description") or v.description
                v.title = sn.get("title") or v.title
                v.duration_s = parse_iso_duration(cd.get("duration", "")) or v.duration_s
                if st.get("viewCount"):
                    v.view_count = int(st["viewCount"])
                thumbs = sn.get("thumbnails") or {}
                best = thumbs.get("maxres") or thumbs.get("standard") or thumbs.get("high") or thumbs.get("medium")
                if best:
                    v.thumbnail_url = best["url"]
    finally:
        if own:
            client.close()


def resolve_channel_data_api(value: str, api_key: str) -> ChannelInfo | None:
    """Alternativa k yt-dlp: channels.list přes handle nebo id."""
    url = normalize_channel_input(value)
    params = {"part": "snippet", "key": api_key}
    m = re.search(r"/channel/(UC[\w-]{22})", url)
    if m:
        params["id"] = m.group(1)
    else:
        m = re.search(r"/@([^/]+)", url)
        if not m:
            return None
        params["forHandle"] = m.group(1)
    r = httpx.get("https://www.googleapis.com/youtube/v3/channels", params=params, timeout=30)
    r.raise_for_status()
    items = r.json().get("items") or []
    if not items:
        return None
    it = items[0]
    return ChannelInfo(id=it["id"], name=it["snippet"]["title"], url=f"https://www.youtube.com/channel/{it['id']}")
