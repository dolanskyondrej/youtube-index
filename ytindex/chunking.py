"""Rozdělení transkriptu (segmenty s časem) na chunky vhodné pro vyhledávání."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Segment:
    start: float   # sekundy
    duration: float
    text: str

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class Chunk:
    start: float
    end: float
    text: str


def normalize_text(text: str) -> str:
    # ">>" = značka změny mluvčího v auto-titulcích YouTube; \xa0 z json3
    text = text.replace("\n", " ").replace("\xa0", " ").replace(">>", " ")
    return " ".join(text.split())


def merge_segments(segments: list[Segment]) -> list[Segment]:
    """Vyhodí prázdné segmenty, normalizuje whitespace, seřadí podle času."""
    out = []
    for s in sorted(segments, key=lambda s: s.start):
        t = normalize_text(s.text)
        if t and t not in ("[Hudba]", "[Music]", "[Applause]", "[Potlesk]"):
            out.append(Segment(s.start, max(s.duration, 0.0), t))
    return out


def chunk_segments(segments: list[Segment], target_chars: int = 450,
                   max_seconds: float = 75.0, overlap_segments: int = 1) -> list[Chunk]:
    """Spojuje po sobě jdoucí segmenty do chunků ~target_chars znaků, s malým překryvem.

    Překryv zajistí, že fráze rozseknutá na hranici dvou chunků je stále nalezitelná.
    """
    segs = merge_segments(segments)
    chunks: list[Chunk] = []
    i = 0
    n = len(segs)
    while i < n:
        j = i
        chars = 0
        while j < n:
            seg_len = len(segs[j].text) + 1
            too_long = (segs[j].end - segs[i].start) > max_seconds
            if j > i and (chars + seg_len > target_chars or too_long):
                break
            chars += seg_len
            j += 1
        window = segs[i:j]
        chunks.append(Chunk(
            start=window[0].start,
            end=max(window[-1].end, window[0].start + 0.5),
            text=" ".join(s.text for s in window),
        ))
        if j >= n:
            break
        i = max(j - overlap_segments, i + 1)
    return chunks


def format_timestamp(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
