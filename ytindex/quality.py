"""Kontrola kvality přepisů: zacyklené modely (stejné slovo / fráze opakovaně za sebou).

  ytindex quality scan            – vypíše, kolik oken je zasažených (nic nemění)
  ytindex quality scan --mark     – zasažená okna zapíše jako mezery (gaps:…) → doplní `ytindex batch fill-gaps`
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass
class Loop:
    video_id: str
    start: float          # čas segmentu, kde smyčka začíná
    text: str             # opakovaná jednotka
    repeats: int
    unit_words: int


def find_loops(words: list[str], min_word_repeats: int = 8, min_phrase_repeats: int = 8,
               max_phrase: int = 8) -> list[tuple[int, int, int]]:
    """V posloupnosti slov najde opakování: → [(index začátku, délka jednotky, počet opakování)].

    Jednoslovná smyčka ("a a a a") vyžaduje min_word_repeats (krátké ne-ne-ne je normální řeč),
    víceslovná fráze ("to je to je to je") min_phrase_repeats.
    """
    n = len(words)
    hits = []
    i = 0
    while i < n:
        best = None
        for k in range(1, max_phrase + 1):
            if i + 2 * k > n:
                break
            unit = words[i:i + k]
            reps = 1
            while i + (reps + 1) * k <= n and words[i + reps * k:i + (reps + 1) * k] == unit:
                reps += 1
            need = min_word_repeats if k == 1 else min_phrase_repeats
            if reps >= need and (best is None or reps * k > best[1] * best[2]):
                best = (i, k, reps)
        if best:
            hits.append(best)
            i = best[0] + best[1] * best[2]
        else:
            i += 1
    return hits


def scan_video(video_id: str, segments, **kw) -> list[Loop]:
    """Prochází segmenty jako souvislý text (smyčka může přetékat přes hranici segmentu)."""
    words, owners = [], []
    for s in segments:
        for w in _WORD.findall(s["text"].lower()):
            words.append(w)
            owners.append(float(s["start"]))
    out = []
    for i, k, reps in find_loops(words, **kw):
        out.append(Loop(video_id, owners[i], " ".join(words[i:i + k]), reps, k))
    return out


def windows_for(loops: list[Loop], duration_s: int | None, window_s: int = 600) -> list[tuple[int, int]]:
    """Smyčky → desetiminutová okna (stejná mřížka jako batch), aby šla znovu přepsat."""
    wins = set()
    for lp in loops:
        a = int(lp.start // window_s) * window_s
        wins.add((a, min(a + window_s, duration_s) if duration_s else a + window_s))
    return sorted(wins)


def scan(db, min_word_repeats: int = 8, min_phrase_repeats: int = 8, window_s: int = 600) -> dict:
    """Projde všechna hotová videa. Vrací souhrn + detaily per video."""
    videos = db.list_videos(status="done", limit=10**6, order="position ASC")
    result = {"videos_scanned": len(videos), "videos_hit": 0, "windows": 0, "hours": 0.0, "loops": 0,
              "by_source": {}, "details": []}
    for v in videos:
        loops = scan_video(v["id"], db.get_segments(v["id"]), min_word_repeats=min_word_repeats,
                           min_phrase_repeats=min_phrase_repeats)
        if not loops:
            continue
        wins = windows_for(loops, v["duration_s"], window_s)
        src = v["transcript_source"] or "-"
        bs = result["by_source"].setdefault(src, {"videos": 0, "windows": 0})
        bs["videos"] += 1
        bs["windows"] += len(wins)
        result["videos_hit"] += 1
        result["windows"] += len(wins)
        result["hours"] += sum(b - a for a, b in wins) / 3600
        result["loops"] += len(loops)
        worst = max(loops, key=lambda l: l.repeats * l.unit_words)
        result["details"].append({
            "video_id": v["id"], "title": v["title"], "source": src, "windows": wins, "loops": len(loops),
            "example": f"{worst.text!r} ×{worst.repeats} @ {int(worst.start)}s",
        })
    result["hours"] = round(result["hours"], 1)
    return result


def mark(db, details: list[dict], reason: str = "loop") -> int:
    """Zasažená okna přidá do gaps:[…] (sloučí s existujícími). Vrací počet přidaných oken."""
    added = 0
    for d in details:
        v = db.get_video(d["video_id"])
        existing = []
        if v["transcript_error"] and v["transcript_error"].startswith("gaps:"):
            try:
                existing = json.loads(v["transcript_error"][5:])
            except json.JSONDecodeError:
                existing = []
        have = {(int(a), int(b)) for a, b, *_ in existing}
        for a, b in d["windows"]:
            if (a, b) not in have:
                existing.append([a, b, reason])
                added += 1
        existing.sort()
        db.set_transcript_status(d["video_id"], "done", source=v["transcript_source"], lang=v["transcript_lang"],
                                 error="gaps:" + json.dumps(existing))
    return added
