"""Nástroje pro psaní: k otázce vrátí souvislé pasáže z přepisů (kontext), volitelně i odpověď Gemini s citacemi."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, asdict

from .chunking import format_timestamp
from .search import search

log = logging.getLogger(__name__)


@dataclass
class Passage:
    n: int                 # pořadí = číslo citace
    video_id: str
    title: str
    published_at: str | None
    start: float
    end: float
    timestamp: str
    url: str
    text: str
    score: float


# „Co si Urza myslí o X?“ → hledat jen X: jméno mluvčího a tázací rámec jinak táhnou k úvodům videí,
# kde se dotyčný představuje, místo k obsahu
_FRAME = re.compile(
    r"^\s*(co|jak|jaký|jaké|proč|kdy|kde)\s+(si\s+)?(\w+\s+){0,2}?(myslí|říká|říkal|řekl|tvrdí|soudí|vnímá|hodnotí|"
    r"vidí|chápe|přemýšlí|odpovídá|argumentuje|obhajuje|kritizuje)\s+(o|na|k|že|,)?\s*", re.IGNORECASE)
_NAME = re.compile(r"\b(urza|urzy|urzou|urzovi|urzův|urzovo|urzova)\b", re.IGNORECASE)


def topic_from_question(question: str) -> str:
    q = _NAME.sub(" ", question)
    q = _FRAME.sub("", q)
    q = re.sub(r"\s+", " ", q).strip(" ?.!")
    return q or question


def gather_context(db, question: str, embed_query=None, max_passages: int = 8, context_s: float = 90.0,
                   per_video: int = 3, mode: str = "hybrid") -> list[Passage]:
    """Najde nejrelevantnější místa a každé rozšíří o okolní řeč (±context_s/2), aby šlo citovat celou myšlenku.

    Diverzita: pasáže se berou střídavě z nejlepších videí (round-robin), překrývající se pasáže se slučují.
    """
    topic = topic_from_question(question)
    hits = search(db, topic, limit=max(3, max_passages), mode=mode, embed_query=embed_query,
                  per_video_matches=per_video)
    # round-robin přes videa: 1. shoda z každého videa, pak 2. shoda …
    queue = []
    for k in range(per_video):
        for h in hits:
            if k < len(h.matches):
                queue.append((h, h.matches[k]))
    passages: list[Passage] = []
    taken: dict[str, list[tuple[float, float]]] = {}
    half = context_s / 2
    for h, m in queue:
        if len(passages) >= max_passages:
            break
        a, b = max(0.0, m.start - half), m.end + half
        if any(a < e and b > s_ for s_, e in taken.get(h.video_id, [])):
            continue  # překryv s už vybranou pasáží stejného videa
        segs = db.get_segments_range(h.video_id, a, b)
        if not segs:
            continue
        text = " ".join(s["text"] for s in segs)
        start = float(segs[0]["start"]); end = float(segs[-1]["start"] + segs[-1]["duration"])
        taken.setdefault(h.video_id, []).append((start, end))
        passages.append(Passage(
            n=len(passages) + 1, video_id=h.video_id, title=h.title, published_at=h.published_at,
            start=start, end=end, timestamp=format_timestamp(start),
            url=f"https://www.youtube.com/watch?v={h.video_id}&t={int(start)}s", text=text, score=h.score,
        ))
    return passages


_ANSWER_PROMPT = """Jsi asistent spisovatele. Níže jsou doslovné pasáže z přepisů videí YouTube kanálu „{channel}“
(převážně mluví Urza, ve studiu bývají i hosté – pokud z textu není jasné, kdo mluví, napiš „ve videu zaznělo“).

Otázka: {question}

Úkol: Odpověz česky POUZE na základě pasáží. Shrň, co k otázce zaznělo: jaké postoje, argumenty a formulace
se objevují, jak dotyčný uvažuje a jak odpovídá (typické obraty, příklady, analogie). Každé tvrzení doplň
citací ve tvaru [n] podle čísla pasáže. Krátké doslovné citace uváděj v uvozovkách. Pokud pasáže otázku
nepokrývají nebo jen okrajově, řekni to otevřeně a nic nedomýšlej.

Pasáže:
{passages}
"""


def answer(cfg, db, question: str, embed_query=None, max_passages: int = 8, context_s: float = 90.0) -> dict:
    """RAG: pasáže → Gemini → odpověď s citacemi [n] + seznam citací s odkazy na čas."""
    passages = gather_context(db, question, embed_query, max_passages=max_passages, context_s=context_s)
    if not passages:
        return {"question": question, "answer": "K této otázce v zaindexovaných videích nic nezaznělo.", "citations": []}
    if not cfg.gemini_api_key:
        return {"question": question, "answer": None, "error": "GEMINI_API_KEY není nastaven",
                "citations": [asdict(p) for p in passages]}
    from google import genai
    from google.genai import types

    ch = db.get_channel(cfg.channel.id or None)
    rendered = "\n\n".join(f"[{p.n}] {p.title} ({p.timestamp}):\n{p.text}" for p in passages)
    prompt = _ANSWER_PROMPT.format(channel=ch["name"] if ch else cfg.channel.name, question=question, passages=rendered)
    client = genai.Client(api_key=cfg.gemini_api_key)
    resp = client.models.generate_content(
        model=cfg.gemini.answer_model, contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    return {"question": question, "answer": (resp.text or "").strip(), "model": cfg.gemini.answer_model,
            "citations": [asdict(p) for p in passages]}
