"""Jednoduchý limiter: token bucket per IP + globální strop pro drahé operace (Gemini).

Stav je v paměti instance (Cloud Run má max. 2 instance, takže reálný limit je ≤ 2× nastavený).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from starlette.requests import Request
from starlette.responses import JSONResponse


@dataclass
class Bucket:
    tokens: float
    updated: float


@dataclass
class Limiter:
    rate_per_min: float       # doplňování tokenů
    burst: int                # kapacita
    buckets: dict[str, Bucket] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def allow(self, key: str) -> tuple[bool, float]:
        """(povoleno, sekundy do dalšího tokenu)"""
        now = time.monotonic()
        with self.lock:
            b = self.buckets.get(key)
            if b is None:
                b = self.buckets[key] = Bucket(tokens=float(self.burst), updated=now)
            b.tokens = min(self.burst, b.tokens + (now - b.updated) * self.rate_per_min / 60.0)
            b.updated = now
            if b.tokens >= 1.0:
                b.tokens -= 1.0
                return True, 0.0
            retry = (1.0 - b.tokens) * 60.0 / self.rate_per_min
            if len(self.buckets) > 10_000:   # úklid starých klíčů
                stale = [k for k, v in self.buckets.items() if now - v.updated > 600]
                for k in stale:
                    self.buckets.pop(k, None)
            return False, retry


# (prefix cesty, limit per IP, globální limit) – pořadí = priorita; první shoda vyhrává
RULES: list[tuple[str, Limiter, Limiter]] = [
    ("/api/ask",     Limiter(rate_per_min=6,  burst=3),  Limiter(rate_per_min=30, burst=10)),   # Gemini generace: drahé
    ("/api/context", Limiter(rate_per_min=30, burst=10), Limiter(rate_per_min=240, burst=60)),
    ("/api/search",  Limiter(rate_per_min=60, burst=20), Limiter(rate_per_min=600, burst=100)),
    ("/mcp",         Limiter(rate_per_min=60, burst=20), Limiter(rate_per_min=600, burst=100)),
    ("/api/",        Limiter(rate_per_min=120, burst=40), Limiter(rate_per_min=1200, burst=200)),
]


def client_ip(request: Request) -> str:
    # Cloud Run / proxy: první IP v X-Forwarded-For je klient
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    for prefix, per_ip, global_ in RULES:
        if path.startswith(prefix):
            ok, retry = per_ip.allow(client_ip(request))
            scope = "ip"
            if ok:
                ok, retry = global_.allow("*")
                scope = "global"
            if not ok:
                return JSONResponse(
                    {"error": "rate_limited", "scope": scope, "retry_after_s": round(retry, 1),
                     "detail": "Příliš mnoho požadavků, zkus to za chvíli."},
                    status_code=429, headers={"Retry-After": str(max(1, int(retry + 0.999)))},
                )
            break
    return await call_next(request)
