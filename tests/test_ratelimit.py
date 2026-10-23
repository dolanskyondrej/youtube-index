from fastapi.testclient import TestClient

from ytindex import ratelimit
from ytindex.api import create_app
from ytindex.ratelimit import Limiter


def test_limiter_token_bucket():
    lim = Limiter(rate_per_min=60, burst=2)
    assert lim.allow("a")[0] and lim.allow("a")[0]
    ok, retry = lim.allow("a")
    assert not ok and 0 < retry <= 1.0
    assert lim.allow("b")[0]   # jiný klient má vlastní kyblík


def test_api_returns_429_after_burst(cfg, db, monkeypatch):
    monkeypatch.setattr(ratelimit, "RULES", [("/api/search", Limiter(rate_per_min=60, burst=2), Limiter(rate_per_min=600, burst=100))])
    client = TestClient(create_app(cfg, db))
    h = {"X-Forwarded-For": "203.0.113.5, 10.0.0.1"}
    assert client.get("/api/search", params={"q": "bitcoin"}, headers=h).status_code == 200
    assert client.get("/api/search", params={"q": "bitcoin"}, headers=h).status_code == 200
    r = client.get("/api/search", params={"q": "bitcoin"}, headers=h)
    assert r.status_code == 429 and r.headers["Retry-After"] and r.json()["scope"] == "ip"
    # jiná IP projde, neomezené cesty projdou
    assert client.get("/api/search", params={"q": "bitcoin"}, headers={"X-Forwarded-For": "198.51.100.9"}).status_code == 200
    assert client.get("/api/channel", headers=h).status_code == 200
