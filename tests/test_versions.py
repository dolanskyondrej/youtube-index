from fastapi.testclient import TestClient

from ytindex.api import create_app


def test_versioned_and_alias_paths(cfg, db):
    c = TestClient(create_app(cfg, db, rate_limit=False))
    v = c.get("/api/versions").json()
    assert v["default"] == 1 and v["paths"] == {1: "/api/v1"} or v["paths"] == {"1": "/api/v1"}
    r1 = c.get("/api/v1/search", params={"q": "bitcoin"})
    r0 = c.get("/api/search", params={"q": "bitcoin"})
    assert r1.status_code == r0.status_code == 200 and r1.json()["total"] == r0.json()["total"] == 2
    assert r1.headers["X-API-Version"] == "1" and r0.headers["X-API-Version"] == "1" and r1.headers["X-Index-Version"]
    assert c.get("/api/v1/videos/vid00000002").json()["video_id"] == "vid00000002"
    assert c.get("/api/v2/search", params={"q": "x"}).status_code == 404
    paths = c.get("/openapi.json").json()["paths"]
    assert "/api/v1/search" in paths and "/api/search" not in paths      # dokumentována jen kanonická verze
