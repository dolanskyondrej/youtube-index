from fastapi.testclient import TestClient

from ytindex.api import create_app


def test_api_endpoints(cfg, db):
    client = TestClient(create_app(cfg, db))
    assert client.get("/health").json()["ok"] is True

    ch = client.get("/api/channel").json()
    assert ch["name"] == "Test kanál" and ch["stats"]["videos"] == 3

    r = client.get("/api/search", params={"q": "bitcoin"}).json()
    assert r["total"] == 2 and r["mode"] == "fts"
    hit = r["results"][0]
    assert {"video_id", "title", "url", "thumbnail", "match", "matches", "score"} <= set(hit)
    assert "<mark>" in hit["match"]["highlighted"] and "&t=" in hit["match"]["url"]

    v = client.get("/api/videos", params={"per_page": 2}).json()
    assert v["total"] == 3 and len(v["items"]) == 2
    db.set_transcript_status("vid00000003", "pending")
    assert client.get("/api/videos").json()["total"] == 2            # výchozí jen done
    assert client.get("/api/videos", params={"status": "all"}).json()["total"] == 3

    d = client.get("/api/videos/vid00000002").json()
    assert d["transcript_status"] == "done" and len(d["segments"]) == 3
    assert client.get("/api/videos/neexistuje").status_code == 404

    assert client.get("/").status_code == 200
    assert "openapi" in client.get("/openapi.json").json()
    assert client.options("/api/search", headers={"Origin": "https://x.example", "Access-Control-Request-Method": "GET"}).headers.get("access-control-allow-origin") == "*"
