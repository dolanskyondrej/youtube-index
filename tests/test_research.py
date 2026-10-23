from ytindex.research import gather_context


def test_gather_context_expands_and_dedupes(db):
    ps = gather_context(db, "bitcoin", embed_query=None, max_passages=5, context_s=20, per_video=3)
    assert ps and ps[0].n == 1
    vids = [p.video_id for p in ps]
    assert set(vids) == {"vid00000001", "vid00000002"}          # round-robin přes videa
    first = ps[0]
    assert "Bitcoin" in first.text and len(first.text) > 40        # rozšířeno o okolní segmenty
    assert first.url.endswith(f"&t={int(first.start)}s") and first.timestamp
    # pasáže v jednom videu se nepřekrývají
    for v in set(vids):
        spans = sorted((p.start, p.end) for p in ps if p.video_id == v)
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))


def test_context_and_ask_endpoints(cfg, db):
    from fastapi.testclient import TestClient
    from ytindex.api import create_app
    c = TestClient(create_app(cfg, db))
    r = c.get("/api/context", params={"q": "inflace"}).json()
    assert r["indexed_videos"] == 3 and r["passages"][0]["video_id"] == "vid00000002"
    r = c.get("/api/ask", params={"q": "inflace"}).json()      # bez klíče → pasáže ano, odpověď ne
    assert r["answer"] is None and r["error"] and r["citations"]


def test_topic_from_question():
    from ytindex.research import topic_from_question
    assert topic_from_question("Co si Urza myslí o demokracii a většinovém rozhodování?") == "demokracii a většinovém rozhodování"
    assert topic_from_question("Jak Urza argumentuje proti daním?") == "proti daním"
    assert topic_from_question("bitcoin") == "bitcoin"
    assert topic_from_question("Proč je stát zbytečný?") == "Proč je stát zbytečný"
