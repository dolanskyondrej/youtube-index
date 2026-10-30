from fastapi.testclient import TestClient

from ytindex.api import create_app
from ytindex.chunking import Segment, chunk_segments, merge_segments
from ytindex.db import handle_from_input
from ytindex.search import search


def _add_second_channel(db):
    db.upsert_channel("UCother0000000000000000x", "Jiný kanál", "https://www.youtube.com/channel/UCother", "@jinykanal")
    with db.tx():
        db.upsert_video({"id": "vidOther001", "channel_id": "UCother0000000000000000x", "title": "Bitcoin jinde", "position": 0})
    segs = merge_segments([Segment(0, 5, "Bitcoin je i tady, v jiném kanálu."), Segment(5, 5, "A daně také.")])
    db.store_transcript("vidOther001", segs, chunk_segments(segs, target_chars=120), "captions", "cs")


def test_handle_parsing():
    assert handle_from_input("https://youtube.com/@kanalsvobodnehopristavu?si=abc") == "@kanalsvobodnehopristavu"
    assert handle_from_input("@kanal") == "@kanal" and handle_from_input("UCxyz") is None


def test_search_filters_by_channel_and_reports_channel(cfg, db):
    _add_second_channel(db)
    all_hits = search(db, "bitcoin", mode="fts")
    assert {h.video_id for h in all_hits} == {"vid00000001", "vid00000002", "vidOther001"}
    assert all(h.channel and h.channel["id"] for h in all_hits)
    mine = search(db, "bitcoin", mode="fts", channel_id=cfg.channel.id)
    assert {h.video_id for h in mine} == {"vid00000001", "vid00000002"}
    other = search(db, "bitcoin", mode="fts", channel_id="UCother0000000000000000x")
    assert [h.video_id for h in other] == ["vidOther001"] and other[0].channel["url"] == "https://www.youtube.com/@jinykanal"


def test_api_channel_param_and_channels_endpoint(cfg, db):
    _add_second_channel(db)
    c = TestClient(create_app(cfg, db, rate_limit=False))
    chans = c.get("/api/channels").json()
    assert chans["primary"] == cfg.channel.id and {x["id"] for x in chans["channels"]} == {cfg.channel.id, "UCother0000000000000000x"}
    assert c.get("/api/search", params={"q": "bitcoin"}).json()["total"] == 3
    r = c.get("/api/search", params={"q": "bitcoin", "channel": "primary"}).json()
    assert r["total"] == 2 and r["channel"] == cfg.channel.id and r["results"][0]["channel"]["id"] == cfg.channel.id
    assert c.get("/api/search", params={"q": "bitcoin", "channel": "@jinykanal"}).json()["total"] == 1
    assert c.get("/api/search", params={"q": "bitcoin", "channel": "UCneexistuje"}).status_code == 404
    v = c.get("/api/videos", params={"channel": "jinykanal"}).json()
    assert v["total"] == 1 and v["items"][0]["channel"]["name"] == "Jiný kanál"
    assert c.get("/api/videos/vidOther001").json()["channel"]["handle"] == "@jinykanal"
    ctx = c.get("/api/context", params={"q": "bitcoin", "channel": "primary"}).json()
    assert ctx["indexed_videos"] == 3 and all(p["channel"]["id"] == cfg.channel.id for p in ctx["passages"])
    ch = c.get("/api/channel").json()
    assert ch["id"] == cfg.channel.id and len(ch["channels"]) == 2
