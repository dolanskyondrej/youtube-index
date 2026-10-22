import pytest

from ytindex.chunking import Segment, chunk_segments, merge_segments
from ytindex.config import Config
from ytindex.db import Database

TRANSCRIPTS = {
    "vid00000001": ("Svoboda a stát: debata o anarchokapitalismu", [
        Segment(0, 4, "Dobrý večer, vítám vás ve Svobodném přístavu."),
        Segment(4, 5, "Dnes se budeme bavit o tom, co je to anarchokapitalismus"),
        Segment(9, 5, "a proč si myslíme, že stát je zbytečný."),
        Segment(14, 6, "Bitcoin je podle mě nejdůležitější vynález posledních let."),
        Segment(20, 5, "Daně jsou krádež, to říkal už Rothbard."),
    ]),
    "vid00000002": ("Ekonomie pro začátečníky", [
        Segment(0, 5, "Inflace je růst množství peněz v oběhu."),
        Segment(5, 5, "Centrální banka tiskne peníze a tím znehodnocuje úspory."),
        Segment(10, 5, "Bitcoin má pevně dané množství, dvacet jedna milionů."),
    ]),
    "vid00000003": ("Rozhovor o vzdělávání", [
        Segment(0, 5, "Domácí vzdělávání je v Česku legální, ale komplikované."),
        Segment(5, 5, "Děti se učí nejlépe, když je to baví."),
    ]),
}


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.db_path = str(tmp_path / "t.sqlite")
    c.channel.id, c.channel.name, c.channel.url = "UCtest00000000000000000x", "Test kanál", "https://www.youtube.com/channel/UCtest"
    return c


@pytest.fixture
def db(cfg):
    d = Database(cfg.db_path, cfg.gemini.embedding_dim)
    d.upsert_channel(cfg.channel.id, cfg.channel.name, cfg.channel.url)
    with d.tx():
        for i, (vid, (title, _)) in enumerate(TRANSCRIPTS.items()):
            d.upsert_video({"id": vid, "channel_id": cfg.channel.id, "title": title,
                            "published_at": f"2024-0{i+1}-01T00:00:00Z", "duration_s": 100 * (i + 1),
                            "thumbnail_url": f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"})
    for vid, (_, segs) in TRANSCRIPTS.items():
        segs = merge_segments(segs)
        d.store_transcript(vid, segs, chunk_segments(segs, target_chars=120), "captions", "cs")
    yield d
    d.close()
