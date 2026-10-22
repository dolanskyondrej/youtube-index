from ytindex.chunking import Segment, chunk_segments, format_timestamp, merge_segments


def test_merge_drops_noise_and_sorts():
    segs = merge_segments([Segment(5, 1, " b "), Segment(0, 1, "a"), Segment(2, 1, "[Hudba]"), Segment(3, 1, "")])
    assert [s.text for s in segs] == ["a", "b"]


def test_chunking_respects_target_and_overlap():
    segs = [Segment(i * 5, 5, f"slovo{i} " * 5) for i in range(20)]
    chunks = chunk_segments(segs, target_chars=100, max_seconds=60, overlap_segments=1)
    assert len(chunks) > 1
    assert all(len(c.text) <= 100 + 40 for c in chunks)
    # překryv: poslední segment chunku N je prvním segmentem chunku N+1
    assert chunks[1].text.split()[0] in chunks[0].text
    assert chunks[0].start == 0 and chunks[-1].end == 100


def test_chunking_max_seconds():
    segs = [Segment(i * 30, 30, "x") for i in range(10)]
    chunks = chunk_segments(segs, target_chars=10_000, max_seconds=75, overlap_segments=0)
    assert all(c.end - c.start <= 90 for c in chunks)
    assert len(chunks) >= 4


def test_format_timestamp():
    assert format_timestamp(65) == "1:05"
    assert format_timestamp(3725) == "1:02:05"
