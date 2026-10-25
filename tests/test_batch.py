import json

from ytindex.batch import Window, assemble_segments, build_windows, ensure_schema, parse_window_items, pending_video_ids


def test_build_windows():
    ws = build_windows("v", 1250, 600)
    assert [(w.start, w.end) for w in ws] == [(0, 600), (600, 1200), (1200, 1250)]


def test_parse_items_absolute_and_relative():
    abs_items = [{"start": "20:00", "text": "a"}, {"start": "29:50", "text": "b"}]
    assert [t for t, _ in parse_window_items(abs_items, 1200, 1800)] == [1200, 1790]
    rel_items = [{"start": "00:05", "text": "a"}, {"start": "09:50", "text": "b"}]
    assert [t for t, _ in parse_window_items(rel_items, 1200, 1800)] == [1205, 1790]
    assert parse_window_items([{"start": "xx", "text": "a"}, {"start": "0:01", "text": ""}], 0, 600) == []


def test_assemble_segments_monotonic():
    segs = assemble_segments([(0, "a"), (12, "b"), (8, "c"), (30, "d")])
    assert [s.start for s in segs] == [0, 12, 12, 30]
    assert segs[-1].duration == 10 and segs[0].duration == 12


def test_pending_ids_from_jobs(db):
    ensure_schema(db)
    db.conn.execute("INSERT INTO batch_jobs(name, model, windows) VALUES (?,?,?)",
                    ("batches/x", "m", json.dumps([["vidA", 0, 600], ["vidA", 600, 900], ["vidB", 0, 100]])))
    db.conn.execute("INSERT INTO batch_jobs(name, model, state, windows) VALUES (?,?,?,?)",
                    ("batches/y", "m", "collected", json.dumps([["vidC", 0, 600]])))
    db.conn.commit()
    assert pending_video_ids(db) == {"vidA", "vidB"}
