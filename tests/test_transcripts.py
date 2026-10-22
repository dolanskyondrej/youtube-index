from ytindex.transcripts import parse_json3, parse_vtt, _pick_sub
from ytindex.youtube import normalize_channel_input, parse_iso_duration


def test_parse_json3():
    data = {"events": [{"tStartMs": 0, "dDurationMs": 2000, "segs": [{"utf8": "Ahoj "}, {"utf8": "světe\n"}]},
                       {"tStartMs": 2000, "dDurationMs": 1000},
                       {"tStartMs": 3000, "dDurationMs": 1500, "segs": [{"utf8": "další"}]}]}
    segs = parse_json3(data)
    assert [(s.start, s.duration, s.text) for s in segs] == [(0.0, 2.0, "Ahoj světe"), (3.0, 1.5, "další")]


def test_parse_vtt():
    vtt = "WEBVTT\nKind: captions\n\n00:00:01.000 --> 00:00:03.500\n<c>Ahoj</c> světe\n\n00:00:03.500 --> 00:00:05.000\ndruhý\n"
    segs = parse_vtt(vtt)
    assert [(s.start, round(s.duration, 1), s.text) for s in segs] == [(1.0, 2.5, "Ahoj světe"), (3.5, 1.5, "druhý")]


def test_pick_sub_priority():
    subs = {"en": [{"ext": "vtt", "url": "e"}], "cs-CZ": [{"ext": "json3", "url": "c"}, {"ext": "vtt", "url": "cv"}]}
    assert _pick_sub(subs, ["cs", "en"]) == ("cs-CZ", {"ext": "json3", "url": "c"})
    assert _pick_sub(subs, ["de", "en"]) == ("en", {"ext": "vtt", "url": "e"})
    assert _pick_sub(subs, ["de"]) is None


def test_normalize_channel_input():
    assert normalize_channel_input("https://youtube.com/@kanal?si=abc") == "https://www.youtube.com/@kanal"
    assert normalize_channel_input("@kanal") == "https://www.youtube.com/@kanal"
    assert normalize_channel_input("kanal") == "https://www.youtube.com/@kanal"
    assert normalize_channel_input("UCvl8a57ja-P4aTGV_O6Q21Q") == "https://www.youtube.com/channel/UCvl8a57ja-P4aTGV_O6Q21Q"
    assert normalize_channel_input("https://www.youtube.com/channel/UCvl8a57ja-P4aTGV_O6Q21Q/videos") == "https://www.youtube.com/channel/UCvl8a57ja-P4aTGV_O6Q21Q"


def test_parse_iso_duration():
    assert parse_iso_duration("PT1H2M3S") == 3723
    assert parse_iso_duration("PT45S") == 45
    assert parse_iso_duration("") is None
