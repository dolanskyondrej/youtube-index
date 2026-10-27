from ytindex.quality import find_loops, scan_video, windows_for


def test_find_loops_word_and_phrase():
    w = "to je a a a a a prostě to je to je to je konec".split()
    hits = find_loops(w, min_word_repeats=4, min_phrase_repeats=3)
    assert (2, 1, 5) in hits            # "a" ×5
    assert (8, 2, 3) in hits            # "to je" ×3
    assert find_loops("ne ne ne ano".split(), min_word_repeats=4) == []          # 3 stejná slova = normální řeč
    assert find_loops("ne ne ne ano".split(), min_word_repeats=3) == [(0, 1, 3)]


def test_scan_video_and_windows():
    segs = [{"start": 10.0, "text": "Dobrý den, vítejte"}, {"start": 700.0, "text": "a tak dále a tak dále a tak dále a tak dále"},
            {"start": 1300.0, "text": "normální věta"}]
    loops = scan_video("v", segs, min_word_repeats=4, min_phrase_repeats=4)
    assert len(loops) == 1 and loops[0].text == "a tak dále" and loops[0].repeats == 4 and loops[0].start == 700.0
    assert windows_for(loops, 1500) == [(600, 1200)]
