import math

from ytindex.search import build_fts_query, search, strip_diacritics


def test_build_fts_query():
    assert build_fts_query('bitcoin "daně jsou krádež" a') == '"daně jsou krádež" AND "bitcoin"*'
    assert build_fts_query('proč je stát zbytečný') == '"stát"* AND "zbytečný"*'
    assert build_fts_query('je to') == '"je"* AND "to"*'   # jen stopslova → nic nevyhazovat
    assert build_fts_query("x; DROP TABLE") == '"DROP"* AND "TABLE"*'
    assert build_fts_query("") == ""


def test_fts_search_diacritics_insensitive(db):
    hits = search(db, "dane kradez", mode="fts")
    assert len(hits) == 1 and hits[0].video_id == "vid00000001"
    assert "<mark>Daně</mark>" in hits[0].match.highlighted
    assert "<mark>krádež</mark>" in hits[0].match.highlighted
    assert hits[0].match.url.startswith("https://www.youtube.com/watch?v=vid00000001&t=")


def test_prefix_matching_handles_inflection(db):
    hits = search(db, "svobod", mode="fts")
    assert hits and hits[0].video_id == "vid00000001"


def test_grouping_per_video_and_counts(db):
    hits = search(db, "bitcoin", mode="fts")
    assert {h.video_id for h in hits} == {"vid00000001", "vid00000002"}
    assert all(h.match_count >= 1 for h in hits)
    assert all(h.thumbnail.endswith("hqdefault.jpg") for h in hits)


def test_and_falls_back_to_or(db):
    hits = search(db, "inflace anarchokapitalismus", mode="fts")
    assert len(hits) == 2


def test_hybrid_with_fake_embedder(db):
    assert db.vec_available
    # deterministický "embedder": vektor podle přítomnosti slov
    vocab = ["bitcoin", "inflace", "vzdelavani", "stat", "deti"]

    def emb(text):
        t = strip_diacritics(text).lower()
        # + konstantní složka, aby žádný vektor nebyl nulový (reálné embeddingy nulové nejsou)
        v = [1.0 if w in t else 0.0 for w in vocab] + [0.05] + [0.0] * (db.embedding_dim - len(vocab) - 1)
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    rows = db.pending_embeddings(limit=1000)
    db.store_embeddings([(r["id"], emb(r["text"])) for r in rows])
    assert db.count_embedded()[0] == len(rows)

    # dotaz, který lexikálně nesedí ("školáci"), ale sémanticky míří na děti/vzdělávání
    hits = search(db, "děti školáci", mode="hybrid", embed_query=emb)
    assert hits[0].video_id == "vid00000003"
    # čistě vektorově
    hits = search(db, "vzdělávání", mode="vector", embed_query=emb)
    assert hits[0].video_id == "vid00000003"


def test_manual_highlight_skips_stopwords():
    from ytindex.search import _manual_highlight
    out = _manual_highlight("Jakým způsobem děti bez trestů vychovávat", "jak vychovávat děti bez trestů")
    assert "<mark>Jakým</mark>" not in out and "<mark>bez</mark>" not in out
    assert "<mark>děti</mark>" in out and "<mark>trestů</mark>" in out and "<mark>vychovávat</mark>" in out
