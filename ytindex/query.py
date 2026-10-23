"""Převod uživatelského dotazu na syntaxi fulltextu daného backendu."""

from __future__ import annotations

import re

_WORD = re.compile(r"[\w]+", re.UNICODE)

# krátká stopslova: ve fulltextu jen přidávají šum (prefix "je"* najde "jenom", "jestli", …)
STOPWORDS = set("""
a i o u v k s z na do od po za ve ke se si je jsou být byl byla bylo to ten ta ty tu tam ale ani
jako že co kdo jak proč kde kdy který která které jaký jaká jaké tento tato toto tak tím ani nebo
aby když už jen jenom také taky mi mě ti tě ho mu ji jí my vy oni ono ona on ne ano
the a an of to in on is are was were be and or for with what why how who where when this that
""".split())


def _keep(word: str) -> bool:
    w = word.lower()
    return w not in STOPWORDS and (len(w) >= 3 or w.isdigit())


def _parse(q: str) -> tuple[list[list[str]], list[str]]:
    """→ (fráze jako seznamy slov, samostatná slova)"""
    phrases = [_WORD.findall(p) for p in re.findall(r'"([^"]+)"', q)]
    rest = re.sub(r'"[^"]*"', " ", q)
    words = [w for w in _WORD.findall(rest) if _keep(w)]
    if not words and not phrases:   # dotaz jen ze stopslov – vezmi vše, ať něco najdeme
        words = _WORD.findall(rest)
    return [p for p in phrases if p], words


def build_fts_query(q: str, mode: str = "and") -> str:
    """SQLite FTS5: fráze v uvozovkách zůstávají frází, slova dostanou prefix (* = skloňování)."""
    phrases, words = _parse(q)
    terms = ['"' + " ".join(p) + '"' for p in phrases]
    terms += [f'"{w}"*' if len(w) >= 2 else f'"{w}"' for w in words]
    return (" AND " if mode == "and" else " OR ").join(terms)


def build_tsquery(q: str, mode: str = "and") -> str:
    """PostgreSQL to_tsquery: fráze = w1 <-> w2, slova = prefix w:*."""
    phrases, words = _parse(q)
    terms = ["(" + " <-> ".join(w for w in p) + ")" for p in phrases]
    terms += [f"{w}:*" if len(w) >= 2 else w for w in words]
    return (" & " if mode == "and" else " | ").join(terms)
