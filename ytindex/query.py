"""Převod uživatelského dotazu na syntaxi fulltextu daného backendu."""

from __future__ import annotations

import re

_WORD = re.compile(r"[\w]+", re.UNICODE)


def _parse(q: str) -> tuple[list[list[str]], list[str]]:
    """→ (fráze jako seznamy slov, samostatná slova)"""
    phrases = [_WORD.findall(p) for p in re.findall(r'"([^"]+)"', q)]
    rest = re.sub(r'"[^"]*"', " ", q)
    return [p for p in phrases if p], _WORD.findall(rest)


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
