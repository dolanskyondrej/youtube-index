"""Embeddingy přes Gemini (gemini-embedding-001 / gemini-embedding-2)."""

from __future__ import annotations

import logging
import math
import time

from .config import Config

log = logging.getLogger(__name__)

BATCH = 64


class Embedder:
    def __init__(self, cfg: Config):
        if not cfg.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY není nastaven – semantické hledání vyžaduje Gemini embeddingy")
        from google import genai
        self._genai_types = __import__("google.genai.types", fromlist=["types"])
        self.client = genai.Client(api_key=cfg.gemini_api_key)
        self.model = cfg.gemini.embedding_model
        self.dim = cfg.gemini.embedding_dim
        # task_type podporuje jen embedding-001; embedding-2 používá textový prefix
        self.legacy = "001" in self.model

    def _embed(self, texts: list[str], task: str) -> list[list[float]]:
        t = self._genai_types
        kwargs = {"output_dimensionality": self.dim}
        if self.legacy:
            kwargs["task_type"] = task
            contents = texts
        else:
            prefix = "task: search result | query: " if task == "RETRIEVAL_QUERY" else "title: none | text: "
            contents = [prefix + x for x in texts]
        for attempt in range(6):
            try:
                res = self.client.models.embed_content(model=self.model, contents=contents,
                                                       config=t.EmbedContentConfig(**kwargs))
                return [_normalize(e.values) for e in res.embeddings]
            except Exception as e:
                msg = str(e)
                if attempt == 5 or not any(x in msg for x in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE")):
                    raise
                wait = 5 * (2 ** attempt)
                log.warning("Embeddings: %s – čekám %ds", msg[:100], wait)
                time.sleep(wait)
        return []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), BATCH):
            out.extend(self._embed(texts[i:i + BATCH], "RETRIEVAL_DOCUMENT"))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]
