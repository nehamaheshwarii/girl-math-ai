"""BM25 retrieval over filing chunks — pure stdlib, no vector DB needed for v1.

Why BM25 first (product decision, documented on purpose):
filings are keyword-dense (accounting terms, item numbers, exact figures).
Lexical retrieval is strong, explainable, and free. Semantic embeddings are
a v2 upgrade path, measured against the eval set in /evals before adopting.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .ingest import Chunk

TOKEN_RE = re.compile(r"[a-z0-9$%.,]+")
STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was",
        "were", "for", "on", "by", "with", "as", "at", "from", "that", "this",
        "what", "why", "how", "which", "who", "when", "where", "does", "do",
        "did", "much", "many", "s"}


def tokenize(text: str) -> list[str]:
    out = []
    for raw in TOKEN_RE.findall(text.lower()):
        t = raw.strip(".,")
        # Drop 1-char tokens: they are mostly possessive fragments ("company's" -> "s")
        if len(t) >= 2 and t not in STOP:
            out.append(t)
    return out


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [tokenize(c.text) for c in chunks]
        self.doc_len = [len(d) for d in self.docs]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0
        df: Counter = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        q = tokenize(query)
        if not q or not self.chunks:
            return []
        scored: list[tuple[Chunk, float]] = []
        for i, chunk in enumerate(self.chunks):
            score = 0.0
            for term in q:
                if term not in self.tf[i]:
                    continue
                f = self.tf[i][term]
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avg_len or 1))
                score += self.idf.get(term, 0.0) * (f * (self.k1 + 1)) / denom
            if score > 0:
                scored.append((chunk, round(score, 4)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
