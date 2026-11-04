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


# v2 (U6): conversational phrasing -> the filing's own language.
# This is deliberately a small, curated table, not a thesaurus: every
# entry is a term a 10-K actually uses for the conversational concept.
# "risk" is the motivating case — Risk Factors sections describe a
# "material adverse effect", so a question about risks should also match
# those words, not only the word "risk" itself.
EXPANSIONS: dict[str, list[str]] = {
    "risk": ["material", "adverse"],
    "risks": ["material", "adverse"],
    "sales": ["revenue"],
    "money": ["cash", "liquidity"],
    "debt": ["borrowings", "liabilities"],
    "lawsuit": ["litigation", "proceedings"],
    "lawsuits": ["litigation", "proceedings"],
    "headquarters": ["properties"],
    "hq": ["properties"],
    "cyber": ["cybersecurity"],
    "hacking": ["cybersecurity", "security"],
    "profit": ["income", "margin"],
    "earnings": ["income"],
}


def _variants(token: str) -> list[str]:
    """Singular/plural variants of *token* (BM25 matches exact tokens).

    Filings switch freely between "supplier" and "suppliers", "risk" and
    "risks". Without this, a question in the singular misses the plural
    passage and vice versa — a pure tokenisation gap, not a meaning gap.
    Short tokens are excluded: "it" -> "its" is a false friend, not a
    plural, and 1-2 letter tokens carry no filing meaning on their own.
    """
    if len(token) < 3:
        return []
    if token.endswith("ies") and len(token) > 3:
        return [token[:-3] + "y"]
    if token.endswith("s") and not token.endswith("ss"):
        return [token[:-1]]
    return [token + "s", token + "es"]


def expand_tokens(tokens: list[str], vocab: set[str]) -> list[str]:
    """Terms to add to a query, in deterministic order.

    Only terms actually present in this filing's vocabulary are added.
    That is the anti-gaming rule: expansion can never invent a match for
    a term the filing does not contain, and nonsense variants ("cashs")
    are dropped instead of polluting the query.
    """
    original = set(tokens)
    added: set[str] = set()
    for token in tokens:
        for candidate in EXPANSIONS.get(token, []) + _variants(token):
            if candidate in vocab and candidate not in original:
                added.add(candidate)
    return sorted(added)


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

    def expand_query(self, query: str) -> str:
        """The query plus its U6 expansion terms, or the query unchanged.

        The original question is always kept first and intact; expansion
        only appends. Callers that need the added terms alone can diff
        the token sets of the input and the output.
        """
        added = expand_tokens(tokenize(query), set(self.idf))
        return query + (" " + " ".join(added) if added else "")

    def search(self, query: str, top_k: int = 5,
               expand: bool = True) -> list[tuple[Chunk, float]]:
        # v2 (U6): by default, rank on the expanded query. Pass
        # expand=False for the unexpanded baseline (used in tests and
        # by callers that already expanded the query themselves).
        q = tokenize(self.expand_query(query)) if expand else tokenize(query)
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
                # Short-chunk penalty: table-of-contents entries in real
                # filings are keyword-perfect ("Item 1A. Risk Factors 5")
                # but content-free. Real passages are substantially longer.
                if len(chunk.text) < 200:
                    score *= 0.25
                scored.append((chunk, round(score, 4)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
