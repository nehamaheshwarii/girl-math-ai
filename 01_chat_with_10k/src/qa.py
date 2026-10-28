"""Question answering with a hard citation contract.

Rules this module enforces (the product, not just the code):
1. If retrieval finds nothing relevant, say so — never answer from memory.
2. Every claim in an LLM answer must be grounded in retrieved chunks,
   which are passed in with their citations.
3. With no API key, the app still works: it returns an extractive answer
   (the top passages, cited) instead of a generated one.
"""
from __future__ import annotations

import os

from .ingest import Chunk
from .retrieval import BM25Index, tokenize

MIN_SCORE = 1.0  # below this, we treat retrieval as "not in the filing"
# Tuned on evals/eval_questions.json: real questions score 2.0+, while
# off-topic questions that only share stray terms score below 1.0.

SPECIFIC_IDF = 1.0  # a matched query term must be at least this specific

QUALITY_RATIO = 0.5  # v2: only show passages scoring within 50% of the top hit.
# A strong top passage plus weak 2nd/3rd ones reads as noise (the ITEM 16
# problem in the MSFT screenshot). Better one strong passage than three.

SNIPPET_CHARS = 500  # v2: trim extractive snippets to this, at sentence ends.


def _has_specific_match(question: str, index: BM25Index,
                        hits: list[tuple[Chunk, float]]) -> bool:
    """True if a *specific* query term actually appears in a retrieved chunk.

    Score alone is not enough: "What is the dividend policy next year?"
    scores >1.0 on generic words (company, next, year) even when 'dividend'
    and 'policy' appear nowhere in the filing. Requiring one high-IDF term
    to match is what makes the refusal trustworthy.
    """
    hit_tokens: set[str] = set()
    for chunk, _ in hits:
        hit_tokens.update(tokenize(chunk.text))
    return any(
        index.idf.get(term, 0.0) >= SPECIFIC_IDF and term in hit_tokens
        for term in tokenize(question)
    )


def _quality_hits(hits: list[tuple[Chunk, float]]) -> list[tuple[Chunk, float]]:
    """Keep only passages within QUALITY_RATIO of the top score (min: 1).

    ``hits`` must be score-sorted descending, as BM25Index.search returns.
    """
    if not hits:
        return []
    top = hits[0][1]
    kept = [(c, s) for c, s in hits if s >= top * QUALITY_RATIO]
    return kept or [hits[0]]


def _trim_snippet(text: str, max_chars: int = SNIPPET_CHARS) -> str:
    """Trim to <= max_chars at a sentence end; fall back to a word boundary.

    Never cuts mid-word. Returns the trimmed text without the ellipsis;
    callers add their own.
    """
    flat = " ".join(text.split())
    if len(flat) <= max_chars:
        return flat
    # Last sentence end within the budget...
    cut = max(flat.rfind(p, 0, max_chars) for p in ".!?")
    if cut > max_chars // 4:
        return flat[: cut + 1]
    # ...else the last word boundary.
    space = flat.rfind(" ", 0, max_chars)
    return flat[: space] if space > 0 else flat[:max_chars]


def answer(question: str, index: BM25Index, top_k: int = 5) -> dict:
    hits = index.search(question, top_k=top_k)
    if not hits or hits[0][1] < MIN_SCORE or not _has_specific_match(question, index, hits):
        return {
            "status": "not_found",
            "answer": "I couldn't find that in this filing. "
                      "Try rephrasing with terms the filing itself would use "
                      "(e.g. 'risk factors', 'revenue recognition', 'liquidity').",
            "citations": [],
            "sources": [],
        }

    # v2 (U2): drop weak trailing passages — show 1 strong passage,
    # not 3 noisy ones.
    hits = _quality_hits(hits)
    sources = [{"citation": c.citation(), "section": c.section,
                "text": c.text, "score": s} for c, s in hits]

    if os.getenv("OPENAI_API_KEY"):
        generated = _llm_answer(question, hits)
        if generated:
            return {"status": "ok", "answer": generated,
                    "citations": [s["citation"] for s in sources],
                    "sources": sources}

    # Extractive fallback — no key needed, nothing invented.
    lines = ["Based only on the passages retrieved from this filing:\n"]
    for s in sources:
        # v2 (U3): trim at a sentence end, never mid-word.
        lines.append(f"- {_trim_snippet(s['text'])}… {s['citation']}")
    return {"status": "ok_extractive", "answer": "\n".join(lines),
            "citations": [s["citation"] for s in sources],
            "sources": sources}


def _llm_answer(question: str, hits: list[tuple[Chunk, float]]) -> str | None:
    try:
        from openai import OpenAI
    except ImportError:
        return None
    context = "\n\n".join(f"{c.citation()}\n{c.text}" for c, _ in hits)
    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content":
                 "Answer ONLY from the provided filing passages. Cite every "
                 "claim with its [SECTION · chunk N] citation. If the passages "
                 "do not contain the answer, say you couldn't find it in this filing."},
                {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0,
        )
        return resp.choices[0].message.content
    except Exception:
        return None
