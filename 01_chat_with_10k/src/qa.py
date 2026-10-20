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
from .retrieval import BM25Index

MIN_SCORE = 1.0  # below this, we treat retrieval as "not in the filing"
# Tuned on evals/eval_questions.json: real questions score 2.0+, while
# off-topic questions that only share stray terms score below 1.0.


def answer(question: str, index: BM25Index, top_k: int = 5) -> dict:
    hits = index.search(question, top_k=top_k)
    if not hits or hits[0][1] < MIN_SCORE:
        return {
            "status": "not_found",
            "answer": "I couldn't find that in this filing. "
                      "Try rephrasing with terms the filing itself would use "
                      "(e.g. 'risk factors', 'revenue recognition', 'liquidity').",
            "citations": [],
            "sources": [],
        }

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
    for s in sources[:3]:
        snippet = " ".join(s["text"].split())[:500]
        lines.append(f"- {snippet}… {s['citation']}")
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
