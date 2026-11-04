"""Question answering with a hard citation contract.

Rules this module enforces (the product, not just the code):
1. If retrieval finds nothing relevant, say so — never answer from memory.
2. Every claim in an LLM answer must be grounded in retrieved chunks,
   which are passed in with their citations.
3. With no API key, the app still works: it returns an extractive answer
   (the top passages, cited) instead of a generated one.
4. (v2, U4) A generated answer is only returned if it cites retrieved
   passages, using their exact citations and no invented ones. Anything
   else falls back to extractive passages — and says so, visibly.
"""
from __future__ import annotations

import os
import re

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

DEFAULT_MODEL = "gpt-4o-mini"  # v2 (U4): model choice, default is the small one.

# Citations look like "[ITEM 1A. RISK FACTORS · chunk 2]".
CITATION_RE = re.compile(r"\[[^\]]*· chunk \d+\]")

NOT_FOUND_TEXT = (
    "I couldn't find that in this filing. "
    "Try rephrasing with terms the filing itself would use "
    "(e.g. 'risk factors', 'revenue recognition', 'liquidity')."
)


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


def _citation_problem(text: str, citations: list[str]) -> str | None:
    """Return why *text* fails the citation check, or None if it passes.

    Passes only if it cites at least one retrieved passage, using that
    passage's exact citation, and cites nothing else in chunk form.
    A prompt alone cannot enforce this — the check runs on the output.
    """
    cited = CITATION_RE.findall(text)
    if not cited:
        return "it did not cite any retrieved passage"
    allowed = set(citations)
    invented = [c for c in cited if c not in allowed]
    if invented:
        return f"it cited a passage that was not retrieved ({invented[0]})"
    return None


def _extractive_answer(sources: list[dict],
                       expanded_query: str | None = None) -> dict:
    lines = ["Based only on the passages retrieved from this filing:\n"]
    for s in sources:
        # v2 (U3): trim at a sentence end, never mid-word.
        lines.append(f"- {_trim_snippet(s['text'])}… {s['citation']}")
    return {"status": "ok_extractive", "answer": "\n".join(lines),
            "citations": [s["citation"] for s in sources],
            "sources": sources, "mode": "extractive", "llm_warning": None,
            "expanded_query": expanded_query}


def answer(question: str, index: BM25Index, top_k: int = 5,
           mode: str = "auto", model: str | None = None,
           llm_client=None, api_key: str | None = None,
           expand: bool = True) -> dict:
    """Answer *question* from retrieved passages.

    mode:
      "auto"       — generated if a key/client is available, else extractive
                     (the no-key default; no warning when no key was ever
                     expected).
      "extractive" — never call an LLM, even if a key is available.
      "generated"  — try a generated answer; fall back to extractive, with
                     a visible warning, if it is unavailable or fails the
                     citation check.

    ``llm_client`` is an OpenAI-compatible client (``client.chat.completions
    .create(...)``); it exists so tests can fake the LLM without a key or
    network. In production it is normally None and the client is built
    from ``api_key`` or the OPENAI_API_KEY environment variable.

    ``expand`` (v2, U6): rank on the filing-language expansion of the
    question. The refusal gate below still uses the *original* question:
    expansion may improve ranking, but it must never manufacture
    eligibility for an answer the user did not actually ask about.
    """
    if mode not in {"auto", "extractive", "generated"}:
        raise ValueError(f"unknown mode: {mode!r} "
                         "(expected 'auto', 'extractive' or 'generated')")
    model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    # v2 (U6): rank on the expanded query; the query the user sees in
    # the result is the expanded one, so the UI can show what was added.
    expanded = index.expand_query(question) if expand else question
    hits = index.search(expanded, top_k=top_k, expand=False)
    if not hits or hits[0][1] < MIN_SCORE or not _has_specific_match(question, index, hits):
        return {
            "status": "not_found",
            "answer": NOT_FOUND_TEXT,
            "citations": [],
            "sources": [],
            "mode": "extractive",
            "llm_warning": None,
            "expanded_query": expanded,
        }

    # v2 (U2): drop weak trailing passages — show 1 strong passage,
    # not 3 noisy ones.
    hits = _quality_hits(hits)
    sources = [{"citation": c.citation(), "section": c.section,
                "text": c.text, "score": s} for c, s in hits]

    if mode == "extractive":
        return _extractive_answer(sources, expanded_query=expanded)

    has_key = bool(llm_client or api_key or os.getenv("OPENAI_API_KEY"))
    if mode == "auto" and not has_key:
        # The ordinary no-key path: extractive, no warning needed.
        return _extractive_answer(sources, expanded_query=expanded)

    generated, problem = _llm_answer(
        question, hits, model=model, llm_client=llm_client, api_key=api_key)
    if generated is not None:
        return {"status": "ok", "answer": generated,
                "citations": [s["citation"] for s in sources],
                "sources": sources, "mode": "generated", "llm_warning": None,
                "expanded_query": expanded}

    # Graceful, visible fallback (v2, U4): never fail silently, and never
    # return an unverified generated answer.
    if mode == "generated" and not has_key:
        problem = "no API key was available"
    result = _extractive_answer(sources, expanded_query=expanded)
    result["llm_warning"] = (
        f"Summarized answer unavailable ({problem}); "
        "showing the retrieved passages instead.")
    return result


def _llm_answer(question: str, hits: list[tuple[Chunk, float]],
                model: str, llm_client=None,
                api_key: str | None = None) -> tuple[str | None, str | None]:
    """Return (generated_text, problem); problem is None on success.

    The prompt is the grounding contract, and _citation_problem is the
    enforcement: a generated answer that does not cite the retrieved
    passages, exactly, is treated as a failure, not returned.
    """
    client = llm_client
    if client is None:
        try:
            from openai import OpenAI
        except ImportError:
            return None, "the OpenAI SDK is not installed"
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            return None, "no API key was available"
        client = OpenAI(api_key=key)

    citations = [c.citation() for c, _ in hits]
    context = "\n\n".join(f"{c.citation()}\n{c.text}" for c, _ in hits)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content":
                 "You are a diligence assistant answering ONLY from the "
                 "filing passages provided. Do not use outside knowledge, "
                 "do not guess figures, dates, or company facts, and do "
                 "not summarise anything that is not in the passages. "
                 "Every factual sentence must end with one of the exact "
                 "citations provided, copied verbatim (for example "
                 "[ITEM 1A. RISK FACTORS · chunk 2]). If the passages do "
                 "not contain the answer, reply exactly: I couldn't find "
                 "that in this filing. Do not invent citations."},
                {"role": "user", "content":
                 f"Allowed citations (copy verbatim, use no others):\n"
                 f"{', '.join(citations)}\n\n"
                 f"Passages:\n{context}\n\n"
                 f"Question: {question}\n\n"
                 "Answer in 2-4 sentences, each factual sentence ending "
                 "with its exact citation."},
            ],
            temperature=0,
        )
        text = resp.choices[0].message.content
    except Exception as e:  # LLM failures must not kill the answer.
        return None, f"the LLM call failed ({type(e).__name__})"
    if not text or not text.strip():
        return None, "the LLM returned an empty answer"
    problem = _citation_problem(text, citations)
    if problem:
        return None, f"the generated answer failed the citation check: {problem}"
    return text, None
