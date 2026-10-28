import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest import chunk_document, load_text
from src.retrieval import BM25Index
from src.qa import answer

SAMPLE = Path(__file__).parent.parent / "sample_data" / "sample_10k_excerpt.txt"


def _index():
    doc = chunk_document(load_text(SAMPLE), title="sample")
    return doc, BM25Index(doc.chunks)


def test_sections_are_detected():
    doc, _ = _index()
    assert any("RISK FACTORS" in s for s in doc.sections)
    assert any("BUSINESS" in s for s in doc.sections)
    assert len(doc.chunks) >= 3


def test_chunks_carry_citations():
    doc, _ = _index()
    for c in doc.chunks:
        assert "chunk" in c.citation()
        assert c.end_char > c.start_char


def test_retrieval_finds_risk_factors():
    _, index = _index()
    hits = index.search("What are the principal risk factors and customer concentration?")
    assert hits, "expected at least one hit"
    assert "RISK" in hits[0][0].section.upper()


def test_retrieval_finds_liquidity():
    _, index = _index()
    hits = index.search("How much cash does the company hold and what credit facility?")
    assert hits
    assert any("84" in c.text for c, _ in hits)


def test_unknown_question_is_not_found():
    _, index = _index()
    result = answer("What is the CEO's favourite colour and home address?", index)
    assert result["status"] == "not_found"
    assert result["citations"] == []


def test_generic_word_overlap_is_not_found():
    # 'dividend'/'policy' are nowhere in the filing; only generic words
    # (company, next, year) overlap. Score alone would wrongly answer.
    _, index = _index()
    result = answer("What is the company's dividend policy for next year?", index)
    assert result["status"] == "not_found"


def test_known_question_returns_cited_sources():
    _, index = _index()
    result = answer("What percentage of revenue do the top five customers represent?", index)
    assert result["status"] in {"ok", "ok_extractive"}
    assert result["citations"]
    assert any("41%" in s["text"] for s in result["sources"])


# --- v2 quick wins (U1–U3) ---

def test_chunks_never_start_or_end_mid_word():
    # U1: sentence-aware chunking — boundaries must fall on whitespace,
    # killing the "…vernance" mid-word cuts from the MSFT screenshot.
    text = load_text(SAMPLE)
    doc = chunk_document(text, title="sample")
    assert len(doc.chunks) >= 3
    for c in doc.chunks:
        if c.start_char > 0:
            prev, cur = text[c.start_char - 1], text[c.start_char]
            assert not (prev.isalnum() and cur.isalnum()), \
                f"chunk {c.chunk_id} starts mid-word: {c.text[:40]!r}"
        if c.end_char < len(text):
            last, nxt = text[c.end_char - 1], text[c.end_char]
            assert not (last.isalnum() and nxt.isalnum()), \
                f"chunk {c.chunk_id} ends mid-word: {c.text[-40:]!r}"


def test_long_sentence_splits_at_word_boundary():
    # U1: a single sentence longer than chunk_chars still never cuts mid-word.
    long_sentence = " ".join(f"word{i}" for i in range(300)) + "."
    text = "ITEM 1. BUSINESS\n\n" + long_sentence + "\n\n" + long_sentence
    doc = chunk_document(text, title="t", chunk_chars=200)
    assert len(doc.chunks) >= 2
    for c in doc.chunks:
        if c.start_char > 0:
            prev, cur = text[c.start_char - 1], text[c.start_char]
            assert not (prev.isalnum() and cur.isalnum())
        if c.end_char < len(text):
            last, nxt = text[c.end_char - 1], text[c.end_char]
            assert not (last.isalnum() and nxt.isalnum())


def test_quality_bar_drops_noisy_trailing_passages():
    # U2: every returned source must be within 50% of the top score —
    # one strong passage instead of three noisy ones.
    _, index = _index()
    result = answer("What percentage of revenue do the top five customers represent?", index)
    assert result["status"] in {"ok", "ok_extractive"}
    scores = [s["score"] for s in result["sources"]]
    assert scores, "expected at least one source"
    top = scores[0]
    assert all(s >= top * 0.5 for s in scores)


def test_snippets_trim_at_sentence_or_word_boundary():
    # U3: snippets are <= 500 chars and never end mid-word.
    from src.qa import _trim_snippet
    long = ("First sentence states the key fact about revenue. "
            "Second sentence adds supporting detail for the reader. ") * 20
    snippet = _trim_snippet(long)
    assert len(snippet) <= 500
    assert not snippet[-1].isalnum() or snippet.endswith((".", "!", "?"))
    # No-sentence-end input falls back to a word boundary, not a hard cut.
    nosent = "word " * 400
    trimmed = _trim_snippet(nosent.strip())
    assert len(trimmed) <= 500
    assert not trimmed.endswith("wor")


def test_answer_snippets_are_sentence_trimmed():
    _, index = _index()
    result = answer("What percentage of revenue do the top five customers represent?", index)
    for line in result["answer"].splitlines():
        if line.startswith("- "):
            body = line[2:].rsplit(" [", 1)[0].rstrip("…")
            assert len(body) <= 500, f"snippet too long: {len(body)}"
