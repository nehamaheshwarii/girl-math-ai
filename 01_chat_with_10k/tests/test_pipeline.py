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
