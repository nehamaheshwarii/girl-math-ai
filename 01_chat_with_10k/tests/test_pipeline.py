import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest import Chunk, chunk_document, load_text
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


# --- v2 (U4): summarized answers, citation-verified, visible fallback ---

class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    """Mimics client.chat.completions; records the call for assertions."""
    def __init__(self, make_text):
        self.make_text = make_text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self.make_text(kwargs["messages"]))


class _FakeChat:
    def __init__(self, make_text):
        self.completions = _FakeCompletions(make_text)


class _FakeClient:
    def __init__(self, make_text):
        self.chat = _FakeChat(make_text)

    @property
    def calls(self):
        return self.chat.completions.calls


def _first_citation(messages):
    user = next(m["content"] for m in messages if m["role"] == "user")
    return re.search(r"\[[^\]]*· chunk \d+\]", user).group(0)


def _cited_answer(messages):
    cit = _first_citation(messages)
    return f"Top five customers accounted for approximately 41% of revenue. {cit}"


QUESTION_41 = "What percentage of revenue do the top five customers represent?"


def test_generated_answer_returned_when_citations_verified():
    # U4: a properly cited generated answer is returned, mode = generated.
    _, index = _index()
    client = _FakeClient(_cited_answer)
    result = answer(QUESTION_41, index, mode="generated", llm_client=client)
    assert result["status"] == "ok"
    assert result["mode"] == "generated"
    assert result["llm_warning"] is None
    assert "41%" in result["answer"]
    assert any(c in result["answer"] for c in result["citations"])


def test_generated_answer_falls_back_when_uncited():
    # U4: no citation in the output = unverified = extractive + warning,
    # never a silently returned generated answer.
    _, index = _index()
    client = _FakeClient(lambda m: "Top five customers were about 41%.")
    result = answer(QUESTION_41, index, mode="generated", llm_client=client)
    assert result["status"] == "ok_extractive"
    assert result["mode"] == "extractive"
    assert result["llm_warning"] and "citation check" in result["llm_warning"]
    assert result["answer"].startswith("Based only on the passages")


def test_generated_answer_falls_back_on_fabricated_citation():
    # U4: a citation that was not retrieved is invented, so it fails too.
    _, index = _index()
    client = _FakeClient(
        lambda m: "Top five customers were about 41%. [ITEM 9. FAKE · chunk 99]")
    result = answer(QUESTION_41, index, mode="generated", llm_client=client)
    assert result["status"] == "ok_extractive"
    assert result["llm_warning"] and "not retrieved" in result["llm_warning"]


def test_generated_answer_falls_back_on_client_error():
    # U4: an LLM failure must not kill the answer — visible fallback.
    _, index = _index()

    def _boom(messages):
        raise RuntimeError("down")

    client = _FakeClient(_boom)
    result = answer(QUESTION_41, index, mode="generated", llm_client=client)
    assert result["status"] == "ok_extractive"
    assert result["llm_warning"] and "LLM call failed" in result["llm_warning"]


def test_explicit_extractive_never_calls_llm():
    # U4: extractive mode is a hard no-LLM choice, even with a client given.
    _, index = _index()

    def _boom(messages):
        raise AssertionError("LLM must not be called in extractive mode")

    client = _FakeClient(_boom)
    result = answer(QUESTION_41, index, mode="extractive", llm_client=client)
    assert result["status"] == "ok_extractive"
    assert result["llm_warning"] is None
    assert client.calls == []


def test_generated_mode_without_key_falls_back_with_warning():
    # U4: asking for summarized with no key is not an error — it is a
    # visible extractive fallback. Env key is cleared so this is deterministic.
    _, index = _index()
    saved = os.environ.pop("OPENAI_API_KEY", None)
    try:
        result = answer(QUESTION_41, index, mode="generated")
    finally:
        if saved is not None:
            os.environ["OPENAI_API_KEY"] = saved
    assert result["status"] == "ok_extractive"
    assert result["llm_warning"] and "api key" in result["llm_warning"].lower()


def test_model_choice_is_passed_to_client():
    # U4: the UI's model choice must actually reach the LLM call.
    _, index = _index()
    client = _FakeClient(_cited_answer)
    answer(QUESTION_41, index, mode="generated", model="gpt-4o",
           llm_client=client)
    assert client.calls and client.calls[0]["model"] == "gpt-4o"
    assert client.calls[0]["temperature"] == 0


# --- v2 (U6): query expansion with filing language ---

def _u6_index():
    # The AAPL failure case, in miniature: "principal" alone points at
    # Controls (principal executive officer), while the Risk Factors
    # passage uses "risks" (plural) and "material adverse effect".
    chunks = [
        Chunk(0, "ITEM 9A. Controls and Procedures",
              "The principal executive officer and principal financial "
              "officer have concluded disclosure controls were effective.",
              0, 100),
        Chunk(1, "ITEM 1A. Risk Factors",
              "The following summarizes factors that could have a material "
              "adverse effect on the business. Risks include supply "
              "interruptions and tariffs.",
              100, 200),
        # Fillers: they raise the IDF of the specific terms above the
        # refusal threshold (which needs a term in 1 of 4+ chunks),
        # without containing any query term themselves.
        Chunk(2, "ITEM 2. Properties",
              "The company owns office buildings and warehouses in several "
              "states for its operations and distribution centers.",
              200, 300),
        Chunk(3, "ITEM 1. Business",
              "The company designs and sells warehouse automation systems "
              "and software platforms to logistics customers.",
              300, 400),
    ]
    return BM25Index(chunks)


def test_expansion_only_adds_terms_actually_in_the_filing():
    # U6 anti-gaming rule: "material"/"adverse" are not in the sample
    # excerpt's vocabulary, so asking about risk there adds nothing.
    _, index = _index()
    assert index.expand_query("risk") == "risk"
    # But the curated "sales" -> "revenue" mapping does fire on the sample.
    assert "revenue" in index.expand_query("sales").split()


def test_expansion_adds_plural_and_filing_terms_not_nonsense():
    # U6: risk -> risks (plural), material, adverse — all in this corpus.
    # Nonsense variants like "riskes" must never appear.
    index = _u6_index()
    expanded = index.expand_query("What are the principal risk factors?")
    for term in ("risks", "material", "adverse"):
        assert term in expanded.split()
    assert "riskes" not in expanded.split()


def test_expansion_flips_principal_risk_query_to_risk_factors():
    # U6, the motivating case: unexpanded, "principal" wins and the top
    # hit is Controls; expanded, the Risk Factors passage wins.
    index = _u6_index()
    q = "What are the principal risk factors?"
    baseline = index.search(q, expand=False)
    assert baseline and "Controls" in baseline[0][0].section
    expanded = index.search(q)
    assert expanded and "Risk Factors" in expanded[0][0].section


def test_answer_uses_expansion_and_reports_it():
    # U6: answer() ranks on the expanded query and reports it, so the
    # UI can show the user what filing terms were added.
    index = _u6_index()
    result = answer("What are the principal risk factors?", index)
    assert result["status"] == "ok_extractive"
    assert "Risk Factors" in result["sources"][0]["section"]
    assert "material" in result["expanded_query"].split()


def test_expansion_does_not_break_refusal():
    # U6: expansion ranks, it never creates eligibility. The dividend
    # question's own terms are still absent, so it still refuses.
    _, index = _index()
    result = answer("What is the company's dividend policy for next year?",
                    index)
    assert result["status"] == "not_found"
    assert result["citations"] == []
