# 01 · Chat with a 10-K

Ask a company's 10-K questions and get answers **cited to the section they came from** — or a clear "I couldn't find that in this filing" instead of a hallucination.

## The problem
A 10-K is 100+ pages of dense, structured disclosure. Analysts, investors, and product/finance teams waste hours hunting for one fact ("what changed in risk factors?", "how is revenue recognised?"), and generic chatbots answer from memory — confidently, sometimes wrongly, and with no way to verify.

**Target user:** anyone doing diligence on a public company who needs a *verifiable* answer fast.

## What it does
1. Get a filing three ways: **enter a ticker** (fetches the latest real 10-K from SEC EDGAR), upload a 10-K (`.txt`, `.md`, or `.pdf`), or try the sample excerpt.
2. The filing is split by ITEM section (Risk Factors, Business, MD&A…) into overlapping, citable chunks.
3. Ask a question. BM25 retrieval finds the relevant passages; the answer is built **only** from those passages, each cited as `[ITEM 1A. RISK FACTORS · chunk 3]`.
4. Nothing relevant found → the app says so. It never answers from general knowledge.

## Run it
Fastest test — no UI, no installs, no API key:
```bash
cd 01_chat_with_10k
python demo.py                     # 3 answerable questions + 1 that must be refused
python demo.py "Your question?"    # ask the sample filing your own question
python demo.py --ticker AAPL "What are the principal risk factors?"   # a real 10-K, from SEC EDGAR
```

Full app:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
No API key needed: without one you get extractive, cited answers. Copy `.env.example` to `.env` and add `OPENAI_API_KEY` for generated answers — same citation contract, same grounding.

Tests: `pytest tests/ -q`

## Architecture
```
ticker → edgar.py (SEC EDGAR: ticker → CIK → latest 10-K → text, cached)
upload → ingest.py (section split + chunking, keeps offsets)
       → retrieval.py (BM25, pure stdlib)
       → qa.py (citation contract: answer only from retrieved chunks,
                or not_found) → app.py (Streamlit UI + sources panel)
```
EDGAR notes: follows SEC fair-access rules (declared User-Agent — set `SEC_EDGAR_USER_AGENT` to your own name/email, never commit a real email; one filing per request; downloads cached under `data/cache/`). Inline XBRL metadata headers are stripped before chunking, and very short chunks (mostly table-of-contents entries) are penalised in retrieval.

## Product decisions (and why)
- **BM25 before embeddings.** Filings are keyword-dense — accounting terms, item numbers, exact figures. Lexical search is strong, explainable, and free. Embeddings are a v2 candidate, to be adopted only if they beat BM25 on the eval set.
- **"Not found" is a feature.** In finance, a wrong confident answer costs more than no answer. The retrieval threshold is tuned on `evals/eval_questions.json`.
- **Works with no API key.** A portfolio project nobody can run helps nobody. The extractive mode also makes the system testable end-to-end in CI.

## Evals
Starter set in `evals/eval_questions.json` (against the synthetic sample excerpt): 4 answerable questions with expected sections/figures, 1 unanswerable question where the correct behaviour is `not_found`. Grow this to 30–50 questions against a real filing before making quality claims.

## Cost & latency
BM25 + extractive mode: $0, milliseconds. LLM mode: one small-model call per question over ~5 chunks; temperature 0.

## Risks & limitations
- Sample data in `sample_data/` is **synthetic**, written for this project — not a real filing. Use `--ticker` / the EDGAR box for real filings.
- Real filings repeat section headings in the table of contents; retrieval on 100+ page filings is good, not perfect. The eval set should grow against real filings before quality claims.
- BM25 can miss questions phrased very differently from the filing's own language.
- This is a research/diligence aid, **not investment advice**, and not a substitute for reading the filing.

## Roadmap
- v2: hybrid BM25 + embeddings (eval-gated) · year-over-year section diffing ("what changed in Risk Factors?") · 10-Q support
