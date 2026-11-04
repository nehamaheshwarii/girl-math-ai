# 02 · Earnings Call Analyst

Paste or upload an earnings call transcript and get a structured report:
a summary, guidance, risks, Q&A themes, and key figures — every item
cited to the segment (speaker turn) it came from, or it isn't in the report.

## The problem
An earnings call is an hour of speech. The numbers that move a stock are
buried in it: what guidance changed, what risks management actually
named, and what analysts kept pressing on in Q&A. Summaries from a
chatbot sound confident and cite nothing, so you can't check them.

**Target user:** anyone doing diligence on a public company who wants
the call's signal, verifiably, in two minutes.

## What it does
1. Get a transcript three ways: **paste it**, upload a `.txt`/`.md`
   file, or try the sample transcript.
2. The transcript is split by speaker into segments, split into
   Prepared Remarks and Q&A at the Q&A marker.
3. The report is **extracted, not generated**: summary sentences,
   guidance sentences, risk sentences, Q&A themes (counted, with an
   example citation each), and verbatim $ / % figures — all cited as
   `[CEO Maya Chen · segment 3]`.
4. Search the transcript from the UI: type a term and see every matching
   segment with its citation.

No API key, no model call, no invented guidance. That is the point of v1.

## Run it
Fastest test — no UI, no installs, no API key:
```bash
cd 02_earnings_call_analyst
python3 demo.py                     # structured report for the sample call
python3 demo.py path/to/transcript.txt
```

Full app:
```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

Tests: `pytest tests/ -q`

## Architecture
```
paste/upload → parse.py (speaker turns + Prepared/Q&A split, keeps offsets)
             → analysis.py (extractive: summary, guidance, risks,
                Q&A themes, figures — all cited)
             → demo.py (terminal report) / app.py (Streamlit UI + search)
```
Deliberate scope decision for v1: "Ask the transcript" is *not* built
here. Project 01 already proves BM25 Q&A, and both projects use a
`src/` package, so importing 01's retrieval into 02 in the same process
collides on the package name. Copying the BM25 code into 02 would make
each project run standalone but duplicate a core algorithm. The honest
v1 choice is structured analysis only; Ask returns as a shared-library
cleanup (or a copied, credited retrieval module) if a third project
needs it — not as hidden duplication now.

## Product decisions (and why)
- **Extracted before generated.** A generated summary that invents a
  guidance number is the worst failure mode for this product. v1 proves
  the structure and the citations first; an LLM summary mode, if added,
  must pass Project 01's citation-verification contract (exact retrieved
  citations only, visible fallback to extractive).
- **Synthetic sample, real uploads.** The sample transcript is fictional
  (Northwind Robotics, the same fictional company as Project 01's sample
  filing), so the demo runs offline with no copyright or attribution
  questions. Paste or upload a real public transcript for real calls.
- **Themes are counted, not asserted.** A Q&A theme appears only if
  segments actually match its keywords, with a count and an example
  citation — so "analysts pressed on AI" is checkable, not a vibe.

## Evals
Starter checks in `evals/eval_questions.json` (against the synthetic
sample): guidance must contain the $320–335 million Q3 range and the
raised 16–18% full-year range, risks must contain the 41% customer
concentration, the summary must contain $312 million revenue up 18%,
and Q&A themes must include AI, Guidance / Outlook, and Capital return.
Grow this against a real public transcript before claiming quality.

## Cost & latency
$0, milliseconds. Pure stdlib analysis; Streamlit is the only runtime
dependency.

## Risks & limitations
- Sample data in `sample_data/` is **synthetic**, written for this
  project — not a real call. Use paste/upload for real transcripts.
- Keyword extraction is literal: paraphrased guidance ("we're raising
  the bar") without the words guidance/outlook/expect/forecast will be
  missed, and a segment can match a theme keyword in passing. The
  citations exist so a human can check each item in seconds.
- Speaker parsing assumes "Name:" turns and a Q&A marker. Messy
  transcripts (no speaker labels, interleaved crosstalk) will parse
  less cleanly — the segment count and speakers list in the UI are the
  quick check that parsing worked.
- This is a research/diligence aid, **not investment advice**.

## Roadmap
- Ask the transcript (BM25 Q&A, shared with Project 01 — see Architecture)
- LLM summary mode with citation verification (Project 01's U4 contract)
- Fetch a public transcript by ticker (source and licensing to be checked first)
- Quarter-over-quarter comparison: same company, guidance then vs now
