"""Test the app end-to-end from the terminal — no UI, no API key, no deps.

Run:  python demo.py
      python demo.py "Your own question here"
      python demo.py --ticker AAPL "What are the principal risk factors?"

Default: loads the synthetic sample filing excerpt, asks three questions
that ARE in it (expect cited answers) and one that is NOT (expect: not
found). That last one is the most important test — in finance, refusing
to invent an answer is the feature.

With --ticker: fetches that company's latest real 10-K from SEC EDGAR
(public data) and caches it under data/cache/ so repeat runs are fast.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.ingest import chunk_document, load_text
from src.retrieval import BM25Index
from src.qa import answer

SAMPLE = Path(__file__).parent / "sample_data" / "sample_10k_excerpt.txt"
CACHE = Path(__file__).parent / "data" / "cache"

QUESTIONS = [
    "What percentage of revenue do the top five customers represent?",
    "How much cash does the company hold and what credit facility does it have?",
    "Why did gross margin decline?",
    "What is the company's dividend policy for next year?",  # not in the excerpt
]

EDGAR_QUESTIONS = [
    "What are the principal risk factors?",
    "What was total net sales and how did it change?",
    "What does the company say about liquidity and capital resources?",
]


def main() -> None:
    args = sys.argv[1:]
    ticker = None
    if "--ticker" in args:
        i = args.index("--ticker")
        ticker = args[i + 1]
        args = args[:i] + args[i + 2:]

    if ticker:
        from src.edgar import fetch_latest_10k_text
        ref, text = fetch_latest_10k_text(ticker, cache_dir=CACHE)
        doc = chunk_document(text, title=f"{ref.company} {ref.form} filed {ref.filing_date}")
        print(f"Fetched from SEC EDGAR: {ref.company} ({ref.ticker}) — {ref.form}, "
              f"filed {ref.filing_date}, period ending {ref.report_date}")
        print(f"Source: {ref.url}\n")
        default_questions = EDGAR_QUESTIONS
    else:
        doc = chunk_document(load_text(SAMPLE), title=SAMPLE.name)
        default_questions = QUESTIONS

    index = BM25Index(doc.chunks)
    print(f"Loaded {doc.title}: {len(doc.chunks)} chunks, "
          f"{len(doc.sections)} sections detected\n")

    questions = [" ".join(args)] if args else default_questions
    for q in questions:
        result = answer(q, index)
        print(f"Q: {q}")
        print(f"Status: {result['status']}")
        print(result["answer"][:700])
        print("-" * 70)


if __name__ == "__main__":
    main()
