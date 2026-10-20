"""Test the app end-to-end from the terminal — no UI, no API key, no deps.

Run:  python demo.py
      python demo.py "Your own question here"

Loads the synthetic sample filing excerpt, asks three questions that ARE
in it (expect cited answers) and one that is NOT (expect: not found).
That last one is the most important test — in finance, refusing to invent
an answer is the feature.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.ingest import chunk_document, load_text
from src.retrieval import BM25Index
from src.qa import answer

SAMPLE = Path(__file__).parent / "sample_data" / "sample_10k_excerpt.txt"

QUESTIONS = [
    "What percentage of revenue do the top five customers represent?",
    "How much cash does the company hold and what credit facility does it have?",
    "Why did gross margin decline?",
    "What is the company's dividend policy for next year?",  # not in the excerpt
]


def main() -> None:
    doc = chunk_document(load_text(SAMPLE), title=SAMPLE.name)
    index = BM25Index(doc.chunks)
    print(f"Loaded {SAMPLE.name}: {len(doc.chunks)} chunks, sections: {doc.sections}\n")

    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else QUESTIONS
    for q in questions:
        result = answer(q, index)
        print(f"Q: {q}")
        print(f"Status: {result['status']}")
        print(result["answer"][:700])
        print("-" * 70)


if __name__ == "__main__":
    main()
