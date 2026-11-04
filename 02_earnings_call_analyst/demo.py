"""Test Project 02 end-to-end from the terminal — no UI, no API key, no deps.

Run:  python demo.py
      python demo.py path/to/transcript.txt

Default: analyses the synthetic sample transcript in sample_data/ and
prints the structured report: summary, guidance, risks, Q&A themes, and
key figures — every item cited to the segment it came from.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.parse import parse_transcript  # noqa: E402
from src.analysis import analyze  # noqa: E402

SAMPLE = Path(__file__).parent / "sample_data" / "sample_earnings_call.txt"


def print_section(title: str, items: list[dict], key: str = "text"):
    print(f"\n## {title} ({len(items)})")
    if not items:
        print("  (none found in this transcript)")
        return
    for item in items:
        print(f"- {item[key]} {item['citation']}")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE
    text = path.read_text(encoding="utf-8", errors="ignore")
    segments = parse_transcript(text)
    report = analyze(segments)

    print(f"Analysed {path.name}: {report['counts']['segments']} segments "
          f"({report['counts']['prepared_remarks']} prepared remarks, "
          f"{report['counts']['qa']} Q&A)")
    # One per line: speaker names themselves contain commas
    # ("Sarah Lin, Meridian Capital"), so joining with ", " is ambiguous.
    print("Speakers:")
    for speaker in report["speakers"]:
        print(f"  - {speaker}")

    print_section("Summary", report["summary"])
    print_section("Guidance", report["guidance"])
    print_section("Risks", report["risks"])

    print(f"\n## Q&A themes ({len(report['themes'])})")
    for theme in report["themes"]:
        print(f"- {theme['theme']}: {theme['count']} segment(s), "
              f"e.g. {theme['example_citation']} "
              f"({theme['example_speaker']})")

    print(f"\n## Key figures ({len(report['figures'])})")
    for figure in report["figures"]:
        print(f"- {figure['figure']} {figure['citation']}: "
              f"{figure['sentence'][:160]}")


if __name__ == "__main__":
    main()
