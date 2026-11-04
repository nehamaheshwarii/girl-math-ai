import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parse import parse_transcript
from src.analysis import analyze

SAMPLE = Path(__file__).parent.parent / "sample_data" / "sample_earnings_call.txt"


def _report():
    return analyze(parse_transcript(SAMPLE.read_text(encoding="utf-8")))


def test_segments_split_by_speaker_and_part():
    segments = parse_transcript(SAMPLE.read_text(encoding="utf-8"))
    assert segments, "expected segments"
    assert any(s.part == "Prepared Remarks" for s in segments)
    assert any(s.part == "Q&A" for s in segments)
    assert any("Maya Chen" in s.speaker for s in segments)
    assert any("Meridian Capital" in s.speaker for s in segments)
    # Segment ids are unique and citations name the speaker.
    ids = [s.segment_id for s in segments]
    assert len(ids) == len(set(ids))
    for s in segments:
        assert s.speaker in s.citation()
        assert str(s.segment_id) in s.citation()


def test_qa_marker_switches_part():
    segments = parse_transcript(
        "CEO Ana: Good morning.\n\n"
        "Question-and-Answer Session\n\n"
        "Analyst Bob: What about margins?\n"
        "CEO Ana: Margins will recover.\n")
    assert segments[0].part == "Prepared Remarks"
    assert all(s.part == "Q&A" for s in segments[1:])


def test_summary_carries_figures_and_citations():
    report = _report()
    assert report["summary"], "expected a summary"
    assert any("$312 million" in item["text"] for item in report["summary"])
    for item in report["summary"]:
        assert item["citation"] and "segment" in item["citation"]


def test_guidance_finds_raise_and_q3_expectation():
    report = _report()
    texts = " ".join(item["text"] for item in report["guidance"])
    assert "raising guidance" in texts
    assert "$320 million" in texts or "$335 million" in texts
    assert "16%" in texts or "18%" in texts
    assert report["guidance"], "expected guidance items"


def test_risks_find_tariffs_and_customer_concentration():
    report = _report()
    texts = " ".join(item["text"] for item in report["risks"])
    assert "41%" in texts  # top-five customer concentration
    assert "tariff" in texts.lower() or "headwind" in texts.lower()


def test_qa_themes_include_ai_guidance_and_buybacks():
    report = _report()
    names = {t["theme"] for t in report["themes"]}
    assert {"AI", "Guidance / Outlook", "Capital return"} <= names
    for theme in report["themes"]:
        assert theme["count"] >= 1
        assert theme["example_citation"]


def test_figures_are_verbatim_with_citations():
    report = _report()
    figures = {f["figure"] for f in report["figures"]}
    assert "$312 million" in figures
    assert "18%" in figures
    assert "45%" in figures
    for figure in report["figures"]:
        assert figure["figure"] in figure["sentence"]
        assert figure["citation"]


def test_empty_transcript_gives_empty_report():
    report = analyze(parse_transcript(""))
    assert report["counts"]["segments"] == 0
    assert report["summary"] == []
    assert report["themes"] == []


def test_preamble_header_is_not_attributed_as_speech():
    # The sample file opens with a disclaimer header before any speaker.
    # That header must not become an "Operator" segment.
    segments = parse_transcript(SAMPLE.read_text(encoding="utf-8"))
    assert not any("SYNTHETIC DATA" in s.text for s in segments)
    assert segments[0].speaker == "Operator"
    assert "Good afternoon" in segments[0].text


def test_unlabelled_transcript_is_kept_as_one_segment():
    segments = parse_transcript(
        "Revenue grew strongly this quarter despite higher costs.")
    assert len(segments) == 1
    assert "Revenue grew" in segments[0].text
