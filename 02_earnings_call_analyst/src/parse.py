"""Parse an earnings call transcript into citable segments.

A segment is one speaker's uninterrupted turn: who said it, whether it
was prepared remarks or Q&A, and the text. Every downstream output cites
a segment id, so a claim can always be traced back to the exact turn
that said it — the same citation discipline as Project 01, applied to
speech instead of a filing.

Supported formats (all common in published transcripts):
  "CEO Maya Chen: Thank you, everyone, and welcome..."
  "Sarah Lin, Meridian Capital: My question is about..."
  "Operator:" on its own line, with the speech on the following lines.
  A "Question-and-Answer Session" / "Q&A" marker line, which switches
  the part from Prepared Remarks to Q&A (the marker itself is not a
  segment).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

SPEAKER_RE = re.compile(
    r"^([A-Za-z][A-Za-z .,'&\-\(\)]{1,60}?):\s*(.*)$")
ROLE_ONLY_RE = re.compile(
    r"^(Operator|CEO|CFO|COO|CTO|President|Chairman|Chairwoman|"
    r"Analyst|Moderator)\b.*$", re.IGNORECASE)
QA_MARKER_RE = re.compile(
    r"^\s*(question-and-answer session|q&a session|q&a|"
    r"questions and answers|question and answer session)\s*:?\s*$",
    re.IGNORECASE)
PREPARED_MARKER_RE = re.compile(
    r"^\s*(prepared remarks|opening remarks)\s*:?\s*$", re.IGNORECASE)


@dataclass
class Segment:
    segment_id: int
    speaker: str
    part: str  # "Prepared Remarks" or "Q&A"
    text: str
    start_char: int = 0
    end_char: int = 0

    def citation(self) -> str:
        return f"[{self.speaker} · segment {self.segment_id}]"


def _is_speaker_line(line: str) -> tuple[str, str] | None:
    """Return (speaker, rest) if *line* starts a new speaker turn."""
    m = SPEAKER_RE.match(line)
    if not m:
        return None
    speaker, rest = m.group(1).strip(), m.group(2).strip()
    # A sentence that happens to contain a colon ("Note: revenue grew")
    # is not a speaker turn: speaker labels are short and contain no
    # sentence-ending punctuation before the colon.
    if any(p in speaker for p in (". ", "! ", "? ")):
        return None
    if len(speaker) > 60:
        return None
    return speaker, rest


def parse_transcript(text: str) -> list[Segment]:
    """Split *text* into segments, in order."""
    segments: list[Segment] = []
    part = "Prepared Remarks"
    offset = 0  # character offset of the current line in *text*

    def flush(speaker: str, lines: list[str], start: int, end: int):
        body = " ".join(l.strip() for l in lines if l.strip())
        if body:
            segments.append(Segment(
                segment_id=len(segments), speaker=speaker, part=part,
                text=body, start_char=start, end_char=end))

    current_speaker: str | None = None
    current_lines: list[str] = []
    current_start = 0

    for raw_line in text.splitlines(keepends=True):
        line = raw_line.strip()
        line_start = offset
        offset += len(raw_line)
        if not line:
            continue
        if QA_MARKER_RE.match(line):
            if current_speaker is not None:
                flush(current_speaker, current_lines, current_start, line_start)
                current_speaker, current_lines = None, []
            part = "Q&A"
            continue
        if PREPARED_MARKER_RE.match(line):
            if current_speaker is not None:
                flush(current_speaker, current_lines, current_start, line_start)
                current_speaker, current_lines = None, []
            part = "Prepared Remarks"
            continue
        speaker_line = _is_speaker_line(line)
        if speaker_line is not None:
            if current_speaker is not None:
                flush(current_speaker, current_lines, current_start, line_start)
            current_speaker, rest = speaker_line
            current_lines = [rest] if rest else []
            current_start = line_start
            continue
        if ROLE_ONLY_RE.match(line) and current_speaker is None:
            # "CEO Maya Chen" on its own line, speech follows.
            current_speaker = line
            current_lines = []
            current_start = line_start
            continue
        if current_speaker is None:
            # Text before the first speaker label is header/metadata
            # (company, date, disclaimers) — skipping it beats
            # misattributing it to the operator as if it were speech.
            continue
        else:
            current_lines.append(line)

    if current_speaker is not None:
        flush(current_speaker, current_lines, current_start, offset)
    if not segments:
        # A transcript with no speaker labels at all: keep it as one
        # segment rather than returning nothing for a real document.
        body = " ".join(text.split())
        if body:
            segments.append(Segment(segment_id=0, speaker="Transcript",
                                    part="Prepared Remarks", text=body,
                                    start_char=0, end_char=len(text)))
    return segments
