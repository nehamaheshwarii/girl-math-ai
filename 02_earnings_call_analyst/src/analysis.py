"""Extractive structured analysis of an earnings call — no LLM, no key.

Everything here is extracted, not generated: every bullet, figure, and
theme points at the segment it came from. That is deliberate. A call
analyst that invents guidance is worse than useless, and an extractive
v1 makes the structured output testable end to end (see tests/) before
any LLM summary mode is added on top, with the same citation contract
Project 01 uses for its generated answers.

Output shape (from analyze()):
  summary   — headline sentences from prepared remarks
  guidance  — segments/sentences about outlook, guidance, expectations
  risks     — segments/sentences about risks, headwinds, challenges
  themes    — Q&A themes, counted, with an example citation each
  figures   — $ and % figures found, each with its sentence + citation
  speakers  — speakers in order of first appearance
  counts    — segment / prepared / Q&A totals
"""
from __future__ import annotations

import re

from .parse import Segment

GUIDANCE_WORDS = ("guidance", "outlook", "expect", "forecast", "guide",
                  "projected", "anticipat")
RISK_WORDS = ("risk", "headwind", "challenge", "uncertain", "decline",
              "pressure", "disruption", "shortage", "volatil")
# Q&A themes: keyword groups a PM/analyst would actually ask about.
THEMES: dict[str, tuple[str, ...]] = {
    "AI": ("ai", "artificial intelligence", "machine learning"),
    "Guidance / Outlook": ("guidance", "outlook", "forecast", "expect"),
    "Margins": ("margin", "gross margin"),
    "Revenue": ("revenue", "sales"),
    "Capital return": ("buyback", "repurchase", "dividend", "shareholder return"),
    "Supply chain": ("supply", "tariff", "manufacturer", "freight"),
    "Costs": ("cost", "expense", "pricing"),
}

FIGURE_RE = re.compile(
    r"\$[\d,]+(?:\.\d+)?\s*(?:billion|million|trillion)?"
    r"|\d+(?:\.\d+)?%")
SENTENCE_RE = re.compile(r"(?<![A-Z][.!?])(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def _sentences(text: str) -> list[str]:
    bounds = [0]
    for m in SENTENCE_RE.finditer(text):
        bounds.append(m.end())
    bounds.append(len(text))
    return [text[a:b].strip() for a, b in zip(bounds, bounds[1:])
            if text[a:b].strip()]


def _contains(text: str, words: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(w in lowered for w in words)


def _trim(text: str, max_chars: int = 500) -> str:
    flat = " ".join(text.split())
    if len(flat) <= max_chars:
        return flat
    cut = max(flat.rfind(p, 0, max_chars) for p in ".!?")
    if cut > max_chars // 4:
        return flat[: cut + 1]
    space = flat.rfind(" ", 0, max_chars)
    return flat[: space] if space > 0 else flat[:max_chars]


def _matching_sentences(segment: Segment,
                         words: tuple[str, ...],
                         context: bool = False) -> list[dict]:
    """Sentences in *segment* matching *words*.

    With context=True, a matching sentence's neighbours in the same
    segment are included too. Risk is the case for this: a speaker
    introduces a risk ("A word on risk."), then describes it over the
    next sentences without repeating the word "risk" — sentence-only
    matching would return the introduction and miss the actual risk.
    Guidance sentences, by contrast, usually state the guidance in the
    sentence itself, so guidance stays sentence-only.
    """
    sentences = _sentences(segment.text)
    matching = {i for i, s in enumerate(sentences) if _contains(s, words)}
    if context:
        expanded = set(matching)
        for i in matching:
            if i > 0:
                expanded.add(i - 1)
            if i + 1 < len(sentences):
                expanded.add(i + 1)
        matching = expanded
    return [{"citation": segment.citation(),
             "speaker": segment.speaker,
             "part": segment.part,
             "text": sentences[i]}
            for i in sorted(matching)]


def analyze(segments: list[Segment]) -> dict:
    prepared = [s for s in segments if s.part == "Prepared Remarks"]
    qa = [s for s in segments if s.part == "Q&A"]

    # Summary: headline sentences from prepared remarks — prefer ones
    # carrying a number or an outlook word, fall back to the opening.
    summary: list[dict] = []
    for segment in prepared:
        for sentence in _sentences(segment.text):
            if FIGURE_RE.search(sentence) or _contains(sentence, GUIDANCE_WORDS):
                summary.append({"citation": segment.citation(),
                                "speaker": segment.speaker,
                                "text": sentence})
            if len(summary) >= 5:
                break
        if len(summary) >= 5:
            break
    if not summary and prepared:
        for sentence in _sentences(prepared[0].text)[:3]:
            summary.append({"citation": prepared[0].citation(),
                            "speaker": prepared[0].speaker,
                            "text": sentence})

    guidance: list[dict] = []
    for segment in segments:
        guidance.extend(_matching_sentences(segment, GUIDANCE_WORDS))

    risks: list[dict] = []
    for segment in segments:
        risks.extend(_matching_sentences(segment, RISK_WORDS, context=True))

    # Q&A themes: count segments per theme (a segment counts once per
    # theme, even if it hits several keywords), most discussed first.
    themes: list[dict] = []
    for name, keywords in THEMES.items():
        matching = [s for s in qa if _contains(s.text, keywords)]
        if matching:
            themes.append({"theme": name, "count": len(matching),
                           "example_citation": matching[0].citation(),
                           "example_speaker": matching[0].speaker})
    themes.sort(key=lambda t: (-t["count"], t["theme"]))

    # Figures: verbatim $ / % figures, each with its sentence + citation.
    figures: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for segment in segments:
        for sentence in _sentences(segment.text):
            for figure in FIGURE_RE.findall(sentence):
                key = (figure, segment.citation())
                if key in seen:
                    continue
                seen.add(key)
                figures.append({"figure": figure,
                                "citation": segment.citation(),
                                "speaker": segment.speaker,
                                "sentence": sentence})

    speakers: list[str] = []
    for segment in segments:
        if segment.speaker not in speakers:
            speakers.append(segment.speaker)

    return {
        "summary": summary,
        "guidance": guidance,
        "risks": risks,
        "themes": themes,
        "figures": figures,
        "speakers": speakers,
        "counts": {"segments": len(segments),
                   "prepared_remarks": len(prepared),
                   "qa": len(qa)},
    }
