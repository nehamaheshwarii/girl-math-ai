"""Ingest a 10-K (or any long filing) into citable chunks.

Design choices (deliberate, for a portfolio project):
- Pure stdlib core. PDF text extraction uses pypdf only if it is installed,
  so the retrieval/QA logic can be tested with no heavy dependencies.
- Chunks keep their section heading and character offsets. Every answer
  downstream must cite a chunk id, which maps back to a section — that is
  the anti-hallucination contract for this project.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# 10-K section headings, e.g. "ITEM 1A. RISK FACTORS"
SECTION_RE = re.compile(r"^\s*(ITEM\s+\d+[A-Z]?\.)\s*(.*)$", re.IGNORECASE | re.MULTILINE)

# Sentence boundary: punctuation + whitespace + a capital/digit/quote/paren
# ahead. The negative lookbehind keeps initials like "U.S." together.
SENTENCE_END_RE = re.compile(r"(?<![A-Z][.!?])(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass
class Chunk:
    chunk_id: int
    section: str          # e.g. "ITEM 1A. RISK FACTORS" or "Preamble"
    text: str
    start_char: int
    end_char: int

    def citation(self) -> str:
        return f"[{self.section} · chunk {self.chunk_id}]"


@dataclass
class Document:
    title: str
    chunks: list[Chunk] = field(default_factory=list)

    @property
    def sections(self) -> list[str]:
        seen: list[str] = []
        for c in self.chunks:
            if c.section not in seen:
                seen.append(c.section)
        return seen


def load_text(path: str | Path) -> str:
    """Load .txt/.md directly, .pdf via pypdf if available."""
    p = Path(path)
    if p.suffix.lower() in {".txt", ".md"}:
        return p.read_text(encoding="utf-8", errors="ignore")
    if p.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise RuntimeError(
                "PDF support needs pypdf: pip install -r requirements.txt"
            ) from e
        reader = PdfReader(str(p))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    raise ValueError(f"Unsupported file type: {p.suffix}")


def split_sections(text: str) -> list[tuple[str, str, int]]:
    """Return (section_title, section_text, start_offset) tuples."""
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return [("Preamble", text, 0)]
    out: list[tuple[str, str, int]] = []
    if matches[0].start() > 0:
        out.append(("Preamble", text[: matches[0].start()], 0))
    for i, m in enumerate(matches):
        remainder = m.group(2).strip()
        if not remainder:
            # Real filings often put the heading on the next line:
            # "Item 1A.\nRisk Factors" — take that line as the title.
            following = text[m.end(): m.end() + 300].strip().splitlines()
            if following:
                candidate = following[0].strip()
                if 0 < len(candidate) <= 90:
                    remainder = candidate
        title = f"{m.group(1).upper()} {remainder}".strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((title, text[start:end], start))
    return out


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) offsets of sentences in *text* (offsets into text)."""
    bounds = [0]
    for m in SENTENCE_END_RE.finditer(text):
        bounds.append(m.end())
    bounds.append(len(text))
    return [(a, b) for a, b in zip(bounds, bounds[1:])
            if text[a:b].strip()]


def _hard_split(piece: tuple[int, int], text: str,
                chunk_chars: int) -> list[tuple[int, int]]:
    """Split an over-long sentence at word boundaries, never mid-word."""
    a, b = piece
    out = []
    while b - a > chunk_chars:
        cut = text.rfind(" ", a, a + chunk_chars)
        if cut <= a:  # pathological: no space at all; unavoidable hard cut
            cut = a + chunk_chars
        out.append((a, cut))
        a = cut
    out.append((a, b))
    return out


def chunk_document(text: str, title: str = "10-K", chunk_chars: int = 1800,
                   overlap_sentences: int = 2) -> Document:
    """Split a filing into overlapping, section-labelled chunks.

    Sentences are never split mid-word: each chunk is a run of whole
    sentences packed to <= chunk_chars, and chunk boundaries always fall
    on whitespace. Consecutive chunks share ``overlap_sentences`` sentences
    so context is not lost at boundaries.
    """
    doc = Document(title=title)
    cid = 0
    for section, sec_text, sec_start in split_sections(text):
        raw = sec_text
        sec_text = sec_text.strip()
        if not sec_text:
            continue
        # sec_text was stripped: shift the base offset past leading whitespace
        pad = len(raw) - len(raw.lstrip())

        pieces: list[tuple[int, int]] = []
        for span in _sentence_spans(sec_text):
            pieces.extend(_hard_split(span, sec_text, chunk_chars))

        i = 0
        while i < len(pieces):
            j = i
            length = 0
            while j < len(pieces):
                pa, pb = pieces[j]
                add = (pb - pa) + (1 if length else 0)
                if length and length + add > chunk_chars:
                    break
                length += add
                j += 1
            start, end = pieces[i][0], pieces[j - 1][1]
            chunk_text = sec_text[start:end].strip()
            if chunk_text:
                doc.chunks.append(Chunk(
                    chunk_id=cid,
                    section=section,
                    text=chunk_text,
                    start_char=sec_start + pad + start,
                    end_char=sec_start + pad + end,
                ))
                cid += 1
            if j >= len(pieces):
                break
            i = max(i + 1, j - overlap_sentences)
    return doc
