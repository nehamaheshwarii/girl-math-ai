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
        title = f"{m.group(1).upper()} {m.group(2).strip()}".strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((title, text[start:end], start))
    return out


def chunk_document(text: str, title: str = "10-K", chunk_chars: int = 1800,
                   overlap: int = 200) -> Document:
    """Split a filing into overlapping, section-labelled chunks."""
    doc = Document(title=title)
    cid = 0
    for section, sec_text, sec_start in split_sections(text):
        sec_text = sec_text.strip()
        if not sec_text:
            continue
        pos = 0
        while pos < len(sec_text):
            piece = sec_text[pos: pos + chunk_chars]
            doc.chunks.append(Chunk(
                chunk_id=cid,
                section=section,
                text=piece.strip(),
                start_char=sec_start + pos,
                end_char=sec_start + pos + len(piece),
            ))
            cid += 1
            if pos + chunk_chars >= len(sec_text):
                break
            pos += max(1, chunk_chars - overlap)
    return doc
