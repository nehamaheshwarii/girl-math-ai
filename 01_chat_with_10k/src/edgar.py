"""Fetch a company's latest 10-K from SEC EDGAR — public data, stdlib only.

Flow: ticker → CIK (company_tickers.json) → submissions API → latest 10-K
accession → Archives document → plain text ready for ingest.chunk_document.

SEC fair-access rules this module follows:
- Declares a User-Agent (set SEC_EDGAR_USER_AGENT to your own name/email —
  SEC asks automated tools to identify themselves).
- No bulk downloading; one filing per call, and callers should cache.
Docs: https://www.sec.gov/edgar/sec-api-documentation
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

# SEC returns 403 unless the User-Agent contains an email-shaped contact.
# Set SEC_EDGAR_USER_AGENT to your own "Name you@example.com" — do not
# commit a real email address to a public repo.
DEFAULT_UA = os.getenv("SEC_EDGAR_USER_AGENT",
                       "girl-math-ai/0.2 contact@example.com")
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVES_URL = ("https://www.sec.gov/Archives/edgar/data/{cik_int}/"
                "{accession_nodash}/{primary_document}")


@dataclass
class FilingRef:
    cik: int
    company: str
    ticker: str
    form: str
    filing_date: str
    report_date: str
    accession: str
    primary_document: str

    @property
    def url(self) -> str:
        return ARCHIVES_URL.format(
            cik_int=self.cik,
            accession_nodash=self.accession.replace("-", ""),
            primary_document=self.primary_document)


def _get(url: str, accept: str = "*/*") -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": DEFAULT_UA, "Accept": accept,
                      "Accept-Encoding": "gzip", "Host":
                      "data.sec.gov" if "data.sec.gov" in url else "www.sec.gov"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            import gzip
            data = gzip.decompress(data)
        return data


def resolve_ticker(ticker: str) -> tuple[int, str]:
    """Return (cik, company name) for a ticker, or raise ValueError."""
    data = json.loads(_get(TICKERS_URL, "application/json"))
    wanted = ticker.strip().upper()
    for row in data.values():
        if row["ticker"].upper() == wanted:
            return int(row["cik_str"]), row["title"]
    raise ValueError(f"Ticker not found in SEC company list: {ticker}")


def find_latest_10k(submissions: dict, cik: int, company: str,
                    ticker: str) -> FilingRef:
    """Pure function — pick the most recent original 10-K from submissions JSON."""
    recent = submissions["filings"]["recent"]
    forms = recent["form"]
    for i, form in enumerate(forms):
        if form == "10-K":  # originals first; 10-K/A amendments are rarer/newer edge
            return FilingRef(
                cik=cik, company=company, ticker=ticker.upper(), form=form,
                filing_date=recent["filingDate"][i],
                report_date=recent["reportDate"][i],
                accession=recent["accessionNumber"][i],
                primary_document=recent["primaryDocument"][i])
    raise ValueError(f"No 10-K found for {company} ({ticker})")


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "head"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        if tag in {"p", "div", "tr", "br", "h1", "h2", "h3", "li"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    # Inline XBRL filings carry a hidden <ix:header> full of taxonomy
    # URLs and context refs. It is metadata, not filing text — drop it
    # before parsing or it pollutes the first chunks and retrieval.
    html = re.sub(r"<ix:header.*?</ix:header>", " ", html,
                  flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r"<head.*?</head>", " ", html,
                  flags=re.IGNORECASE | re.DOTALL)
    parser = _TextExtractor()
    parser.feed(html)
    text = "".join(parser.parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def fetch_latest_10k_text(ticker: str, cache_dir: str | Path | None = None) -> tuple[FilingRef, str]:
    """Ticker → (filing reference, full plain text of the latest 10-K)."""
    cik, company = resolve_ticker(ticker)
    submissions = json.loads(_get(SUBMISSIONS_URL.format(cik=cik), "application/json"))
    ref = find_latest_10k(submissions, cik, company, ticker)

    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / f"{ticker.upper()}_{ref.accession}.txt"
        if cache_path.exists():
            return ref, cache_path.read_text(encoding="utf-8", errors="ignore")

    raw = _get(ref.url).decode("utf-8", errors="ignore")
    text = html_to_text(raw) if ref.primary_document.lower().endswith((".htm", ".html")) else raw

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
    return ref, text
