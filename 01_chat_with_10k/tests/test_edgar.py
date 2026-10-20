import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.edgar import FilingRef, find_latest_10k, html_to_text

SUBMISSIONS = {
    "filings": {"recent": {
        "form": ["10-Q", "10-K/A", "10-K", "10-K"],
        "filingDate": ["2026-05-01", "2026-03-01", "2025-10-31", "2024-11-01"],
        "reportDate": ["2026-03-28", "2025-09-27", "2025-09-27", "2024-09-28"],
        "accessionNumber": ["0001-26-1", "0001-26-2", "0001-25-3", "0001-24-4"],
        "primaryDocument": ["q.htm", "ka.htm", "k25.htm", "k24.htm"],
    }}
}


def test_find_latest_10k_skips_amendments_and_quarters():
    ref = find_latest_10k(SUBMISSIONS, cik=320193, company="Apple Inc.", ticker="aapl")
    assert isinstance(ref, FilingRef)
    assert ref.accession == "0001-25-3"
    assert ref.ticker == "AAPL"
    assert ref.url.endswith("/0001253/k25.htm")
    assert "data/320193/" in ref.url


def test_html_to_text_strips_ix_header_and_scripts():
    html = ("<html><head><style>x</style></head><body>"
            "<ix:header>taxonomy-junk http://fasb.org/xyz</ix:header>"
            "<p>ITEM 1A.</p><p>Risk Factors are material.</p>"
            "<script>evil()</script></body></html>")
    text = html_to_text(html)
    assert "Risk Factors are material." in text
    assert "taxonomy-junk" not in text
    assert "evil" not in text
