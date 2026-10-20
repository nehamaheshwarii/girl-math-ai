"""Chat with a 10-K — Streamlit UI.

Run:  streamlit run app.py
Works with no API key (extractive, cited answers). Set OPENAI_API_KEY in
.env for generated answers — still grounded only in retrieved passages.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st  # noqa: E402

from src.ingest import chunk_document, load_text  # noqa: E402
from src.retrieval import BM25Index  # noqa: E402
from src.qa import answer  # noqa: E402

st.set_page_config(page_title="Chat with a 10-K", page_icon="📄")
st.title("Chat with a 10-K")
st.caption("Ask a filing questions. Every answer cites its section — "
           "or tells you it isn't in there.")

SAMPLE = Path(__file__).parent / "sample_data" / "sample_10k_excerpt.txt"

with st.sidebar:
    st.header("Filing")
    uploaded = st.file_uploader("Upload a 10-K (.txt, .md, .pdf)", type=["txt", "md", "pdf"])
    use_sample = st.button("Use the sample filing excerpt")
    st.divider()
    ticker = st.text_input("…or fetch from SEC EDGAR by ticker", placeholder="AAPL")
    fetch = st.button("Fetch latest 10-K")

if "doc" not in st.session_state:
    st.session_state.doc = None
    st.session_state.index = None

if uploaded is not None:
    tmp = Path("/tmp") / uploaded.name
    tmp.write_bytes(uploaded.getbuffer())
    st.session_state.doc = chunk_document(load_text(tmp), title=uploaded.name)
    st.session_state.index = BM25Index(st.session_state.doc.chunks)
elif use_sample:
    st.session_state.doc = chunk_document(load_text(SAMPLE), title=SAMPLE.name)
    st.session_state.index = BM25Index(st.session_state.doc.chunks)
elif fetch and ticker:
    from src.edgar import fetch_latest_10k_text
    with st.spinner(f"Fetching {ticker.upper()}'s latest 10-K from SEC EDGAR…"):
        ref, text = fetch_latest_10k_text(
            ticker, cache_dir=Path(__file__).parent / "data" / "cache")
    st.session_state.doc = chunk_document(
        text, title=f"{ref.company} {ref.form} filed {ref.filing_date}")
    st.session_state.index = BM25Index(st.session_state.doc.chunks)
    st.sidebar.caption(f"Source: {ref.url}")

doc, index = st.session_state.doc, st.session_state.index
if doc:
    st.sidebar.success(f"{doc.title}: {len(doc.chunks)} chunks")
    st.sidebar.write("Sections:", ", ".join(doc.sections[:12]))

    q = st.text_input("Your question", placeholder="What are the principal risk factors?")
    if q:
        result = answer(q, index)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("Sources"):
                for s in result["sources"]:
                    st.markdown(f"**{s['citation']}** (score {s['score']})")
                    st.write(" ".join(s["text"].split())[:600] + "…")
else:
    st.info("Upload a filing or use the sample excerpt to start.")
