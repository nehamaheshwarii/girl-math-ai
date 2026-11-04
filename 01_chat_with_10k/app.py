"""Chat with a 10-K — Streamlit UI.

Run:  streamlit run app.py
Works with no API key (extractive, cited answers). For summarized answers,
pick "Summarized" below and either set OPENAI_API_KEY before launching or
enter a key in the sidebar (kept in this session only, never saved to disk)
— still grounded only in retrieved passages, with citations verified.
"""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st  # noqa: E402

from src.ingest import chunk_document, load_text  # noqa: E402
from src.retrieval import BM25Index, tokenize  # noqa: E402
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

    # v2 (U4): answer style + model choice. Extractive stays the default,
    # so the app still works with no key at all.
    st.sidebar.header("Answer style")
    style = st.sidebar.radio(
        "Answer style",
        ["Extractive (no key needed)", "Summarized (needs API key)"],
        index=0,
    )
    model = st.sidebar.selectbox(
        "Model (summarized mode)", ["gpt-4o-mini", "gpt-4o"], index=0)
    api_key = st.sidebar.text_input(
        "OpenAI API key (session only, not saved)", type="password",
        help="Or set OPENAI_API_KEY before launching Streamlit.")
    summarized = style.startswith("Summarized")
    if summarized and not (api_key or os.getenv("OPENAI_API_KEY")):
        st.sidebar.warning(
            "No API key found — summarized answers will fall back to "
            "the retrieved passages.")

    q = st.text_input("Your question", placeholder="What are the principal risk factors?")
    if q:
        result = answer(
            q, index,
            mode="generated" if summarized else "extractive",
            model=model, api_key=api_key or None)
        st.markdown(result["answer"])
        # v2 (U6): show what filing-language terms were added to the
        # search, so expansion is transparent rather than hidden magic.
        expanded = result.get("expanded_query") or q
        added = sorted(set(tokenize(expanded)) - set(tokenize(q)))
        if added:
            st.caption(f"Search also used filing terms: {', '.join(added)}")
        if result.get("llm_warning"):
            st.warning(result["llm_warning"])
        elif result.get("mode") == "generated":
            st.caption("Summarized from the retrieved passages only — "
                       "citations verified against the sources below.")
        if result["sources"]:
            with st.expander("Sources"):
                for s in result["sources"]:
                    st.markdown(f"**{s['citation']}** (score {s['score']})")
                    st.write(" ".join(s["text"].split())[:600] + "…")
else:
    st.info("Upload a filing or use the sample excerpt to start.")
