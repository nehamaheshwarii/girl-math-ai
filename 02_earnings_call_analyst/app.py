"""Earnings Call Analyst — Streamlit UI.

Run:  streamlit run app.py
Works with no API key: the structured report is extracted, not generated,
so every bullet, figure, and theme cites the segment it came from.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st  # noqa: E402

from src.parse import parse_transcript  # noqa: E402
from src.analysis import analyze  # noqa: E402

st.set_page_config(page_title="Earnings Call Analyst", page_icon="🎙️")
st.title("Earnings Call Analyst")
st.caption("Paste or upload a transcript. Get a structured summary, "
           "guidance, risks, and Q&A themes — every item cited to the "
           "segment it came from.")

SAMPLE = Path(__file__).parent / "sample_data" / "sample_earnings_call.txt"

with st.sidebar:
    st.header("Transcript")
    uploaded = st.file_uploader("Upload a transcript (.txt, .md)",
                                type=["txt", "md"])
    use_sample = st.button("Use the sample transcript")
    pasted = st.text_area("…or paste a transcript here", height=200)

if "segments" not in st.session_state:
    st.session_state.segments = None

if uploaded is not None:
    text = uploaded.getvalue().decode("utf-8", errors="ignore")
    st.session_state.segments = parse_transcript(text)
elif use_sample:
    st.session_state.segments = parse_transcript(
        SAMPLE.read_text(encoding="utf-8"))
elif pasted.strip():
    st.session_state.segments = parse_transcript(pasted)

segments = st.session_state.segments
if not segments:
    st.info("Upload a transcript, paste one, or use the sample to start.")
else:
    report = analyze(segments)
    st.sidebar.success(
        f"{report['counts']['segments']} segments "
        f"({report['counts']['prepared_remarks']} prepared remarks, "
        f"{report['counts']['qa']} Q&A)")
    st.sidebar.write("Speakers:")
    for speaker in report["speakers"]:
        st.sidebar.markdown(f"- {speaker}")

    search = st.text_input("Find in transcript",
                           placeholder="e.g. tariff, buyback, AI")
    if search.strip():
        hits = [s for s in segments if search.lower() in s.text.lower()]
        st.subheader(f"Matches for “{search}” ({len(hits)})")
        for segment in hits:
            st.markdown(f"**{segment.citation()}** — {segment.speaker}")
            st.write(segment.text[:600] + ("…" if len(segment.text) > 600 else ""))
    else:
        st.subheader("Summary")
        for item in report["summary"]:
            st.markdown(f"- {item['text']} {item['citation']}")

        st.subheader("Guidance")
        for item in report["guidance"]:
            st.markdown(f"- {item['text']} {item['citation']}")

        st.subheader("Risks")
        for item in report["risks"]:
            st.markdown(f"- {item['text']} {item['citation']}")

        st.subheader("Q&A themes")
        for theme in report["themes"]:
            st.markdown(
                f"- **{theme['theme']}** — {theme['count']} segment(s), "
                f"e.g. {theme['example_citation']} "
                f"({theme['example_speaker']})")

        st.subheader("Key figures")
        for figure in report["figures"]:
            st.markdown(
                f"- **{figure['figure']}** {figure['citation']}: "
                f"{figure['sentence']}")
