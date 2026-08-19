"""
TruthLens -- Streamlit interface.

Wires together:
- person1_clickbait.py  (analyze_url)
- person2_verification.py (verify_headline)
- person3_pipeline.py    (run_person3_pipeline, trim_trailing_boilerplate)

Place this file in the same project folder as those three modules and the
models/ directory before running `streamlit run app.py`.
"""
import os
import sys
import tempfile

import streamlit as st

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from person1_clickbait import analyze_url
from person2_verification import verify_headline
from person3_pipeline import run_person3_pipeline, trim_trailing_boilerplate

st.set_page_config(page_title="TruthLens", page_icon="🔍", layout="centered")

# ---------------------------------------------------------------------------
# Theme / styling
# ---------------------------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(180deg, #F5F7FF 0%, #FFFFFF 320px);
    }

    /* Hide default Streamlit chrome for a cleaner look */
    #MainMenu, footer, header { visibility: hidden; }

    .block-container { padding-top: 2.5rem; max-width: 760px; }

    /* Header */
    .tl-header { display: flex; align-items: center; gap: 14px; margin-bottom: 4px; }
    .tl-logo {
        width: 46px; height: 46px; border-radius: 12px;
        background: linear-gradient(135deg, #4F46E5, #7C3AED);
        display: flex; align-items: center; justify-content: center;
        font-size: 22px; flex-shrink: 0;
    }
    .tl-title { font-size: 2rem; font-weight: 800; color: #111827; margin: 0; line-height: 1.1; }
    .tl-caption { color: #6B7280; font-size: 0.95rem; margin: 6px 0 28px 0; }

    /* Card containers (st.container(border=True)) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px !important;
        border: 1px solid #E5E7EB !important;
        background: #FFFFFF;
        box-shadow: 0 1px 3px rgba(17, 24, 39, 0.04);
    }

    /* Section labels inside cards */
    .tl-section-label {
        font-size: 0.78rem; font-weight: 700; letter-spacing: 0.06em;
        text-transform: uppercase; color: #9CA3AF; margin-bottom: 6px;
    }

    /* Text inputs */
    .stTextInput input {
        border-radius: 10px !important;
        border: 1.5px solid #E5E7EB !important;
        padding: 10px 14px !important;
    }
    .stTextInput input:focus {
        border-color: #4F46E5 !important;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.12) !important;
    }

    /* Buttons */
    .stButton button, .stFormSubmitButton button {
        background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 22px !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(79, 70, 229, 0.25);
    }
    .stButton button:hover, .stFormSubmitButton button:hover { opacity: 0.92; }

    .stDownloadButton button {
        background: #111827 !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
        font-weight: 600 !important;
    }

    /* Verdict pill badges */
    .tl-badge {
        display: inline-block; padding: 5px 14px; border-radius: 999px;
        font-weight: 700; font-size: 0.85rem; margin-top: 2px;
    }
    .tl-badge-green { background: #DCFCE7; color: #15803D; }
    .tl-badge-amber { background: #FEF3C7; color: #B45309; }
    .tl-badge-red   { background: #FEE2E2; color: #B91C1C; }
    .tl-badge-gray  { background: #F3F4F6; color: #6B7280; }

    .tl-evidence {
        border-left: 3px solid #E5E7EB; padding-left: 12px; margin: 6px 0;
        color: #374151; font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="tl-header">
    <div class="tl-logo">🔍</div>
    <p class="tl-title">TruthLens</p>
</div>
<p class="tl-caption">Clickbait detection · Headline verification · Research relevance · Summary</p>
""", unsafe_allow_html=True)


def badge(text: str, kind: str) -> str:
    return f'<span class="tl-badge tl-badge-{kind}">{text}</span>'


with st.container(border=True):
    with st.form("truthlens_form"):
        url = st.text_input("Article URL", placeholder="https://example.com/news-story")
        research_topic = st.text_input("Research topic", placeholder="e.g. AI in healthcare")
        submitted = st.form_submit_button("Analyze")

if submitted:
    if not url.strip() or not research_topic.strip():
        st.error("Please provide both a URL and a research topic.")
        st.stop()

    try:
        with st.spinner("Extracting article and checking for clickbait..."):
            p1 = analyze_url(url)
            if not p1.get("success"):
                st.error(f"Couldn't extract the article: {p1.get('error')}")
                st.stop()

        cleaned_article_text = trim_trailing_boilerplate(p1["article_text"])

        with st.spinner("Verifying headline against article content..."):
            p2 = verify_headline(p1["headline"], cleaned_article_text)

        person1_output = {
            "clickbait_label": p1["clickbait_label"],
            "clickbait_score": p1["clickbait_score"],
        }
        person2_output = {
            "verification_label": p2["stance"],
            "verification_confidence": p2["confidence"].get(p2["stance"]),
            "evidence_sentences": [sentence for sentence, score in p2["evidence"]],
        }

        with st.spinner("Scoring research relevance and summarizing..."):
            output_path = os.path.join(tempfile.gettempdir(), "truthlens_report.pdf")
            result, pdf_path = run_person3_pipeline(
                url=url,
                headline=p1["headline"],
                article_text=cleaned_article_text,
                research_topic=research_topic,
                person1_output=person1_output,
                person2_output=person2_output,
                output_path=output_path,
            )
    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.stop()

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="tl-section-label">Headline</div>', unsafe_allow_html=True)
        st.markdown(f"**{result.headline}**")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown('<div class="tl-section-label">Clickbait</div>', unsafe_allow_html=True)
            if result.clickbait_label:
                conf = f" ({result.clickbait_score * 100:.1f}%)" if result.clickbait_score is not None else ""
                kind = "green" if "not" in result.clickbait_label.lower() else "red"
                st.markdown(badge(f"{result.clickbait_label.upper()}{conf}", kind), unsafe_allow_html=True)
            else:
                st.markdown(badge("N/A", "gray"), unsafe_allow_html=True)

    with col2:
        with st.container(border=True):
            st.markdown('<div class="tl-section-label">Verification</div>', unsafe_allow_html=True)
            if result.verification_label:
                conf = f" ({result.verification_confidence * 100:.1f}%)" if result.verification_confidence is not None else ""
                label_lower = result.verification_label.lower()
                if "supported" in label_lower:
                    kind = "green"
                elif "contradicted" in label_lower:
                    kind = "red"
                else:
                    kind = "amber"
                st.markdown(badge(f"{result.verification_label}{conf}", kind), unsafe_allow_html=True)
            else:
                st.markdown(badge("N/A", "gray"), unsafe_allow_html=True)

    if result.evidence_sentences:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="tl-section-label">Supporting Evidence</div>', unsafe_allow_html=True)
            for sent in result.evidence_sentences[:3]:
                st.markdown(f'<div class="tl-evidence">{sent}</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="tl-section-label">Research Relevance</div>', unsafe_allow_html=True)
        st.markdown(f"Topic: *{result.research_topic}*")
        rel_kind = "green" if result.relevance_label == "Relevant" else ("amber" if result.relevance_label == "Somewhat Relevant" else "red")
        st.markdown(badge(f"{result.relevance_label} · score {result.relevance_score}", rel_kind), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="tl-section-label">Article Summary</div>', unsafe_allow_html=True)
        st.write(result.summary)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    with open(pdf_path, "rb") as f:
        st.download_button(
            "⬇️  Download PDF report",
            f,
            file_name="truthlens_report.pdf",
            mime="application/pdf",
        )
