"""
Person 3 -- Research Relevance + Summarization + Report generation.

Exported from Final_Report.ipynb (Sections 2-7) so Streamlit's app.py can import it
directly, the same way person1_clickbait.py and person2_verification.py were exported.
If you change the pipeline in the notebook, copy the updated cells back into this file.

Put this file in the same project folder as person1_clickbait.py, person2_verification.py,
and app.py.
"""
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fpdf import FPDF

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Models (loaded once at import time, reused across every request)
# ---------------------------------------------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

SUMMARY_MODEL_ID = "sshleifer/distilbart-cnn-12-6"
summary_tokenizer = AutoTokenizer.from_pretrained(SUMMARY_MODEL_ID)
summary_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARY_MODEL_ID).to(DEVICE)
summary_model.eval()


# ---------------------------------------------------------------------------
# Research relevance (Sentence-BERT)
# ---------------------------------------------------------------------------
@dataclass
class RelevanceResult:
    topic: str
    score: float
    label: str


def compute_research_relevance(
    research_topic: str,
    summary_text: str,
    relevant_threshold: float = 0.40,
    somewhat_threshold: float = 0.20,
) -> RelevanceResult:
    """Score how relevant the article is to `research_topic` using Sentence-BERT
    cosine similarity, embedding the article's SUMMARY rather than its full text.
    """
    if not research_topic or not research_topic.strip():
        raise ValueError("research_topic must not be empty.")
    if not summary_text or not summary_text.strip():
        raise ValueError("summary_text must not be empty.")

    topic_emb = embedder.encode(research_topic.strip(), convert_to_tensor=True)
    summary_emb = embedder.encode(summary_text.strip(), convert_to_tensor=True)

    score = util.cos_sim(topic_emb, summary_emb).item()
    score = round(score, 3)

    if score >= relevant_threshold:
        label = "Relevant"
    elif score >= somewhat_threshold:
        label = "Somewhat Relevant"
    else:
        label = "Not Relevant"

    return RelevanceResult(topic=research_topic.strip(), score=score, label=label)


# ---------------------------------------------------------------------------
# Summarization (DistilBART)
# ---------------------------------------------------------------------------
def summarize_article(
    article_text: str,
    max_new_tokens: int = 130,
    min_new_tokens: int = 30,
) -> str:
    """Generate a short abstractive summary of `article_text`."""
    if not article_text or not article_text.strip():
        return "Summary unavailable: no article text provided."

    trimmed = article_text.strip()
    effective_min = min(min_new_tokens, max(5, len(trimmed.split()) // 2))

    try:
        inputs = summary_tokenizer(
            trimmed,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(DEVICE)

        with torch.inference_mode():
            output_ids = summary_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=effective_min,
                num_beams=4,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        summary = summary_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return summary.strip()
    except Exception as exc:
        return f"Summary unavailable: summarization failed ({exc})."


def trim_trailing_boilerplate(text: str) -> str:
    """Cut scraped article text at 'Read more / related links' teaser sections."""
    if not text:
        return text
    lines = text.split("\n")
    cutoff = None
    boundary_markers = ("read more", "related articles", "related:", "see also", "you might also like")
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if stripped in boundary_markers or any(stripped.startswith(m) for m in boundary_markers):
            cutoff = i
            break
    if cutoff is not None:
        lines = lines[:cutoff]
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Shared result schema
# ---------------------------------------------------------------------------
@dataclass
class TruthLensResult:
    url: str
    headline: str
    article_text: str

    clickbait_label: Optional[str] = None
    clickbait_score: Optional[float] = None

    verification_label: Optional[str] = None
    verification_confidence: Optional[float] = None
    evidence_sentences: list = field(default_factory=list)

    research_topic: Optional[str] = None
    relevance_score: Optional[float] = None
    relevance_label: Optional[str] = None
    summary: Optional[str] = None

    generated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# PDF report generation (fpdf2)
# ---------------------------------------------------------------------------
class TruthLensPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 30, 30)
        self.cell(0, 10, _safe_text("TruthLens - Article Analysis Report"), new_x="LMARGIN", new_y="NEXT", align="C")
        self.set_draw_color(200, 200, 200)
        self.line(10, 20, 200, 20)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def _safe_text(text: str) -> str:
    if text is None:
        return ""
    replacements = {
        "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2026": "...",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _section_title(pdf: FPDF, title: str):
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 8, _safe_text(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)


def _verdict_badge(pdf: FPDF, label: str, positive_words=("supported", "not clickbait", "relevant")):
    if label is None:
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 7, "Not available", new_x="LMARGIN", new_y="NEXT")
        return

    lower = label.lower()
    if any(w in lower for w in positive_words):
        pdf.set_text_color(30, 140, 60)
    elif "neutral" in lower or "somewhat" in lower or "discussed" in lower or "unrelated" in lower:
        pdf.set_text_color(200, 150, 0)
    else:
        pdf.set_text_color(200, 40, 40)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, _safe_text(label), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)


def generate_report(result: TruthLensResult, output_path: str = "outputs/final_reports/report.pdf") -> str:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    pdf = TruthLensPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 6, _safe_text(f"Generated: {result.generated_at}    |    Source URL: {result.url}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    _section_title(pdf, "Headline")
    pdf.multi_cell(0, 6, _safe_text(result.headline or "N/A"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    _section_title(pdf, "Clickbait Detection")
    badge = None
    if result.clickbait_label is not None:
        confidence_str = f" ({result.clickbait_score * 100:.1f}% confidence)" if result.clickbait_score is not None else ""
        badge = f"{result.clickbait_label.upper()}{confidence_str}"
    _verdict_badge(pdf, badge, positive_words=("not clickbait",))
    pdf.ln(2)

    _section_title(pdf, "Headline-Article Verification")
    _verdict_badge(pdf, result.verification_label, positive_words=("supported",))
    if result.evidence_sentences:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "Supporting evidence:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for sent in result.evidence_sentences[:3]:
            pdf.multi_cell(0, 6, _safe_text(f"-  {sent}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    _section_title(pdf, "Research Relevance")
    pdf.multi_cell(0, 6, _safe_text(f"Topic: {result.research_topic or 'N/A'}"), new_x="LMARGIN", new_y="NEXT")
    rel_badge = None
    if result.relevance_label is not None:
        score_str = f" (score: {result.relevance_score})" if result.relevance_score is not None else ""
        rel_badge = f"{result.relevance_label}{score_str}"
    _verdict_badge(pdf, rel_badge, positive_words=("relevant",))
    pdf.ln(2)

    _section_title(pdf, "Article Summary")
    pdf.multi_cell(0, 6, _safe_text(result.summary or "N/A"))

    pdf.output(output_path)
    return output_path


# ---------------------------------------------------------------------------
# End-to-end pipeline entry point
# ---------------------------------------------------------------------------
def run_person3_pipeline(
    url: str,
    headline: str,
    article_text: str,
    research_topic: str,
    person1_output: Optional[dict] = None,
    person2_output: Optional[dict] = None,
    output_path: str = "outputs/final_reports/report.pdf",
) -> tuple[TruthLensResult, str]:
    """Run the full Person-3 stage and produce a PDF report.

    Returns:
        (result, pdf_path)
    """
    summary = summarize_article(article_text)
    relevance = compute_research_relevance(research_topic, summary)

    p1 = person1_output or {}
    p2 = person2_output or {}

    result = TruthLensResult(
        url=url,
        headline=headline,
        article_text=article_text,
        clickbait_label=p1.get("clickbait_label"),
        clickbait_score=p1.get("clickbait_score"),
        verification_label=p2.get("verification_label"),
        verification_confidence=p2.get("verification_confidence"),
        evidence_sentences=p2.get("evidence_sentences", []),
        research_topic=relevance.topic,
        relevance_score=relevance.score,
        relevance_label=relevance.label,
        summary=summary,
    )

    pdf_path = generate_report(result, output_path=output_path)
    return result, pdf_path
