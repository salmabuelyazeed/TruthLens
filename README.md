\# TruthLens 🔍



TruthLens is an end-to-end fake news and clickbait detection pipeline. It analyzes a news article URL and a research topic to produce a clickbait verdict, headline-vs-content verification, evidence extraction, research relevance scoring, and a downloadable PDF report — all through a simple Streamlit interface.



\*\*Live app:\*\* https://truthlens-cnxz3cbrjxw9hetozuxdvz.streamlit.app/



\## Features



\- \*\*Clickbait Detection\*\* — Fine-tuned DistilBERT classifier trained on the Webis Clickbait Challenge 2017 dataset, flags sensationalized headlines with a confidence score.

\- \*\*Headline-Article Verification\*\* — Checks whether a headline is supported by, contradicted by, or unrelated to its article body, using an NLI-based stance model trained on FNC-1 data, with supporting evidence sentences extracted.

\- \*\*Research Relevance Scoring\*\* — Scores how relevant an article is to a given research topic.

\- \*\*PDF Report Generation\*\* — Produces a downloadable summary report combining all of the above.



\## Architecture



The pipeline is split into three components, wired together in `app.py`:



| Module | Responsibility |

|---|---|

| `person1\_clickbait.py` | Article extraction + clickbait classification (DistilBERT) |

| `person2\_verification.py` | Headline-article stance verification + evidence extraction |

| `person3\_pipeline.py` | Research relevance scoring, summarization, and PDF report generation |



The clickbait model is hosted on Hugging Face (too large for GitHub) and loaded at runtime:

👉 \[salmabuelyazeed/truthlens-clickbait](https://huggingface.co/salmabuelyazeed/truthlens-clickbait)



\## Tech Stack



Python · Streamlit · Hugging Face Transformers · PyTorch · Sentence-Transformers · scikit-learn · BeautifulSoup / Trafilatura / Newspaper3k · FPDF2



\## Running Locally



```bash

git clone https://github.com/salmabuelyazeed/TruthLens.git

cd TruthLens

pip install -r requirements.txt

streamlit run app.py

```



\## Screenshots



\*(add screenshots here)\*



\## Project Background



Built as a capstone project for the \*\*NTI x Huawei ICT Academy (ETA) AI Track\*\*, an intensive program covering machine learning, NLP, RAG, fine-tuning, and computer vision.

