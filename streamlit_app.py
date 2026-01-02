# ============================================================
# LegalEaseAI – Complete Legal Analysis System (Final)
# ============================================================

import os
import re
import nltk
import torch
import pdfplumber
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from langdetect import detect, LangDetectException

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import google.generativeai as genai

# ------------------------------------------------------------
# SAFE NLTK RESOURCE LOADER
# ------------------------------------------------------------
def download_nltk_resources():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")

download_nltk_resources()

# ------------------------------------------------------------
# INITIAL SETUP
# ------------------------------------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

device = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="LegalEaseAI", layout="wide")

# ------------------------------------------------------------
# LOAD MODELS (CACHED)
# ------------------------------------------------------------
@st.cache_resource
def load_models():
    sentence_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-large"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-large"
    ).to(device)

    return sentence_model, tokenizer, model

sentence_model, tokenizer, model = load_models()

# ------------------------------------------------------------
# SAFE LANGUAGE DETECTION
# ------------------------------------------------------------
def safe_detect_language(text):
    try:
        clean = re.sub(r"[^A-Za-z ]", " ", text)
        if len(clean.split()) < 10:
            return "en"
        return detect(clean)
    except LangDetectException:
        return "en"

# ------------------------------------------------------------
# ROBUST PDF TEXT EXTRACTION (AS REQUESTED)
# ------------------------------------------------------------
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if text.strip():
            return text.strip()

    except Exception as e:
        print(f"Direct text extraction failed: {e}")

    return text.strip()

# ------------------------------------------------------------
# LOAD DOCUMENT (PDF / TXT)
# ------------------------------------------------------------
def load_document(file):
    if file.name.endswith(".pdf"):
        return extract_text_from_pdf(file)

    if file.name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")

    return ""

# ------------------------------------------------------------
# LOAD CONSTITUTION
# ------------------------------------------------------------
@st.cache_data
def load_constitution():
    text = ""
    with pdfplumber.open("constitution.pdf") as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

def extract_articles(text):
    articles = {}
    parts = re.split(r"Article\s+(\d+[A-Z]?)", text)

    for i in range(1, len(parts), 2):
        art = f"Article {parts[i]}"
        explanation = re.sub(r"\s+", " ", parts[i + 1]).strip()[:600]
        articles[art] = explanation

    return articles

# ------------------------------------------------------------
# EXTRACTIVE SUMMARY
# ------------------------------------------------------------
def extractive_summary(text, top_k=8):
    sentences = sent_tokenize(text)
    if len(sentences) <= top_k:
        return text

    doc_emb = sentence_model.encode([text])[0]
    sent_embs = sentence_model.encode(sentences)
    sims = cosine_similarity([doc_emb], sent_embs)[0]

    ranked = sorted(
        enumerate(zip(sentences, sims)),
        key=lambda x: x[1][1],
        reverse=True
    )

    selected_idx = sorted([i for i, _ in ranked[:top_k]])
    return " ".join([sentences[i] for i in selected_idx])

# ------------------------------------------------------------
# ABSTRACTIVE SUMMARY (GROUNDED)
# ------------------------------------------------------------
def grounded_text(text, top_k=40):
    sentences = sent_tokenize(text)
    doc_emb = sentence_model.encode([text])[0]
    sent_embs = sentence_model.encode(sentences)
    sims = cosine_similarity([doc_emb], sent_embs)[0]

    ranked = sorted(
        enumerate(zip(sentences, sims)),
        key=lambda x: x[1][1],
        reverse=True
    )

    selected_idx = sorted([i for i, _ in ranked[:top_k]])
    return " ".join([sentences[i] for i in selected_idx])

def abstractive_summary(text):
    g_text = grounded_text(text)

    prompt = f"""
Summarize the following Indian legal judgment.

Rules:
- Preserve legal meaning
- Do not add facts
- Use formal legal language
- Mention issue, reasoning, and final decision

Judgment:
{g_text}
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=2048,
        truncation=True
    ).to(device)

    output = model.generate(
        **inputs,
        max_length=250,
        min_length=120,
        num_beams=4,
        length_penalty=1.2,
        no_repeat_ngram_size=3
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# ------------------------------------------------------------
# CONSTITUTION MAPPING
# ------------------------------------------------------------
def map_constitution(summary, articles):
    matches = []
    sw = set(summary.lower().split())

    for art, expl in articles.items():
        aw = set(expl.lower().split())
        if len(sw & aw) > 12:
            matches.append({
                "article": art,
                "explanation": expl
            })
    return matches

# ------------------------------------------------------------
# CASE WIN PREDICTION
# ------------------------------------------------------------
def predict_case_win(text, matched_articles):
    score = 50
    reasons = []

    if "penalty" in text.lower():
        score -= 10
        reasons.append("Penalty clauses detected")

    if "liable" in text.lower():
        score -= 10
        reasons.append("High liability wording")

    if matched_articles:
        score += 15
        reasons.append("Constitutional alignment present")

    score = max(5, min(score, 95))

    outcome = (
        "High chance of success" if score >= 70
        else "Moderate chance of success" if score >= 50
        else "Low chance of success"
    )

    return score, outcome, reasons

# ------------------------------------------------------------
# GEMINI Q&A
# ------------------------------------------------------------
def gemini_answer(question, context):
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
You are a legal assistant AI.
Answer only from the given context.

Context:
{context}

Question:
{question}
"""

    response = model.generate_content(prompt)
    return response.text

# ------------------------------------------------------------
# EVALUATION METRICS
# ------------------------------------------------------------
def evaluation_metrics(document, summary):
    doc_emb = sentence_model.encode([document])[0]
    sum_emb = sentence_model.encode([summary])[0]

    cosine_sim = cosine_similarity(
        [doc_emb], [sum_emb]
    )[0][0]

    compression_ratio = len(summary.split()) / max(
        1, len(document.split())
    )

    return {
        "Semantic Similarity": round(float(cosine_sim), 3),
        "Compression Ratio": round(float(compression_ratio), 3),
        "Document Length": len(document.split()),
        "Summary Length": len(summary.split())
    }

# ------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------
st.title("⚖️ LegalEaseAI – Complete Legal Analysis")

uploaded_file = st.file_uploader(
    "Upload Legal Document (PDF or TXT)",
    type=["pdf", "txt"]
)

mode = st.radio(
    "Select Summarization Mode",
    ["Extractive", "Abstractive"]
)

if uploaded_file:
    doc_text = load_document(uploaded_file)

    if len(doc_text) < 200:
        st.error("Unable to extract sufficient text from document.")
        st.stop()

    constitution_text = load_constitution()
    articles = extract_articles(constitution_text)

    with st.spinner("Analyzing document..."):
        summary = (
            extractive_summary(doc_text)
            if mode == "Extractive"
            else abstractive_summary(doc_text)
        )

        matched_articles = map_constitution(summary, articles)
        win_score, win_result, win_reasons = predict_case_win(
            doc_text, matched_articles
        )

        metrics = evaluation_metrics(doc_text, summary)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Document",
        "🧠 Summary",
        "📜 Constitution",
        "📊 Case Win Prediction",
        "📈 Evaluation Dashboard"
    ])

    with tab1:
        st.text_area("Document Text", doc_text, height=400)

    with tab2:
        st.markdown(summary)

    with tab3:
        for item in matched_articles:
            with st.expander(item["article"]):
                st.write(item["explanation"])

    with tab4:
        st.metric("Win Probability (%)", win_score)
        st.write(win_result)
        for r in win_reasons:
            st.write("•", r)

    with tab5:
        st.metric("Semantic Similarity", metrics["Semantic Similarity"])
        st.metric("Compression Ratio", metrics["Compression Ratio"])
        st.metric("Document Length (words)", metrics["Document Length"])
        st.metric("Summary Length (words)", metrics["Summary Length"])

    st.divider()
    st.subheader("💬 Ask Questions (Gemini AI)")

    user_q = st.text_input("Ask a question about this case")

    if user_q:
        answer = gemini_answer(user_q, summary)
        st.markdown(answer)
