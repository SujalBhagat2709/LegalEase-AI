# Import necessary libraries

import os
import re
import nltk
import torch
import pdfplumber
import json
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
from dotenv import load_dotenv
from langdetect import detect, LangDetectException
from werkzeug.utils import secure_filename

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import google.generativeai as genai

# Initialize Flask app and load environment variables
load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    print("WARNING: GOOGLE_API_KEY not found in .env file")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# NLTK resource download
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

# Load models
print("Loading models...")
try:
    sentence_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-large"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-large"
    ).to(device)
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    sentence_model = None
    tokenizer = None
    model = None

# Helper functions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def safe_detect_language(text):
    try:
        clean = re.sub(r"[^A-Za-z ]", " ", text)
        if len(clean.split()) < 10:
            return "en"
        return detect(clean)
    except LangDetectException:
        return "en"

def extract_text_from_pdf(pdf_path):
    """Robust PDF text extraction (as requested)"""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if text.strip():
            return text.strip()

    except Exception as e:
        print(f"Direct text extraction failed: {e}")

    return text.strip()

def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read().strip()
    except Exception as e:
        print(f"TXT extraction failed: {e}")
        return ""

# Load constitution text
def load_constitution():
    try:
        text = ""
        # Check for constitution.pdf in current directory
        constitution_path = "constitution.pdf"
        if not os.path.exists(constitution_path):
            print("WARNING: constitution.pdf not found in current directory")
            return ""
            
        with pdfplumber.open(constitution_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        return text
    except Exception as e:
        print(f"Error loading constitution: {e}")
        return ""

def extract_articles(text):
    articles = {}
    parts = re.split(r"Article\s+(\d+[A-Z]?)", text)

    for i in range(1, len(parts), 2):
        art = f"Article {parts[i]}"
        explanation = re.sub(r"\s+", " ", parts[i + 1]).strip()[:600]
        articles[art] = explanation

    return articles

# Extractive Summary
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

# Abstractive Summary
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

# Constitution Mapping
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

# Case Win Prediction
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

# Gemini Q&A
def gemini_answer(question, context):
    if not GOOGLE_API_KEY:
        return "Google API key not configured. Please set GOOGLE_API_KEY in .env file."
    
    try:
        # Try gemini-2.0-flash-exp first, fall back to gemini-pro
        try:
            genai_model = genai.GenerativeModel("gemini-2.5-flash")
        except:
            genai_model = genai.GenerativeModel("gemini-2.5-pro")
        
        prompt = f"""
You are a legal assistant AI.
Answer only from the given context.

Context:
{context}

Question:
{question}
"""

        response = genai_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        return f"AI analysis is currently unavailable. Error: {str(e)}"

# Evaluation Metrics
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

# Complete Legal Analysis
def complete_legal_analysis(doc_text, mode="Extractive"):
    """Performs complete legal analysis as in Streamlit"""
    
    # Load constitution
    constitution_text = load_constitution()
    articles = extract_articles(constitution_text)
    
    # Generate summary based on mode
    if mode == "Extractive":
        summary = extractive_summary(doc_text)
    else:  # Abstractive
        summary = abstractive_summary(doc_text)
    
    # Map to constitution
    matched_articles = map_constitution(summary, articles)
    
    # Predict case win
    win_score, win_result, win_reasons = predict_case_win(
        doc_text, matched_articles
    )
    
    # Calculate evaluation metrics
    metrics = evaluation_metrics(doc_text, summary)
    
    return {
        "summary": summary,
        "matched_articles": matched_articles,
        "win_prediction": {
            "score": win_score,
            "result": win_result,
            "reasons": win_reasons
        },
        "evaluation_metrics": metrics,
        "mode": mode
    }

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload and process document (exact Streamlit logic)"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text based on file type
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(filepath)
        else:
            text = extract_text_from_txt(filepath)
        
        # Clean up file
        try:
            os.remove(filepath)
        except:
            pass
        
        if len(text) < 200:
            return jsonify({'error': 'Unable to extract sufficient text from document (minimum 200 characters required).'}), 400
        
        # Return basic document info
        return jsonify({
            'success': True,
            'filename': filename,
            'content': text,
            'document_length': len(text.split())
        })
    
    return jsonify({'error': 'Invalid file type. Please upload PDF or TXT.'}), 400

@app.route('/analyze', methods=['POST'])
def analyze_document():
    """Complete legal analysis endpoint (exact Streamlit logic)"""
    data = request.json
    text = data.get('text', '')
    mode = data.get('mode', 'Extractive')  # Extractive or Abstractive
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    if len(text.split()) < 50:
        return jsonify({'error': 'Text too short for analysis (minimum 50 words required)'}), 400
    
    try:
        # Perform complete legal analysis
        analysis_results = complete_legal_analysis(text, mode)
        
        # Also calculate readability metrics
        sentences = sent_tokenize(text)
        words = text.split()
        avg_sentence_length = len(words) / max(len(sentences), 1)
        
        # Calculate complex word ratio
        def count_syllables(word):
            vowels = "aeiouy"
            count = 0
            word = word.lower()
            if word.endswith('e'):
                word = word[:-1]
            if len(word) == 0:
                return 0
            prev_char = word[0]
            for char in word[1:]:
                if char in vowels and prev_char not in vowels:
                    count += 1
                prev_char = char
            return max(1, count)
        
        complex_words = [word for word in words if count_syllables(word) >= 3]
        complex_word_ratio = len(complex_words) / max(len(words), 1)
        
        # Flesch reading ease
        flesch_score = 206.835 - 1.015 * avg_sentence_length - 84.6 * complex_word_ratio
        
        readability_metrics = {
            "num_sentences": len(sentences),
            "num_words": len(words),
            "avg_sentence_length": round(avg_sentence_length, 2),
            "avg_word_length": sum(len(w) for w in words) / max(len(words), 1),
            "complex_word_ratio": round(complex_word_ratio, 3),
            "flesch_score": round(flesch_score, 2)
        }
        
        return jsonify({
            'success': True,
            'analysis': analysis_results,
            'readability_metrics': readability_metrics,
            'risk_analysis': {
                'identified_risks': analysis_results['win_prediction']['reasons'],
                'risk_count': len(analysis_results['win_prediction']['reasons'])
            }
        })
        
    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Gemini Q&A endpoint (exact Streamlit logic)"""
    data = request.json
    question = data.get('question', '')
    context = data.get('context', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    if not context:
        return jsonify({'error': 'No document context provided'}), 400
    
    try:
        answer = gemini_answer(question, context)
        return jsonify({
            'success': True,
            'question': question,
            'answer': answer
        })
    except Exception as e:
        print(f"Q&A error: {e}")
        return jsonify({'error': f'Failed to get answer: {str(e)}'}), 500

@app.route('/constitution', methods=['GET'])
def get_constitution_articles():
    """Get constitution articles for mapping"""
    try:
        constitution_text = load_constitution()
        if not constitution_text:
            return jsonify({'error': 'Constitution file not found'}), 404
        
        articles = extract_articles(constitution_text)
        return jsonify({
            'success': True,
            'articles': articles,
            'total_articles': len(articles)
        })
    except Exception as e:
        print(f"Constitution error: {e}")
        return jsonify({'error': f'Failed to load constitution: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': sentence_model is not None,
        'gemini_configured': GOOGLE_API_KEY is not None,
        'device': device
    })

# Error Handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 10MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# Run the Flask app
# ============================================================
# RENDER DEPLOYMENT SETTINGS
# ============================================================
if __name__ == '__main__':
    # Get port from environment variable (Render provides this)
    port = int(os.environ.get('PORT', 5000))
    
    # Use 0.0.0.0 to make server publicly accessible
    app.run(debug=False, host='0.0.0.0', port=port)