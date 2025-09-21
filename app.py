import os
from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
import google.generativeai as genai
import PyPDF2
from docx import Document
from utils.data_processing import process_legal_document
from utils.visualization import create_complexity_radar, generate_wordcloud
import pandas as pd
import json
from utils.model_integration import initialize_models, summarize_with_gemini


models = initialize_models()
gemini_client = models["gemini"]


# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure Gemini AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def extract_text_from_pdf(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_txt(txt_path):
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except Exception as e:
        print(f"Error extracting text from TXT: {e}")
        return ""

# Load abbreviation dictionary
try:
    abbreviation_df = pd.read_csv('Abbreviation_dict.csv')
    abbreviation_dict = dict(zip(abbreviation_df['Abbreviation'], abbreviation_df['Full Form']))
except:
    abbreviation_dict = {}

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text.strip():
            return jsonify({"error": "No text provided"}), 400

        # 🎯 Custom Prompt for Better Summary
        prompt = f"""
        You are an expert legal assistant AI. Your job is to answer user questions about legal documents in a **clear, beginner-friendly way**, without generating unnecessary long summaries.

Instructions:
1. Only answer based on the provided document.
2. Reference relevant sections if necessary.
3. Structure your answer depending on the query:
   - Summary → Key Points → Important Clauses → Explanation
4. Be concise, simple, and practical for someone with no legal knowledge.


        Document:
        {text}
        """

        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)

        summary = response.text if response and response.text else "Gemini returned empty response"
        return jsonify({"gemini": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/summarize", methods=["POST"])
def summarize():
    text = request.json.get("text")
    summary = summarize_with_gemini(text)
    return jsonify({"summary": summary})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_document():
    # Check if file was uploaded
    if 'file' in request.files:
        file = request.files['file']
        if file and file.filename != '':
            # Save the file temporarily
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            
            # Extract text based on file type
            if file.filename.endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
            elif file.filename.endswith('.docx'):
                text = extract_text_from_docx(file_path)
            elif file.filename.endswith('.txt'):
                text = extract_text_from_txt(file_path)
            else:
                return jsonify({'error': 'Unsupported file format'})
            
            # Clean up
            os.remove(file_path)
        else:
            return jsonify({'error': 'No file provided'})
    else:
        # Get text from JSON request
        data = request.get_json()
        text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'})
    
    try:
        # Process the document
        processed_data = process_legal_document(text, abbreviation_dict)
        
        # Generate summaries with Gemini
        model = genai.GenerativeModel('gemini-pro')
        
        summaries = {}
        
        # Easy understanding summary
        easy_prompt = f"Explain this legal document in simple language that a high school student can understand:\n\n{text[:3000]}"
        easy_response = model.generate_content(easy_prompt)
        summaries['easy'] = easy_response.text
        
        # Medium understanding summary
        medium_prompt = f"Summarize this legal document for an educated adult:\n\n{text[:3000]}"
        medium_response = model.generate_content(medium_prompt)
        summaries['medium'] = medium_response.text
        
        # Professional summary
        pro_prompt = f"Provide a detailed professional analysis of this legal document:\n\n{text[:3000]}"
        pro_response = model.generate_content(pro_prompt)
        summaries['professional'] = pro_response.text
        
        # Generate visualizations
        create_complexity_radar(processed_data['readability_stats'], 'user_document')
        generate_word_cloud(processed_data['processed_text'], 'user_document')
        
        # Prepare response
        response = {
            'success': True,
            'readability_metrics': processed_data['readability_stats'],
            'summaries': summaries,
            'visualizations': {
                'radar_chart': 'data/visualizations/radar_user_document.png',
                'word_cloud': 'data/visualizations/wordcloud_user_document.png'
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/sample/<doc_type>')
def get_sample(doc_type):
    samples = {
        'rental': 'samples/rental_agreement.txt',
        'loan': 'samples/loan_agreement.txt',
        'terms': 'samples/terms_of_service.txt'
    }
    
    if doc_type not in samples:
        return jsonify({'error': 'Invalid sample type'})
    
    try:
        with open(samples[doc_type], 'r', encoding='utf-8') as f:
            content = f.read()
        
        return jsonify({
            'success': True,
            'content': content,
            'type': doc_type
        })
    except Exception as e:
        return jsonify({'error': f'Error loading sample: {str(e)}'})

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    document_text = data.get('text', '')
    question = data.get('question', '')
    
    if not document_text or not question:
        return jsonify({'error': 'Document text and question are required'})
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"Based on this legal document: {document_text[:3000]}\n\nAnswer this question: {question}"
        response = model.generate_content(prompt)
        
        return jsonify({
            'success': True,
            'answer': response.text
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/download_report', methods=['POST'])
def download_report():
    data = request.json
    analysis = data.get('analysis', '')
    
    # Create a simple text report
    report_content = f"LegalEase AI Analysis Report\n\n{analysis}"
    
    # Save to a temporary file
    report_path = os.path.join('reports', 'legal_analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return send_file(report_path, as_attachment=True)

if __name__ == '__main__':
    # Create necessary directories
    for folder in ['uploads', 'data/raw', 'data/processed', 'data/visualizations', 'reports']:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    app.run(debug=True, host='0.0.0.0', port=5000)