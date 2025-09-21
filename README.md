# LegalEase AI – Generative AI for Demystifying Legal Documents

LegalEase AI is a prototype built for the Gen AI Exchange Hackathon (Hack2Skills). Our mission is to turn complex legal text into plain-language summaries, clause explanations, and trustworthy answers.
<br>The tool empowers renters, loan applicants, small business owners, gig workers, and everyday consumers to confidently understand legal documents - without costly lawyer consultations.

<b>How it works:</b>
1. Upload a contract or legal case document.
2. The system parses the text, extracts clauses, and generates multi-level summaries (easy, medium, professional).
3. Users can ask questions and receive clause-cited answers with risks flagged.
4. Visual dashboards highlight readability, complexity, and legal red flags.

# Problem Statement

Legal documents like rental agreements, loan contracts, and terms of service are often filled with jargon that’s nearly impossible for non - experts to understand.
<br>This leads to information asymmetry, where individuals unknowingly agree to unfavorable terms - exposing themselves to legal and financial risks.

There is a pressing need for a solution that makes legal documents transparent, simple, and safe for everyone.

# Our AI-Powered Solution
<b>LegalEase AI is designed as a hybrid generative AI system with:</b>
1. Multi-level Summaries: Tailored for students (10th grade), educated professionals, and lawyers.
2. Clause Extraction & Risk Highlighting: Clickable clauses with plain-language explanation + risk tags.
3. Interactive Q&A: Users ask questions → AI provides cited answers linked to the original text.
4. Readability & Complexity Dashboard: Flesch scores, radar charts, and wordclouds for quick insights.
5. Export Options: Simplified PDF, redline changes, risk checklist, CSV export of clauses.
6. Privacy & Security: PII masking and local redaction before processing.
7. Safety-first AI: “I don’t know” responses and flagged ambiguous items routed to a human legal reviewer.

<b>Unique Selling Points (USP):</b>
1. Accuracy + Simplicity → Demystifies without changing legal meaning.
2. Multi-tier outputs → From plain English for citizens to professional detail for lawyers.
3. Ensemble AI → Vertex AI (Gemini) as primary + HuggingFace Pegasus/BART fallback for robustness.
4. Trust & Safety → Confidence scores, source citations, and human-in-loop for edge cases.

# Technologies Used
<b>Frontend & Backend</b>
1. React / Flask + Bootstrap frontend
2. Flask REST API endpoints (/analyze, /ask)
3. Hosting: Cloud Run / App Engine

<b>AI & Data</b>
1. Generative AI: Gemini (Vertex AI) for clause simplification + Q&A
2. Summarization Ensemble: Pegasus (google/pegasus-xsum) & BART (facebook/bart-large-cnn)
3. Hugging Face Transformers (PyTorch / TensorFlow)
4. NLP preprocessing: PyPDF2, python-docx, NLTK, spaCy

<b>Infrastructure & Tools</b>
1. Cloud Storage for uploads & assets
2. Firestore/Postgres for metadata
3. Firebase Auth / OAuth for secure login
4. Matplotlib, Plotly, WordCloud for visual dashboards

<b>Security & DevOps</b>
1. TLS encryption, IAM roles, and Cloud Logging/Monitoring
2. CI/CD with GitHub Actions / Cloud Build

# Project Demonstration:
1. YouTube: []()
2. Web Demo: []()

# Team: Coding Masters
1. Sujal Bhagat: [https://www.linkedin.com/in/sujal-bhagat/](https://www.linkedin.com/in/sujal-bhagat/)
2. Nency Rana: [https://www.linkedin.com/in/nency-rana-a22454376/](https://www.linkedin.com/in/nency-rana-a22454376/)
3. Nishchal Kansara: [https://www.linkedin.com/in/nishchal-kansara/](https://www.linkedin.com/in/nishchal-kansara/)
