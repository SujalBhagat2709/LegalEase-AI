# # from transformers import PegasusForConditionalGeneration, PegasusTokenizer
# # from transformers import BartForConditionalGeneration, BartTokenizer
# # import google.generativeai as genai
# # import os

# # # Initialize models
# # def initialize_models():
# #     models = {}
    
# #     try:
# #         # Pegasus Model
# #         print("Loading Pegasus model...")
# #         pegasus_model_name = "google/pegasus-xsum"
# #         models['pegasus_tokenizer'] = PegasusTokenizer.from_pretrained(pegasus_model_name)
# #         models['pegasus_model'] = PegasusForConditionalGeneration.from_pretrained(pegasus_model_name)
# #         print("Pegasus model loaded successfully!")
# #     except Exception as e:
# #         print(f"Error loading Pegasus model: {str(e)}")
# #         models['pegasus_model'] = None
# #         models['pegasus_tokenizer'] = None
    
# #     try:
# #         # BART Model
# #         print("Loading BART model...")
# #         bart_model_name = "facebook/bart-large-cnn"
# #         models['bart_tokenizer'] = BartTokenizer.from_pretrained(bart_model_name)
# #         models['bart_model'] = BartForConditionalGeneration.from_pretrained(bart_model_name)
# #         print("BART model loaded successfully!")
# #     except Exception as e:
# #         print(f"Error loading BART model: {str(e)}")
# #         models['bart_model'] = None
# #         models['bart_tokenizer'] = None
    
# #     try:
# #         # Gemini Model - Using the correct library
# #         print("Initializing Gemini...")
# #         genai.configure(api_key=os.getenv('GEMINI_API_KEY', 'AIzaSyAk7fp-zNnZ1kfmCy1km9NhcsGXgwobKco'))
# #         models['gemini_model'] = genai.GenerativeModel('gemini-pro')
# #         print("Gemini initialized successfully!")
# #     except Exception as e:
# #         print(f"Error initializing Gemini: {str(e)}")
# #         models['gemini_model'] = None
    
# #     return models

# # # Summarize with Pegasus
# # def summarize_with_pegasus(text, model, tokenizer, max_length=150):
# #     try:
# #         if len(text.split()) < 10:
# #             return "Text is too short for meaningful summarization with Pegasus."
            
# #         inputs = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
# #         summary_ids = model.generate(**inputs, max_length=max_length, min_length=30, length_penalty=2.0)
# #         return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
# #     except Exception as e:
# #         return f"Pegasus summarization error: {str(e)}"

# # # Summarize with BART
# # def summarize_with_bart(text, model, tokenizer, max_length=150):
# #     try:
# #         if len(text.split()) < 10:
# #             return "Text is too short for meaningful summarization with BART."
            
# #         inputs = tokenizer(text, return_tensors="pt", truncation=True)
# #         summary_ids = model.generate(**inputs, max_length=max_length, min_length=50, length_penalty=2.0)
# #         return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
# #     except Exception as e:
# #         return f"BART summarization error: {str(e)}"

# # # Summarize with Gemini
# # def summarize_with_gemini(text, model, complexity_level='medium'):
# #     try:
# #         if len(text.split()) < 10:
# #             return "Text is too short for meaningful summarization."
            
# #         level_prompts = {
# #             'easy': "Summarize this legal document in very simple language that a 10th grader can understand. Focus on the key points and avoid legal jargon.",
# #             'medium': "Summarize this legal document in clear language for an educated adult. Balance between simplicity and professional terminology.",
# #             'professional': "Provide a detailed professional summary of this legal document for legal experts. Use appropriate legal terminology and include all critical details."
# #         }
        
# #         prompt = f"{level_prompts[complexity_level]}\n\nDocument:\n{text[:3000]}"
        
# #         response = model.generate_content(prompt)
        
# #         if hasattr(response, 'text'):
# #             return response.text
# #         else:
# #             return "Summary generated successfully. Please check the results."
# #     except Exception as e:
# #         return f"Gemini summarization error: {str(e)}"

# # # Answer questions using Gemini
# # def answer_question(document_text, question):
# #     try:
# #         # Configure Gemini with the API key directly
# #         genai.configure(api_key=os.getenv('GEMINI_API_KEY', 'AIzaSyAk7fp-zNnZ1kfmCy1km9NhcsGXgwobKco'))
# #         model = genai.GenerativeModel('gemini-pro')
        
# #         prompt = f"""
# #         Based on the following legal document, please answer the user's question in simple, clear language.
# #         If the answer isn't clear from the document, say so rather than guessing.

# #         Document:
# #         {document_text[:3000]}

# #         Question: {question}

# #         Answer:
# #         """
        
# #         response = model.generate_content(prompt)
        
# #         if hasattr(response, 'text'):
# #             return response.text
# #         else:
# #             return "Answer generated successfully. Please check the response."
# #     except Exception as e:
# #         return f"Error answering question: {str(e)}"

# import google.generativeai as genai
# import os

# # Only use Gemini - it's the most reliable
# def initialize_models():
#     print("Initializing Gemini AI...")
#     try:
#         # Use your API key directly (replace with your actual key)
#         genai.configure(api_key='AIzaSyC-V3pBqhDBFXaYSsR8hdOKz1XxVx1WBOw')
#         gemini_model = genai.GenerativeModel('gemini-pro')
#         print("✅ Gemini AI Ready!")
#         return {'gemini': gemini_model}
#     except Exception as e:
#         print(f"❌ Gemini Error: {e}")
#         return {'gemini': None}

# # Simple summarization
# def summarize_with_gemini(text, model, complexity_level='medium'):
#     try:
#         prompts = {
#             'easy': "Explain this legal document like I'm 15 years old. Keep it very simple:",
#             'medium': "Summarize this legal document clearly for an average adult:",
#             'professional': "Provide a detailed professional legal analysis of this document:"
#         }
        
#         prompt = f"{prompts[complexity_level]}\n\n{text[:2000]}"
#         response = model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         return f"Summary: {str(e)}"

# # Simple question answering
# def answer_question(document_text, question):
#     try:
#         genai.configure(api_key='AIzaSyC-V3pBqhDBFXaYSsR8hdOKz1XxVx1WBOw')
#         model = genai.GenerativeModel('gemini-pro')
        
#         prompt = f"Document: {document_text[:1500]}\n\nQuestion: {question}\n\nAnswer:"
#         response = model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         return f"Answer: {str(e)}"

# # Placeholder functions for other models
# def summarize_with_pegasus(text, model, tokenizer):
#     return "Pegasus: Legal analysis optimized for clarity"

# def summarize_with_bart(text, model, tokenizer):
#     return "BART: Comprehensive document insights available"

# from transformers import PegasusForConditionalGeneration, PegasusTokenizer
# from transformers import BartForConditionalGeneration, BartTokenizer
import google.generativeai as genai
import os
from config import GEMINI_API_KEY

# Initialize Gemini
def initialize_models():
    models = {}
    try:
        print("Initializing Gemini...")
        genai.configure(api_key=os.getenv('GEMINI_API_KEY', GEMINI_API_KEY))
        models['gemini'] = genai.GenerativeModel('gemini-pro')
        print("Gemini initialized successfully!")
    except Exception as e:
        print(f"Error initializing Gemini: {str(e)}")
        models['gemini'] = None
    return models


# Summarize with Gemini
def summarize_with_gemini(text, model=None, complexity_level='medium'):
    try:
        if not model:
            model = genai.GenerativeModel('gemini-pro')

        if len(text.split()) < 10:
            return "Text is too short for meaningful summarization."

        level_prompts = {
            'easy': "Summarize this legal document in very simple language that a 10th grader can understand.",
            'medium': "Summarize this legal document in clear language for an educated adult.",
            'professional': "Provide a detailed professional summary of this legal document for legal experts."
        }

        prompt = f"{level_prompts.get(complexity_level, level_prompts['medium'])}\n\nDocument:\n{text[:3000]}"
        response = model.generate_content(prompt)

        if hasattr(response, "text"):
            return response.text
        return "Summary generated successfully, but response format was unexpected."
    except Exception as e:
        return f"Gemini summarization error: {str(e)}"


# Answer questions using Gemini
def answer_question(document_text, question, model=None):
    try:
        if not model:
            model = genai.GenerativeModel('gemini-pro')

        prompt = f"""
        Based on the following legal document, please answer the user's question clearly.
        If the answer isn't in the document, say so.

        Document:
        {document_text[:3000]}

        Question: {question}

        Answer:
        """
        response = model.generate_content(prompt)

        if hasattr(response, "text"):
            return response.text
        return "Answer generated successfully, but response format was unexpected."
    except Exception as e:
        return f"Error answering question: {str(e)}"