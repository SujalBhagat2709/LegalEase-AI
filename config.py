import os

# Flask configuration
DEBUG = True
HOST = '0.0.0.0'
PORT = 5000
SECRET_KEY = 'b36168c300b9e6f7945e90333a9fac90b08b882bea4abc57'

# File upload configuration
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'data/raw'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}

# Path configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, 'data')
PROCESSED_DATA_FOLDER = os.path.join(DATA_FOLDER, 'processed')
VISUALIZATION_FOLDER = os.path.join(DATA_FOLDER, 'visualizations')
MODEL_FOLDER = os.path.join(BASE_DIR, 'models', 'trained_models')

# AI model configuration
PEGASUS_MODEL_NAME = "google/pegasus-xsum"
BART_MODEL_NAME = "facebook/bart-large-cnn"
GEMINI_API_KEY = "AIzaSyAk7fp-zNnZ1kfmCy1km9NhcsGXgwobKco"

# Visualization configuration
WORDCLOUD_WIDTH = 800
WORDCLOUD_HEIGHT = 600
CHART_WIDTH = 800
CHART_HEIGHT = 400