import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
DATABASE_NAME = os.getenv('DATABASE_NAME', 'smart_parking')
CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')
USE_YOLO = os.getenv("USE_YOLO", "true").lower() == "true"
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

