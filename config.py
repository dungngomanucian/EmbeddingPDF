import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load biến môi trường
load_dotenv()

class Config:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    BUCKET_NAME = os.getenv("SUPABASE_BUCKET")
    SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret_key")
    
    # Cấu hình Gemini
    GEMINI_API_KEY = os.getenv("API_KEY")
    
    # Cấu hình tìm kiếm
    SIMILARITY_THRESHOLD = 0.6
    TOP_K_SEARCH = 10

# Khởi tạo Supabase Client (Singleton)
try:
    supabase: Client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
except Exception as e:
    print(f"Lỗi khởi tạo Supabase: {e}")
    supabase = None