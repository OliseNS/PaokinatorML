import os
from dotenv import load_dotenv

load_dotenv()

# --- Server ---
PORT = int(os.getenv("PORT", 8000)) # Default to 8000 for FastAPI

# --- Supabase ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# --- Admin ---
RELOAD_SECRET_TOKEN = os.getenv("RELOAD_SECRET_TOKEN")

# --- Redis ---
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
SESSION_TTL_SECONDS = 10 * 60  # 10 minutes

if not all([SUPABASE_URL, SUPABASE_KEY, REDIS_HOST, REDIS_PASSWORD]):
    print("WARNING: One or more environment variables are missing.")
    print("Please check your .env file or system environment.")
