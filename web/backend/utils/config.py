import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/app.db")
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8080"))
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"

config = Config()
