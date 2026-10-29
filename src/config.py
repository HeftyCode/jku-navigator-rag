import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load .env variables into the system environment immediately
load_dotenv()
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

class Settings(BaseSettings):
    PROJECT_NAME: str = "JKU & Austrian Student Navigator"
    QDRANT_PATH: str = "./qdrant_storage"
    COLLECTION_NAME: str = "jku_documents"
    
    # Free local HuggingFace models
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Free Groq LLM settings
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL: str = "openai/gpt-oss-120b"
    HF_HUB_DISABLE_SYMLINKS_WARNING: str = "1"

    class Config:
        env_file = ".env"
        extra = "ignore"  # Prevents Pydantic from crashing on extra .env variables

settings = Settings()