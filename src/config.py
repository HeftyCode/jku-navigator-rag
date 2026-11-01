import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "JKU & Austrian Student Navigator"
    QDRANT_PATH: str = str(BASE_DIR / "qdrant_storage")
    CHUNKS_PATH: str = str(BASE_DIR / "data" / "processed" / "chunks.json")
    COLLECTION_NAME: str = "jku_documents"

    # Multilingual Dense Embedder (max_seq_length = 128 tokens)
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # Multilingual Cross-Encoder Reranker (trained on German + English mMARCO)
    RERANKER_MODEL: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    # Minimum Cross-Encoder logit score before abstention gate triggers
    MIN_RERANK_SCORE: float = -4.5

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL: str = "openai/gpt-oss-120b"
    HF_HUB_DISABLE_SYMLINKS_WARNING: str = "1"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()