from contextlib import asynccontextmanager
from typing import Literal, Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.config import settings
from src.generation.generator import JKUNavigatorGenerator

rag_engine: Optional[JKUNavigatorGenerator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the embedding model, reranker, and Qdrant index once on server startup."""
    global rag_engine
    rag_engine = JKUNavigatorGenerator()
    yield
    if rag_engine:
        rag_engine.close()


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Bilingual (EN/DE) Hybrid RAG + Cross-Encoder Reranking API for JKU Linz Regulations & Curricula.",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, example="How does credit transfer work between Informatik and AI?")
    target_language: Literal["auto", "en", "de"] = Field(default="auto")
    top_n_chunks: int = Field(default=5, ge=1, le=10)


class QueryResponse(BaseModel):
    user_query: str
    detected_language: str
    reply_language: str
    bilingual_search_query: str
    answer: str
    is_supported: bool
    citations: List[Dict[str, Any]]


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration and uptime monitoring."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "indexed_chunks": len(rag_engine.retriever.chunks) if rag_engine else 0,
    }


@app.post("/ask", response_model=QueryResponse)
async def ask_navigator(request: QueryRequest):
    """End-to-end Bilingual RAG endpoint returning grounded answers and page-level citations."""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG Engine is still initializing.")
    try:
        result = rag_engine.answer_question(
            user_query=request.query,
            target_language=request.target_language,
            top_n_chunks=request.top_n_chunks,
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))