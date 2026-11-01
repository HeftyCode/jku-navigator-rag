import asyncio
from contextlib import asynccontextmanager
from typing import Literal, Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, ConfigDict
from groq import RateLimitError
from src.config import settings
from src.generation.generator import JKUNavigatorGenerator

rag_engine: Optional[JKUNavigatorGenerator] = None
concurrency_gate = asyncio.Semaphore(2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine
    rag_engine = JKUNavigatorGenerator()
    yield
    if rag_engine:
        rag_engine.close()


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Bilingual (EN/DE) Hybrid RAG + Multilingual Cross-Encoder Reranking API for JKU Linz.",
    version="1.1.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    target_language: Literal["auto", "en", "de"] = Field(default="auto")
    top_n_chunks: int = Field(default=5, ge=1, le=10)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "How does credit transfer work between Informatik and AI at JKU?",
                "target_language": "auto",
                "top_n_chunks": 5,
            }
        }
    )


class QueryResponse(BaseModel):
    user_query: str
    detected_language: str
    reply_language: str
    bilingual_search_query: str
    answer: str
    is_supported: bool
    citations: List[Dict[str, Any]]
    citation_audit: Dict[str, Any]
    model_used: str
    latency_ms: float


@app.get("/health")
async def health_liveness():
    """Non-blocking liveness probe for container orchestrators."""
    return {"status": "alive", "service": settings.PROJECT_NAME}


@app.get("/ready")
async def readiness_check():
    """Readiness probe verifying Qdrant index and chunk map are loaded and synchronized."""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG Engine is still initializing.")
    return {
        "status": "ready",
        "indexed_chunks": len(rag_engine.retriever.chunks),
        "primary_model": rag_engine.last_used_model,
    }


@app.post("/ask", response_model=QueryResponse)
async def ask_navigator(request: QueryRequest):
    """Non-blocking Bilingual RAG endpoint executed in worker threadpool with bounded concurrency."""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="Service initializing.")
    try:
        async with concurrency_gate:
            result = await run_in_threadpool(
                rag_engine.answer_question,
                request.query,
                request.target_language,
                request.top_n_chunks,
            )
        return QueryResponse(**result)
    except RateLimitError:
        raise HTTPException(status_code=429, detail="Upstream LLM rate limit reached. Please retry shortly.")
    except Exception:
        raise HTTPException(status_code=502, detail="Upstream generation error occurred.")