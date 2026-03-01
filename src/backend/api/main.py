"""FastAPI application for Jason RAG API."""
import asyncio
import json
from functools import lru_cache
from typing import List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config.database import VectorDatabase
from config.config import OPENAI_API_KEY, LLM_MODEL, EMBEDDING_MODEL
from ingestion.embedder import Embedder
from retrieval.query import QueryEngine
from retrieval.prompt import PromptBuilder

app = FastAPI(title="Jason RAG API", version="1.0.0")

# Initialize components
embedder = Embedder(model_name=EMBEDDING_MODEL)
vector_db = VectorDatabase()
vector_db.connect()

query_engine = QueryEngine(embedder, vector_db)
prompt_builder = PromptBuilder(openai_api_key=OPENAI_API_KEY, model=LLM_MODEL)


@lru_cache(maxsize=128)
def _cached_rag_query(question: str, top_k: int):
    retrieved_docs = query_engine.search(question, top_k=top_k)
    return prompt_builder.answer_question(question, retrieved_docs)


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    answer: str
    sources: List[Dict]


@app.get("/")
def root():
    """Root endpoint."""
    return {"message": "Jason RAG API is running"}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Answer a question using RAG."""
    try:
        return await asyncio.to_thread(_cached_rag_query, request.question, request.top_k)
    except Exception as e:
        print(f"Error in query endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Answer a question using RAG with streaming LLM response."""
    try:
        retrieved_docs = await asyncio.to_thread(
            query_engine.search, request.question, request.top_k
        )
        context = prompt_builder.build_context(retrieved_docs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    def generate():
        yield f"data: {json.dumps({'type': 'sources', 'sources': retrieved_docs})}\n\n"
        for chunk in prompt_builder.generate_answer_stream(request.question, context):
            yield f"data: {json.dumps({'type': 'text', 'content': chunk})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
