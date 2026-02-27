from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
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


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]


@app.get("/")
def root():
    return {"message": "Jason RAG API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Answer a question using RAG."""
    try:
        # Retrieve similar documents
        retrieved_docs = query_engine.search(request.question, top_k=request.top_k)
        # Generate answer
        result = prompt_builder.answer_question(request.question, retrieved_docs)

        return result
    except Exception as e:
        print(f"Error in query endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
