# Jason RAG System

A production-ready RAG (Retrieval-Augmented Generation) system that answers questions based on your Medium posts and LinkedIn profile.

## Features

- 📝 Scrapes Medium posts via RSS
- 👔 Scrapes LinkedIn profile
- 🔍 Semantic search using sentence-transformers
- 🗄️ Vector storage with Qdrant
- 🤖 GPT-4o-mini for answer generation
- 🚀 FastAPI backend
- 💬 Streamlit chat interface

## Project Structure

```
jason-rag/
├── src/
│   ├── ingestion/
│   │   ├── scrapers/
│   │   │   ├── medium.py         # Fetch Medium posts via RSS
│   │   │   └── linkedin.py       # Scrape LinkedIn profile
│   │   ├── chunker.py            # Split text into chunks
│   │   ├── embedder.py           # Embed chunks + store in Qdrant
│   │   └── main.py               # Ingestion pipeline script
│   │
│   ├── retrieval/
│   │   ├── query.py              # Embed user question + search Qdrant
│   │   └── prompt.py             # Build prompt + generate answer
│   │
│   ├── api/
│   │   └── main.py               # FastAPI app, exposes /query endpoint
│   │
│   ├── frontend/
│   │   └── app.py                # Streamlit chat UI
│   │
│   └── config/
│       ├── config.py             # Configuration settings
│       └── database.py           # Qdrant vector database client
│
├── docker-compose.yml            # Spins up Qdrant
├── Dockerfile                    # Optional: containerize the app
├── requirements.txt              # Python dependencies
├── .env                          # API keys (create from .env.example)
└── README.md
```

## Setup

### 1. Clone and Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Qdrant

```bash
docker-compose up -d
```

This will start Qdrant on:
- HTTP API: http://localhost:6333
- gRPC API: http://localhost:6334
- Dashboard: http://localhost:6333/dashboard

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

Required variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `MEDIUM_USERNAME`: Your Medium username (e.g., @yourname)
- `LINKEDIN_PROFILE_URL`: Your LinkedIn profile URL
- `QDRANT_HOST`: Qdrant host (default: localhost)
- `QDRANT_PORT`: Qdrant port (default: 6333)
- `QDRANT_COLLECTION_NAME`: Collection name (default: jason_documents)

### 4. Run Ingestion Pipeline

```bash
python -m src.ingestion.main
```

This will:
1. Scrape your Medium posts and LinkedIn profile
2. Chunk the content into smaller pieces
3. Generate embeddings for each chunk
4. Store vectors in Qdrant

### 5. Start the API

```bash
python -m src.api.main
```

API will be available at http://localhost:8000

API endpoints:
- `GET /` - Health check
- `GET /health` - Health status
- `POST /query` - Ask a question

### 6. Launch the Frontend

```bash
streamlit run src/frontend/app.py
```

Frontend will open in your browser at http://localhost:8501

## Usage

### Ask Questions via Frontend

Open http://localhost:8501 and start chatting!

Example questions:
- "What did Jason learn about CI/CD pipelines?"
- "Summarize his experience with Terraform."
- "What technologies does Jason work with?"

### API Usage

**POST /query**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What did Jason learn about CI/CD?",
    "top_k": 5
  }'
```

Response:
```json
{
  "answer": "Based on the content...",
  "sources": [
    {
      "title": "Article Title",
      "content": "...",
      "source": "medium",
      "url": "...",
      "similarity": 0.85
    }
  ]
}
```

## Architecture

### Ingestion Pipeline

1. **Scrapers** (`src/ingestion/scrapers/`) - Fetch content from Medium RSS and LinkedIn
2. **Chunker** (`src/ingestion/chunker.py`) - Split documents into overlapping chunks
3. **Embedder** (`src/ingestion/embedder.py`) - Generate embeddings and store in Qdrant

### Retrieval Pipeline

1. **Query Engine** (`src/retrieval/query.py`) - Embed user question and search Qdrant
2. **Prompt Builder** (`src/retrieval/prompt.py`) - Build context and generate answer with LLM

## Tech Stack

- **Scraping**: requests, BeautifulSoup, feedparser
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: Qdrant
- **LLM**: GPT-4o-mini via OpenAI API
- **API**: FastAPI
- **Frontend**: Streamlit

## Configuration

### Adjust Chunk Size

Edit `src/ingestion/chunker.py`:
```python
chunker = TextChunker(chunk_size=500, overlap=50)
```

### Change Embedding Model

Edit `src/config/config.py`:
```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```

### Modify Retrieval Count

In API or frontend, adjust `top_k`:
```python
top_k: int = 5  # Number of similar chunks to retrieve
```

## Development

### Run Tests
```bash
pytest tests/
```

### Format Code
```bash
black src/
```

### Lint Code
```bash
pylint src/
```

## Troubleshooting

**Qdrant connection error**: Ensure Qdrant is running via `docker-compose up -d`

**Check Qdrant dashboard**: Visit http://localhost:6333/dashboard

**Embedding model download slow**: First run downloads ~80MB model

**LinkedIn scraping fails**: Use manual data export instead (Settings > Data Privacy > Get a copy of your data)

**API not responding**: Check if it's running on port 8000

**Import errors**: Make sure you're running from the project root directory

## Notes

### LinkedIn Scraping
LinkedIn has strong anti-scraping measures. For production:
- Export your LinkedIn data manually (Settings > Data Privacy)
- Use LinkedIn's official API
- Copy profile text into a file

### Re-running Ingestion
To update your content:
```bash
python -m src.ingestion.main
```

This will scrape new content and add it to Qdrant.

## License

MIT
