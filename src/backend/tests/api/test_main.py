import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from api.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_query_engine():
    """Mock the query engine."""
    with patch("api.main.query_engine") as mock:
        yield mock


@pytest.fixture
def mock_prompt_builder():
    """Mock the prompt builder."""
    with patch("api.main.prompt_builder") as mock:
        yield mock


def test_root(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Jason RAG API is running"}


def test_health(client):
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_query_success(client, mock_query_engine, mock_prompt_builder):
    """Test successful query."""
    # Mock retrieved documents
    mock_retrieved_docs = [
        {"content": "Document 1 content", "metadata": {"source": "doc1.txt"}},
        {"content": "Document 2 content", "metadata": {"source": "doc2.txt"}}
    ]
    mock_query_engine.search.return_value = mock_retrieved_docs

    # Mock answer
    mock_answer = {
        "answer": "This is the answer to the question.",
        "sources": [{"source": "doc1.txt"}, {"source": "doc2.txt"}]
    }
    mock_prompt_builder.answer_question.return_value = mock_answer

    # Make request
    response = client.post("/query", json={"question": "What is the answer?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "This is the answer to the question."
    assert len(data["sources"]) == 2

    # Verify mocks were called correctly
    mock_query_engine.search.assert_called_once_with("What is the answer?", top_k=5)
    mock_prompt_builder.answer_question.assert_called_once_with(
        "What is the answer?", mock_retrieved_docs
    )


def test_query_with_custom_top_k(client, mock_query_engine, mock_prompt_builder):
    """Test query with custom top_k parameter."""
    mock_query_engine.search.return_value = []
    mock_prompt_builder.answer_question.return_value = {
        "answer": "Answer",
        "sources": []
    }

    response = client.post("/query", json={"question": "Test question", "top_k": 10})

    assert response.status_code == 200
    mock_query_engine.search.assert_called_once_with("Test question", top_k=10)


def test_query_missing_question(client):
    """Test query with missing question field."""
    response = client.post("/query", json={})
    assert response.status_code == 422  # Validation error


def test_query_invalid_top_k(client):
    """Test query with invalid top_k type."""
    response = client.post("/query", json={"question": "Test", "top_k": "invalid"})
    assert response.status_code == 422  # Validation error


def test_query_search_error(client, mock_query_engine, mock_prompt_builder):
    """Test query when search raises an exception."""
    mock_query_engine.search.side_effect = Exception("Database error")

    response = client.post("/query", json={"question": "Test question"})

    assert response.status_code == 500
    assert "Database error" in response.json()["detail"]


def test_query_answer_generation_error(client, mock_query_engine, mock_prompt_builder):
    """Test query when answer generation raises an exception."""
    mock_query_engine.search.return_value = []
    mock_prompt_builder.answer_question.side_effect = Exception("LLM error")

    response = client.post("/query", json={"question": "Test question"})

    assert response.status_code == 500
    assert "LLM error" in response.json()["detail"]
