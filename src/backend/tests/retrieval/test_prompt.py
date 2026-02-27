import pytest
from unittest.mock import Mock, patch, MagicMock
from retrieval.prompt import PromptBuilder


@pytest.fixture
def prompt_builder():
    """Create a PromptBuilder instance with a mock API key."""
    return PromptBuilder(openai_api_key="test-api-key", model="gpt-4o-mini")


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "title": "Jason's Background",
            "source": "profile.txt",
            "content": "Jason is a software engineer with 10 years of experience."
        },
        {
            "title": "Jason's Skills",
            "source": "resume.txt",
            "content": "Jason specializes in Python and machine learning."
        }
    ]


def test_prompt_builder_initialization():
    """Test PromptBuilder initialization."""
    builder = PromptBuilder(openai_api_key="test-key", model="gpt-4")
    assert builder.model == "gpt-4"
    assert builder.client is not None


def test_prompt_builder_default_model():
    """Test PromptBuilder with default model."""
    builder = PromptBuilder(openai_api_key="test-key")
    assert builder.model == "gpt-4o-mini"


def test_build_context_single_document(prompt_builder):
    """Test building context with a single document."""
    documents = [
        {
            "title": "Test Document",
            "source": "test.txt",
            "content": "This is test content."
        }
    ]

    context = prompt_builder.build_context(documents)

    assert "[Source 1]" in context
    assert "Test Document" in context
    assert "test.txt" in context
    assert "This is test content." in context


def test_build_context_multiple_documents(prompt_builder, sample_documents):
    """Test building context with multiple documents."""
    context = prompt_builder.build_context(sample_documents)

    assert "[Source 1]" in context
    assert "[Source 2]" in context
    assert "Jason's Background" in context
    assert "Jason's Skills" in context
    assert "profile.txt" in context
    assert "resume.txt" in context
    assert "software engineer" in context
    assert "Python and machine learning" in context


def test_build_context_empty_documents(prompt_builder):
    """Test building context with empty document list."""
    context = prompt_builder.build_context([])
    assert context == ""


def test_build_context_formatting(prompt_builder):
    """Test that context is properly formatted."""
    documents = [
        {
            "title": "Doc1",
            "source": "source1.txt",
            "content": "Content 1"
        },
        {
            "title": "Doc2",
            "source": "source2.txt",
            "content": "Content 2"
        }
    ]

    context = prompt_builder.build_context(documents)
    lines = context.split("\n")

    # Should have proper structure
    assert any("[Source 1] Doc1 (source1.txt)" in line for line in lines)
    assert any("[Source 2] Doc2 (source2.txt)" in line for line in lines)


@patch("retrieval.prompt.OpenAI")
def test_generate_answer(mock_openai_class, prompt_builder):
    """Test answer generation with mocked OpenAI API."""
    # Mock the API response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "This is the generated answer."

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    prompt_builder.client = mock_client

    question = "What does Jason do?"
    context = "Jason is a software engineer."

    answer = prompt_builder.generate_answer(question, context)

    assert answer == "This is the generated answer."

    # Verify API was called correctly
    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args

    assert call_args.kwargs["model"] == "gpt-4o-mini"
    assert call_args.kwargs["temperature"] == 0.7
    assert call_args.kwargs["max_tokens"] == 500
    assert len(call_args.kwargs["messages"]) == 2
    assert "Jason is a software engineer" in call_args.kwargs["messages"][1]["content"]
    assert "What does Jason do?" in call_args.kwargs["messages"][1]["content"]


@patch("retrieval.prompt.OpenAI")
def test_generate_answer_prompt_structure(mock_openai_class, prompt_builder):
    """Test that the prompt includes all required elements."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Answer"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    prompt_builder.client = mock_client

    prompt_builder.generate_answer("Test question?", "Test context")

    call_args = mock_client.chat.completions.create.call_args
    user_message = call_args.kwargs["messages"][1]["content"]

    # Check prompt structure
    assert "Context:" in user_message
    assert "Question:" in user_message
    assert "Answer:" in user_message
    assert "Test context" in user_message
    assert "Test question?" in user_message
    assert "Jason" in user_message  # Should mention Jason


@patch("retrieval.prompt.OpenAI")
def test_answer_question(mock_openai_class, prompt_builder, sample_documents):
    """Test the complete answer_question workflow."""
    # Mock the API response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Jason is a software engineer."

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    prompt_builder.client = mock_client

    question = "What does Jason do?"
    result = prompt_builder.answer_question(question, sample_documents)

    # Check result structure
    assert "answer" in result
    assert "sources" in result
    assert result["answer"] == "Jason is a software engineer."
    assert result["sources"] == sample_documents


@patch("retrieval.prompt.OpenAI")
def test_answer_question_empty_documents(mock_openai_class, prompt_builder):
    """Test answer_question with no documents."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "I do not know him well enough to answer."

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    prompt_builder.client = mock_client

    result = prompt_builder.answer_question("Who is Jason?", [])

    assert result["answer"] == "I do not know him well enough to answer."
    assert result["sources"] == []


@patch("retrieval.prompt.OpenAI")
def test_answer_question_api_error(mock_openai_class, prompt_builder, sample_documents):
    """Test handling of API errors."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API Error")
    prompt_builder.client = mock_client

    with pytest.raises(Exception, match="API Error"):
        prompt_builder.answer_question("Test question", sample_documents)
