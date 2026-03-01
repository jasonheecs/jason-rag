import pytest
from ingestion.chunker import TextChunker


class TestTextChunker:
    """Test suite for TextChunker class."""

    def test_init_default_values(self):
        """Test chunker initialization with default values."""
        chunker = TextChunker()
        assert chunker.chunk_size == 256
        assert chunker.overlap == 25

    def test_init_custom_values(self):
        """Test chunker initialization with custom values."""
        chunker = TextChunker(chunk_size=100, overlap=10)
        assert chunker.chunk_size == 100
        assert chunker.overlap == 10

    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        chunker = TextChunker(chunk_size=5, overlap=2)
        text = "one two three four five six seven eight nine ten"
        chunks = chunker.chunk_text(text)

        assert len(chunks) == 3
        assert chunks[0] == "one two three four five"
        assert chunks[1] == "four five six seven eight"
        assert chunks[2] == "seven eight nine ten"

    def test_chunk_text_no_overlap(self):
        """Test chunking with no overlap."""
        chunker = TextChunker(chunk_size=3, overlap=0)
        text = "one two three four five six"
        chunks = chunker.chunk_text(text)

        assert len(chunks) == 2
        assert chunks[0] == "one two three"
        assert chunks[1] == "four five six"

    def test_chunk_text_empty_string(self):
        """Test chunking empty string."""
        chunker = TextChunker()
        chunks = chunker.chunk_text("")
        assert chunks == []

    def test_chunk_text_single_word(self):
        """Test chunking single word."""
        chunker = TextChunker(chunk_size=5, overlap=2)
        chunks = chunker.chunk_text("hello")

        assert len(chunks) == 1
        assert chunks[0] == "hello"

    def test_chunk_text_shorter_than_chunk_size(self):
        """Test text shorter than chunk size."""
        chunker = TextChunker(chunk_size=100, overlap=10)
        text = "short text here"
        chunks = chunker.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_documents_single_doc(self):
        """Test chunking single document with metadata preservation."""
        chunker = TextChunker(chunk_size=5, overlap=2)
        documents = [
            {
                'content': 'one two three four five six seven eight',
                'title': 'Test Doc',
                'url': 'https://example.com',
                'source': 'test'
            }
        ]

        result = chunker.chunk_documents(documents)

        assert len(result) == 2
        assert result[0]['content'] == 'one two three four five'
        assert result[0]['title'] == 'Test Doc'
        assert result[0]['url'] == 'https://example.com'
        assert result[0]['source'] == 'test'
        assert result[0]['chunk_index'] == 0

        assert result[1]['content'] == 'four five six seven eight'
        assert result[1]['chunk_index'] == 1

    def test_chunk_documents_multiple_docs(self):
        """Test chunking multiple documents."""
        chunker = TextChunker(chunk_size=3, overlap=1)
        documents = [
            {
                'content': 'one two three four',
                'title': 'Doc 1'
            },
            {
                'content': 'alpha beta gamma delta',
                'title': 'Doc 2'
            }
        ]

        result = chunker.chunk_documents(documents)

        assert len(result) == 4
        assert result[0]['title'] == 'Doc 1'
        assert result[0]['chunk_index'] == 0
        assert result[1]['title'] == 'Doc 1'
        assert result[1]['chunk_index'] == 1
        assert result[2]['title'] == 'Doc 2'
        assert result[2]['chunk_index'] == 0
        assert result[3]['title'] == 'Doc 2'
        assert result[3]['chunk_index'] == 1

    def test_chunk_documents_empty_list(self):
        """Test chunking empty document list."""
        chunker = TextChunker()
        result = chunker.chunk_documents([])
        assert result == []

    def test_chunk_documents_metadata_isolation(self):
        """Test that modifying one chunk's scalar fields doesn't affect other chunks."""
        chunker = TextChunker(chunk_size=3, overlap=1)
        documents = [
            {
                'content': 'one two three four',
                'source': 'test',
            }
        ]

        result = chunker.chunk_documents(documents)
        result[0]['source'] = 'modified'

        # Other chunks should not be affected
        assert result[1]['source'] == 'test'

    @pytest.mark.parametrize("chunk_size,overlap,word_count,expected_chunks", [
        (10, 2, 25, 3),
        (5, 0, 20, 4),
        (100, 10, 50, 1),
    ])
    def test_chunk_text_various_sizes(self, chunk_size, overlap, word_count, expected_chunks):
        """Test chunking with various size parameters."""
        chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        text = ' '.join([f'word{i}' for i in range(word_count)])
        chunks = chunker.chunk_text(text)
        assert len(chunks) == expected_chunks

    def test_create_chunked_document(self):
        """Test the _create_chunked_document helper method."""
        chunker = TextChunker()
        original_doc = {
            'content': 'original content',
            'title': 'Test',
            'url': 'https://example.com'
        }

        result = chunker._create_chunked_document(original_doc, 'new chunk content', 2)

        assert result['content'] == 'new chunk content'
        assert result['chunk_index'] == 2
        assert result['title'] == 'Test'
        assert result['url'] == 'https://example.com'
        assert original_doc['content'] == 'original content'  # Original unchanged
