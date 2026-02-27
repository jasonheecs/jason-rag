import pytest
from unittest.mock import Mock, patch
import numpy as np
from ingestion.embedder import Embedder


class TestEmbedder:
    """Test suite for Embedder class."""

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
    DEFAULT_EMBEDDING_DIM = 384

    @pytest.fixture
    def mock_embedder_setup(self):
        """Fixture to create mock TextEmbedding and Embedder instance."""
        with patch('ingestion.embedder.TextEmbedding') as mock_text_embedding:
            mock_model = Mock()
            mock_text_embedding.return_value = mock_model
            yield mock_text_embedding, mock_model

    def _create_embedder(self, mock_setup, model_name=None):
        """Helper to create Embedder instance with mocked dependencies."""
        return Embedder(model_name=model_name) if model_name else Embedder()

    def test_init_default_model(self, mock_embedder_setup):
        """Test embedder initialization with default model."""
        mock_text_embedding, mock_model = mock_embedder_setup
        embedder = self._create_embedder(mock_embedder_setup)

        mock_text_embedding.assert_called_once_with(model_name=self.DEFAULT_MODEL)
        assert embedder.model == mock_model
        assert embedder.embedding_dim == self.DEFAULT_EMBEDDING_DIM

    def test_init_custom_model(self, mock_embedder_setup):
        """Test embedder initialization with custom model."""
        mock_text_embedding, _ = mock_embedder_setup
        embedder = self._create_embedder(mock_embedder_setup, "custom-model")

        mock_text_embedding.assert_called_once_with(model_name="custom-model")
        assert embedder.embedding_dim == self.DEFAULT_EMBEDDING_DIM

    def test_embed_text(self, mock_embedder_setup):
        """Test embedding single text."""
        _, mock_model = mock_embedder_setup
        expected_embedding = [0.1, 0.2, 0.3]
        mock_model.embed.return_value = iter([expected_embedding])

        embedder = self._create_embedder(mock_embedder_setup)
        result = embedder.embed_text("test text")

        mock_model.embed.assert_called_once_with(["test text"])
        np.testing.assert_array_equal(result, np.array(expected_embedding))

    def test_embed_batch(self, mock_embedder_setup):
        """Test embedding multiple texts."""
        _, mock_model = mock_embedder_setup
        expected_embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_model.embed.return_value = iter(expected_embeddings)

        embedder = self._create_embedder(mock_embedder_setup)
        texts = ["text1", "text2", "text3"]
        result = embedder.embed_batch(texts)

        mock_model.embed.assert_called_once_with(texts)
        np.testing.assert_array_equal(result, np.array(expected_embeddings))

    def test_embed_batch_empty_list(self, mock_embedder_setup):
        """Test embedding empty list of texts."""
        _, mock_model = mock_embedder_setup
        mock_model.embed.return_value = iter([])

        embedder = self._create_embedder(mock_embedder_setup)

        assert len(embedder.embed_batch([])) == 0

    def test_add_embeddings_to_documents(self, mock_embedder_setup):
        """Test the _add_embeddings_to_documents helper method."""
        embedder = self._create_embedder(mock_embedder_setup)
        documents = [{'content': 'text1'}, {'content': 'text2'}]
        embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])

        embedder._add_embeddings_to_documents(documents, embeddings)

        for i, doc in enumerate(documents):
            np.testing.assert_array_equal(doc['embedding'], embeddings[i])

    def test_embed_documents(self, mock_embedder_setup, capsys):
        """Test embedding documents adds embeddings in-place."""
        _, mock_model = mock_embedder_setup
        embeddings_list = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_model.embed.return_value = iter(embeddings_list)

        embedder = self._create_embedder(mock_embedder_setup)
        documents = [
            {'content': 'text1', 'title': 'Doc 1'},
            {'content': 'text2', 'title': 'Doc 2'}
        ]

        embedder.embed_documents(documents)

        for i, doc in enumerate(documents):
            np.testing.assert_array_equal(doc['embedding'], np.array(embeddings_list[i]))

        assert "Generating embeddings for 2 chunks" in capsys.readouterr().out

    def test_store(self, mock_embedder_setup, capsys):
        """Test embedding and storing documents."""
        _, mock_model = mock_embedder_setup
        embeddings_list = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_model.embed.return_value = iter(embeddings_list)

        embedder = self._create_embedder(mock_embedder_setup)
        documents = [
            {'content': 'text1', 'title': 'Doc 1'},
            {'content': 'text2', 'title': 'Doc 2'}
        ]
        mock_vector_db = Mock()

        embedder.store(documents, mock_vector_db)

        for i, doc in enumerate(documents):
            np.testing.assert_array_equal(doc['embedding'], np.array(embeddings_list[i]))

        mock_vector_db.insert_documents.assert_called_once_with(documents)

        output = capsys.readouterr().out
        assert "Generating embeddings for 2 chunks" in output
        assert "Storing embeddings in Qdrant" in output
        assert "Successfully embedded and stored 2 chunks" in output

    def test_store_single_document(self, mock_embedder_setup):
        """Test embedding and storing single document."""
        _, mock_model = mock_embedder_setup
        embedding_list = [[0.1, 0.2, 0.3]]
        mock_model.embed.return_value = iter(embedding_list)

        embedder = self._create_embedder(mock_embedder_setup)
        documents = [{'content': 'test text'}]
        mock_vector_db = Mock()

        embedder.store(documents, mock_vector_db)

        assert 'embedding' in documents[0]
        np.testing.assert_array_equal(documents[0]['embedding'], np.array(embedding_list[0]))
        mock_vector_db.insert_documents.assert_called_once()

    def test_store_preserves_metadata(self, mock_embedder_setup):
        """Test that store preserves document metadata."""
        _, mock_model = mock_embedder_setup
        mock_model.embed.return_value = iter([[0.1, 0.2]])

        embedder = self._create_embedder(mock_embedder_setup)
        expected_metadata = {
            'content': 'text',
            'title': 'Test',
            'url': 'https://example.com',
            'source': 'test',
            'custom_field': 'value'
        }
        documents = [expected_metadata.copy()]
        mock_vector_db = Mock()

        embedder.store(documents, mock_vector_db)

        doc = documents[0]
        for key, value in expected_metadata.items():
            assert doc[key] == value
        assert 'embedding' in doc

    def test_store_extracts_content_correctly(self, mock_embedder_setup):
        """Test that content is correctly extracted for batch embedding."""
        _, mock_model = mock_embedder_setup
        mock_model.embed.return_value = iter([[0.1], [0.2], [0.3]])

        embedder = self._create_embedder(mock_embedder_setup)
        documents = [
            {'content': 'first'},
            {'content': 'second'},
            {'content': 'third'}
        ]

        embedder.store(documents, Mock())

        assert mock_model.embed.call_args[0][0] == ['first', 'second', 'third']
