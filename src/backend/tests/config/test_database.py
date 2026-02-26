import pytest
from unittest.mock import MagicMock, patch
from config.database import VectorDatabase


class TestVectorDatabase:
    """Test suite for the VectorDatabase class."""

    @pytest.fixture
    def setup_mock_database(self):
        """Fixture to provide a mock VectorDatabase instance."""
        vector_db = VectorDatabase()
        vector_db.client = MagicMock()
        return vector_db

    def test_connect_creates_qdrant_client(self):
        """Test that connect initializes the Qdrant client."""
        with patch("config.database.QdrantClient") as mock_qdrant_client:
            vector_db = VectorDatabase()
            client = vector_db.connect()
            assert vector_db.client == client
            mock_qdrant_client.assert_called_once_with(host=vector_db.host, port=vector_db.port)

    def test_setup_database_behavior(self, setup_mock_database):
        """Test setup_database creates collection if not exists, skips otherwise."""
        vector_db = setup_mock_database

        # Test when collection does not exist
        vector_db.client.get_collections.return_value.collections = []
        vector_db.setup_database(embedding_dim=384)
        vector_db.client.create_collection.assert_called_once()
        vector_db.client.create_collection.reset_mock()

        # Test when collection already exists
        mock_collection = MagicMock()
        mock_collection.name = vector_db.collection_name
        vector_db.client.get_collections.return_value.collections = [mock_collection]
        vector_db.setup_database(embedding_dim=384)
        vector_db.client.create_collection.assert_not_called()

    def test_insert_documents_calls_upsert(self, setup_mock_database):
        """Test insert_documents calls upsert with the correct points."""
        vector_db = setup_mock_database
        documents = [
            {
                "embedding": MagicMock(),
                "title": "Document Title",
                "content": "Document Content",
                "source": "Document Source",
                "url": "http://example.com",
                "published_date": MagicMock(),
                "chunk_index": 1,
            }
        ]

        with patch.object(vector_db, "_create_point_from_document", return_value="mock_point") as mock_create_point:
            vector_db.insert_documents(documents)
            mock_create_point.assert_called_once_with(documents[0])
            vector_db.client.upsert.assert_called_once_with(
                collection_name=vector_db.collection_name,
                points=["mock_point"],
            )

    def test_search_similar_calls_search(self, setup_mock_database):
        """Test search_similar performs search with expected parameters and returns results."""
        vector_db = setup_mock_database
        query_embedding = MagicMock()
        vector_db.client.search.return_value = []

        results = vector_db.search_similar(query_embedding=query_embedding, top_k=5)

        vector_db.client.search.assert_called_once_with(
            collection_name=vector_db.collection_name,
            query_vector=query_embedding.tolist(),
            limit=5,
        )
        assert results == []

    def test_get_last_scraped_date_without_client(self):
        """Test get_last_scraped_date returns None when client is None."""
        vector_db = VectorDatabase()
        result = vector_db.get_last_scraped_date(source="source_name")
        assert result is None

    def test_get_last_scraped_date_collection_not_exists(self, setup_mock_database):
        """Test get_last_scraped_date returns None when collection doesn't exist."""
        vector_db = setup_mock_database
        vector_db.client.get_collections.return_value.collections = []

        result = vector_db.get_last_scraped_date(source="source_name")

        assert result is None
        vector_db.client.scroll.assert_not_called()

    def test_get_last_scraped_date_collection_empty(self, setup_mock_database):
        """Test get_last_scraped_date returns None when collection is empty."""
        vector_db = setup_mock_database

        # Mock collection exists
        mock_collection = MagicMock()
        mock_collection.name = vector_db.collection_name
        vector_db.client.get_collections.return_value.collections = [mock_collection]

        # Mock collection is empty
        vector_db.client.count.return_value.count = 0

        result = vector_db.get_last_scraped_date(source="source_name")

        assert result is None
        vector_db.client.scroll.assert_not_called()

    def test_get_last_scraped_date_with_results(self, setup_mock_database):
        """Test get_last_scraped_date returns the most recent date."""
        from datetime import datetime

        vector_db = setup_mock_database

        # Mock collection exists
        mock_collection = MagicMock()
        mock_collection.name = vector_db.collection_name
        vector_db.client.get_collections.return_value.collections = [mock_collection]

        # Mock collection has documents
        vector_db.client.count.return_value.count = 10

        # Mock scroll returns a document
        mock_point = MagicMock(payload={"published_date": "2023-01-01T12:00:00"})
        vector_db.client.scroll.return_value = ([mock_point], None)

        result = vector_db.get_last_scraped_date(source="source_name")

        assert result == datetime.fromisoformat("2023-01-01T12:00:00")
        vector_db.client.scroll.assert_called_once()

    def test_get_last_scraped_date_no_matching_source(self, setup_mock_database):
        """Test get_last_scraped_date returns None when no documents match the source."""
        vector_db = setup_mock_database

        # Mock collection exists and has documents
        mock_collection = MagicMock()
        mock_collection.name = vector_db.collection_name
        vector_db.client.get_collections.return_value.collections = [mock_collection]
        vector_db.client.count.return_value.count = 10

        # Mock scroll returns empty results
        vector_db.client.scroll.return_value = ([], None)

        result = vector_db.get_last_scraped_date(source="source_name")

        assert result is None

    def test_close(self, setup_mock_database):
        """Test close calls client.close()"""
        vector_db = setup_mock_database
        vector_db.close()
        vector_db.client.close.assert_called_once()