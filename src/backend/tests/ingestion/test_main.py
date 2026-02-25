import pytest
from unittest.mock import Mock, patch, MagicMock
from ingestion.main import IngestionPipeline


class TestIngestionPipeline:
    """Test suite for IngestionPipeline class."""

    def test_init_with_default_parameters(self):
        """Test pipeline initialization with default parameters."""
        pipeline = IngestionPipeline()

        assert pipeline.medium_username is None
        assert pipeline.embedding_model == 'BAAI/bge-small-en-v1.5'
        assert pipeline.vector_db is None
        assert pipeline.chunker is not None
        assert pipeline.embedder is None

    def test_init_with_custom_parameters(self):
        """Test pipeline initialization with custom parameters."""
        pipeline = IngestionPipeline(
            medium_username='testuser',
            embedding_model='custom-model'
        )

        assert pipeline.medium_username == 'testuser'
        assert pipeline.embedding_model == 'custom-model'

    @patch('ingestion.main.VectorDatabase')
    def test_context_manager_connects_and_closes_db(self, mock_vector_db):
        """Test context manager properly connects and closes database."""
        db_instance = Mock()
        mock_vector_db.return_value = db_instance

        with IngestionPipeline() as pipeline:
            assert pipeline.vector_db == db_instance
            db_instance.connect.assert_called_once()

        db_instance.close.assert_called_once()

    @patch('ingestion.main.VectorDatabase')
    def test_context_manager_closes_db_on_exception(self, mock_vector_db):
        """Test context manager closes database even when exception occurs."""
        db_instance = Mock()
        mock_vector_db.return_value = db_instance

        with pytest.raises(ValueError):
            with IngestionPipeline() as pipeline:
                raise ValueError("Test exception")

        db_instance.close.assert_called_once()

    @patch('ingestion.main.MediumScraper')
    @patch('ingestion.main.VectorDatabase')
    def test_scrape_content_with_medium_only(self, mock_vector_db, mock_medium_scraper):
        """Test scraping content from Medium only."""
        db_instance = Mock()
        db_instance.get_last_scraped_date.return_value = None
        mock_vector_db.return_value = db_instance

        medium_instance = Mock()
        medium_instance.scrape.return_value = [
            {'content': 'post1', 'title': 'Post 1'},
            {'content': 'post2', 'title': 'Post 2'}
        ]
        mock_medium_scraper.return_value = medium_instance

        pipeline = IngestionPipeline(medium_username='testuser')
        pipeline.vector_db = db_instance

        documents = pipeline._scrape_content()

        assert len(documents) == 2
        mock_medium_scraper.assert_called_once_with('testuser')
        medium_instance.scrape.assert_called_once_with(last_scraped_date=None)

    @patch('ingestion.main.VectorDatabase')
    def test_scrape_content_with_no_sources(self, mock_vector_db):
        """Test scraping with no sources configured."""
        db_instance = Mock()
        mock_vector_db.return_value = db_instance

        pipeline = IngestionPipeline()
        pipeline.vector_db = db_instance

        documents = pipeline._scrape_content()

        assert len(documents) == 0

    @patch('ingestion.main.MediumScraper')
    @patch('ingestion.main.VectorDatabase')
    def test_scrape_content_uses_last_scraped_date(self, mock_vector_db, mock_medium_scraper):
        """Test that scraping uses last scraped date from database."""
        db_instance = Mock()
        db_instance.get_last_scraped_date.return_value = '2024-01-01'
        mock_vector_db.return_value = db_instance

        medium_instance = Mock()
        medium_instance.scrape.return_value = [{'content': 'new post'}]
        mock_medium_scraper.return_value = medium_instance

        pipeline = IngestionPipeline(medium_username='testuser')
        pipeline.vector_db = db_instance

        pipeline._scrape_content()

        db_instance.get_last_scraped_date.assert_called_once_with('medium')
        medium_instance.scrape.assert_called_once_with(last_scraped_date='2024-01-01')

    def test_chunk_documents(self):
        """Test chunking documents."""
        pipeline = IngestionPipeline()
        pipeline.chunker = Mock()
        pipeline.chunker.chunk_documents.return_value = [
            {'content': 'chunk1', 'chunk_index': 0},
            {'content': 'chunk2', 'chunk_index': 1}
        ]

        documents = [{'content': 'doc1'}, {'content': 'doc2'}]
        chunks = pipeline._chunk_documents(documents)

        assert len(chunks) == 2
        pipeline.chunker.chunk_documents.assert_called_once_with(documents)

    @patch('ingestion.main.Embedder')
    def test_embed_and_store(self, mock_embedder):
        """Test embedding and storing documents."""
        embedder_instance = Mock()
        embedder_instance.embedding_dim = 384
        mock_embedder.return_value = embedder_instance

        db_instance = Mock()

        pipeline = IngestionPipeline(embedding_model='test-model')
        pipeline.vector_db = db_instance

        chunks = [{'content': 'chunk1'}, {'content': 'chunk2'}]
        pipeline._embed_and_store(chunks)

        mock_embedder.assert_called_once_with(model_name='test-model')
        db_instance.setup_database.assert_called_once_with(embedding_dim=384)
        embedder_instance.store.assert_called_once_with(chunks, db_instance)

    @patch('ingestion.main.Embedder')
    @patch('ingestion.main.MediumScraper')
    @patch('ingestion.main.VectorDatabase')
    def test_run_full_pipeline_success(self, mock_vector_db, mock_medium_scraper, mock_embedder):
        """Test successful execution of full pipeline."""
        # Setup mocks
        db_instance = Mock()
        db_instance.get_last_scraped_date.return_value = None
        mock_vector_db.return_value = db_instance

        medium_instance = Mock()
        medium_instance.scrape.return_value = [{'content': 'post'}]
        mock_medium_scraper.return_value = medium_instance

        embedder_instance = Mock()
        embedder_instance.embedding_dim = 384
        mock_embedder.return_value = embedder_instance

        # Create pipeline and run
        with IngestionPipeline(medium_username='testuser') as pipeline:
            pipeline.chunker = Mock()
            pipeline.chunker.chunk_documents.return_value = [{'content': 'chunk'}]
            pipeline.run()

        # Verify all steps executed
        medium_instance.scrape.assert_called_once()
        pipeline.chunker.chunk_documents.assert_called_once()
        embedder_instance.store.assert_called_once()

    @patch('ingestion.main.VectorDatabase')
    def test_run_pipeline_with_no_documents(self, mock_vector_db, capsys):
        """Test pipeline exits gracefully when no documents found."""
        db_instance = Mock()
        mock_vector_db.return_value = db_instance

        with IngestionPipeline() as pipeline:
            pipeline.run()

        captured = capsys.readouterr()
        assert "No documents found" in captured.out

    @patch('ingestion.main.Embedder')
    @patch('ingestion.main.MediumScraper')
    @patch('ingestion.main.VectorDatabase')
    def test_run_pipeline_skips_embedding_if_no_documents(self, mock_vector_db, mock_medium_scraper, mock_embedder):
        """Test pipeline doesn't call embedder when no documents scraped."""
        db_instance = Mock()
        db_instance.get_last_scraped_date.return_value = None
        mock_vector_db.return_value = db_instance

        medium_instance = Mock()
        medium_instance.scrape.return_value = []
        mock_medium_scraper.return_value = medium_instance

        with IngestionPipeline(medium_username='testuser') as pipeline:
            pipeline.run()

        mock_embedder.assert_not_called()
