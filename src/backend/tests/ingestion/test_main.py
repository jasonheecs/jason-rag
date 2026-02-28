import pytest
import os
from unittest.mock import Mock, patch
from datetime import datetime, timezone
from ingestion.main import IngestionPipeline


def _create_mock_env(medium_username=None, github_username=None):
    """Helper to create a mock os.getenv function."""
    def getenv_side_effect(key, default=None):
        env_vars = {}
        if medium_username:
            env_vars['MEDIUM_USERNAME'] = medium_username
        if github_username:
            env_vars['GITHUB_USERNAME'] = github_username
        return env_vars.get(key, default)
    return getenv_side_effect


class TestIngestionPipeline:
    """Test suite for IngestionPipeline class."""

    def test_init_with_default_parameters(self):
        """Test pipeline initialization with default parameters."""
        pipeline = IngestionPipeline()

        assert pipeline.embedding_model == 'BAAI/bge-small-en-v1.5'
        assert pipeline.vector_db is None
        assert pipeline.chunker is not None
        assert pipeline.embedder is None

    def test_init_with_custom_parameters(self):
        """Test pipeline initialization with custom parameters."""
        pipeline = IngestionPipeline(embedding_model='custom-model')

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

    @patch('ingestion.main.os.getenv')
    @patch('ingestion.main.Embedder')
    @patch('ingestion.main.MediumScraper')
    @patch('ingestion.main.VectorDatabase')
    def test_run_with_medium_source(self, mock_vector_db, mock_medium_scraper, mock_embedder, mock_getenv):
        """Test full pipeline execution with Medium source."""
        db_instance = Mock()
        db_instance.get_last_scraped_date.return_value = None
        mock_vector_db.return_value = db_instance

        medium_instance = Mock()
        medium_instance.scrape.return_value = [
            {'content': 'post1', 'title': 'Post 1'},
            {'content': 'post2', 'title': 'Post 2'}
        ]
        mock_medium_scraper.return_value = medium_instance

        embedder_instance = Mock()
        embedder_instance.embedding_dim = 384
        mock_embedder.return_value = embedder_instance

        mock_getenv.side_effect = _create_mock_env(medium_username='testuser')

        with IngestionPipeline() as pipeline:
            pipeline.run()

        # Verify all major steps were executed
        medium_instance.scrape.assert_called_once()
        embedder_instance.store.assert_called_once()
        db_instance.close.assert_called_once()

    @patch('ingestion.main.VectorDatabase')
    def test_run_with_no_sources(self, mock_vector_db, capsys):
        """Test pipeline exits gracefully when no sources configured."""
        db_instance = Mock()
        mock_vector_db.return_value = db_instance

        with patch.dict(os.environ, {}, clear=True):
            with IngestionPipeline() as pipeline:
                pipeline.run()

        captured = capsys.readouterr()
        assert "No documents found" in captured.out
        db_instance.close.assert_called_once()

    @patch('ingestion.main.os.getenv')
    @patch('ingestion.main.Embedder')
    @patch('ingestion.main.MediumScraper')
    @patch('ingestion.main.VectorDatabase')
    def test_run_skips_embedding_when_no_documents(self, mock_vector_db, mock_medium_scraper, mock_embedder, mock_getenv):
        """Test pipeline skips embedding when scraper returns no documents."""
        db_instance = Mock()
        db_instance.get_last_scraped_date.return_value = None
        mock_vector_db.return_value = db_instance

        medium_instance = Mock()
        medium_instance.scrape.return_value = []
        mock_medium_scraper.return_value = medium_instance

        mock_getenv.side_effect = _create_mock_env(medium_username='testuser')

        with IngestionPipeline() as pipeline:
            pipeline.run()

        # Embedder should not be instantiated when there are no documents
        mock_embedder.assert_not_called()
        db_instance.close.assert_called_once()

    @patch('ingestion.main.os.getenv')
    @patch('ingestion.main.Embedder')
    @patch('ingestion.main.MediumScraper')
    @patch('ingestion.main.VectorDatabase')
    def test_run_uses_last_scraped_date(self, mock_vector_db, mock_medium_scraper, mock_embedder, mock_getenv):
        """Test pipeline passes last scraped date to scraper."""
        db_instance = Mock()
        last_scraped_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        db_instance.get_last_scraped_date.return_value = last_scraped_date
        mock_vector_db.return_value = db_instance

        medium_instance = Mock()
        medium_instance.scrape.return_value = [{'content': 'new post'}]
        mock_medium_scraper.return_value = medium_instance

        embedder_instance = Mock()
        embedder_instance.embedding_dim = 384
        mock_embedder.return_value = embedder_instance

        mock_getenv.side_effect = _create_mock_env(medium_username='testuser')

        with IngestionPipeline() as pipeline:
            pipeline.run()

        # Verify scraper was called with the last scraped date
        medium_instance.scrape.assert_called_once_with(last_scraped_date=last_scraped_date)
        embedder_instance.store.assert_called_once()
