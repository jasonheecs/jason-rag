"""Tests for the Resume PDF scraper."""
import pytest
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from ingestion.scrapers.resume import ResumeScraper


class TestResumeScraper:
    """Test suite for ResumeScraper class."""

    @pytest.fixture
    def temp_resume_dir(self, tmp_path):
        """Create a temporary directory for test resume files."""
        return str(tmp_path)

    @pytest.fixture
    def scraper(self, temp_resume_dir):
        """Fixture providing a ResumeScraper instance."""
        return ResumeScraper(resume_dir=temp_resume_dir)

    def test_init_with_valid_directory(self, temp_resume_dir):
        """Test scraper initialization with valid directory."""
        scraper = ResumeScraper(resume_dir=temp_resume_dir)
        assert scraper.username == "user"
        assert scraper.resume_dir == temp_resume_dir
        assert scraper.SOURCE_NAME == "resume"

    def test_init_with_invalid_directory(self):
        """Test scraper initialization with non-existent directory."""
        with pytest.raises(ValueError, match="Resume directory not found"):
            ResumeScraper(resume_dir="/nonexistent/path")

    def test_init_with_default_directory(self):
        """Test scraper initialization uses default directory."""
        with patch("os.path.exists", return_value=True):
            scraper = ResumeScraper()
            assert scraper.resume_dir == "src/files"

    def test_scrape_no_pdf_files(self, scraper, capsys):
        """Test scraping when no PDF files are present."""
        documents = scraper.scrape()
        
        assert documents == []
        captured = capsys.readouterr()
        assert "No PDF files found" in captured.out

    @patch("pdfplumber.open")
    def test_scrape_with_pdf_file(self, mock_pdf_open, scraper, temp_resume_dir, capsys):
        """Test scraping with a valid PDF file."""
        # Create a mock PDF fixture
        pdf_name = "resume.pdf"
        pdf_path = Path(temp_resume_dir) / pdf_name
        
        # Create a mock PDF file and pdfplumber response
        with patch("pathlib.Path.glob") as mock_glob:
            mock_glob.return_value = [pdf_path]
            
            # Mock pdfplumber PDF object
            mock_pdf = MagicMock()
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "John Doe\nSoftware Engineer\nExperience: 5 years"
            mock_pdf.pages = [mock_page]
            mock_pdf_open.return_value.__enter__.return_value = mock_pdf
            
            # Mock file modification time
            with patch("os.path.getmtime", return_value=1609459200):  # 2021-01-01
                documents = scraper.scrape()
        
        assert len(documents) == 1
        assert documents[0]["title"] == "resume"
        assert "John Doe" in documents[0]["content"]
        assert documents[0]["source"] == "resume"
        assert documents[0]["metadata"]["filename"] == pdf_name
        assert documents[0]["metadata"]["pages"] == 1

    @patch("pdfplumber.open")
    def test_scrape_with_extraction_error(self, mock_pdf_open, scraper, temp_resume_dir, capsys):
        """Test scraping handles extraction errors gracefully."""
        pdf_path = Path(temp_resume_dir) / "resume.pdf"
        
        with patch("pathlib.Path.glob") as mock_glob:
            mock_glob.return_value = [pdf_path]
            mock_pdf_open.side_effect = Exception("PDF corruption")
            
            documents = scraper.scrape()
        
        assert documents == []
        captured = capsys.readouterr()
        assert "Failed to extract text from" in captured.out
        assert "PDF corruption" in captured.out

    @patch("pdfplumber.open")
    def test_filter_by_date(self, mock_pdf_open, scraper, temp_resume_dir):
        """Test that scraper can filter documents by date."""
        pdf_path = Path(temp_resume_dir) / "resume.pdf"
        
        with patch("pathlib.Path.glob") as mock_glob:
            mock_glob.return_value = [pdf_path]
            
            mock_pdf = MagicMock()
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "Resume content"
            mock_pdf.pages = [mock_page]
            mock_pdf_open.return_value.__enter__.return_value = mock_pdf
            
            # Set document date to 2021-01-01
            with patch("os.path.getmtime", return_value=1609459200):
                # Filter with a date after document creation
                future_date = datetime(2022, 1, 1)
                documents = scraper.scrape(last_scraped_date=future_date)
        
        assert documents == []

    def test_create_document_structure(self, scraper):
        """Test that created documents have correct structure."""
        doc = scraper._create_document(
            title="Test Resume",
            content="Test content",
            url="file:///path/to/resume.pdf",
            published_date=datetime.now(),
            metadata={"pages": 2}
        )
        
        assert "title" in doc
        assert "content" in doc
        assert "url" in doc
        assert "published_date" in doc
        assert "source" in doc
        assert "metadata" in doc
        assert doc["source"] == "resume"
