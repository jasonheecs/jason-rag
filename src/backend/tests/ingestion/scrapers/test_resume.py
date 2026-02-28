"""Tests for the Resume PDF scraper."""
import hashlib
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from ingestion.scrapers.resume import ResumeScraper


FAKE_URL = "https://example.com/resume.pdf"
FAKE_PDF_BYTES = b"%PDF-1.4 fake pdf content"
FAKE_HASH = hashlib.sha256(FAKE_PDF_BYTES).hexdigest()


class TestResumeScraper:
    """Test suite for ResumeScraper class."""

    @pytest.fixture
    def scraper(self):
        return ResumeScraper(url=FAKE_URL)

    # --- Initialisation ---

    def test_init_with_explicit_url(self):
        scraper = ResumeScraper(url=FAKE_URL)
        assert scraper.url == FAKE_URL
        assert scraper.SOURCE_NAME == "resume"

    def test_init_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("RESUME_URL", FAKE_URL)
        scraper = ResumeScraper()
        assert scraper.url == FAKE_URL

    def test_init_raises_without_url(self, monkeypatch):
        monkeypatch.delenv("RESUME_URL", raising=False)
        with pytest.raises(ValueError, match="RESUME_URL"):
            ResumeScraper()

    # --- scrape() ---

    @patch("ingestion.scrapers.resume.ResumeScraper._fetch_and_extract", return_value=None)
    def test_scrape_returns_empty_when_fetch_fails(self, _mock, scraper):
        assert scraper.scrape() == []

    @patch("ingestion.scrapers.resume.ResumeScraper._fetch_and_extract")
    def test_scrape_returns_document(self, mock_fetch, scraper):
        fake_doc = {"title": "resume", "content": "text", "content_hash": FAKE_HASH}
        mock_fetch.return_value = fake_doc
        docs = scraper.scrape()
        assert docs == [fake_doc]

    @patch("ingestion.scrapers.resume.ResumeScraper._fetch_and_extract")
    def test_scrape_ignores_last_scraped_date(self, mock_fetch, scraper):
        """last_scraped_date param is accepted but unused (hash-based dedup)."""
        mock_fetch.return_value = {"title": "r", "content": "c", "content_hash": FAKE_HASH}
        docs = scraper.scrape(last_scraped_date=datetime(2020, 1, 1))
        assert len(docs) == 1

    # --- _download_pdf() ---

    @patch("ingestion.scrapers.resume.requests.get")
    def test_download_pdf_plain_url(self, mock_get, scraper):
        mock_response = MagicMock()
        mock_response.content = FAKE_PDF_BYTES
        mock_get.return_value = mock_response
        pdf_bytes, response = scraper._download_pdf()
        assert pdf_bytes == FAKE_PDF_BYTES
        assert response is mock_response

    @patch("ingestion.scrapers.resume.ResumeScraper._download_from_google_drive")
    def test_download_pdf_routes_google_drive(self, mock_gdrive, scraper):
        scraper.url = "https://drive.google.com/file/d/abc/view"
        mock_gdrive.return_value = FAKE_PDF_BYTES
        pdf_bytes, response = scraper._download_pdf()
        assert pdf_bytes == FAKE_PDF_BYTES
        assert response is None
        mock_gdrive.assert_called_once()

    # --- _fetch_and_extract() ---

    @patch("ingestion.scrapers.resume.pdfplumber.open")
    @patch("ingestion.scrapers.resume.ResumeScraper._download_pdf")
    def test_fetch_and_extract_returns_doc_with_hash(self, mock_dl, mock_pdf_open, scraper):
        mock_dl.return_value = (FAKE_PDF_BYTES, None)
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Jason Hee — Software Engineer"
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf

        doc = scraper._fetch_and_extract()

        assert doc is not None
        assert doc["content_hash"] == FAKE_HASH
        assert "Jason Hee" in doc["content"]
        assert doc["source"] == "resume"

    @patch("ingestion.scrapers.resume.pdfplumber.open")
    @patch("ingestion.scrapers.resume.ResumeScraper._download_pdf")
    def test_fetch_and_extract_returns_none_on_empty_text(self, mock_dl, mock_pdf_open, scraper):
        mock_dl.return_value = (FAKE_PDF_BYTES, None)
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf

        assert scraper._fetch_and_extract() is None

    @patch("ingestion.scrapers.resume.ResumeScraper._download_pdf", side_effect=Exception("network error"))
    def test_fetch_and_extract_returns_none_on_error(self, _mock, scraper, capsys):
        assert scraper._fetch_and_extract() is None
        assert "Failed to fetch resume" in capsys.readouterr().out

    # --- document_is_new() ---

    def test_document_is_new_when_no_stored_hash(self):
        doc = {"content_hash": FAKE_HASH}
        assert ResumeScraper.document_is_new(doc, None) is True

    def test_document_is_new_when_hash_differs(self):
        doc = {"content_hash": FAKE_HASH}
        assert ResumeScraper.document_is_new(doc, "oldhash") is True

    def test_document_is_not_new_when_hash_matches(self):
        doc = {"content_hash": FAKE_HASH}
        assert ResumeScraper.document_is_new(doc, FAKE_HASH) is False

    # --- helpers ---

    def test_is_google_drive_url(self):
        assert ResumeScraper._is_google_drive_url("https://drive.google.com/file/d/abc/view")
        assert not ResumeScraper._is_google_drive_url("https://example.com/resume.pdf")

    def test_parse_filename_from_url(self, scraper):
        assert scraper._parse_filename("https://example.com/my-resume.pdf") == "my-resume"
        assert scraper._parse_filename("https://example.com/resume") == "resume"

    def test_parse_last_modified_from_header(self, scraper):
        mock_response = MagicMock()
        mock_response.headers = {"Last-Modified": "Wed, 01 Jan 2025 00:00:00 GMT"}
        dt = scraper._parse_last_modified(mock_response)
        assert dt.year == 2025

    def test_parse_last_modified_falls_back_to_now(self, scraper):
        dt = scraper._parse_last_modified(None)
        assert dt.tzinfo == timezone.utc

    def test_create_document_structure(self, scraper):
        doc = scraper._create_document(
            title="Test Resume",
            content="Test content",
            url=FAKE_URL,
            published_date=datetime.now(),
            metadata={"pages": 2},
        )
        assert doc["source"] == "resume"
        for key in ("title", "content", "url", "published_date", "metadata"):
            assert key in doc
