import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from ingestion.scrapers.medium import MediumScraper


class TestMediumScraper:
    """Test suite for MediumScraper class."""

    @pytest.fixture
    def scraper(self):
        """Fixture providing a MediumScraper instance."""
        return MediumScraper("testuser")

    @staticmethod
    def _create_mock_entry(title, description, link, published_tuple):
        """Create a mock feed entry with specified attributes."""
        mock_entry = Mock()
        mock_entry.title = title
        mock_entry.description = description
        mock_entry.link = link
        mock_entry.published_parsed = published_tuple
        return mock_entry

    def test_init(self, scraper):
        """Test scraper initialization with username."""
        assert scraper.username == "testuser"
        assert scraper.rss_url == "https://medium.com/feed/@testuser"

    def test_rss_url_template(self):
        """Test RSS URL template constant."""
        assert MediumScraper.RSS_URL_TEMPLATE == "https://medium.com/feed/@{username}"

    def test_extract_text_from_html(self, scraper):
        """Test HTML text extraction."""
        html = "<p>Hello <strong>World</strong></p><div>Test</div>"
        result = scraper._extract_text_from_html(html)
        assert result == "Hello World Test"

    def test_extract_text_from_html_with_whitespace(self, scraper):
        """Test HTML text extraction handles whitespace correctly."""
        html = "<p>  Multiple   spaces  </p>"
        result = scraper._extract_text_from_html(html)
        assert result == "Multiple   spaces"

    def test_extract_text_from_html_empty(self, scraper):
        """Test HTML text extraction with empty input."""
        result = scraper._extract_text_from_html("")
        assert result == ""

    def test_parse_entry(self, scraper):
        """Test parsing a feed entry into post dictionary."""
        mock_entry = self._create_mock_entry(
            "Test Post Title",
            "<p>Test content</p>",
            "https://medium.com/@testuser/test-post",
            (2024, 1, 15, 10, 30, 45, 0, 15, 0)
        )

        result = scraper._parse_entry(mock_entry)

        assert result['title'] == "Test Post Title"
        assert result['content'] == "Test content"
        assert result['url'] == "https://medium.com/@testuser/test-post"
        assert result['published_date'] == datetime(2024, 1, 15, 10, 30, 45)
        assert result['source'] == 'medium'

    @patch('ingestion.scrapers.medium.feedparser.parse')
    @patch('builtins.print')
    def test_scrape_success(self, mock_print, mock_parse, scraper):
        """Test successful scraping of Medium posts."""
        mock_entry1 = self._create_mock_entry(
            "Post 1", "<p>Content 1</p>",
            "https://medium.com/@testuser/post-1",
            (2024, 1, 15, 10, 30, 45, 0, 15, 0)
        )
        mock_entry2 = self._create_mock_entry(
            "Post 2", "<p>Content 2</p>",
            "https://medium.com/@testuser/post-2",
            (2024, 1, 16, 14, 20, 30, 0, 16, 0)
        )

        mock_feed = Mock()
        mock_feed.entries = [mock_entry1, mock_entry2]
        mock_parse.return_value = mock_feed

        result = scraper.scrape()

        assert len(result) == 2
        assert result[0]['title'] == "Post 1"
        assert result[1]['title'] == "Post 2"
        mock_parse.assert_called_once_with("https://medium.com/feed/@testuser")
        mock_print.assert_called_once_with("Scraped 2 new Medium posts")

    @patch('ingestion.scrapers.medium.feedparser.parse')
    @patch('builtins.print')
    def test_scrape_empty_feed(self, mock_print, mock_parse, scraper):
        """Test scraping when feed has no entries."""
        mock_feed = Mock()
        mock_feed.entries = []
        mock_parse.return_value = mock_feed

        result = scraper.scrape()

        assert len(result) == 0
        mock_print.assert_called_once_with("Scraped 0 new Medium posts")

    @patch('ingestion.scrapers.medium.feedparser.parse')
    @patch('builtins.print')
    def test_scrape_with_last_scraped_date(self, mock_print, mock_parse, scraper):
        """Test scraping filters posts older than last_scraped_date."""
        mock_entry1 = self._create_mock_entry(
            "Old Post", "<p>Old content</p>",
            "https://medium.com/@testuser/old-post",
            (2024, 1, 10, 10, 0, 0, 0, 10, 0)
        )
        mock_entry2 = self._create_mock_entry(
            "New Post", "<p>New content</p>",
            "https://medium.com/@testuser/new-post",
            (2024, 1, 20, 10, 0, 0, 0, 20, 0)
        )

        mock_feed = Mock()
        mock_feed.entries = [mock_entry1, mock_entry2]
        mock_parse.return_value = mock_feed

        last_scraped = datetime(2024, 1, 15, 0, 0, 0)
        result = scraper.scrape(last_scraped_date=last_scraped)

        assert len(result) == 1
        assert result[0]['title'] == "New Post"
        mock_print.assert_called_once_with("Scraped 1 new Medium posts")

    def test_parse_entry_with_complex_html(self, scraper):
        """Test parsing entry with nested HTML tags."""
        mock_entry = self._create_mock_entry(
            "Complex Post",
            """
            <div>
                <h1>Header</h1>
                <p>Paragraph with <a href="#">link</a></p>
                <ul><li>Item 1</li><li>Item 2</li></ul>
            </div>
            """,
            "https://medium.com/@testuser/complex",
            (2024, 2, 1, 12, 0, 0, 0, 32, 0)
        )

        result = scraper._parse_entry(mock_entry)

        assert 'Header' in result['content']
        assert 'Paragraph with link' in result['content']
        assert 'Item 1' in result['content']
        assert 'Item 2' in result['content']
