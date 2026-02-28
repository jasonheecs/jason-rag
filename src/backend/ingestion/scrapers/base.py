"""Base scraper class with common functionality for all scrapers."""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from datetime import datetime


class BaseScraper(ABC):
    """Abstract base class for all content scrapers."""

    SOURCE_NAME: str = None  # Override in child classes

    def __init__(self, username: str):
        """
        Initialize the base scraper.

        Args:
            username: Username for the service being scraped
        """
        self.username = username

    @abstractmethod
    def scrape(self, last_scraped_date: Optional[datetime] = None) -> List[Dict]:
        """
        Scrape content from the source.

        Args:
            last_scraped_date: Only return documents newer than this date

        Returns:
            List of documents with title, content, url, published_date, source, and metadata
        """
        pass

    def _filter_by_date(
        self, documents: List[Dict], last_scraped_date: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Filter documents to only include those newer than last_scraped_date.

        Args:
            documents: List of documents to filter
            last_scraped_date: Cutoff date (exclusive)

        Returns:
            Filtered list of documents
        """
        if last_scraped_date is None:
            return documents

        return [
            doc for doc in documents
            if doc['published_date'] > last_scraped_date
        ]

    def _create_document(
        self,
        title: str,
        content: str,
        url: str,
        published_date: datetime,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Create a standardized document dictionary.

        Args:
            title: Document title
            content: Document content
            url: Document URL
            published_date: Publication date
            metadata: Optional metadata dictionary

        Returns:
            Standardized document dictionary
        """
        return {
            'title': title,
            'content': content,
            'url': url,
            'published_date': published_date,
            'source': self.SOURCE_NAME,
            'metadata': metadata or {},
        }
