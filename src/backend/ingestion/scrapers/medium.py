import os
import feedparser
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from datetime import datetime
from ingestion.scrapers.base import BaseScraper


class MediumScraper(BaseScraper):
    """Scrapes Medium posts via RSS feed."""

    RSS_URL_TEMPLATE = "https://medium.com/feed/@{username}"
    HTML_PARSER = 'html.parser'
    SOURCE_NAME = 'medium'
    TEXT_SEPARATOR = ' '

    def __init__(self, username: Optional[str] = None):
        """
        Initialize Medium scraper.

        Args:
            username: Medium username to scrape (defaults to MEDIUM_USERNAME env var)
        """
        if username is None:
            username = os.getenv("MEDIUM_USERNAME")
        
        if not username:
            raise ValueError("Medium username must be provided or set MEDIUM_USERNAME env var")
        
        super().__init__(username)
        self.rss_url = self.RSS_URL_TEMPLATE.format(username=username)

    def scrape(self, last_scraped_date: Optional[datetime] = None) -> List[Dict]:
        """Fetch and parse Medium posts newer than last_scraped_date."""
        feed = feedparser.parse(self.rss_url)
        posts = self._parse_posts(feed.entries)
        filtered_posts = self._filter_by_date(posts, last_scraped_date)
        print(f"Scraped {len(filtered_posts)} new Medium posts")
        return filtered_posts

    def _parse_posts(self, entries) -> List[Dict]:
        """Parse feed entries into post dictionaries."""
        posts = []
        for entry in entries:
            post = self._parse_entry(entry)
            posts.append(post)
        return posts

    def _parse_entry(self, entry) -> Dict:
        """Transform a feed entry into a post dictionary."""
        return self._create_document(
            title=entry.title,
            content=self._extract_text_from_html(entry.description),
            url=entry.link,
            published_date=datetime(*entry.published_parsed[:6]),
        )

    def _extract_text_from_html(self, html: str) -> str:
        """Extract clean text content from HTML."""
        soup = BeautifulSoup(html, self.HTML_PARSER)
        return soup.get_text(separator=self.TEXT_SEPARATOR, strip=True)
