import feedparser
from bs4 import BeautifulSoup
from typing import List, Dict
from datetime import datetime


class MediumScraper:
    """Scrapes Medium posts via RSS feed."""

    RSS_URL_TEMPLATE = "https://medium.com/feed/@{username}"
    HTML_PARSER = 'html.parser'
    SOURCE_NAME = 'medium'
    TEXT_SEPARATOR = ' '

    def __init__(self, username: str):
        self.username = username
        self.rss_url = self.RSS_URL_TEMPLATE.format(username=username)

    def scrape(self) -> List[Dict]:
        """Fetch and parse Medium posts."""
        feed = feedparser.parse(self.rss_url)
        posts = [self._parse_entry(entry) for entry in feed.entries]
        print(f"Scraped {len(posts)} Medium posts")
        return posts

    def _parse_entry(self, entry) -> Dict:
        """Transform a feed entry into a post dictionary."""
        return {
            'title': entry.title,
            'content': self._extract_text_from_html(entry.description),
            'url': entry.link,
            'published_date': datetime(*entry.published_parsed[:6]),
            'source': self.SOURCE_NAME
        }

    def _extract_text_from_html(self, html: str) -> str:
        """Extract clean text content from HTML."""
        soup = BeautifulSoup(html, self.HTML_PARSER)
        return soup.get_text(separator=self.TEXT_SEPARATOR, strip=True)
