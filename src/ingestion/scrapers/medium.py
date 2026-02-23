import feedparser
from bs4 import BeautifulSoup
from typing import List, Dict
from datetime import datetime


class MediumScraper:
    """Scrapes Medium posts via RSS feed."""

    def __init__(self, username: str):
        self.username = username
        self.rss_url = f"https://medium.com/feed/@{username}"

    def scrape(self) -> List[Dict]:
        """Fetch and parse Medium posts."""
        feed = feedparser.parse(self.rss_url)
        posts = []

        for entry in feed.entries:
            # Clean HTML content
            soup = BeautifulSoup(entry.description, 'html.parser')
            content = soup.get_text(separator=' ', strip=True)

            posts.append({
                'title': entry.title,
                'content': content,
                'url': entry.link,
                'published_date': datetime(*entry.published_parsed[:6]),
                'source': 'medium'
            })

        print(f"Scraped {len(posts)} Medium posts")
        return posts
