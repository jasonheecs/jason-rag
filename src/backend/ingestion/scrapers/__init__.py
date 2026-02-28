"""Scrapers module for collecting content from various sources."""

from ingestion.scrapers.base import BaseScraper
from ingestion.scrapers.github import GitHubScraper
from ingestion.scrapers.medium import MediumScraper
from ingestion.scrapers.resume import ResumeScraper

__all__ = [
    "BaseScraper",
    "GitHubScraper",
    "MediumScraper",
    "ResumeScraper",
]
