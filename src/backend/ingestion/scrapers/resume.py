"""Resume PDF scraper for fetching resume PDFs from a local directory."""
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pdfplumber
from ingestion.scrapers.base import BaseScraper


class ResumeScraper(BaseScraper):
    """Scrapes resume PDFs from a local directory."""

    SOURCE_NAME = "resume"
    DEFAULT_RESUME_DIR = "src/files"

    def __init__(self, resume_dir: Optional[str] = None):
        """
        Initialize Resume scraper.

        Args:
            resume_dir: Path to directory containing resume PDFs
                        (defaults to DEFAULT_RESUME_DIR)
        """
        self.resume_dir = resume_dir or self.DEFAULT_RESUME_DIR
        if not os.path.exists(self.resume_dir):
            raise ValueError(f"Resume directory not found: {self.resume_dir}")
        super().__init__("user")

    def scrape(self, last_scraped_date: Optional[datetime] = None) -> List[Dict]:
        """
        Find and parse all PDF files in the resume directory.

        Args:
            last_scraped_date: If provided, only return documents modified after this date

        Returns:
            List of resume documents extracted from PDFs
        """
        pdf_files = list(Path(self.resume_dir).glob("*.pdf"))
        if not pdf_files:
            print("No PDF files found in resume directory")
            return []

        documents = []
        for pdf_path in pdf_files:
            doc = self._extract_from_pdf(pdf_path)
            if doc is None:
                continue
            if last_scraped_date is not None:
                pub_date = doc["published_date"]
                if pub_date.tzinfo is None:
                    pub_date = pub_date.replace(tzinfo=timezone.utc)
                if last_scraped_date.tzinfo is None:
                    last_scraped_date = last_scraped_date.replace(tzinfo=timezone.utc)
                if pub_date <= last_scraped_date:
                    continue
            documents.append(doc)

        print(f"Scraped {len(documents)} resume document(s)")
        return documents

    def _extract_from_pdf(self, pdf_path: Path) -> Optional[Dict]:
        """Extract text and metadata from a single PDF file."""
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                mtime = os.path.getmtime(str(pdf_path))
                published_date = datetime.fromtimestamp(mtime, tz=timezone.utc)
                text_content = self._extract_text_from_pdf(pdf)
                if not text_content.strip():
                    print(f"No text extracted from {pdf_path.name}")
                    return None

                return self._create_document(
                    title=pdf_path.stem,
                    content=text_content.strip(),
                    url=f"file://{pdf_path.resolve()}",
                    published_date=published_date,
                    metadata={
                        "filename": pdf_path.name,
                        "pages": len(pdf.pages),
                    },
                )
        except Exception as e:
            print(f"Failed to extract text from {pdf_path.name}: {str(e)}")
            return None

    def _extract_text_from_pdf(self, pdf) -> str:
        """Extract text from all pages of a PDF."""
        return "".join(page.extract_text() or "" for page in pdf.pages)
