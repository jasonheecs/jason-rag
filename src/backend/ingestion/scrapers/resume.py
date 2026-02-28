"""Resume PDF scraper for fetching a resume PDF from a remote URL."""
import hashlib
import io
import os
import tempfile
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import gdown
import pdfplumber
import requests
from ingestion.scrapers.base import BaseScraper


class ResumeScraper(BaseScraper):
    """Scrapes a resume PDF from a remote URL."""

    SOURCE_NAME = "resume"

    def __init__(self, url: Optional[str] = None):
        """
        Initialize Resume scraper.

        Args:
            url: Direct URL to a PDF resume (defaults to RESUME_URL env var)
        """
        self.url = url or os.getenv("RESUME_URL")
        if not self.url:
            raise ValueError("Resume URL must be provided or set RESUME_URL env var")
        super().__init__("user")

    def scrape(self, last_scraped_date: Optional[datetime] = None) -> List[Dict]:  # noqa: ARG002
        """
        Fetch and parse the remote resume PDF.

        Args:
            last_scraped_date: Unused; resume deduplication is hash-based

        Returns:
            List containing a single resume document (with content_hash), or empty list
        """
        doc = self._fetch_and_extract()
        if not doc:
            return []

        print("Scraped 1 resume document")
        return [doc]

    def _fetch_and_extract(self) -> Optional[Dict]:
        """Download the PDF, compute its hash, and extract text."""
        try:
            pdf_bytes, response = self._download_pdf()
            content_hash = hashlib.sha256(pdf_bytes).hexdigest()
            published_date = self._parse_last_modified(response)

            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                text_content = self._extract_text_from_pdf(pdf)
                if not text_content.strip():
                    print(f"No text extracted from resume at {self.url}")
                    return None

                doc = self._create_document(
                    title=self._parse_filename(self.url),
                    content=text_content.strip(),
                    url=self.url,
                    published_date=published_date,
                    metadata={"pages": len(pdf.pages)},
                )
                doc['content_hash'] = content_hash
                return doc
        except Exception as e:
            print(f"Failed to fetch resume from {self.url}: {str(e)}")
            return None

    def _download_pdf(self) -> Tuple[bytes, Optional[requests.Response]]:
        """Download PDF bytes from the URL, using gdown for Google Drive links."""
        if self._is_google_drive_url(self.url):
            return self._download_from_google_drive(), None

        response = requests.get(self.url, timeout=30)
        response.raise_for_status()
        return response.content, response

    def _download_from_google_drive(self) -> bytes:
        """Download a PDF from a Google Drive sharing link via gdown."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = gdown.download(self.url, tmp_path, quiet=True, fuzzy=True)
            if not result:
                raise ValueError("gdown failed to download the file")
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @staticmethod
    def document_is_new(doc: Dict, stored_hash: Optional[str]) -> bool:
        """Return True if the document content differs from what is stored."""
        return doc.get('content_hash') != stored_hash

    @staticmethod
    def _is_google_drive_url(url: str) -> bool:
        """Return True if the URL points to Google Drive."""
        return "drive.google.com" in url

    def _extract_text_from_pdf(self, pdf) -> str:
        """Extract text from all pages of a PDF."""
        return "".join(page.extract_text() or "" for page in pdf.pages)

    def _parse_last_modified(self, response: Optional[requests.Response]) -> datetime:
        """Return the Last-Modified header date, or now if unavailable."""
        if response:
            last_modified = response.headers.get("Last-Modified")
            if last_modified:
                try:
                    return parsedate_to_datetime(last_modified)
                except Exception:
                    pass
        return datetime.now(tz=timezone.utc)

    def _parse_filename(self, url: str) -> str:
        """Extract a filename stem from the URL path."""
        name = url.rstrip("/").split("/")[-1]
        return name.rsplit(".", 1)[0] if "." in name else name or "resume"
