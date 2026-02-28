"""
Main ingestion pipeline script.
Scrapes content, chunks it, embeds it, and stores in Qdrant.
"""
import os
from ingestion.scrapers.medium import MediumScraper
from ingestion.scrapers.github import GitHubScraper
from ingestion.scrapers.resume import ResumeScraper
from ingestion.chunker import TextChunker
from ingestion.embedder import Embedder
from config.database import VectorDatabase
from config.config import EMBEDDING_MODEL


class IngestionPipeline:
    """Orchestrates the full ingestion pipeline for scraping, chunking, and embedding content."""

    def __init__(self, embedding_model=EMBEDDING_MODEL):
        """
        Initialize the ingestion pipeline.

        Args:
            embedding_model: Name of the embedding model to use
        """
        self.embedding_model = embedding_model
        self.vector_db = None
        self.chunker = TextChunker()
        self.embedder = None

    def __enter__(self):
        """Context manager entry: connect to database."""
        self._connect_to_database()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: close database connection."""
        self._close_database()
        return False

    def _connect_to_database(self):
        """Connect to Qdrant vector database."""
        print("\n[1/5] Connecting to Qdrant...")
        self.vector_db = VectorDatabase()
        self.vector_db.connect()

    def _scrape_content(self):
        """Scrape content from sources."""
        print("\n[2/5] Scraping content...")
        documents = []

        if os.getenv("MEDIUM_USERNAME"):
            documents.extend(self._scrape_source(MediumScraper, "medium"))

        if os.getenv("GITHUB_USERNAME"):
            documents.extend(self._scrape_source(GitHubScraper, "github"))

        if os.getenv("RESUME_URL"):
            resume_docs = self._scrape_source(ResumeScraper, "resume")
            stored_hash = self.vector_db.get_content_hash("resume")
            if self._should_scrape_resume(resume_docs, stored_hash):
                documents.extend(resume_docs)
            elif resume_docs:
                print("Resume unchanged (hash match), skipping ingestion")

        print(f"Scraped {len(documents)} documents")
        return documents
    
    def _should_scrape_resume(self, resume_docs, stored_hash) -> bool:
        """Determine if the resume should be scraped based on content hash."""
        if resume_docs and ResumeScraper.document_is_new(resume_docs[0], stored_hash):
            return True
        
        return False

    def _scrape_source(self, scraper_class, source_name):
        """Scrape content from a source using the provided scraper class.
        
        Args:
            scraper_class: The scraper class to instantiate
            source_name: The source name for tracking last scraped date
            
        Returns:
            List of documents from the scraper
        """
        last_scraped = self.vector_db.get_last_scraped_date(source_name)
        if last_scraped:
            print(f"Last {source_name.title()} scrape: {last_scraped}")
        scraper = scraper_class()
        return scraper.scrape(last_scraped_date=last_scraped)

    def _chunk_documents(self, documents):
        """Chunk documents into smaller pieces."""
        print("\n[3/5] Chunking documents...")
        chunked_docs = self.chunker.chunk_documents(documents)
        print(f"✓ Created {len(chunked_docs)} chunks")
        return chunked_docs

    def _embed_and_store(self, chunked_docs):
        """Generate embeddings and store in database."""
        print("\n[4/5] Generating embeddings and storing...")
        self.embedder = Embedder(model_name=self.embedding_model)
        self.vector_db.setup_database(embedding_dim=self.embedder.embedding_dim)
        self.embedder.store(chunked_docs, self.vector_db)

    def _close_database(self):
        """Close database connection."""
        if self.vector_db:
            print("\n[5/5] Closing connection...")
            self.vector_db.close()

    def run(self):
        """Execute the full ingestion pipeline."""
        print("=" * 60)
        print("Starting Ingestion Pipeline")
        print("=" * 60)

        documents = self._scrape_content()

        if not documents:
            print("No documents found. Please check your configuration.")
            return

        chunked_docs = self._chunk_documents(documents)
        self._embed_and_store(chunked_docs)

        print("\n" + "=" * 60)
        print("✓ Ingestion Pipeline Complete!")
        print("=" * 60)


if __name__ == "__main__":
    with IngestionPipeline() as pipeline:
        pipeline.run()
