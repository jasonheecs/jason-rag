"""
Main ingestion pipeline script.
Scrapes content, chunks it, embeds it, and stores in Qdrant.
"""
from ingestion.scrapers.medium import MediumScraper
from ingestion.scrapers.github import GitHubScraper
from ingestion.chunker import TextChunker
from ingestion.embedder import Embedder
from config.database import VectorDatabase
from config.config import MEDIUM_USERNAME, GITHUB_USERNAME, GITHUB_TOKEN, EMBEDDING_MODEL


class IngestionPipeline:
    """Orchestrates the full ingestion pipeline for scraping, chunking, and embedding content."""

    def __init__(self, medium_username=None, github_username=None, github_token=None, embedding_model=EMBEDDING_MODEL):
        """
        Initialize the ingestion pipeline.

        Args:
            medium_username: Medium username to scrape posts from
            github_username: GitHub username to scrape profile and repos from
            github_token: Optional GitHub personal access token for higher rate limits
            embedding_model: Name of the embedding model to use
        """
        self.medium_username = medium_username
        self.github_username = github_username
        self.github_token = github_token
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

        if self.medium_username:
            documents.extend(self._scrape_medium())

        if self.github_username:
            documents.extend(self._scrape_github())

        print(f"Scraped {len(documents)} documents")
        return documents

    def _scrape_medium(self):
        last_scraped = self.vector_db.get_last_scraped_date("medium")
        if last_scraped:
            print(f"Last Medium scrape: {last_scraped}")
        medium_scraper = MediumScraper(self.medium_username)
        return medium_scraper.scrape(last_scraped_date=last_scraped)

    def _scrape_github(self):
        last_scraped = self.vector_db.get_last_scraped_date("github")
        if last_scraped:
            print(f"Last GitHub scrape: {last_scraped}")
        github_scraper = GitHubScraper(self.github_username, token=self.github_token)
        return github_scraper.scrape(last_scraped_date=last_scraped)

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
    with IngestionPipeline(
        medium_username=MEDIUM_USERNAME,
        github_username=GITHUB_USERNAME,
        github_token=GITHUB_TOKEN,
        embedding_model=EMBEDDING_MODEL
    ) as pipeline:
        pipeline.run()
