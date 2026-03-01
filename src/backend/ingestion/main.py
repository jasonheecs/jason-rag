"""
Main ingestion pipeline script.
Scrapes content, chunks it, embeds it, and stores in Qdrant.
"""
import argparse
import os
from ingestion.source_registry import SourceRegistry
from ingestion.chunker import TextChunker
from ingestion.embedder import Embedder
from config.database import VectorDatabase
from config.config import EMBEDDING_MODEL


class IngestionPipeline:
    """Orchestrates the full ingestion pipeline for scraping, chunking, and embedding content."""

    def __init__(self, embedding_model=EMBEDDING_MODEL, sources=None):
        self.embedding_model = embedding_model
        self.sources = set(sources) if sources else set(SourceRegistry.get_sources())
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
        print(f"\n[2/5] Scraping content from: {', '.join(sorted(self.sources))}...")
        documents = []

        for source in SourceRegistry.get_sources():
            if source not in self.sources:
                continue

            env_var = SourceRegistry.get_env_var(source)
            if not os.getenv(env_var):
                print(f"Skipping {source}: {env_var} not set")
                continue
            documents.extend(self._scrape_source(SourceRegistry.get_scraper_class(source), source))

        print(f"Scraped {len(documents)} documents")
        return documents

    def _scrape_source(self, scraper_class, source_name):
        """Scrape content from a source using the provided scraper class."""
        last_scraped = self.vector_db.get_last_scraped_date(source_name)
        if last_scraped:
            print(f"Last {source_name.title()} scrape: {last_scraped}")
        scraper = scraper_class()
        kwargs = {"last_scraped_date": last_scraped}
        if source_name == "resume":
            kwargs["stored_hash"] = self.vector_db.get_content_hash("resume")
        return scraper.scrape(**kwargs)

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

def _parse_args():
    parser = argparse.ArgumentParser(description="Run the ingestion pipeline.")
    parser.add_argument(
        "-s", "--source",
        dest="sources",
        action="append",
        choices=list(SourceRegistry.get_sources()),
        metavar="SOURCE",
        help=f"Source(s) to scrape. Choices: {', '.join(SourceRegistry.get_sources())}. Can be repeated. Defaults to all.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()

    with IngestionPipeline(sources=args.sources) as pipeline:
        pipeline.run()
