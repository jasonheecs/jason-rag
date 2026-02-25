"""
Main ingestion pipeline script.
Scrapes content, chunks it, embeds it, and stores in Qdrant.
"""
from src.ingestion.scrapers.medium import MediumScraper
from src.ingestion.scrapers.linkedin import LinkedInScraper
from src.ingestion.chunker import TextChunker
from src.ingestion.embedder import Embedder
from src.config.database import VectorDatabase
from src.config.config import MEDIUM_USERNAME, LINKEDIN_PROFILE_URL, EMBEDDING_MODEL


def run_ingestion():
    """Execute the full ingestion pipeline."""

    print("=" * 60)
    print("Starting Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Scrape content
    print("\n[1/4] Scraping content...")
    documents = []

    if MEDIUM_USERNAME:
        medium_scraper = MediumScraper(MEDIUM_USERNAME)
        medium_posts = medium_scraper.scrape()
        documents.extend(medium_posts)

    if LINKEDIN_PROFILE_URL:
        linkedin_scraper = LinkedInScraper(LINKEDIN_PROFILE_URL)
        linkedin_data = linkedin_scraper.scrape()
        documents.extend(linkedin_data)

    if not documents:
        print("No documents found. Please check your configuration.")
        return

    print(f"Scraped {len(documents)} documents")

    # Step 2: Chunk documents
    print("\n[2/4] Chunking documents...")
    chunker = TextChunker(chunk_size=500, overlap=50)
    chunked_docs = chunker.chunk_documents(documents)
    print(f"✓ Created {len(chunked_docs)} chunks")

    # Step 3: Connect to database
    print("\n[3/4] Connecting to Qdrant...")
    vector_db = VectorDatabase()
    vector_db.connect()

    # Step 4: Embed and store
    print("\n[4/4] Generating embeddings and storing...")
    embedder = Embedder(model_name=EMBEDDING_MODEL)
    vector_db.setup_database(embedding_dim=embedder.embedding_dim)
    embedder.embed_and_store(chunked_docs, vector_db)

    vector_db.close()

    print("\n" + "=" * 60)
    print("✓ Ingestion Pipeline Complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_ingestion()
