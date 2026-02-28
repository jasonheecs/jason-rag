"""Query engine module for semantic search over documents."""
from typing import Dict, List

from config.database import VectorDatabase
from ingestion.embedder import Embedder


class QueryEngine:  # pylint: disable=too-few-public-methods
    """Handles query embedding and vector search."""

    def __init__(self, embedder: Embedder, vector_db: VectorDatabase):
        self.embedder = embedder
        self.vector_db = vector_db

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar documents given a query.

        Args:
            query: User's question
            top_k: Number of similar documents to retrieve

        Returns:
            List of similar documents with metadata and similarity scores
        """
        # Embed the query
        query_embedding = self.embedder.embed_text(query)

        # Search vector database
        results = self.vector_db.search_similar(query_embedding, top_k=top_k)

        return results
