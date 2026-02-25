from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from src.config.database import VectorDatabase


class Embedder:
    """Generates embeddings and stores them in Qdrant."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    def embed_and_store(self, documents: List[dict], vector_db: VectorDatabase):
        """
        Generate embeddings for documents and store in vector database.

        Args:
            documents: List of document dicts with 'content' field
            vector_db: Connected VectorDatabase instance
        """
        print(f"Generating embeddings for {len(documents)} chunks...")

        # Extract text content
        texts = [doc['content'] for doc in documents]

        # Generate embeddings
        embeddings = self.embed_batch(texts)

        # Add embeddings to documents
        for doc, embedding in zip(documents, embeddings):
            doc['embedding'] = embedding

        # Store in vector database
        print("Storing embeddings in Qdrant...")
        vector_db.insert_documents(documents)

        print(f"✓ Successfully embedded and stored {len(documents)} chunks")
