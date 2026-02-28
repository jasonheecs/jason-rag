"""Embedding module for generating and managing text embeddings."""
from typing import List

import numpy as np
from fastembed import TextEmbedding
from config.database import VectorDatabase


class Embedder:
    """Generates embeddings and stores them in Qdrant."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)
        # FastEmbed models have standard dimensions - bge-small is 384
        self.embedding_dim = 384

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        embedding = list(self.model.embed([text]))[0]
        return np.array(embedding)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        embeddings = list(self.model.embed(texts))
        return np.array(embeddings)

    def _add_embeddings_to_documents(self, documents: List[dict], embeddings: np.ndarray) -> None:
        """Add embeddings to document dictionaries in-place."""
        for doc, embedding in zip(documents, embeddings):
            doc['embedding'] = embedding

    def embed_documents(self, documents: List[dict]) -> None:
        """
        Generate embeddings for documents and add them in-place.

        Args:
            documents: List of document dicts with 'content' field
        """
        print(f"Generating embeddings for {len(documents)} chunks...")
        embeddings = self.embed_batch([doc['content'] for doc in documents])
        self._add_embeddings_to_documents(documents, embeddings)

    def store(self, documents: List[dict], vector_db: VectorDatabase):
        """
        Generate embeddings for documents and store in vector database.

        Args:
            documents: List of document dicts with 'content' field
            vector_db: Connected VectorDatabase instance
        """
        self.embed_documents(documents)

        print("Storing embeddings in Qdrant...")
        vector_db.insert_documents(documents)
        print(f"Successfully embedded and stored {len(documents)} chunks")
