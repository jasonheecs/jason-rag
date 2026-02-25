from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict
import numpy as np
from uuid import uuid4
from src.config.config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME


class VectorDatabase:
    """Manages Qdrant vector database."""

    def __init__(self, host: str = QDRANT_HOST, port: int = QDRANT_PORT):
        self.host = host
        self.port = port
        self.collection_name = QDRANT_COLLECTION_NAME
        self.client = None

    def connect(self):
        """Establish connection to Qdrant."""
        self.client = QdrantClient(host=self.host, port=self.port)
        print(f"Connected to Qdrant at {self.host}:{self.port}")
        return self.client

    def setup_database(self, embedding_dim: int = 384):
        """Initialize Qdrant collection."""
        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection: {self.collection_name}")
        else:
            print(f"Collection {self.collection_name} already exists")

    def insert_documents(self, documents: List[Dict]):
        """Insert documents with embeddings into Qdrant."""
        points = []

        for doc in documents:
            point_id = str(uuid4())

            # Prepare payload (metadata)
            payload = {
                'title': doc['title'],
                'content': doc['content'],
                'source': doc['source'],
                'url': doc['url'],
                'published_date': doc['published_date'].isoformat(),
                'chunk_index': doc['chunk_index']
            }

            # Create point
            point = PointStruct(
                id=point_id,
                vector=doc['embedding'].tolist(),
                payload=payload
            )
            points.append(point)

        # Batch upload
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        print(f"Inserted {len(documents)} document chunks into Qdrant")

    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search for similar documents using cosine similarity."""
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )

        results = []
        for hit in search_result:
            results.append({
                'id': hit.id,
                'title': hit.payload['title'],
                'content': hit.payload['content'],
                'source': hit.payload['source'],
                'url': hit.payload['url'],
                'published_date': hit.payload['published_date'],
                'similarity': hit.score
            })

        return results

    def close(self):
        """Close connection to Qdrant."""
        # Qdrant client doesn't require explicit closing
        print("Qdrant connection closed")
