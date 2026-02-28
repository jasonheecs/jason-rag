"""Database module for vector database operations."""
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, PayloadSchemaType, OrderBy

from config.config import QDRANT_COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT
from config.db_helper import check_payload_index_exists


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
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
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
        points = [self._create_point_from_document(doc) for doc in documents]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Inserted {len(documents)} document chunks into Qdrant")

    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search for similar documents using cosine similarity."""
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=top_k
        )

        return [self._format_search_result(hit) for hit in search_result.points]

    def get_last_scraped_date(self, source: str) -> Optional[datetime]:
        """Get the most recent published_date for a given source."""
        if not self.client or not self._collection_exists() or self._collection_is_empty():
            return None

        if not check_payload_index_exists(self.client, self.collection_name, "published_date"):
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="published_date",
                field_schema=PayloadSchemaType.DATETIME,
            )

        return self._query_latest_document(source)

    def close(self):
        """Close connection to Qdrant."""
        if self.client:
            self.client.close()
            print("Qdrant connection closed")

    def get_content_hash(self, source: str) -> Optional[str]:
        """Get the stored content hash for a given source."""
        if not self.client or not self._collection_exists() or self._collection_is_empty():
            return None

        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter={
                "must": [
                    {"key": "source", "match": {"value": source}}
                ]
            },
            limit=1,
            with_payload=["content_hash"]
        )
        points, _ = results
        if points:
            return points[0].payload.get('content_hash')
        return None

    def _create_point_from_document(self, doc: Dict) -> PointStruct:
        """Convert document to Qdrant point."""
        payload = {
            'title': doc['title'],
            'content': doc['content'],
            'source': doc['source'],
            'url': doc['url'],
            'published_date': doc['published_date'].isoformat(),
            'chunk_index': doc['chunk_index'],
        }
        if 'content_hash' in doc:
            payload['content_hash'] = doc['content_hash']
        return PointStruct(
            id=str(uuid4()),
            vector=doc['embedding'].tolist(),
            payload=payload
        )

    def _format_search_result(self, hit) -> Dict:
        """Format search result hit into dictionary."""
        return {
            'id': hit.id,
            'title': hit.payload['title'],
            'content': hit.payload['content'],
            'source': hit.payload['source'],
            'url': hit.payload['url'],
            'published_date': hit.payload['published_date'],
            'similarity': hit.score
        }

    def _collection_exists(self) -> bool:
        """Check if the collection exists in Qdrant."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        return self.collection_name in collection_names

    def _collection_is_empty(self) -> bool:
        """Check if the collection has any documents."""
        count = self.client.count(collection_name=self.collection_name)
        return count.count == 0

    def _query_latest_document(self, source: str) -> Optional[datetime]:
        """Query for the most recent document from a given source."""
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter={
                "must": [
                    {"key": "source", "match": {"value": source}}
                ]
            },
            limit=1,
            order_by=OrderBy(
                key="published_date",
                direction="desc"
            ),
            with_payload=["published_date"]
        )

        points, _ = results
        if points:
            return datetime.fromisoformat(points[0].payload['published_date'])

        return None
