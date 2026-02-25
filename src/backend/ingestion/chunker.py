from typing import List


class TextChunker:
    """Splits text into overlapping chunks for better retrieval."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk:
                chunks.append(chunk)

        return chunks

    def chunk_documents(self, documents: List[dict]) -> List[dict]:
        """
        Chunk multiple documents and preserve metadata.

        Args:
            documents: List of dicts with 'content' and other metadata

        Returns:
            List of dicts with chunked content and preserved metadata
        """
        chunked_docs = []

        for doc in documents:
            chunks = self.chunk_text(doc['content'])

            for i, chunk in enumerate(chunks):
                chunked_doc = doc.copy()
                chunked_doc['content'] = chunk
                chunked_doc['chunk_index'] = i
                chunked_docs.append(chunked_doc)

        print(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        return chunked_docs
