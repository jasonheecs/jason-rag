"""Text chunking module for splitting documents into overlapping chunks."""
import copy
from typing import Dict, List


class TextChunker:
    """Splits text into overlapping chunks for better retrieval."""

    def __init__(self, chunk_size: int = 256, overlap: int = 25):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        step_size = self.chunk_size - self.overlap

        i = 0
        while i < len(words):
            chunk_words = words[i:i + self.chunk_size]
            chunk = ' '.join(chunk_words)
            if chunk:
                chunks.append(chunk)

            # Break if we've consumed all words
            if i + self.chunk_size >= len(words):
                break

            i += step_size

        return chunks

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Chunk multiple documents and preserve metadata.

        Args:
            documents: List of dicts with 'content' and other metadata

        Returns:
            List of dicts with chunked content and preserved metadata
        """
        chunked_docs = []

        for doc in documents:
            text_chunks = self.chunk_text(doc['content'])

            for chunk_index, chunk_content in enumerate(text_chunks):
                chunked_doc = self._create_chunked_document(doc, chunk_content, chunk_index)
                chunked_docs.append(chunked_doc)

        return chunked_docs

    def _create_chunked_document(
        self,
        original_doc: Dict,
        chunk_content: str,
        chunk_index: int,
    ) -> Dict:
        """Create a new document with chunked content and metadata."""
        chunked_doc = copy.deepcopy(original_doc)
        chunked_doc['content'] = chunk_content
        chunked_doc['chunk_index'] = chunk_index
        return chunked_doc
