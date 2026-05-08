import numpy as np
from typing import List
from app.contract_rag.schemas import EmbeddedChunk, RetrievalResult


class VectorStore:
    """Local in-memory or on-disk vector store for embedded chunks."""
    
    def __init__(self) -> None:
        self._store: list[EmbeddedChunk] = []
    
    def add(self, embedded_chunks: List[EmbeddedChunk]) -> None:
        """Store new embedded chunks."""
        self._store.extend(embedded_chunks)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[RetrievalResult]:
        """Score every chunk, sort, and return top k."""
        results = []

        for item in self._store:
            results.append(RetrievalResult(
                chunk = item.chunk,
                score = self._cosine_similarity(a = query_embedding, 
                                                b = item.embedding)
            )
        )

        return sorted(results, key = lambda e: e.score, reverse = True)[:top_k]

    def clear(self) -> None:
        """Reset the vector store."""
        self._store = []

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculates and returns cosine similarity between two vectors."""
        va, vb = np.array(a), np.array(b)
        
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
