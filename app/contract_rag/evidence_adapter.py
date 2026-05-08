"""Converts RAG RetrievalResult objects into the canonical Evidence schema.

Keeps RAG internals (DocumentChunk, RetrievalResult) isolated from the main
pipeline schemas. Agents and the orchestrator import Evidence from
app.schemas.common, not from this module.

Metadata preservation:
  - text          → chunk text (the quoted contract passage)
  - source        → original filename, required for citation
  - page          → 1-based page number, required for citation
  - chunk_id      → unique chunk identifier, supports traceability
  - relevance_score → cosine similarity clamped to [0.0, 1.0]

Limitation: cosine similarity can be negative for dissimilar vectors. Negative
scores are clamped to 0.0 rather than propagated, since a score < 0 means the
chunk has no relevance to the query.
"""
from typing import List

from app.contract_rag.schemas import RetrievalResult
from app.schemas.common import Evidence


def retrieval_results_to_evidence(results: List[RetrievalResult]) -> List[Evidence]:
    """Convert a list of RAG retrieval results to canonical Evidence objects."""
    evidence_list = []
    for result in results:
        score = max(0.0, min(1.0, result.score))
        evidence_list.append(Evidence(
            text=result.chunk.text,
            source=result.chunk.source,
            page=result.chunk.page,
            chunk_id=result.chunk.chunk_id,
            relevance_score=score,
        ))
    return evidence_list
