from typing import Optional, List
from pydantic import BaseModel


class DocumentChunk(BaseModel):
    chunk_id: str
    contract_id: str
    text: str
    chunk_index: int
    source: str        # original filename — required for citation
    page: int          # 1-based page number — required for citation
    extra_metadata: Optional[dict] = None


class EmbeddedChunk(BaseModel):
    chunk: DocumentChunk
    embedding: List[float]


class RetrievalResult(BaseModel):
    chunk: DocumentChunk
    score: float


class RetrievalQuery(BaseModel):
    query_text: str
    contract_id: Optional[str] = None
    top_k: int = 5
