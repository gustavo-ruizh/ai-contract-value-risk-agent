from typing import List
from app.contract_rag.schemas import RetrievalQuery, RetrievalResult
from app.contract_rag.embeddings import EmbeddingModel
from app.contract_rag.vector_store import VectorStore


class ContractRetriever:
    """
    Retrieves relevant contract chunks for a natural-language query.
    Owned entirely by the contract_rag module; agents do not call this directly.
    """

    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Embed the query text and match with embeddings in store."""
        
        query_vector = self.embedding_model.embed_query(query.query_text)
        
        # TODO: filter by query.contract_id when VectorStore supports it
        return self.vector_store.search(query_vector, top_k = query.top_k)

