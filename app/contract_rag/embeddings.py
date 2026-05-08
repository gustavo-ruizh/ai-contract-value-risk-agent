from typing import List
from openai import OpenAI
from app.contract_rag.schemas import DocumentChunk, EmbeddedChunk


class EmbeddingModel:
    """Generates vector embeddings for document chunks."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None) -> None:
        self.model_name = model
        self.client = OpenAI(api_key = api_key)

    def embed(self, chunk: DocumentChunk) -> EmbeddedChunk:
        """Embed a single chunk, return EmbeddedChunk."""
        response = self.client.embeddings.create(input=chunk.text, model=self.model_name)
        vector = response.data[0].embedding #list[float]

        return EmbeddedChunk(
            chunk = chunk,
            embedding = vector
        )

    def embed_batch(self, chunks: List[DocumentChunk]) -> List[EmbeddedChunk]:
        """Embed multiple chunks in one API call."""
        texts = [chunk.text for chunk in chunks]
        response = self.client.embeddings.create(input=texts, model=self.model_name)

        return [
            EmbeddedChunk(chunk=chunk, embedding=item.embedding)
            for chunk, item in zip(chunks, response.data)
        ]

    def embed_query(self, query_text: str) -> List[float]:
        """Embed a raw query string (no chunk wrapping needed)."""
        response = self.client.embeddings.create(input=query_text, model=self.model_name)
        vector = response.data[0].embedding #list[float]

        return vector

