from typing import List
from app.contract_rag.schemas import DocumentChunk
from app.contract_rag.document_loader import PageContent


class ContractChunker:
    """Splits contract text into overlapping chunks suitable for embedding."""

    def __init__(self, chunk_size: int = 200, overlap: int = 40) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_pages(self, pages: list[PageContent], contract_id: str) -> List[DocumentChunk]:
        """Iterates through pages in document to split their text into chunks."""
        chunks = []

        for page in pages:
            page_number = page.page_number
            text = page.text
            source = page.source
            page_chunks = self._chunk_text(text, page_number, source, contract_id)
            chunks.extend(page_chunks)

        return chunks

    def _chunk_text(self, text: str, page_number: int, source: str, contract_id: str) -> List[DocumentChunk]:
        """Splits given page text into chunks."""
        step = self.chunk_size - self.overlap
        words = text.split()
        page_chunks = []
        chunk_n = 1

        for j in range(0, len(words), step):
            chunk = words[j:j+self.chunk_size]
            page_chunks.append(DocumentChunk(
                chunk_id=f"{contract_id}_p{page_number}_c{chunk_n}",
                contract_id=contract_id,
                text=" ".join(chunk),
                chunk_index=chunk_n,
                source=source,
                page=page_number,
                extra_metadata=None,
            ))

            if j + self.chunk_size >= len(words):
                break

            chunk_n += 1

        return page_chunks
