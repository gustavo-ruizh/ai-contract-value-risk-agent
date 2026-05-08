from app.contract_rag.schemas import DocumentChunk, EmbeddedChunk
from app.contract_rag.vector_store import VectorStore


def test_top_k_descending_order():
    query = [1.0, 0.0]
    chunk_a = [1.0, 1.0]    # cosine ≈ 0.71 → second result
    chunk_b = [1.0, 0.0]    # cosine = 1.0  → top result
    chunk_c = [-1.0, 1.0]   # cosine ≈ -0.71 → lowest
    chunk_d = [0.0, 1.0]    # cosine = 0.0  → third result
    chunks = [chunk_a, chunk_b, chunk_c, chunk_d]

    vector_store = VectorStore()

    embedded_chunks: list[EmbeddedChunk] = []
    for i, chunk in enumerate(chunks):
        doc_chunk = DocumentChunk(
            chunk_id=f"test_doc_p1_c{i + 1}",
            contract_id="test_doc",
            text=f"text_{i + 1}",
            chunk_index=i + 1,
            source="test_doc.txt",
            page=1,
            extra_metadata=None,
        )
        embedded_chunks.append(EmbeddedChunk(chunk=doc_chunk, embedding=chunk))

    vector_store.add(embedded_chunks)

    results = vector_store.search(query, top_k=2)
    assert results[0].score == 1.0
    assert len(results) == 2
    for result in results:
        assert result.chunk.source
        assert result.chunk.page == 1
        assert result.chunk.text
        assert result.chunk.chunk_id
