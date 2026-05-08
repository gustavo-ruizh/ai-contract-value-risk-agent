from app.contract_rag.document_loader import PageContent
from app.contract_rag.chunker import ContractChunker


def test_chunk_count():
    page = PageContent(page_number=1, text="word " * 300, source="contract.txt")
    chunker = ContractChunker()
    chunks = chunker.chunk_pages([page], contract_id="doc1")

    # 300 words, step=160: positions 0 and 160 → 2 chunks
    assert len(chunks) == 2


def test_chunk_metadata_preserved():
    page = PageContent(page_number=1, text="word " * 300, source="contract.txt")
    chunker = ContractChunker()
    chunks = chunker.chunk_pages([page], contract_id="doc1")

    for chunk in chunks:
        assert chunk.text
        assert chunk.source == "contract.txt"
        assert chunk.page == 1  # all chunks came from page 1
        assert chunk.chunk_id


def test_chunk_id_format():
    page = PageContent(page_number=1, text="word " * 300, source="contract.txt")
    chunker = ContractChunker()
    chunks = chunker.chunk_pages([page], contract_id="doc1")

    assert chunks[0].chunk_id == "doc1_p1_c1"


def test_empty_page_produces_no_chunks():
    empty_page = PageContent(page_number=1, text=" ", source="contract2.txt")
    chunker = ContractChunker()
    empty_chunks = chunker.chunk_pages([empty_page], contract_id="doc2")

    assert len(empty_chunks) == 0
