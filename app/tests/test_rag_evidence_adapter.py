"""Tests for the RAG → Evidence adapter and metadata preservation."""
from app.contract_rag.evidence_adapter import retrieval_results_to_evidence
from app.contract_rag.schemas import DocumentChunk, RetrievalResult
from app.schemas.common import Evidence


def _make_result(chunk_id="c-001", contract_id="doc", text="some text",
                 source="contract.pdf", page=2, score=0.87) -> RetrievalResult:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        contract_id=contract_id,
        text=text,
        chunk_index=1,
        source=source,
        page=page,
    )
    return RetrievalResult(chunk=chunk, score=score)


def test_converts_to_evidence_list():
    results = [_make_result()]
    evidence = retrieval_results_to_evidence(results)
    assert len(evidence) == 1
    assert isinstance(evidence[0], Evidence)


def test_preserves_text():
    results = [_make_result(text="Payment due on the 1st of each month.")]
    ev = retrieval_results_to_evidence(results)[0]
    assert ev.text == "Payment due on the 1st of each month."


def test_preserves_source_and_page():
    results = [_make_result(source="master_agreement.pdf", page=5)]
    ev = retrieval_results_to_evidence(results)[0]
    assert ev.source == "master_agreement.pdf"
    assert ev.page == 5


def test_preserves_chunk_id():
    results = [_make_result(chunk_id="doc_p3_c7")]
    ev = retrieval_results_to_evidence(results)[0]
    assert ev.chunk_id == "doc_p3_c7"


def test_preserves_relevance_score():
    results = [_make_result(score=0.73)]
    ev = retrieval_results_to_evidence(results)[0]
    assert ev.relevance_score == 0.73


def test_clamps_negative_score_to_zero():
    results = [_make_result(score=-0.4)]
    ev = retrieval_results_to_evidence(results)[0]
    assert ev.relevance_score == 0.0


def test_clamps_score_above_one():
    results = [_make_result(score=1.2)]
    ev = retrieval_results_to_evidence(results)[0]
    assert ev.relevance_score == 1.0


def test_empty_input_returns_empty_list():
    assert retrieval_results_to_evidence([]) == []


def test_multiple_results_preserve_order():
    r1 = _make_result(chunk_id="c-001", score=0.9)
    r2 = _make_result(chunk_id="c-002", score=0.7)
    evidence = retrieval_results_to_evidence([r1, r2])
    assert evidence[0].chunk_id == "c-001"
    assert evidence[1].chunk_id == "c-002"
