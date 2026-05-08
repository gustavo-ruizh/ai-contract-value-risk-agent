import sys
from app.contract_rag.document_loader import DocumentLoader
from unittest.mock import patch, MagicMock
from pathlib import Path


SAMPLE_DIR = Path(__file__).parent.parent.parent / "sample_data"
SAMPLE_TXT = SAMPLE_DIR / "sample_contract.txt"

def test_load_txt_pagecount():
    loader = DocumentLoader()
    pages = loader._load_text(str(SAMPLE_TXT))
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].source == "sample_contract.txt"

def test_load_pdf_page_count():
    mock_page_1 = MagicMock()
    mock_page_1.extract_text.return_value = "Page one text."
    mock_page_2 = MagicMock()
    mock_page_2.extract_text.return_value = "Page two text."

    mock_pypdf = MagicMock()
    mock_pypdf.PdfReader.return_value.pages = [mock_page_1, mock_page_2]

    with patch.dict(sys.modules, {"pypdf": mock_pypdf}):
        loader = DocumentLoader()
        pages = loader._load_pdf("contract.pdf")

    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[1].page_number == 2