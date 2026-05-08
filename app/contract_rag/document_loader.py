import os
from dataclasses import dataclass


@dataclass
class PageContent:
    page_number: int
    text: str
    source: str

class DocumentLoader:
    """Loads contract documents from disk into raw text."""
    
    def load(self, path: str) -> list[PageContent]:
        """Load a contract document from disk."""
        if path.endswith('.pdf'):
            return self._load_pdf(path)
        else:
            return self._load_text(path)

    def _ocr(self, path: str) -> list[PageContent]:
        """Extract text from a PDF document using OCR."""
        import ocrmypdf # lazy import — only needed for scanned PDFs
        from pypdf import PdfReader # lazy import — re-read after OCR rewrites the file
        ocrmypdf.ocr(
            path,
            path,
            force_ocr=True,
            invalidate_digital_signatures=True,
            output_type="pdf",
        )
        reader = PdfReader(path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            pages.append(PageContent(page_number=i+1, text=text, source=os.path.basename(path)))
        
        return pages

    def _load_text(self, path: str) -> list[PageContent]:
        """Load a plain-text contract file."""
        pages = []
        with open(path, 'r') as file:
            text = file.read()
            pages.append(PageContent(page_number=1, text=text, source=os.path.basename(path)))
        
        return pages

    def _load_pdf(self, path: str) -> list[PageContent]:
        """Extract text from a PDF contract file."""
        from pypdf import PdfReader # lazy import — not needed for text files
        reader = PdfReader(path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            pages.append(PageContent(page_number=i+1, text=text, source=os.path.basename(path)))

        # If content is empty, call ocr method
        if all(not p.text.strip() for p in pages):
            pages = self._ocr(path)

        return pages
