# utils/document_loaders/loader_factory.py

import os
from typing import Type
from .base import BaseDocumentLoader
from .pdf_loader import PDFLoader
from .pdf_ocr_loader import PDFOCRLoader
from .docx_loader import DocxLoader
from .epub_loader import EpubLoader

# Map extension→loader class. Add more as needed (.txt, .md, etc.)
_LOADER_MAP: dict[str, Type[BaseDocumentLoader]] = {
    ".pdf": PDFLoader,
    ".docx": DocxLoader,
    ".doc": DocxLoader,   # .doc→.docx conversion is built into DocxLoader
    ".epub": EpubLoader,
    # you could add ".txt": TxtLoader, etc.
    # you could add ".md": MarkdownLoader, etc.
}

def is_pdf_text_based(path: str, min_char_threshold: int = 100) -> bool:
    """
    Quick check: opens the PDF with PyMuPDF, tries to extract
    raw text from the first few pages. If the total extracted
    characters exceed `min_char_threshold`, we assume it's
    a “text-based” PDF. Otherwise, we treat it as scanned and
    OCR-only.

    - `path`: local PDF file path
    - `min_char_threshold`: if total extracted characters < threshold, use OCR
    """
    import fitz  # pip install pymupdf

    try:
        pdf = fitz.open(path)
    except Exception:
        # If PDF can’t even be opened, fallback to OCR (or raise)
        return False

    extracted = ""
    # Only check the first 3 pages (to save time)
    for i in range(min(3, len(pdf))):
        page = pdf.load_page(i)
        extracted += page.get_text()
        # Stop early if we already know it’s text‐heavy
        if len(extracted.strip()) >= min_char_threshold:
            pdf.close()
            return True

    pdf.close()
    return len(extracted.strip()) >= min_char_threshold


def get_loader_for(path: str) -> BaseDocumentLoader:
    """
    Return an instance of the appropriate loader based on file extension
    *and*, for PDFs, whether it is text‐based or needs OCR.
    """
    ext = os.path.splitext(path.lower())[1]

    if ext == ".pdf":
        # Decide between PDFLoader (text) vs. PDFOCRLoader (scan)
        if is_pdf_text_based(path):
            return PDFLoader()
        else:
            return PDFOCRLoader()

    # Non‐PDF cases: look up in the “static” map
    LoaderCls = _LOADER_MAP.get(ext)
    if LoaderCls is None:
        raise ValueError(
            f"Unsupported document type '{ext}'. Available: {list(_LOADER_MAP.keys()) + ['.pdf']}"
        )
    return LoaderCls()
