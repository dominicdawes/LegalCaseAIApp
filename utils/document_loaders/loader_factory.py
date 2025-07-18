# utils/document_loaders/loader_factory.py

import os
import io
from typing import Type, Union
from .base import BaseDocumentLoader
from .pdf_loader import PDFLoader
from .pdf_ocr_loader import PDFOCRLoader
from .docx_loader import DocxLoader
from .epub_loader import EpubLoader

# Map extension→loader class remains the same.
_LOADER_MAP: dict[str, Type[BaseDocumentLoader]] = {
    ".pdf": PDFLoader,
    ".docx": DocxLoader,
    # .doc is intentionally removed. See explanation below.
    ".epub": EpubLoader,
    # you could add ".txt": TxtLoader, etc.
    # you could add ".md": MarkdownLoader, etc.
}

def is_pdf_text_based(file_stream: io.BytesIO, min_char_threshold: int = 100) -> bool:
    """
    ### MODIFIED: Now checks a file stream instead of a path.
    Quickly checks if a PDF in an in-memory stream is text-based.
    """
    import fitz  # pip install pymupdf

    try:
        # ### CHANGE: Open PDF from stream, not path.
        pdf = fitz.open(stream=file_stream, filetype="pdf")
    except Exception:
        return False # Fallback to OCR if the stream can't be opened.

    extracted_chars = 0
    # Check the first 3 pages to save time
    for page in pdf.pages(stop=3):
        extracted_chars += len(page.get_text())
        if extracted_chars >= min_char_threshold:
            pdf.close()
            return True

    pdf.close()
    return extracted_chars >= min_char_threshold


def get_loader_for(filename: str, file_like_object: io.BytesIO) -> BaseDocumentLoader:
    """
    ### MODIFIED: Signature changed to accept a filename and file-like object.
    Return:
        An instance of the appropriate loader for the given in-memory document.
    """
    # Get the extension from the original filename.
    ext = os.path.splitext(filename.lower())[1]

    if ext == ".pdf":
        # Reset stream position before checking, as it might have been read before.
        file_like_object.seek(0)
        if is_pdf_text_based(file_like_object):
            loader = PDFLoader()
        else:
            loader = PDFOCRLoader()
        # Reset stream position again so the loader can read it from the beginning.
        file_like_object.seek(0)
        return loader

    LoaderCls = _LOADER_MAP.get(ext)
    if LoaderCls is None:
        # Added .doc to the message to be clear about what's not supported in-memory.
        raise ValueError(
            f"Unsupported document type '{ext}'. In-memory supported: {list(_LOADER_MAP.keys())}"
        )
    return LoaderCls()