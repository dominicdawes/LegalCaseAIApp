# USE LATEST OCR FRAMWORK (> Tessaract)

# utils/document_loaders/pdf_ocr_loader.py

from typing import List, Iterator
from langchain.schema import Document
from .base import BaseDocumentLoader

import fitz  # pip install pymupdf
import pytesseract  # pip install pytesseract
from PIL import Image
import io


class PDFOCRLoader(BaseDocumentLoader):
    """
    A loader that forces OCR on every page of a PDF. Use this
    when PyPDF2 (or your normal PDFLoader) fails to extract any
    text (i.e. scanned/image‐only PDFs).
    """

    def load_documents(self, path: str) -> List[Document]:
        """
        Opens the PDF via PyMuPDF, renders each page to an image,
        runs Tesseract OCR, and returns a list of Document objects
        (one per page) containing whatever text was recognized.

        - `path`: filesystem path to the PDF file.
        """
        documents: list[Document] = []
        # Open with PyMuPDF
        pdf = fitz.open(path)

        for page_number in range(len(pdf)):
            page = pdf.load_page(page_number)

            # Render page to a pixmap (PNG bytes)
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")

            # Load PNG bytes into PIL so Tesseract can consume it
            image = Image.open(io.BytesIO(img_bytes))

            # Run Tesseract OCR on the PIL image
            text = pytesseract.image_to_string(image)

            # If OCR found anything, add it as a Document
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": path, "page": page_number + 1},
                    )
                )

        pdf.close()
        return documents

    def stream_documents(self, path: str) -> Iterator[Document]:
        pdf = fitz.open(path)
        try:
            for page_number in range(len(pdf)):
                page = pdf.load_page(page_number)
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img)
                yield Document(
                    page_content=text,
                    metadata={"source": path, "page": page_number + 1, "ocr": True},
                )
        finally:
            pdf.close()
