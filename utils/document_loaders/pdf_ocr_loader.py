"""For the new 'True In-Memory Streaming' method"""

# utils/document_loaders/pdf_ocr_loader.py

from typing import Iterator, Union
import io
from langchain.schema import Document
from .base import BaseDocumentLoader

import fitz
import pytesseract
from PIL import Image

class PDFOCRLoader(BaseDocumentLoader):
    """
    ### MODIFIED: Forces OCR on a PDF stream from memory.
    """
    def stream_documents(self, source: Union[str, io.BytesIO]) -> Iterator[Document]:
        # ### CHANGE: Opens the PDF from a stream for OCR.
        pdf = fitz.open(stream=source, filetype="pdf") if isinstance(source, io.BytesIO) else fitz.open(source)

        try:
            for page_number, page in enumerate(pdf):
                # Increased DPI for better OCR accuracy.
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img)
                
                if not text.strip():
                    continue

                yield Document(
                    page_content=text,
                    metadata={"page": page_number + 1, "ocr": True},
                )
        finally:
            pdf.close()
