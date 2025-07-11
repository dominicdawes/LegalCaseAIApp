# utils/document_loaders/pdf_loader.py

from langchain_community.document_loaders import PyPDFLoader
from typing import List, Iterator
from langchain.schema import Document
import fitz
from .base import BaseDocumentLoader


class PDFLoader(BaseDocumentLoader):
    def load_documents(self, path: str) -> List:
        """
        Use PyPDFLoader to read a PDF from `path`.
        Returns a list of langchain.schema.Document (with page_content + metadata.page).
        """
        loader = PyPDFLoader(path)
        return loader.load()

    def stream_documents(self, path: str) -> Iterator[Document]:
        pdf = fitz.open(path)
        try:
            for i in range(len(pdf)):
                page = pdf.load_page(i)
                text = page.get_text()
                yield Document(
                    page_content=text,
                    metadata={"source": path, "page": i + 1},
                )
        finally:
            pdf.close()
