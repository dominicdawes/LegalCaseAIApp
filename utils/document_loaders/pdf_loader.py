"""For the new 'True In-Memory Streaming' method"""

# utils/document_loaders/pdf_loader.py

from typing import List, Iterator, Union
import io
from langchain.schema import Document
import fitz
from .base import BaseDocumentLoader

class PDFLoader(BaseDocumentLoader):
    """
    ### MODIFIED: This loader now primarily works with in-memory streams.
    """
    def load_documents(self, source: Union[str, io.BytesIO]) -> List:
        # This method is less used in streaming pipelines but is updated for consistency.
        docs = []
        for doc in self.stream_documents(source):
            docs.append(doc)
        return docs

    def stream_documents(self, source: Union[str, io.BytesIO]) -> Iterator[Document]:
        # ### CHANGE: The core logic now opens the PDF from a stream.
        # It can still accept a path for backward compatibility.
        pdf = fitz.open(stream=source, filetype="pdf") if isinstance(source, io.BytesIO) else fitz.open(source)
        
        try:
            for i, page in enumerate(pdf):
                yield Document(
                    page_content=page.get_text(),
                    # If it's a stream, we don't have a source path, which is expected.
                    metadata={"page": i + 1},
                )
        finally:
            pdf.close()