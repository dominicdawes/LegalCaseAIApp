"""For the new 'True In-Memory Streaming' method"""

# utils/document_loaders/docx_loader.py

import textract
import io
import tempfile
from typing import Iterator, Union, List
from langchain.schema import Document
from .base import BaseDocumentLoader
import docx # python-docx

## OLD IMPORTS
import os
import textract

class DocxLoader(BaseDocumentLoader):
    """
    ### MODIFIED: A loader for .docx files that operates on in-memory streams.
    This loader does NOT support the old binary .doc format for in-memory processing.
    """
    def stream_documents(self, source: Union[str, io.BytesIO]) -> Iterator[Document]:
        """
        ### CHANGE: Uses docx.Document() with a file-like object.
        """
        try:
            document = docx.Document(source)
            for i, para in enumerate(document.paragraphs):
                text = para.text.strip()
                if not text:
                    continue
                yield Document(
                    page_content=text,
                    metadata={"paragraph_index": i},
                )
        except Exception as e:
            raise RuntimeError(f"Failed to process DOCX stream: {e}")

class LegacyDocLoader(BaseDocumentLoader):
    def stream_documents(self, source: io.BytesIO) -> Iterator[Document]:
        # Minimal temp file usage - deleted immediately
        with tempfile.NamedTemporaryFile(suffix='.doc', delete=False) as tmp:
            source.seek(0)
            tmp.write(source.read())
            tmp.flush()
            
            try:
                raw_bytes = textract.process(tmp.name)
                text = raw_bytes.decode('utf-8', errors='ignore')
            finally:
                os.unlink(tmp.name)  # Clean up immediately
            
            # Stream the results
            for i, para in enumerate(text.split('\n\n')):
                if para.strip():
                    yield Document(
                        page_content=para.strip(), 
                        metadata={"paragraph": i}
                    )

class DocxLoader_deprecated(BaseDocumentLoader):
    """
    OLD VERSION: A loader for Microsoft Word files (.doc and .docx) that uses Textract
    under the hood to extract plain text. Textract auto-detects whether the
    file is binary .doc or XML-based .docx and invokes the correct converter.
    """

    def load_documents(self, path: str) -> List[Document]:
        """
        Returns a list of Document(page_content, metadata) for the given file.
        - Uses textract.process(...) to get the full Unicode text.
        - Splits on two consecutive newlines (you can adjust this logic).
        - Wraps each non-empty paragraph in a Document.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: '{path}'")

        # 1) Let Textract pull out all text from .doc or .docx
        try:
            raw_bytes = textract.process(path)
            raw_text = raw_bytes.decode("utf-8", errors="ignore")
        except Exception as e:
            raise RuntimeError(f"Textract failed to extract text from '{path}': {e}")

        # 2) Split raw_text into paragraphs (split on two newlines)
        paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]

        # 3) Wrap each paragraph in a Document(...) with metadata
        docs: List[Document] = []
        for idx, para in enumerate(paragraphs):
            docs.append(
                Document(
                    page_content=para,
                    metadata={
                        "source_path": os.path.basename(path),
                        "paragraph_index": idx,
                    },
                )
            )

        return docs

    def stream_documents(self, path: str) -> Iterator[Document]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: '{path}'")

        document = docx.Document(path)
        for idx, para in enumerate(document.paragraphs):
            text = para.text.strip()
            if not text:
                continue
            yield Document(
                page_content=text,
                metadata={
                    "source_path": os.path.basename(path),
                    "paragraph_index": idx,
                },
            )
