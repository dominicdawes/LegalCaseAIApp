"""For the new 'True In-Memory Streaming' method"""

# utils/document_loaders/base.py

import io
from typing import List, Any, Iterator, Union
from langchain.schema import Document


class BaseDocumentLoader:
    """
    The interface now supports a Union of path or stream.
    Subclasses must implement methods to handle document loading from a source.
    """
    def load_documents(self, source: Union[str, io.BytesIO]) -> List[Any]:
        """Given a source, return a list of "documents" (could be pages or just one big chunk)."""
        raise NotImplementedError("Subclasses must override load_documents().")

    def stream_documents(self, source: Union[str, io.BytesIO]) -> Iterator[Document]:
        """Yield Document objects one by one from the source."""
        # This default implementation is rarely used but kept for interface consistency.
        for doc in self.load_documents(source):
            yield doc
