# utils/document_loaders/epub_loader.py

from typing import List, Iterator
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
from langchain.schema import Document
from .base import BaseDocumentLoader


class EpubLoader(BaseDocumentLoader):
    def load_documents(self, path: str) -> List[Document]:
        """
        Extracts all <p> tags text from an EPUB’s items,
        concatenates them into Document objects. You could also break per chapter.
        """
        book = epub.read_epub(path)
        documents = []
        for item in book.get_items_of_type(ITEM_DOCUMENT):  # More efficient iteration
            # item.get_content() is HTML bytes
            soup = BeautifulSoup(item.get_content(), "html.parser")

            # Find all <p> tags and extract their text
            text = "\n".join(
                p.get_text() for p in soup.find_all("p") if p.get_text().strip()
            )

            # Skip items that result in no text after extraction
            if not text.strip():
                continue

            # You could use item.get_id() or item.get_name() as “metadata” for ordering
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": path, "chapter": item.get_name()},
                )
            )
        return documents

    def stream_documents(self, path: str) -> Iterator[Document]:
        book = epub.read_epub(path)
        for item in book.get_items_of_type(ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = "\n".join(
                p.get_text() for p in soup.find_all("p") if p.get_text().strip()
            )
            if not text.strip():
                continue
            yield Document(
                page_content=text,
                metadata={"source": path, "chapter": item.get_name()},
            )
