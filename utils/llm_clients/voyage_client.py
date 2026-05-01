# utils/llm_clients/voyage_client.py

import os
import logging
from typing import List

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "").strip()
VOYAGE_MODEL = "voyage-law-2"
VOYAGE_EMBEDDING_DIM = 1024
_VOYAGE_MAX_BATCH = 128  # Voyage API limit


class VoyageEmbeddingsClient:
    """
    Voyage AI embeddings for voyage-law-2 (1024-dim).
    Drop-in replacement for OpenAI ada-002 in the embedding pipeline.

    Differences from ada-002:
      - output dim: 1024 (vs 1536)
      - input_type: "document" for indexing, "query" for retrieval
      - max batch size: 128 texts
    """

    def __init__(self, model: str = VOYAGE_MODEL, batch_size: int = _VOYAGE_MAX_BATCH):
        if not VOYAGE_API_KEY:
            raise ValueError("VOYAGE_API_KEY not set in environment")
        import voyageai  # lazy import — missing dep won't break other importers
        self._client = voyageai.Client(api_key=VOYAGE_API_KEY)
        self.model = model
        self.batch_size = batch_size
        self.embedding_dim = VOYAGE_EMBEDDING_DIM
        logger.info(f"🌊 VoyageEmbeddingsClient ready: {model} ({VOYAGE_EMBEDDING_DIM}-dim)")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document texts, auto-batching to respect the API limit.
        Uses input_type='document' which is optimised for indexing.
        """
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self._client.embed(batch, model=self.model, input_type="document")
            all_embeddings.extend(response.embeddings)
        logger.debug(f"🌊 Voyage embedded {len(all_embeddings)} texts")
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.
        Uses input_type='query' for retrieval-optimised representation.
        """
        response = self._client.embed([text], model=self.model, input_type="query")
        return response.embeddings[0]

    async def embed_documents_async(self, texts: List[str]) -> List[List[float]]:
        """Async wrapper — runs the sync client in the default executor."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)

    async def embed_query_async(self, text: str) -> List[float]:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, text)
