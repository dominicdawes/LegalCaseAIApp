# agents/tools/crossdoc.py
"""
Cross-document tools — synthesise across multiple source documents.
"""

import asyncio
import json
import logging
import os
from typing import Optional

from langchain_core.tools import tool

from .base import ToolContext

logger = logging.getLogger(__name__)


def build_crossdoc_tools(ctx: ToolContext) -> list:

    @tool
    async def find_concept_across_docs(concept: str, k_per_doc: int = 3) -> str:
        """
        Search for a legal concept in every source document independently,
        returning the top-k passages per document.

        Use when you need to compare how different documents treat the same
        concept (e.g. how three cases each define "reasonable person").
        More expensive than search_passages because it runs one search per
        document; prefer search_passages for single-document or unstructured
        multi-document queries.

        Args:
            concept: Legal concept or doctrine to search for
            k_per_doc: Passages to return per document (1-5, default 3)

        Returns JSON object keyed by source_id, each with a list of
        [{chunk_id, content, similarity, page_number}].
        """
        from utils.retrieval import vector_search, SearchFilters
        from tasks.database import get_global_async_db_pool, init_async_pools
        import httpx

        k_per_doc = max(1, min(k_per_doc, 5))
        loop = asyncio.get_event_loop()

        # Embed the concept once
        if ctx.use_voyage:
            from utils.llm_clients.voyage_client import VoyageEmbeddingsClient
            client = VoyageEmbeddingsClient()
            embedding = await loop.run_in_executor(None, client.embed_query, concept)
        else:
            async with httpx.AsyncClient(timeout=30.0) as http:
                resp = await http.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"},
                    json={"model": "text-embedding-ada-002", "input": concept},
                )
                resp.raise_for_status()
                embedding = resp.json()["data"][0]["embedding"]

        # Determine which docs to search
        if ctx.source_ids:
            doc_ids = ctx.source_ids
        else:
            await init_async_pools()
        pool = get_global_async_db_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT id FROM document_sources WHERE project_id = $1",
                    ctx.project_id,
                )
            doc_ids = [str(r["id"]) for r in rows]

        if not doc_ids:
            return json.dumps({})

        # Run one vector_search per document in parallel
        async def _search_one(doc_id: str):
            filters: SearchFilters = {"source_ids": [doc_id]}
            return doc_id, await vector_search(
                embedding, ctx.project_id, k=k_per_doc, filters=filters
            )

        results = await asyncio.gather(*[_search_one(d) for d in doc_ids])

        out = {}
        for doc_id, chunks in results:
            out[doc_id] = [
                {
                    "chunk_id": c["id"],
                    "page_number": c.get("page_number"),
                    "similarity": round(c.get("similarity", 0.0), 4),
                    "content": c["content"][:400],
                }
                for c in chunks
            ]
        return json.dumps(out)

    return [find_concept_across_docs]
