# agents/tools/chunk.py
"""
Chunk-level retrieval tools — the primary search surface.
Most agent nodes will use search_passages or hybrid_search as their
main retrieval step.
"""

import asyncio
import json
import logging
import os
from typing import List, Optional

from langchain_core.tools import tool

from .base import ToolContext

logger = logging.getLogger(__name__)

_CHUNK_SUMMARY_KEYS = ("id", "source_id", "chunk_type", "section_path",
                       "page_number", "chunk_index", "similarity",
                       "bm25_score", "rrf_score", "rerank_score")


def _summarise_chunks(chunks: list, include_content: bool = True) -> list:
    out = []
    for c in chunks:
        item = {k: c.get(k) for k in _CHUNK_SUMMARY_KEYS if c.get(k) is not None}
        if include_content:
            item["content"] = c.get("content", "")
        out.append(item)
    return out


async def _embed(text: str, ctx: ToolContext) -> list:
    loop = asyncio.get_event_loop()
    if ctx.use_voyage:
        from utils.llm_clients.voyage_client import VoyageEmbeddingsClient
        client = VoyageEmbeddingsClient()
        return await loop.run_in_executor(None, client.embed_query, text)
    import httpx
    async with httpx.AsyncClient(timeout=30.0) as http:
        resp = await http.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"},
            json={"model": "text-embedding-ada-002", "input": text},
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]


def build_chunk_tools(ctx: ToolContext) -> list:

    @tool
    async def search_passages(query: str, k: int = 20) -> str:
        """
        Vector search for passages most semantically similar to a query.

        The primary retrieval tool.  Use for conceptual / semantic matching
        (e.g. "elements of promissory estoppel").  Use hybrid_search instead
        when you need exact term matching (statutes, case names, defined terms).

        Results are restricted to ctx.source_ids if set.  Returns chunk
        content + metadata sorted by cosine similarity.

        Args:
            query: Legal research question or concept description
            k: Number of passages to return (5–50, default 20)

        Returns JSON array of chunks with content and metadata.
        """
        from utils.retrieval import vector_search, SearchFilters

        k = max(5, min(k, 50))
        embedding = await _embed(query, ctx)
        filters: SearchFilters = {}
        if ctx.source_ids:
            filters["source_ids"] = ctx.source_ids

        chunks = await vector_search(embedding, ctx.project_id, k=k, filters=filters)
        return json.dumps(_summarise_chunks(chunks))

    @tool
    async def hybrid_search(query: str, k: int = 20) -> str:
        """
        Hybrid BM25 + vector search fused with Reciprocal Rank Fusion.

        Use when the query contains specific legal terms, statute citations,
        case names, or defined phrases that need exact keyword matching in
        addition to semantic similarity.  Slower than search_passages but
        more precise for terminology-heavy queries.

        Args:
            query: Query string (can contain exact terms or phrases)
            k: Candidate pool per modality (5-50, default 20); final output
               will be ≤ k after RRF fusion

        Returns JSON array of fused, deduplicated chunks.
        """
        from utils.retrieval import vector_search, bm25_search, rrf_fuse, SearchFilters

        k = max(5, min(k, 50))
        embedding = await _embed(query, ctx)
        filters: SearchFilters = {}
        if ctx.source_ids:
            filters["source_ids"] = ctx.source_ids

        vec_task = vector_search(embedding, ctx.project_id, k=k, filters=filters)
        bm25_task = bm25_search(query, ctx.project_id, k=k, filters=filters)
        vec_res, bm25_res = await asyncio.gather(vec_task, bm25_task)

        fused = rrf_fuse(vec_res, bm25_res, top_n=k)
        return json.dumps(_summarise_chunks(fused))

    @tool
    async def expand_query(query: str, k: int = 30) -> str:
        """
        Multi-query expansion: generate rephrasings, search each, fuse with RRF.

        Use when a single query may miss relevant passages due to terminology
        variation.  Internally generates 3 alternative phrasings via Haiku,
        runs vector_search for each, and fuses results.  More expensive than
        search_passages but improves recall for broad or ambiguous queries.

        Args:
            query: Original legal research question
            k: Final pool size after fusion (10-50, default 30)

        Returns JSON array of fused chunks ordered by rrf_score.
        """
        from utils.retrieval import multi_query_expand, SearchFilters

        k = max(10, min(k, 50))
        embedding = await _embed(query, ctx)
        filters: SearchFilters = {}
        if ctx.source_ids:
            filters["source_ids"] = ctx.source_ids

        chunks = await multi_query_expand(
            query=query,
            project_id=ctx.project_id,
            query_embedding=embedding,
            n=3,
            k_per_query=k,
            top_n=k,
            filters=filters,
        )
        return json.dumps(_summarise_chunks(chunks))

    @tool
    async def get_parents(chunk_ids: List[str]) -> str:
        """
        Expand child chunks to their parent chunks for broader context.

        Use when search_passages returns narrow snippet-level chunks and you
        need the surrounding paragraph or section.  Substitutes each child
        that has a parent_chunk_id with the parent row, keeping scores.

        Do NOT call this on chunks that are already parent-level (no
        parent_chunk_id) — it's a no-op for those.

        Args:
            chunk_ids: List of chunk UUIDs from a previous search call

        Returns JSON array of expanded (parent-substituted) chunks.
        """
        from utils.retrieval import parent_expand

        # Build minimal RetrievedChunk stubs so parent_expand can work
        stubs = [{"id": cid, "parent_chunk_id": None} for cid in chunk_ids]
        # parent_expand queries DB for real data; it fills in missing fields
        expanded = await parent_expand(stubs)
        return json.dumps(_summarise_chunks(expanded))

    @tool
    async def get_neighbors(chunk_id: str, window: int = 2) -> str:
        """
        Fetch the chunks immediately before and after a given chunk.

        Use when a retrieved chunk is cut off mid-argument and you need the
        surrounding context.  Returns up to `window` chunks on each side
        within the same source document and section, ordered by chunk_index.

        Prefer get_section when you need an entire section rather than just
        the surrounding window.

        Args:
            chunk_id: UUID of the anchor chunk
            window: Chunks to fetch on each side (1-5, default 2)

        Returns JSON: {anchor_id, before: [...], after: [...]}
        """
        from tasks.database import get_global_async_db_pool, init_async_pools

        window = max(1, min(window, 5))
        await init_async_pools()
        pool = get_global_async_db_pool()
        async with pool.acquire() as conn:
            anchor = await conn.fetchrow(
                """
                SELECT id, source_id, section_path, chunk_index
                FROM document_vector_store
                WHERE id = $1 AND project_id = $2
                """,
                chunk_id,
                ctx.project_id,
            )
            if not anchor:
                return json.dumps({"error": "chunk not found"})

            rows = await conn.fetch(
                """
                SELECT id, chunk_index, content, chunk_type, page_number
                FROM document_vector_store
                WHERE source_id = $1
                  AND project_id = $2
                  AND section_path = $3
                  AND chunk_index BETWEEN $4 AND $5
                  AND id != $6
                ORDER BY chunk_index
                """,
                str(anchor["source_id"]),
                ctx.project_id,
                anchor["section_path"],
                anchor["chunk_index"] - window,
                anchor["chunk_index"] + window,
                chunk_id,
            )

        pivot = anchor["chunk_index"]
        before = [r for r in rows if r["chunk_index"] < pivot]
        after = [r for r in rows if r["chunk_index"] > pivot]

        def _fmt(r):
            return {
                "id": str(r["id"]),
                "chunk_index": r["chunk_index"],
                "chunk_type": r["chunk_type"],
                "page_number": r["page_number"],
                "content": r["content"],
            }

        return json.dumps({
            "anchor_id": chunk_id,
            "before": [_fmt(r) for r in before],
            "after": [_fmt(r) for r in after],
        })

    return [search_passages, hybrid_search, expand_query, get_parents, get_neighbors]
