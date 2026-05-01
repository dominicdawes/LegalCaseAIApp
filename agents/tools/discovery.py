# agents/tools/discovery.py
"""
Discovery tools — use these FIRST to understand what documents exist
and how they are structured before running retrieval.
"""

import asyncio
import json
import logging
import os
from typing import List

from langchain_core.tools import tool

from .base import ToolContext

logger = logging.getLogger(__name__)


def build_discovery_tools(ctx: ToolContext) -> list:
    """Return discovery tools bound to ctx."""

    @tool
    async def list_sources() -> str:
        """
        List all source documents available in this project.

        Use this FIRST in any research session to discover what documents
        are loaded. Returns a compact summary (id, filename, page count,
        doc_summary) for each source.  Use get_doc_outline or
        find_docs_about for deeper exploration.

        Returns JSON array: [{id, filename, page_count, doc_summary}]
        """
        from tasks.database import get_global_async_db_pool, init_async_pools
        await init_async_pools()
        pool = get_global_async_db_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, source_filename, total_pages, doc_summary
                FROM document_sources
                WHERE project_id = $1
                ORDER BY created_at DESC
                """,
                ctx.project_id,
            )
        return json.dumps([
            {
                "id": str(r["id"]),
                "filename": r["source_filename"],
                "page_count": r["total_pages"],
                "doc_summary": (r["doc_summary"] or "")[:200],
            }
            for r in rows
        ])

    @tool
    async def get_doc_outline(source_id: str) -> str:
        """
        Get the section outline (table of contents) for a specific document.

        Use after list_sources when you need to understand a document's
        structure before deciding which sections to retrieve.  Prefer this
        over find_sections_about when you already know the document and want
        a structural overview rather than a semantic match.

        Args:
            source_id: Document UUID from list_sources

        Returns JSON with toc list and doc_concepts array.
        """
        from tasks.database import get_global_async_db_pool, init_async_pools
        await init_async_pools()
        pool = get_global_async_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT toc, doc_concepts, doc_summary, source_filename
                FROM document_sources
                WHERE id = $1 AND project_id = $2
                """,
                source_id,
                ctx.project_id,
            )
        if not row:
            return json.dumps({"error": "document not found"})

        toc = row["toc"] or []
        if isinstance(toc, str):
            try:
                toc = json.loads(toc)
            except Exception:
                toc = []

        concepts = row["doc_concepts"] or []
        if isinstance(concepts, str):
            try:
                concepts = json.loads(concepts)
            except Exception:
                concepts = []

        return json.dumps({
            "source_id": source_id,
            "filename": row["source_filename"],
            "doc_summary": row["doc_summary"],
            "toc": toc[:50],
            "doc_concepts": concepts[:30],
        })

    @tool
    async def find_docs_about(query: str, k: int = 5) -> str:
        """
        Find which documents are most relevant to a topic or legal concept.

        Use when you have multiple documents and need to decide which ones
        to focus retrieval on.  Returns doc-level relevance, not passage
        text.  Prefer find_sections_about when you already know the docs
        and want section-level precision.

        Args:
            query: Free-text topic description (e.g. "promissory estoppel elements")
            k: Max documents to return (1-10, default 5)

        Returns JSON array: [{source_id, filename, doc_summary, score}]
        """
        from utils.retrieval import find_docs
        from utils.llm_clients.voyage_client import VoyageEmbeddingsClient
        import httpx

        k = max(1, min(k, 10))

        loop = asyncio.get_event_loop()
        if ctx.use_voyage:
            client = VoyageEmbeddingsClient()
            embedding = await loop.run_in_executor(None, client.embed_query, query)
        else:
            async with httpx.AsyncClient(timeout=30.0) as http:
                resp = await http.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"},
                    json={"model": "text-embedding-ada-002", "input": query},
                )
                resp.raise_for_status()
                embedding = resp.json()["data"][0]["embedding"]

        docs = await find_docs(embedding, ctx.project_id, k=k)
        return json.dumps([
            {
                "source_id": d["source_id"],
                "filename": d.get("filename", ""),
                "doc_summary": (d.get("doc_summary") or "")[:300],
                "score": round(d.get("similarity", 0.0), 4),
            }
            for d in docs
        ])

    return [list_sources, get_doc_outline, find_docs_about]
