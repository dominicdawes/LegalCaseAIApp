# agents/tools/section.py
"""
Section tools — navigate document structure at section granularity.
Use these when you need a broader context window than individual passages.
"""

import asyncio
import json
import logging
import os
from typing import List, Optional

from langchain_core.tools import tool

from .base import ToolContext

logger = logging.getLogger(__name__)


def build_section_tools(ctx: ToolContext) -> list:

    @tool
    async def find_sections_about(query: str, k: int = 5, source_id: Optional[str] = None) -> str:
        """
        Find document sections semantically relevant to a legal topic.

        Returns section-level results (summaries + paths), not individual
        passages.  Use this when you need to identify the right section
        before pulling individual chunks, or when a question spans multiple
        paragraphs.  Use get_section to fetch the actual chunk content once
        you have the section_path.

        Prefer search_passages when you need fine-grained chunk retrieval
        directly.

        Args:
            query: Legal topic or question
            k: Max sections to return (1-10, default 5)
            source_id: Restrict to a single document (optional)

        Returns JSON array: [{section_path, section_summary, source_id, similarity}]
        """
        from utils.retrieval import find_sections
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

        source_ids = [source_id] if source_id else (ctx.source_ids or None)
        sections = await find_sections(embedding, ctx.project_id, k=k, source_ids=source_ids)

        return json.dumps([
            {
                "source_id": str(s["source_id"]),
                "section_path": s["section_path"],
                "section_summary": (s.get("section_summary") or "")[:400],
                "similarity": round(s.get("similarity", 0.0), 4),
            }
            for s in sections
        ])

    @tool
    async def get_section(source_id: str, section_path: str, max_chunks: int = 20) -> str:
        """
        Retrieve all chunks belonging to a specific section of a document.

        Use after find_sections_about to fetch the full content of a
        matched section.  Returns chunks ordered by chunk_index.
        Limit with max_chunks if only a summary is needed.

        Args:
            source_id: Document UUID
            section_path: Exact section path string from find_sections_about
            max_chunks: Cap on returned chunks (default 20)

        Returns JSON: {section_path, chunks: [{chunk_index, content, chunk_type, page_number}]}
        """
        from tasks.database import get_global_async_db_pool, init_async_pools

        max_chunks = max(1, min(max_chunks, 100))
        await init_async_pools()
        pool = get_global_async_db_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, chunk_index, content, chunk_type, page_number
                FROM document_vector_store
                WHERE source_id = $1
                  AND project_id = $2
                  AND section_path = $3
                ORDER BY chunk_index
                LIMIT $4
                """,
                source_id,
                ctx.project_id,
                section_path,
                max_chunks,
            )

        return json.dumps({
            "source_id": source_id,
            "section_path": section_path,
            "chunks": [
                {
                    "id": str(r["id"]),
                    "chunk_index": r["chunk_index"],
                    "chunk_type": r["chunk_type"],
                    "page_number": r["page_number"],
                    "content": r["content"],
                }
                for r in rows
            ],
        })

    return [find_sections_about, get_section]
