# agents/tools/tables.py
"""
Table tools — semantic table discovery + structured QA.

Tables in the vector store have chunk_type="table", with the full markdown
stored in `content` and structured data in metadata["table_data"].
"""

import asyncio
import json
import logging
import os
from typing import Optional

from langchain_core.tools import tool

from .base import ToolContext

logger = logging.getLogger(__name__)


def build_table_tools(ctx: ToolContext) -> list:

    @tool
    async def find_tables_about(query: str, k: int = 5) -> str:
        """
        Find tables in the documents relevant to a legal concept or topic.

        Use when the question involves statutory schedules, sentencing grids,
        comparison tables, or other structured data.  Returns table summaries
        and IDs.  Use read_table to get the full markdown, or query_table to
        ask a natural-language question against the data.

        Prefer search_passages when looking for prose passages, not tables.

        Args:
            query: Description of the data you expect (e.g. "damages caps by tort type")
            k: Max tables to return (1-10, default 5)

        Returns JSON array: [{chunk_id, source_id, chunk_summary, similarity}]
        """
        from utils.retrieval import find_tables, SearchFilters

        k = max(1, min(k, 10))

        loop = asyncio.get_event_loop()
        if ctx.use_voyage:
            from utils.llm_clients.voyage_client import VoyageEmbeddingsClient
            client = VoyageEmbeddingsClient()
            embedding = await loop.run_in_executor(None, client.embed_query, query)
        else:
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as http:
                resp = await http.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"},
                    json={"model": "text-embedding-ada-002", "input": query},
                )
                resp.raise_for_status()
                embedding = resp.json()["data"][0]["embedding"]

        filters: SearchFilters = {}
        if ctx.source_ids:
            filters["source_ids"] = ctx.source_ids

        tables = await find_tables(embedding, ctx.project_id, k=k, filters=filters)
        return json.dumps([
            {
                "chunk_id": t["id"],
                "source_id": str(t["source_id"]),
                "chunk_summary": (t.get("chunk_summary") or "")[:300],
                "similarity": round(t.get("similarity", 0.0), 4),
            }
            for t in tables
        ])

    @tool
    async def read_table(chunk_id: str) -> str:
        """
        Retrieve the full markdown content and structured data for a table chunk.

        Use after find_tables_about when you need the raw table to include in
        a prompt or answer.  Returns markdown in `content` and row/column
        data in `table_data` (if available).

        Args:
            chunk_id: Table chunk UUID from find_tables_about

        Returns JSON: {chunk_id, content (markdown), table_data, caption,
                       headers, source_id, page_number}
        """
        from tasks.database import get_global_async_db_pool, init_async_pools

        await init_async_pools()
        pool = get_global_async_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, source_id, content, chunk_summary, metadata, page_number
                FROM document_vector_store
                WHERE id = $1 AND project_id = $2 AND chunk_type = 'table'
                """,
                chunk_id,
                ctx.project_id,
            )

        if not row:
            return json.dumps({"error": "table chunk not found"})

        meta = row["metadata"] or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}

        return json.dumps({
            "chunk_id": chunk_id,
            "source_id": str(row["source_id"]),
            "page_number": row["page_number"],
            "chunk_summary": row["chunk_summary"],
            "content": row["content"],
            "table_data": meta.get("table_data"),
            "caption": meta.get("caption"),
            "headers": meta.get("table_headers"),
        })

    @tool
    async def query_table(chunk_id: str, question: str) -> str:
        """
        Ask a natural-language question about a specific table's data.

        Use when you need to extract a value, compare rows, or reason over
        structured data in a table.  Internally fetches the table markdown
        and asks Haiku to answer the question.  More expensive than read_table
        but avoids passing raw table markdown into your prompt when only a
        specific fact is needed.

        Args:
            chunk_id: Table chunk UUID from find_tables_about
            question: Specific question about the table contents

        Returns JSON: {answer, chunk_id, table_summary}
        """
        import httpx

        # Re-use read_table logic inline
        from tasks.database import get_global_async_db_pool, init_async_pools

        await init_async_pools()
        pool = get_global_async_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT content, chunk_summary
                FROM document_vector_store
                WHERE id = $1 AND project_id = $2
                """,
                chunk_id,
                ctx.project_id,
            )

        if not row:
            return json.dumps({"error": "chunk not found"})

        table_text = row["content"] or ""
        summary = row["chunk_summary"] or ""

        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not anthropic_key:
            return json.dumps({"error": "ANTHROPIC_API_KEY not set"})

        prompt = (
            f"You are a legal research assistant. Answer the following question "
            f"using ONLY the table data provided. Be concise and precise.\n\n"
            f"Table:\n{table_text}\n\nQuestion: {question}"
        )

        async with httpx.AsyncClient(timeout=30.0) as http:
            resp = await http.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 512,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            answer = resp.json()["content"][0]["text"].strip()

        return json.dumps({
            "chunk_id": chunk_id,
            "table_summary": summary[:200],
            "answer": answer,
        })

    return [find_tables_about, read_table, query_table]
