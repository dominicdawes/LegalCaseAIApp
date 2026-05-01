# agents/tools/housekeeping.py
"""
Housekeeping tools — metadata inspection and token budgeting.
"""

import json
import logging
from typing import List

from langchain_core.tools import tool

from .base import ToolContext

logger = logging.getLogger(__name__)


def build_housekeeping_tools(ctx: ToolContext) -> list:

    @tool
    async def get_doc_metadata(source_id: str) -> str:
        """
        Retrieve full metadata for a source document.

        Use when the agent needs jurisdiction, document type, upload date,
        or other header information before deciding retrieval strategy.

        Args:
            source_id: Document UUID from list_sources

        Returns JSON with all metadata fields for the document.
        """
        from tasks.database import get_global_async_db_pool, init_async_pools

        await init_async_pools()
        pool = get_global_async_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, source_filename, cdn_url, total_pages,
                       doc_summary, doc_concepts, toc, created_at,
                       file_type, file_size
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
            "id": str(row["id"]),
            "filename": row["source_filename"],
            "cdn_url": row["cdn_url"],
            "total_pages": row["total_pages"],
            "file_type": row.get("file_type"),
            "file_size": row.get("file_size"),
            "doc_summary": row["doc_summary"],
            "doc_concepts": concepts[:30],
            "toc_length": len(toc),
            "created_at": str(row["created_at"]) if row["created_at"] else None,
        })

    @tool
    def count_tokens(text: str) -> str:
        """
        Count the approximate token count for a string using tiktoken.

        Use before injecting large context blocks into a prompt to check
        you will not exceed the model's context window.

        Args:
            text: Any string to measure

        Returns JSON: {token_count, char_count}
        """
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            token_count = len(enc.encode(text))
        except Exception:
            # Rough fallback: ~4 chars per token
            token_count = len(text) // 4

        return json.dumps({
            "token_count": token_count,
            "char_count": len(text),
        })

    return [get_doc_metadata, count_tokens]
