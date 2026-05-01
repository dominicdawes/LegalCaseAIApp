# utils/retrieval/parent_expand.py

"""
parent_expand — fetch parent chunks and deduplicate the candidate pool.

For any chunk in the input list that has a non-null parent_chunk_id, this
function substitutes the parent chunk (fetched in a single batched query).
If both the child and parent already appear in the list, the child is dropped
to avoid duplicating content at generation time.

This is the "embed small, return large" pattern from the spec: small chunks
are used for precise retrieval but the wider parent section is passed to the LLM.
"""

import logging
from typing import List, Optional

from tasks.database import get_db_connection, get_global_async_db_pool, init_async_pools
from .types import RetrievedChunk
from .search import _row_to_chunk

logger = logging.getLogger(__name__)


async def parent_expand(
    chunks: List[RetrievedChunk],
    max_parents: Optional[int] = None,
) -> List[RetrievedChunk]:
    """
    Replace child chunks with their parent chunk where available.

    Steps:
      1. Collect all parent_chunk_ids referenced in the input.
      2. Batch-fetch those parent rows in a single query.
      3. Substitute: child → parent, carrying the child's best score forward.
      4. Deduplicate: if a parent was already in the input list, drop the child.
      5. Preserve original order by rrf_score > similarity > position.

    Args:
        chunks      — input candidate list (from rrf_fuse or vector_search)
        max_parents — hard cap on how many parent substitutions to make;
                      None = no cap (substitute all that have parents)

    Returns:
        Expanded list with parents substituted, deduplicated.
    """
    if not chunks:
        return []

    # Build lookup of chunks already in the list
    existing_ids = {c["id"] for c in chunks}

    # Collect parent IDs that need fetching (not already in the list)
    parent_ids_needed: List[str] = []
    for chunk in chunks:
        pid = chunk.get("parent_chunk_id")
        if pid and pid not in existing_ids:
            parent_ids_needed.append(pid)

    if not parent_ids_needed:
        return chunks  # Nothing to expand

    if max_parents is not None:
        parent_ids_needed = parent_ids_needed[:max_parents]

    # Batch-fetch parent rows
    pool = get_global_async_db_pool()
    if not pool:
        await init_async_pools()

    parent_map: dict = {}
    async with get_db_connection() as conn:
        rows = await conn.fetch(
            """
            SELECT id, source_id, project_id, content,
                   chunk_summary, section_path, chunk_type, parent_chunk_id,
                   metadata, page_number, chunk_index
            FROM document_vector_store
            WHERE id = ANY($1::uuid[])
            """,
            parent_ids_needed,
        )
    for row in rows:
        parent_map[str(row["id"])] = _row_to_chunk(row)

    # Substitute children with parents, preserving scores
    seen_ids: set = set()
    result: List[RetrievedChunk] = []

    for chunk in chunks:
        pid = chunk.get("parent_chunk_id")
        if pid and pid in parent_map:
            # Use parent content but keep the child's retrieval scores
            parent = dict(parent_map[pid])
            parent["similarity"]    = parent.get("similarity")    or chunk.get("similarity")
            parent["bm25_score"]    = parent.get("bm25_score")    or chunk.get("bm25_score")
            parent["rrf_score"]     = chunk.get("rrf_score")      or parent.get("rrf_score")
            parent["rerank_score"]  = chunk.get("rerank_score")   or parent.get("rerank_score")

            if pid not in seen_ids:
                result.append(parent)  # type: ignore[arg-type]
                seen_ids.add(pid)
            # child is dropped (its content is a subset of the parent)
        else:
            if chunk["id"] not in seen_ids:
                result.append(chunk)
                seen_ids.add(chunk["id"])

    logger.info(
        f"📦 parent_expand: {len(chunks)} → {len(result)} chunks "
        f"({len(parent_map)} parents fetched)"
    )
    return result
