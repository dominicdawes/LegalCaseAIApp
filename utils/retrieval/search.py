# utils/retrieval/search.py

"""
vector_search  — HNSW cosine similarity via match_document_chunks_hnsw RPC
bm25_search    — Postgres tsvector / BM25 via websearch_to_tsquery

Both functions accept SearchFilters and return List[RetrievedChunk].
The k parameter is the *candidate pool* size before reranking (default 50).
"""

import json
import logging
import os
from typing import List, Optional

from dotenv import load_dotenv

from tasks.database import get_db_connection, get_global_async_db_pool, init_async_pools
from .types import RetrievedChunk, SearchFilters, vec_to_pg

load_dotenv()

logger = logging.getLogger(__name__)

USE_VOYAGE_EMBEDDINGS = os.getenv("USE_VOYAGE_EMBEDDINGS", "false").lower() == "true"


# ——— Row conversion ——————————————————————————————————————————————————————————


def _row_to_chunk(row, score_key: str = "similarity", score_val: Optional[float] = None) -> RetrievedChunk:
    """Convert an asyncpg Row or dict to a RetrievedChunk TypedDict."""
    d = dict(row) if not isinstance(row, dict) else row
    meta = d.get("metadata") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}

    chunk: RetrievedChunk = {
        "id":              str(d.get("id", "")),
        "source_id":       str(d.get("source_id", "")),
        "project_id":      str(d.get("project_id", "")),
        "content":         d.get("content", ""),
        "chunk_type":      d.get("chunk_type") or "prose",
        "chunk_summary":   d.get("chunk_summary"),
        "section_path":    d.get("section_path"),
        "parent_chunk_id": str(d["parent_chunk_id"]) if d.get("parent_chunk_id") else None,
        "metadata":        meta,
        "page_number":     d.get("page_number"),
        "chunk_index":     d.get("chunk_index"),
        "similarity":      None,
        "bm25_score":      None,
        "rrf_score":       None,
        "rerank_score":    None,
    }

    # Apply the score value
    if score_val is not None:
        chunk[score_key] = score_val
    elif score_key in d:
        chunk[score_key] = float(d[score_key]) if d[score_key] is not None else None

    return chunk


# ——— vector_search ————————————————————————————————————————————————————————————


async def vector_search(
    query_embedding: List[float],
    project_id: str,
    k: int = 50,
    filters: Optional[SearchFilters] = None,
) -> List[RetrievedChunk]:
    """
    HNSW cosine-similarity search via the match_document_chunks_hnsw RPC.

    When USE_VOYAGE_EMBEDDINGS=true the RPC uses the embedding_voyage_2 column
    (voyage-law-2, 1024-dim); otherwise uses embedding_ada_legacy (ada-002, 1536-dim).

    Args:
        query_embedding — pre-computed embedding vector
        project_id      — scope results to this project
        k               — candidate pool size (retrieve before reranking)
        filters         — optional source_ids / chunk_types / section_path_prefix

    Returns:
        List[RetrievedChunk] ordered by descending cosine similarity.
    """
    filters = filters or {}
    vector_str = vec_to_pg(query_embedding)

    source_ids  = filters.get("source_ids")   or None
    chunk_types = filters.get("chunk_types")  or None
    sec_prefix  = filters.get("section_path_prefix") or None

    pool = get_global_async_db_pool()
    if not pool:
        await init_async_pools()

    async with get_db_connection() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM match_document_chunks_hnsw(
                $1, $2::vector, $3,
                $4::text[],
                $5::uuid[],
                $6::text,
                $7::boolean
            )
            """,
            project_id,
            vector_str,
            k,
            chunk_types,
            source_ids,
            sec_prefix,
            USE_VOYAGE_EMBEDDINGS,
        )

    results = [_row_to_chunk(row, score_key="similarity") for row in rows]
    logger.info(
        f"🔍 vector_search: {len(results)} results "
        f"(k={k}, filters={bool(filters)}, voyage={USE_VOYAGE_EMBEDDINGS})"
    )
    return results


# ——— bm25_search ————————————————————————————————————————————————————————————


async def bm25_search(
    query_text: str,
    project_id: str,
    k: int = 50,
    filters: Optional[SearchFilters] = None,
) -> List[RetrievedChunk]:
    """
    BM25 full-text search using Postgres tsvector (bm25_tsvector generated column).

    Uses websearch_to_tsquery for robust query parsing (handles phrases, -, |, etc.).
    Falls back to plainto_tsquery if websearch_to_tsquery returns nothing.

    Args:
        query_text — natural-language search string
        project_id — scope results to this project
        k          — max results
        filters    — optional source_ids / chunk_types / section_path_prefix

    Returns:
        List[RetrievedChunk] ordered by descending ts_rank_cd score.
    """
    filters = filters or {}

    source_ids  = filters.get("source_ids")          or None
    chunk_types = filters.get("chunk_types")          or None
    sec_prefix  = filters.get("section_path_prefix")  or None

    # websearch_to_tsquery ANDs every term; long queries (30+ words) or queries
    # containing stop-like tokens (numbers, prepositions) can match zero chunks.
    # Truncate to the first 8 meaningful words to keep the query achievable.
    bm25_query = " ".join(query_text.split()[:8])

    pool = get_global_async_db_pool()
    if not pool:
        await init_async_pools()

    async with get_db_connection() as conn:
        # Debug: confirm rows exist with bm25_tsvector populated for this project
        debug_row = await conn.fetchrow(
            """
            SELECT
                COUNT(*) FILTER (WHERE bm25_tsvector IS NOT NULL) AS populated,
                COUNT(*) FILTER (WHERE bm25_tsvector IS NULL)     AS nulls,
                COUNT(*)                                           AS total,
                websearch_to_tsquery('english', $2)::text          AS tsquery_text
            FROM document_vector_store
            WHERE project_id = $1::uuid
            """,
            project_id,
            bm25_query,
        )
        logger.info(
            f"🔍 [BM25-DEBUG] project={project_id[:8]} "
            f"query={repr(query_text)[:80]} "
            f"bm25_query={repr(bm25_query)!r} "
            f"tsquery={debug_row['tsquery_text']!r} "
            f"rows={debug_row['total']} populated={debug_row['populated']} nulls={debug_row['nulls']}"
        )

        rows = await conn.fetch(
            """
            SELECT
                id, source_id, project_id, content,
                chunk_summary, section_path, chunk_type, parent_chunk_id,
                metadata, page_number, chunk_index,
                ts_rank_cd(bm25_tsvector,
                    COALESCE(
                        NULLIF(websearch_to_tsquery('english', $2)::text, ''),
                        plainto_tsquery('english', $2)::text
                    )::tsquery,
                    32
                ) AS bm25_score
            FROM document_vector_store
            WHERE project_id = $1::uuid
              AND bm25_tsvector @@ (
                    COALESCE(
                        NULLIF(websearch_to_tsquery('english', $2)::text, ''),
                        plainto_tsquery('english', $2)::text
                    )::tsquery
                  )
              AND ($3::text[]   IS NULL OR chunk_type  = ANY($3))
              AND ($4::uuid[]   IS NULL OR source_id   = ANY($4::uuid[]))
              AND ($5::text     IS NULL OR section_path LIKE $5 || '%')
            ORDER BY bm25_score DESC
            LIMIT $6
            """,
            project_id,
            bm25_query,
            chunk_types,
            source_ids,
            sec_prefix,
            k,
        )

    results = [_row_to_chunk(row, score_key="bm25_score") for row in rows]
    logger.info(f"📖 bm25_search: {len(results)} results (k={k})")
    return results
