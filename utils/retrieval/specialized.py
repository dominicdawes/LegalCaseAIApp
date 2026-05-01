# utils/retrieval/specialized.py

"""
Specialized retrieval functions for the hierarchical index layers.

find_tables   — vector search scoped to chunk_type='table'
find_sections — vector search against document_sections embeddings
find_docs     — vector search against document_sources doc_summary embeddings
               (falls back to keyword match on doc_concepts when no embedding exists)
"""

import json
import logging
import os
from typing import List, Optional

from tasks.database import get_db_connection, get_global_async_db_pool, init_async_pools
from .types import DocResult, RetrievedChunk, SearchFilters, SectionResult, vec_to_pg
from .search import _row_to_chunk

logger = logging.getLogger(__name__)

USE_VOYAGE_EMBEDDINGS = os.getenv("USE_VOYAGE_EMBEDDINGS", "false").lower() == "true"


# ——— find_tables ————————————————————————————————————————————————————————————


async def find_tables(
    query_embedding: List[float],
    project_id: str,
    k: int = 10,
    filters: Optional[SearchFilters] = None,
) -> List[RetrievedChunk]:
    """
    Vector search restricted to chunk_type='table'.

    Table chunks embed their LLM-written summary (not the raw markdown),
    so this search finds tables by *meaning*, not keyword.
    At generation time, small tables (≤30 rows, ≤8 cols) are rendered as
    markdown inline; larger ones use the chunk_summary with on-demand row
    quoting.

    Returns chunks with table_data available in metadata['table'].
    """
    table_filter: SearchFilters = dict(filters or {})  # type: ignore[assignment]
    table_filter["chunk_types"] = ["table"]

    # Reuse vector_search with the table filter
    from .search import vector_search
    results = await vector_search(
        query_embedding=query_embedding,
        project_id=project_id,
        k=k,
        filters=table_filter,
    )
    logger.info(f"📊 find_tables: {len(results)} table chunks")
    return results


# ——— find_sections ————————————————————————————————————————————————————————————


async def find_sections(
    query_embedding: List[float],
    project_id: str,
    k: int = 5,
    source_ids: Optional[List[str]] = None,
) -> List[SectionResult]:
    """
    Vector search against document_sections (mid-level summaries).

    Sections sit between document-level and chunk-level in the hierarchy.
    Use find_sections to identify which parts of a document are relevant
    before pulling individual chunks via vector_search with section_path_prefix.

    Returns:
        List[SectionResult] ordered by descending cosine similarity.
    """
    vector_str = vec_to_pg(query_embedding)

    pool = get_global_async_db_pool()
    if not pool:
        await init_async_pools()

    async with get_db_connection() as conn:
        rows = await conn.fetch(
            """
            SELECT
                id, source_id, project_id,
                section_path, section_summary,
                start_chunk_idx, end_chunk_idx,
                1 - (embedding <=> $1::vector) AS similarity
            FROM document_sections
            WHERE project_id = $2::uuid
              AND ($3::uuid[] IS NULL OR source_id = ANY($3::uuid[]))
              AND embedding IS NOT NULL
            ORDER BY embedding <=> $1::vector
            LIMIT $4
            """,
            vector_str,
            project_id,
            source_ids,
            k,
        )

    results: List[SectionResult] = [
        {
            "id":              str(row["id"]),
            "source_id":       str(row["source_id"]),
            "project_id":      str(row["project_id"]),
            "section_path":    row["section_path"] or "",
            "section_summary": row["section_summary"] or "",
            "similarity":      float(row["similarity"]) if row["similarity"] is not None else 0.0,
            "start_chunk_idx": row["start_chunk_idx"] or 0,
            "end_chunk_idx":   row["end_chunk_idx"] or 0,
        }
        for row in rows
    ]
    logger.info(f"🗂️  find_sections: {len(results)} sections")
    return results


# ——— find_docs ——————————————————————————————————————————————————————————————


async def find_docs(
    query_embedding: List[float],
    project_id: str,
    k: int = 5,
) -> List[DocResult]:
    """
    Document-level retrieval using doc_summary embeddings stored in document_sources.

    The doc_summary is a ~140-char sentence describing the document's primary legal
    topic.  This is the coarsest layer of the hierarchy; use it to identify which
    source documents are relevant before drilling into sections or chunks.

    Note: doc_summary embeddings are NOT stored in document_sources yet (Phase 1
    stores the text only).  This function embeds the doc_summary on-the-fly using
    the active embedding model and performs an in-process cosine similarity sort.
    A dedicated embedding column can be added in a later migration if this becomes
    a bottleneck.

    Returns:
        List[DocResult] ordered by descending similarity to the query.
    """
    pool = get_global_async_db_pool()
    if not pool:
        await init_async_pools()

    async with get_db_connection() as conn:
        rows = await conn.fetch(
            """
            SELECT id, filename, doc_summary, doc_concepts, total_chunks
            FROM document_sources
            WHERE project_id = $1::uuid
              AND vector_embed_status = 'COMPLETE'
              AND doc_summary IS NOT NULL
            LIMIT 100
            """,
            project_id,
        )

    if not rows:
        return []

    # Embed all doc_summaries and compute cosine similarity in-process
    summaries = [row["doc_summary"] for row in rows]
    similarities = _cosine_similarities(query_embedding, summaries)

    docs: List[DocResult] = []
    for row, sim in zip(rows, similarities):
        docs.append({
            "id":           str(row["id"]),
            "project_id":   project_id,
            "filename":     row["filename"] or "",
            "doc_summary":  row["doc_summary"],
            "doc_concepts": row["doc_concepts"] or [],
            "similarity":   sim,
            "total_chunks": row["total_chunks"],
        })

    docs.sort(key=lambda d: d["similarity"], reverse=True)
    docs = docs[:k]

    logger.info(f"📚 find_docs: {len(docs)} documents")
    return docs


# ——— Cosine similarity helper ————————————————————————————————————————————————


def _cosine_similarities(query: List[float], texts: List[str]) -> List[float]:
    """
    Embed a list of text strings and return cosine similarities against query.
    Uses the active embedding model (Voyage or OpenAI).
    """
    import asyncio

    async def _embed():
        if USE_VOYAGE_EMBEDDINGS:
            from utils.llm_clients.voyage_client import VoyageEmbeddingsClient
            loop = asyncio.get_event_loop()
            client = VoyageEmbeddingsClient()
            return await loop.run_in_executor(None, client.embed_documents, texts)
        else:
            import httpx
            import os as _os
            vecs = []
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                async with httpx.AsyncClient(timeout=60.0) as http:
                    resp = await http.post(
                        "https://api.openai.com/v1/embeddings",
                        headers={"Authorization": f"Bearer {_os.getenv('OPENAI_API_KEY')}",
                                 "Content-Type": "application/json"},
                        json={"model": "text-embedding-ada-002", "input": batch},
                    )
                    resp.raise_for_status()
                    vecs.extend([item["embedding"] for item in resp.json()["data"]])
            return vecs

    loop = asyncio.get_event_loop()
    try:
        doc_vecs = loop.run_until_complete(_embed())
    except RuntimeError:
        # Already inside a running event loop — use nest_asyncio pattern
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            doc_vecs = pool.submit(asyncio.run, _embed()).result()

    import math

    def dot(a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def norm(v: List[float]) -> float:
        return math.sqrt(sum(x * x for x in v))

    q_norm = norm(query)
    sims = []
    for vec in doc_vecs:
        v_norm = norm(vec)
        if q_norm == 0 or v_norm == 0:
            sims.append(0.0)
        else:
            sims.append(dot(query, vec) / (q_norm * v_norm))
    return sims
