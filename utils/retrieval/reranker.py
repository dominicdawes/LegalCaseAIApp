# utils/retrieval/reranker.py

"""
rerank — cross-encoder reranking of a candidate pool.

Primary:  Voyage Rerank (voyage-rerank-2) — matches the voyage-law-2 embedding family.
Fallback: Cohere Rerank 3 — used when RERANKER=cohere in env or when Voyage is unavailable.

Typical usage in the pipeline:
    candidates = rrf_fuse(vector_results, bm25_results, top_n=50)
    final      = await rerank(query_text, candidates, top_n=10)
"""

import asyncio
import logging
import os
from typing import List, Optional

from .types import RetrievedChunk

logger = logging.getLogger(__name__)

VOYAGE_API_KEY  = os.getenv("VOYAGE_API_KEY",  "").strip()
COHERE_API_KEY  = os.getenv("COHERE_API_KEY",  "").strip()

# Set RERANKER=cohere in env to prefer Cohere over Voyage
_RERANKER = os.getenv("RERANKER", "voyage").lower()

VOYAGE_RERANK_MODEL = "rerank-2"
COHERE_RERANK_MODEL = "rerank-english-v3.0"


# ——— Voyage Rerank ——————————————————————————————————————————————————————————


def _voyage_rerank_sync(
    query: str,
    documents: List[str],
    top_n: int,
) -> List[dict]:
    """
    Call Voyage Rerank synchronously (runs in executor).
    Returns list of {index, relevance_score} dicts ordered by score desc.
    """
    import voyageai
    client = voyageai.Client(api_key=VOYAGE_API_KEY)
    result = client.rerank(
        query=query,
        documents=documents,
        model=VOYAGE_RERANK_MODEL,
        top_k=top_n,
    )
    # result.results is a list of RerankingObject with .index and .relevance_score
    return [{"index": r.index, "relevance_score": r.relevance_score} for r in result.results]


# ——— Cohere Rerank ——————————————————————————————————————————————————————————


def _cohere_rerank_sync(
    query: str,
    documents: List[str],
    top_n: int,
) -> List[dict]:
    """
    Call Cohere Rerank synchronously (runs in executor).
    Returns list of {index, relevance_score} dicts ordered by score desc.
    """
    import cohere
    co = cohere.Client(api_key=COHERE_API_KEY)
    result = co.rerank(
        query=query,
        documents=documents,
        model=COHERE_RERANK_MODEL,
        top_n=top_n,
    )
    return [{"index": r.index, "relevance_score": r.relevance_score} for r in result.results]


# ——— Public API —————————————————————————————————————————————————————————————


async def rerank(
    query: str,
    chunks: List[RetrievedChunk],
    top_n: int = 10,
) -> List[RetrievedChunk]:
    """
    Rerank a candidate pool with a cross-encoder model.

    Tries Voyage Rerank first; falls back to Cohere if configured or Voyage
    errors; falls back to rrf_score / similarity ordering if both fail.

    Args:
        query  — the original user query text
        chunks — candidate pool (typically k=50 from rrf_fuse)
        top_n  — number of results to return

    Returns:
        List[RetrievedChunk] of length ≤ top_n, ordered by rerank_score desc.
    """
    if not chunks:
        return []

    if len(chunks) <= top_n:
        # No point reranking — just return as-is
        return chunks[:top_n]

    documents = [
        (c.get("chunk_summary") or "") + "\n\n" + c.get("content", "")
        for c in chunks
    ]

    loop = asyncio.get_event_loop()
    ranked: Optional[List[dict]] = None

    # ── Primary reranker ──────────────────────────────────────────────────────
    if _RERANKER == "voyage" and VOYAGE_API_KEY:
        try:
            ranked = await loop.run_in_executor(
                None, _voyage_rerank_sync, query, documents, top_n
            )
        except Exception as e:
            logger.warning(f"⚠️ Voyage rerank failed: {e} — falling back to Cohere")

    if ranked is None and COHERE_API_KEY:
        try:
            ranked = await loop.run_in_executor(
                None, _cohere_rerank_sync, query, documents, top_n
            )
        except Exception as e:
            logger.warning(f"⚠️ Cohere rerank failed: {e} — using score order")

    # ── Fallback: sort by existing score ─────────────────────────────────────
    if ranked is None:
        logger.warning("⚠️ rerank: no reranker available; using rrf_score/similarity order")
        fallback = sorted(
            chunks,
            key=lambda c: (c.get("rrf_score") or c.get("similarity") or 0.0),
            reverse=True,
        )
        return fallback[:top_n]

    # ── Reconstruct ordered list with rerank_score ────────────────────────────
    output: List[RetrievedChunk] = []
    for hit in ranked:
        chunk = dict(chunks[hit["index"]])
        chunk["rerank_score"] = hit["relevance_score"]
        output.append(chunk)  # type: ignore[arg-type]

    logger.info(f"🏆 rerank: {len(chunks)} → {len(output)} (reranker={_RERANKER})")
    return output
