# utils/retrieval/query_expansion.py

"""
multi_query_expand — generate N query rephrasings with a cheap LLM,
                     then run vector_search for each and fuse with RRF.

This implements the "multi-query rephrase → RRF" pattern from the spec.
HyDE (Hypothetical Document Embeddings) is noted as optional and not
included here — add it in Phase 3 if needed.

Typical usage in the pipeline:
    all_chunks = await multi_query_expand(
        query="elements of promissory estoppel",
        project_id=project_id,
        source_ids=source_ids,
        k_per_query=30,
        top_n=50,
    )
    final = await rerank(query, all_chunks, top_n=10)
"""

import asyncio
import logging
import os
from typing import List, Optional

import httpx

from .types import RetrievedChunk, SearchFilters

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
_EXPAND_MODEL = "claude-haiku-4-5-20251001"


# ——— LLM rephrase helper ————————————————————————————————————————————————————


async def _rephrase_query(query: str, n: int = 3) -> List[str]:
    """
    Ask Haiku to generate n rephrasings of the query.
    Returns [original] + rephrasings (deduplicated).
    """
    if not ANTHROPIC_API_KEY:
        logger.warning("⚠️ ANTHROPIC_API_KEY not set — returning original query only")
        return [query]

    prompt = (
        f"Generate {n} alternative phrasings of the following legal research query. "
        "Each rephrasing should capture the same intent but use different terminology "
        "or framing that might appear in legal documents, case law, or academic writing.\n\n"
        f"Original query: {query}\n\n"
        f"Return exactly {n} alternatives as a JSON array of strings, e.g. "
        '["alt 1", "alt 2", "alt 3"]. No extra text.'
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": _EXPAND_MODEL,
                    "max_tokens": 256,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            raw = resp.json()["content"][0]["text"].strip()

        import json as _json
        alts: List[str] = _json.loads(raw)
        if not isinstance(alts, list):
            raise ValueError("not a list")
        alts = [str(a).strip() for a in alts if a][:n]
    except Exception as e:
        logger.warning(f"⚠️ Query expansion failed: {e}")
        alts = []

    # Original first, then rephrasings, deduplicated
    seen = {query.strip().lower()}
    result = [query]
    for alt in alts:
        if alt.lower() not in seen:
            seen.add(alt.lower())
            result.append(alt)
    return result


# ——— Public API —————————————————————————————————————————————————————————————


async def multi_query_expand(
    query: str,
    project_id: str,
    query_embedding: Optional[List[float]] = None,
    n: int = 3,
    k_per_query: int = 30,
    top_n: int = 50,
    filters: Optional[SearchFilters] = None,
) -> List[RetrievedChunk]:
    """
    Expand the query into n+1 variants, run vector_search for each,
    and fuse with RRF.

    Args:
        query           — original user query (text)
        project_id      — project scope
        query_embedding — pre-computed embedding for the original query;
                          if None it will be computed here
        n               — number of rephrasings to generate
        k_per_query     — candidate pool size per query variant
        top_n           — final pool size after RRF fusion
        filters         — passed through to vector_search

    Returns:
        List[RetrievedChunk] of length ≤ top_n, ordered by rrf_score.
    """
    from .search import vector_search
    from .fusion import rrf_fuse

    # ── 1. Generate query variants ────────────────────────────────────────────
    variants = await _rephrase_query(query, n=n)
    logger.info(f"🔤 multi_query_expand: {len(variants)} query variants")

    # ── 2. Embed all variants ────────────────────────────────────────────────
    embeddings = await _embed_queries(variants, query_embedding)

    # ── 3. Parallel vector search for each variant ───────────────────────────
    search_tasks = [
        vector_search(emb, project_id, k=k_per_query, filters=filters)
        for emb in embeddings
    ]
    results_per_variant = await asyncio.gather(*search_tasks, return_exceptions=True)

    valid_lists: List[List[RetrievedChunk]] = []
    for i, res in enumerate(results_per_variant):
        if isinstance(res, Exception):
            logger.warning(f"⚠️ vector_search failed for variant {i}: {res}")
        else:
            valid_lists.append(res)

    if not valid_lists:
        return []

    # ── 4. Fuse with RRF ──────────────────────────────────────────────────────
    fused = rrf_fuse(*valid_lists, top_n=top_n)
    logger.info(f"🔀 multi_query_expand: {sum(len(l) for l in valid_lists)} raw → {len(fused)} fused")
    return fused


# ——— Embedding helper ————————————————————————————————————————————————————————


async def _embed_queries(
    queries: List[str],
    first_embedding: Optional[List[float]] = None,
) -> List[List[float]]:
    """
    Embed all query variants.
    Reuses first_embedding for the first query to avoid a redundant API call.
    """
    if not queries:
        return []

    start_idx = 0
    embeddings: List[Optional[List[float]]] = [None] * len(queries)

    if first_embedding is not None:
        embeddings[0] = first_embedding
        start_idx = 1

    remaining = queries[start_idx:]
    if not remaining:
        return embeddings  # type: ignore[return-value]

    loop = asyncio.get_event_loop()
    use_voyage = os.getenv("USE_VOYAGE_EMBEDDINGS", "false").lower() == "true"

    if use_voyage:
        from utils.llm_clients.voyage_client import VoyageEmbeddingsClient
        client = VoyageEmbeddingsClient()

        async def _voyage_embed():
            return await loop.run_in_executor(None, client.embed_documents, remaining)

        vecs = await _voyage_embed()
    else:
        async with httpx.AsyncClient(timeout=60.0) as http:
            resp = await http.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}",
                    "Content-Type": "application/json",
                },
                json={"model": "text-embedding-ada-002", "input": remaining},
            )
            resp.raise_for_status()
            vecs = [item["embedding"] for item in resp.json()["data"]]

    for i, vec in enumerate(vecs):
        embeddings[start_idx + i] = vec

    return embeddings  # type: ignore[return-value]
