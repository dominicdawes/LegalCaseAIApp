# utils/retrieval/fusion.py

"""
rrf_fuse — Reciprocal Rank Fusion over any number of ranked result lists.

RRF score for a document d across ranked lists L1…Ln:
    score(d) = Σ  1 / (k + rank_i(d))
where k=60 is the standard constant that dampens high-rank outliers.

Usage:
    fused = rrf_fuse(vector_results, bm25_results)
    fused = rrf_fuse(vec_results, bm25_results, query_exp_results, top_n=50)
"""

import logging
from typing import Dict, List, Optional

from .types import RetrievedChunk

logger = logging.getLogger(__name__)

_RRF_K = 60  # standard constant; lowering to 10-20 amplifies rank differences


def rrf_fuse(
    *result_lists: List[RetrievedChunk],
    k: int = _RRF_K,
    top_n: Optional[int] = None,
) -> List[RetrievedChunk]:
    """
    Merge N ranked lists into one using Reciprocal Rank Fusion.

    Args:
        *result_lists — any number of ranked RetrievedChunk lists
        k             — RRF constant (default 60)
        top_n         — truncate output to top_n results; None → return all

    Returns:
        Deduplicated list ordered by descending rrf_score, with rrf_score set.
    """
    if not result_lists:
        return []

    # Accumulate RRF scores and keep one copy of each chunk (first seen wins
    # for metadata; scores accumulate across lists)
    rrf_scores: Dict[str, float] = {}
    canonical: Dict[str, RetrievedChunk] = {}

    for ranked_list in result_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            cid = chunk["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in canonical:
                canonical[cid] = chunk

    # Write rrf_score back and sort
    merged: List[RetrievedChunk] = []
    for cid, score in rrf_scores.items():
        c = dict(canonical[cid])   # shallow copy to avoid mutating caller's list
        c["rrf_score"] = score
        merged.append(c)  # type: ignore[arg-type]

    merged.sort(key=lambda c: c.get("rrf_score") or 0.0, reverse=True)

    if top_n is not None:
        merged = merged[:top_n]

    logger.info(
        f"🔀 rrf_fuse: {sum(len(l) for l in result_lists)} raw → "
        f"{len(merged)} fused (k={k}, top_n={top_n})"
    )
    return merged
