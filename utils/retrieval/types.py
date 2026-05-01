# utils/retrieval/types.py

"""
Shared TypedDicts and dataclasses for the retrieval module.

All retrieval functions produce and consume RetrievedChunk dicts so that
callers can mix results from vector_search, bm25_search, find_tables, etc.
without conversion.
"""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict, NotRequired


# ——— Input types —————————————————————————————————————————————————————————————


class SearchFilters(TypedDict, total=False):
    """
    Optional filters applied to all search functions.
    All keys are optional; omitting a key means no filter on that dimension.

    source_ids          — restrict to specific document UUIDs
    chunk_types         — e.g. ['prose', 'table'], defaults to all types
    section_path_prefix — prefix match on section_path hierarchy
                          e.g. "I. Background" matches "I. Background > A. Facts"
    """
    source_ids: List[str]
    chunk_types: List[str]
    section_path_prefix: str


# ——— Output types ————————————————————————————————————————————————————————————


class RetrievedChunk(TypedDict):
    """
    Unified chunk representation returned by all retrieval functions.

    similarity   — cosine similarity from vector search (0–1); None for BM25-only
    bm25_score   — ts_rank_cd score from BM25 search; None for vector-only
    rrf_score    — reciprocal rank fusion score; set by rrf_fuse()
    rerank_score — cross-encoder score; set by rerank()
    """
    id: str
    source_id: str
    project_id: str
    content: str
    chunk_summary: NotRequired[Optional[str]]
    section_path: NotRequired[Optional[str]]
    chunk_type: str
    parent_chunk_id: NotRequired[Optional[str]]
    metadata: NotRequired[Dict[str, Any]]
    page_number: NotRequired[Optional[int]]
    chunk_index: NotRequired[Optional[int]]
    # scoring
    similarity: NotRequired[Optional[float]]
    bm25_score: NotRequired[Optional[float]]
    rrf_score: NotRequired[Optional[float]]
    rerank_score: NotRequired[Optional[float]]


class SectionResult(TypedDict):
    """Returned by find_sections()."""
    id: str
    source_id: str
    project_id: str
    section_path: str
    section_summary: str
    similarity: float
    start_chunk_idx: int
    end_chunk_idx: int


class DocResult(TypedDict):
    """Returned by find_docs()."""
    id: str
    project_id: str
    filename: str
    doc_summary: Optional[str]
    doc_concepts: Optional[List[str]]
    similarity: float
    total_chunks: Optional[int]


# ——— Helpers —————————————————————————————————————————————————————————————————


def vec_to_pg(embedding: List[float]) -> str:
    """Convert a Python float list to the pgvector string literal '[x,y,...]'."""
    return "[" + ",".join(map(str, embedding)) + "]"
