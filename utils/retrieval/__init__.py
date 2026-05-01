# utils/retrieval/__init__.py
#
# Public API for the hierarchical retrieval module.
# Import from here; internal module structure may change.

from .types import (
    DocResult,
    RetrievedChunk,
    SearchFilters,
    SectionResult,
    vec_to_pg,
)

from .search import (
    bm25_search,
    vector_search,
)

from .fusion import rrf_fuse

from .reranker import rerank

from .parent_expand import parent_expand

from .specialized import (
    find_docs,
    find_sections,
    find_tables,
)

from .query_expansion import multi_query_expand

__all__ = [
    # types
    "DocResult",
    "RetrievedChunk",
    "SearchFilters",
    "SectionResult",
    "vec_to_pg",
    # search
    "vector_search",
    "bm25_search",
    # fusion
    "rrf_fuse",
    # rerank
    "rerank",
    # parent
    "parent_expand",
    # specialized
    "find_tables",
    "find_sections",
    "find_docs",
    # query expansion
    "multi_query_expand",
]
