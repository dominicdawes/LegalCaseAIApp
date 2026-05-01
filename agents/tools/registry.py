# agents/tools/registry.py
"""
Per-node tool subsets for the exam-questions agent.

Each LangGraph node gets only the tools it needs — this limits the
tool choice space and reduces hallucination of irrelevant tool calls.
"""

# Planner sees discovery + housekeeping to understand available material
PLANNER_TOOLS = [
    "list_sources",
    "get_doc_outline",
    "find_docs_about",
    "get_doc_metadata",
    "count_tokens",
]

# Source profiler maps each document's key concepts + structure
PROFILER_TOOLS = [
    "get_doc_outline",
    "find_sections_about",
    "find_docs_about",
    "get_doc_metadata",
]

# Retriever pulls the evidence bundles for each question
RETRIEVER_TOOLS = [
    "search_passages",
    "hybrid_search",
    "expand_query",
    "find_sections_about",
    "get_section",
    "get_parents",
    "get_neighbors",
    "find_tables_about",
    "read_table",
    "find_concept_across_docs",
]

# Grounder checks factual claims in drafted questions
VERIFIER_TOOLS = [
    "verify_claim",
    "find_supporting_evidence",
    "get_citations_for",
    "search_passages",
]

# Full set — only used when explicit node routing is unavailable
ALL_TOOLS = [
    "list_sources",
    "get_doc_outline",
    "find_docs_about",
    "find_sections_about",
    "get_section",
    "search_passages",
    "hybrid_search",
    "expand_query",
    "get_parents",
    "get_neighbors",
    "find_tables_about",
    "read_table",
    "query_table",
    "verify_claim",
    "find_supporting_evidence",
    "get_citations_for",
    "find_concept_across_docs",
    "get_doc_metadata",
    "count_tokens",
]
