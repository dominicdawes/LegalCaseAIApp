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

# Concept synthesizer maps shared concepts across all source documents
SYNTHESIZER_TOOLS = [
    "find_concept_across_docs",
    "search_passages",
]

# ── Attack-outline agent tool subsets ─────────────────────────────────────────

# head_orchestrator sees discovery + housekeeping to survey available material
ATTACK_PLANNER_TOOLS = [
    "list_sources",
    "get_doc_outline",
    "find_docs_about",
    "get_doc_metadata",
    "count_tokens",
]

# source_profiler maps each document's structure and doctrine coverage
ATTACK_PROFILER_TOOLS = [
    "get_doc_outline",
    "find_sections_about",
    "find_docs_about",
    "get_doc_metadata",
]

# research_agent (compact pipeline) — one multistep tool-calling agent that
# compresses profiling + topic mapping + retrieval planning + extraction +
# clustering. Union of the profiler and retriever sets: discovery to survey the
# corpus, then BM25/vector/section drilldown to pull evidence.
ATTACK_RESEARCH_TOOLS = [
    "list_sources",
    "get_doc_outline",
    "find_docs_about",
    "get_doc_metadata",
    "find_sections_about",
    "get_section",
    "get_parents",
    "get_neighbors",
    "search_passages",
    "hybrid_search",
    "expand_query",
    "find_concept_across_docs",
]

# planned_retriever executes multi-intent retrieval (BM25, vector, section drilldown)
ATTACK_RETRIEVER_TOOLS = [
    "search_passages",
    "hybrid_search",
    "expand_query",
    "find_sections_about",
    "get_section",
    "get_parents",
    "get_neighbors",
    "find_concept_across_docs",
]

# legal_artifact_extractor reads chunk context and nearby chunks for artifact grounding
ATTACK_EXTRACTOR_TOOLS = [
    "search_passages",
    "hybrid_search",
    "get_section",
    "get_neighbors",
    "find_concept_across_docs",
]

# grounding_verifier checks claims and fetches citations
ATTACK_VERIFIER_TOOLS = [
    "verify_claim",
    "find_supporting_evidence",
    "get_citations_for",
    "search_passages",
]

# ── Case-brief agent tool subsets ─────────────────────────────────────────────

# head_orchestrator surveys available sources
BRIEF_PLANNER_TOOLS = [
    "list_sources",
    "get_doc_outline",
    "find_docs_about",
    "get_doc_metadata",
    "count_tokens",
]

# source_profiler maps each document's structure and case metadata
BRIEF_PROFILER_TOOLS = [
    "get_doc_outline",
    "find_sections_about",
    "find_docs_about",
    "get_doc_metadata",
]

# research_agent (compact pipeline) — one multistep tool-calling agent that
# compresses profiling + corpus orientation + retrieval planning + retrieval +
# evidence carding + artifact extraction. Union of the profiler and retriever
# sets: discovery to pin down the primary opinion, then case-name-anchored
# keyword/vector retrieval and structural drill-down into it.
BRIEF_RESEARCH_TOOLS = [
    "list_sources",
    "get_doc_outline",
    "get_doc_metadata",
    "find_docs_about",
    "find_sections_about",
    "get_section",
    "get_parents",
    "get_neighbors",
    "search_passages",
    "hybrid_search",
    "expand_query",
    "find_concept_across_docs",
]

# planned_retriever executes multi-method retrieval per brief artifact
BRIEF_RETRIEVER_TOOLS = [
    "search_passages",
    "hybrid_search",
    "expand_query",
    "find_sections_about",
    "get_section",
    "get_parents",
    "get_neighbors",
    "find_concept_across_docs",
]

# section_grounder verifies section claims against source chunks
BRIEF_VERIFIER_TOOLS = [
    "verify_claim",
    "find_supporting_evidence",
    "get_citations_for",
    "search_passages",
]

# ── Cold-call agent tool subsets ─────────────────────────────────────────────

# head_orchestrator surveys available sources
COLD_CALL_PLANNER_TOOLS = [
    "list_sources",
    "get_doc_outline",
    "find_docs_about",
    "get_doc_metadata",
    "count_tokens",
]

# source_profiler inventories each document for cases / statutes / doctrine sections
COLD_CALL_PROFILER_TOOLS = [
    "get_doc_outline",
    "find_sections_about",
    "find_docs_about",
    "get_doc_metadata",
]

# case_rule_extractor + socratic_thread_builder + socratic_answer_agent retrieve evidence
COLD_CALL_RETRIEVER_TOOLS = [
    "search_passages",
    "hybrid_search",
    "expand_query",
    "find_sections_about",
    "get_section",
    "get_parents",
    "get_neighbors",
    "find_concept_across_docs",
]

# grounder_agent verifies answer claims against source
COLD_CALL_VERIFIER_TOOLS = [
    "verify_claim",
    "find_supporting_evidence",
    "get_citations_for",
    "search_passages",
]

# ── Quiz agent tool subsets ───────────────────────────────────────────────────

# head_orchestrator surveys available sources; source_profiler outlines each doc
QUIZ_PLANNER_TOOLS = [
    "list_sources",
    "get_doc_outline",
    "find_docs_about",
    "get_doc_metadata",
    "count_tokens",
]

# source_profiler and case_rule_extractor map each document's structure and cases
QUIZ_PROFILER_TOOLS = [
    "get_doc_outline",
    "find_sections_about",
    "find_docs_about",
    "get_doc_metadata",
]

# question_drafter + grounder retrieve evidence per question spec
QUIZ_RETRIEVER_TOOLS = [
    "search_passages",
    "hybrid_search",
    "expand_query",
    "find_sections_about",
    "get_section",
    "get_parents",
    "get_neighbors",
    "find_concept_across_docs",
]

# grounder verifies question/answer claims against source chunks
QUIZ_VERIFIER_TOOLS = [
    "verify_claim",
    "find_supporting_evidence",
    "get_citations_for",
    "search_passages",
]

# ── Flashcard agent tool subsets ─────────────────────────────────────────────

# head_orchestrator surveys sources; source_profiler outlines each doc
FLASHCARD_PLANNER_TOOLS = [
    "list_sources",
    "get_doc_outline",
    "find_docs_about",
    "get_doc_metadata",
    "count_tokens",
]

# source_profiler maps each document's structure and identified cases
FLASHCARD_PROFILER_TOOLS = [
    "get_doc_outline",
    "find_sections_about",
    "find_docs_about",
    "get_doc_metadata",
]

# concept_extractor + flashcard_drafter retrieve evidence per concept / card spec
FLASHCARD_RETRIEVER_TOOLS = [
    "search_passages",
    "hybrid_search",
    "expand_query",
    "find_sections_about",
    "get_section",
    "get_parents",
    "get_neighbors",
    "find_concept_across_docs",
]

# ── Compact-pipeline research sets ───────────────────────────────────────────
# One multistep tool-calling research agent replaces each legacy pipeline's
# profiler + orientation + retrieval-planning + retrieval + extraction nodes,
# so it needs the union of the profiler and retriever sets: discovery to orient,
# then keyword/vector retrieval and structural drill-down. Mirrors
# ATTACK_RESEARCH_TOOLS / BRIEF_RESEARCH_TOOLS above.

_RESEARCH_UNION = [
    "list_sources",
    "get_doc_outline",
    "get_doc_metadata",
    "find_docs_about",
    "find_sections_about",
    "get_section",
    "get_parents",
    "get_neighbors",
    "search_passages",
    "hybrid_search",
    "expand_query",
    "find_concept_across_docs",
]

COLD_CALL_RESEARCH_TOOLS = list(_RESEARCH_UNION)
EXAM_RESEARCH_TOOLS      = list(_RESEARCH_UNION) + ["find_tables_about"]
QUIZ_RESEARCH_TOOLS      = list(_RESEARCH_UNION)
FLASHCARD_RESEARCH_TOOLS = list(_RESEARCH_UNION)

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
