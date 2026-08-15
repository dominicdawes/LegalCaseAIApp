# agents/attack_outline/state.py
"""
Typed state for the attack-outline LangGraph agent.

Fields that receive parallel Send fan-out updates use
  Annotated[List[T], operator.add]
so LangGraph concatenates them across branches instead of last-writer-wins.

Fields that are set once by a sequential node use NotRequired[T] (plain replace).
"""

import operator
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict, NotRequired


# ── Per-document source profile ────────────────────────────────────────────────

class ExamDoctrine(TypedDict):
    doctrine: str
    sections: List[str]
    priority: str          # "high" | "medium" | "low"
    reason: str


class SectionEntry(TypedDict):
    section_id: str
    heading: str
    summary: str


class SourceProfile(TypedDict):
    source_id: str
    course_area: str
    document_type: str           # lecture_notes | casebook | outline | statute | other
    document_summary: str
    likely_exam_doctrines: List[ExamDoctrine]
    section_map: List[SectionEntry]


# ── Corpus-level topic map ─────────────────────────────────────────────────────

class TopicEntry(TypedDict):
    topic_id: str
    label: str
    source_ids: List[str]
    sections: List[str]
    priority: int               # 1=high, 2=medium, 3=low


# ── Retrieval plan per concept ─────────────────────────────────────────────────

class RetrievalIntent(TypedDict):
    intent: str                 # black_letter_rule | elements_test | exceptions | issue_triggers | cases | policy | exam_traps
    bm25_queries: List[str]
    vector_queries: List[str]
    regex_patterns: List[str]


class ConceptRetrievalPlan(TypedDict):
    concept_id: str
    concept_label: str
    retrieval_intents: List[RetrievalIntent]
    section_filters: List[str]
    source_ids: List[str]


# ── Retrieved and ranked chunk bundles ────────────────────────────────────────

class RankedBundle(TypedDict):
    concept_id: str
    concept_label: str
    chunks: List[Dict[str, Any]]   # each: {chunk_id, content, source_id, page_number, score}


# ── Legal artifacts extracted from chunks ─────────────────────────────────────

class SourceRef(TypedDict):
    source_id: str
    chunk_id: str
    page: int


class LegalArtifact(TypedDict):
    artifact_type: str           # rule_card | element_card | exception_card | issue_trigger_card | defense_card | case_card | policy_card | remedy_card | exam_trap_card
    concept_id: str
    text: str
    elements: List[str]
    exceptions: List[str]
    source_refs: List[SourceRef]
    confidence: float


# ── Normalised / deduplicated artifacts ───────────────────────────────────────

class NormalizedArtifact(TypedDict):
    concept_id: str
    canonical_name: str
    rules: List[str]
    elements: List[str]
    exceptions: List[str]
    conflicts: List[str]
    source_refs: List[SourceRef]


# ── Concept clusters (doctrine modules) ───────────────────────────────────────

class ConceptCluster(TypedDict):
    cluster_id: str
    label: str
    artifact_ids: List[str]       # concept_ids included
    parent_topic: str
    priority: int


# ── Doctrine graph ────────────────────────────────────────────────────────────

class DoctrineNode(TypedDict):
    id: str
    label: str


class DoctrineEdge(TypedDict):
    from_node: str              # renamed from "from" (reserved keyword)
    to_node: str
    condition: str
    transition_text: str


class DoctrineGraph(TypedDict):
    nodes: List[DoctrineNode]
    edges: List[DoctrineEdge]


# ── Attack block (one per doctrine) ───────────────────────────────────────────

class AttackStep(TypedDict):
    step: int
    label: str
    rule: str
    ask: List[str]
    arguments_for: List[str]
    arguments_against: List[str]
    source_refs: List[str]      # chunk_ids
    # T-14 depth fields (optional — gracefully absent from pre-upgrade blocks)
    key_facts_for: NotRequired[List[str]]      # key facts supporting this element
    key_facts_against: NotRequired[List[str]]  # key facts cutting against
    exam_analysis: NotRequired[str]            # 2-3 sentence application paragraph
    if_then_logic: NotRequired[List[str]]      # ["If X → Then Y", ...]


class AttackBlock(TypedDict):
    concept_id: str
    title: str
    trigger: str
    attack_steps: List[AttackStep]
    exceptions_or_limits: List[str]
    exam_traps: List[str]
    source_refs: List[str]      # chunk_ids
    revised: bool               # set True by revision_agent
    # T-14 depth fields (optional — gracefully absent from pre-upgrade blocks)
    big_exam_takeaway: NotRequired[str]           # 1-2 sentence doctrine overview
    claims_and_defenses: NotRequired[List[str]]   # causes of action / defenses covered
    elements_checklist: NotRequired[List[str]]    # flat list of all required elements
    exam_ready_rule_statement: NotRequired[str]   # complete single-sentence rule for exam use
    one_paragraph_application: NotRequired[str]   # model application paragraph


# ── Grounding verification report per block ───────────────────────────────────

class VerificationReport(TypedDict):
    block_id: str               # concept_id of the block
    supported_claims: List[str]
    unsupported_claims: List[str]
    weak_claims: List[str]
    missing_citations: List[str]
    overall_verdict: str        # "pass" | "warn" | "fail"


# ── Critic output ─────────────────────────────────────────────────────────────

class CritiqueResult(TypedDict):
    rule_density_score: float             # 0–10
    checklist_structure_score: float
    issue_trigger_score: float
    exception_coverage_score: float
    counterargument_score: float
    depth_and_completeness_score: float   # replaces concision_score — rewards T-14 depth
    t14_format_compliance_score: float    # rewards correct T-14 section structure
    overall_score: float
    must_revise: bool
    revision_targets: List[str]           # concept_ids of blocks to revise
    revision_instructions: str


# ── Full agent state ───────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # ── inputs ──────────────────────────────────────────────────────────────
    request: str
    project_id: str
    source_ids: List[str]
    use_voyage: NotRequired[bool]

    # ── ledger identifiers ───────────────────────────────────────────────────
    job_id: NotRequired[str]
    run_id: NotRequired[str]
    user_id: NotRequired[str]

    # ── head_orchestrator output ─────────────────────────────────────────────
    job_plan: NotRequired[Dict[str, Any]]

    # ── parallel fan-out accumulation (operator.add reducer) ─────────────────
    # Each parallel branch appends its result; sequential nodes read the full list.
    source_profiles: Annotated[List[SourceProfile], operator.add]
    retrieval_bundles: Annotated[List[RankedBundle], operator.add]
    raw_artifacts: Annotated[List[LegalArtifact], operator.add]
    raw_blocks: Annotated[List[AttackBlock], operator.add]
    verification_reports: Annotated[List[VerificationReport], operator.add]

    # ── sequential pipeline outputs (replace reducer) ────────────────────────
    topic_map: NotRequired[List[TopicEntry]]
    retrieval_plans: NotRequired[List[ConceptRetrievalPlan]]
    normalized_artifacts: NotRequired[List[NormalizedArtifact]]
    concept_clusters: NotRequired[List[ConceptCluster]]
    doctrine_graph: NotRequired[DoctrineGraph]

    # attack_blocks = sorted working list set by assembler, updated by revision_agent
    attack_blocks: NotRequired[List[AttackBlock]]

    # assembled_outline = markdown outline, set by assembler, updated by revision_agent
    assembled_outline: NotRequired[str]

    # ── critique + revision loop ─────────────────────────────────────────────
    critique: NotRequired[CritiqueResult]
    revision_count: NotRequired[int]

    # ── final output ─────────────────────────────────────────────────────────
    final_output: NotRequired[str]

    # ── budget tracking ──────────────────────────────────────────────────────
    budget: NotRequired[Dict[str, Any]]

    # ── compact 4-stage pipeline (compact_nodes.py) ──────────────────────────
    # research_agent output: clusters + artifact cards + evidence references.
    research_dossier: NotRequired[Dict[str, Any]]
    # Every chunk payload harvested from the research agent's tool calls,
    # keyed by chunk_id — generators ground against this without re-retrieving.
    evidence_store: NotRequired[Dict[str, Dict[str, Any]]]
    # block_generator fan-out accumulation (one dict per cluster section).
    compact_blocks: Annotated[List[Dict[str, Any]], operator.add]
