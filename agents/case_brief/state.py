# agents/case_brief/state.py
"""
Typed state for the case-brief LangGraph agent.

Fields that receive parallel Send fan-out updates use
  Annotated[List[T], operator.add]
so LangGraph concatenates them across branches instead of last-writer-wins.

Fields set once by a sequential node use NotRequired[T] (plain replace).
"""

import operator
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict, NotRequired


# ── Per-source profile ────────────────────────────────────────────────────────

class SectionEntry(TypedDict):
    section_id: str
    heading: str
    summary: str


class SourceProfile(TypedDict):
    source_id: str
    doc_type_guess: str          # full_case_opinion | casebook_excerpt | lecture_notes | secondary | other
    case_name_guess: str
    court_guess: str
    year_guess: str
    has_dissent_guess: bool
    document_summary: str
    section_map: List[SectionEntry]
    confidence: float


# ── Corpus orientation ────────────────────────────────────────────────────────

class SourceRole(TypedDict):
    source_id: str
    role: str                    # primary_opinion | casebook_excerpt | lecture_notes | secondary | duplicate | irrelevant


class CorpusOrientation(TypedDict):
    primary_case_source_id: str
    supporting_source_ids: List[str]
    case_brief_scope: str        # single_case | multi_case | casebook_excerpt | doctrine_packet | mixed_source
    source_roles: List[SourceRole]
    primary_case_name: str
    briefing_notes: str


# ── Retrieval plan per brief artifact ─────────────────────────────────────────

class RetrievalPlan(TypedDict):
    target: str                  # material_facts | procedural_posture | issue | holding | rule | reasoning | dissent | case_identity | pedagogy
    methods: List[str]           # bm25 | vector | section_summary | regex | metadata
    queries: List[str]
    regex_patterns: List[str]
    section_filters: List[str]
    source_filters: List[str]
    max_chunks: int
    priority: str                # high | medium | low


# ── Retrieved and bundled chunks ──────────────────────────────────────────────

class RetrievalBundle(TypedDict):
    target: str
    source_id: str
    chunks: List[Dict[str, Any]]  # each: {chunk_id, content, source_id, page_number, score}


# ── Evidence cards ────────────────────────────────────────────────────────────

class EvidenceCard(TypedDict):
    card_id: str
    role: str                    # holding | facts | rule | reasoning | dissent | posture | issue | citation | pedagogy | exam_trigger
    source_id: str
    chunk_id: str
    page_range: str
    quote: str
    confidence: float


# ── Legal artifact from each parallel extractor ───────────────────────────────

class ExtractedArtifact(TypedDict):
    artifact_type: str           # facts_posture | issue_holding | rule_reasoning | dissent
    content: Dict[str, Any]      # type-specific payload (see below)
    supporting_card_ids: List[str]
    confidence: float

# facts_posture content shape:
#   procedural_posture: {lower_court, current_stage, standard_or_frame, disposition_below}
#   material_facts: [{fact, why_material, supporting_card_ids}]
#   background_facts: [str]
#   uncertainties: [str]
#
# issue_holding content shape:
#   issue: str
#   holding_narrow: str
#   disposition: str  (affirmed|reversed|remanded|vacated)
#   winner: str
#   confidence: float
#
# rule_reasoning content shape:
#   rule: {black_letter, test, elements, exceptions, limitations}
#   reasoning: [{type, text}]  (application|precedent|policy|textual|institutional|fairness|administrability)
#
# dissent content shape:
#   has_dissent: bool
#   has_concurrence: bool
#   dissent_summary: str
#   concurrence_summary: str
#   alternative_rule: str
#   key_disagreement: str


# ── Doctrinal synthesis ───────────────────────────────────────────────────────

class DoctrinalSynthesis(TypedDict):
    doctrinal_role: str          # introduces | refines | limits | overrules | applies | distinguishes
    narrow_holding: str
    broad_rule: str
    rule_limits: List[str]
    exam_triggers: List[str]
    cold_call_traps: List[str]
    common_misreadings: List[str]
    related_doctrines: List[str]
    pedagogical_note: str


# ── Section draft (output of each section_writer) ────────────────────────────

class Claim(TypedDict):
    claim: str
    supporting_card_ids: List[str]


class SectionDraft(TypedDict):
    section_id: str              # case_identity | procedural_posture | facts | issue_holding | rule | reasoning | dissent | pedagogy | exam_translation | cold_call
    title: str
    draft_text: str
    claims: List[Claim]
    word_count: int
    warnings: List[str]


# ── Grounding report per section ──────────────────────────────────────────────

class GroundingReport(TypedDict):
    section_id: str
    grounding_pass: bool
    unsupported_claims: List[str]
    overbroad_claims: List[str]
    missing_evidence: List[str]
    required_fixes: List[str]


# ── Critic output ─────────────────────────────────────────────────────────────

class BriefCritique(TypedDict):
    quality_pass: bool
    scores: Dict[str, float]     # accuracy, briefing_quality, exam_usefulness, citation_discipline
    critique: List[str]
    revise: bool
    sections_to_revise: List[str]
    revision_instructions: str


# ── Full agent state ──────────────────────────────────────────────────────────

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
    source_profiles:     Annotated[List[SourceProfile], operator.add]
    retrieval_bundles:   Annotated[List[RetrievalBundle], operator.add]
    evidence_cards:      Annotated[List[EvidenceCard], operator.add]
    extracted_artifacts: Annotated[List[ExtractedArtifact], operator.add]
    raw_sections:        Annotated[List[SectionDraft], operator.add]
    grounding_reports:   Annotated[List[GroundingReport], operator.add]

    # ── sequential pipeline outputs (replace reducer) ────────────────────────
    corpus_orientation:  NotRequired[CorpusOrientation]
    retrieval_plans:     NotRequired[List[RetrievalPlan]]
    doctrinal_synthesis: NotRequired[DoctrinalSynthesis]
    drafting_manifest:   NotRequired[Dict[str, Any]]

    # final_sections = raw_sections with failed sections replaced by revisions
    final_sections:      NotRequired[List[SectionDraft]]

    assembled_brief:     NotRequired[str]
    coherence_edit:      NotRequired[Dict[str, Any]]  # {edited_markdown, contradictions_found, ...}
    critique:            NotRequired[BriefCritique]
    revision_count:      NotRequired[int]

    # ── final output ─────────────────────────────────────────────────────────
    final_output: NotRequired[str]

    # ── budget tracking ──────────────────────────────────────────────────────
    budget: NotRequired[Dict[str, Any]]
