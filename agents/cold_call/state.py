# agents/cold_call/state.py
"""
Typed state for the cold-call LangGraph agent.

Fields that receive parallel Send fan-out updates use
  Annotated[List[T], operator.add]
so LangGraph concatenates them across branches instead of last-writer-wins.

Fields set once by a sequential node use NotRequired[T] (plain replace).
"""

import operator
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict, NotRequired


# ── Source profile (inventory pass) ──────────────────────────────────────────

class SourceProfile(TypedDict):
    source_id: str
    doc_type_guess: str       # full_case_opinion | casebook_excerpt | lecture_notes | secondary | other
    case_name_guess: str
    court_guess: str
    year_guess: str
    has_dissent_guess: bool
    document_summary: str
    identified_cases: List[str]      # list of case name strings found in this source
    identified_statutes: List[str]
    confidence: float


# ── Structured case/rule object ───────────────────────────────────────────────

class RuleObject(TypedDict):
    rule_statement: str
    elements: List[str]
    exceptions: List[str]
    burdens: List[str]
    rule_type: str   # categorical | balancing | element_based | factor_based


class CaseRuleObject(TypedDict):
    case_id: str          # e.g. "case_001"
    source_id: str
    case_name: str
    court: str
    year: str
    procedural_posture: str
    facts: List[str]
    legally_relevant_facts: List[str]
    issue: str
    holding: str
    rule: RuleObject
    reasoning: List[str]
    dicta: List[str]
    dissent: str
    policy_concerns: List[str]
    source_refs: List[str]  # chunk_ids


# ── Doctrine map ──────────────────────────────────────────────────────────────

class DoctrineEdge(TypedDict):
    from_id: str
    to_id: str
    rel_type: str    # DEFINES | DISTINGUISHES | EXPANDS | LIMITS | EXCEPTION_TO | ANALOGOUS_TO | CONFLICTS_WITH
    rationale: str
    source_refs: List[str]


class DoctrineMap(TypedDict):
    topic: str
    subdoctrine: str
    nodes: List[Dict[str, Any]]   # {id, type: case|rule|doctrine, label}
    edges: List[DoctrineEdge]


# ── Question type selection ───────────────────────────────────────────────────

class QuestionTypeSelection(TypedDict):
    selected_question_types: Dict[str, List[str]]   # early/middle/deep → [QUESTION_TYPE_TAG, ...]
    required_types: List[str]
    optional_types: List[str]


# ── Cold-call seed ────────────────────────────────────────────────────────────

class Seed(TypedDict):
    seed_id: str
    case_id: str
    sequence_theme: str     # CORE_UNDERSTANDING | RULE_BOUNDARY | COMPARE_DISTINGUISH | POLICY | EXAM_TRANSFER | COUNTERARGUMENT | HYPO_START
    question_type: str      # LEGALLY_RELEVANT_FACTS | RULE_EXTRACTION | FACT_CHANGE_HYPO | ...
    question: str
    target_skill: str
    difficulty: str         # early | middle | deep


# ── Individual question in a sequence ────────────────────────────────────────

class SequenceQuestion(TypedDict):
    question_id: str        # e.g. "1A", "1B"
    question_index: int
    question_type: str
    depth_position: str     # early | middle | deep
    question_text: str
    target_skill: str
    difficulty_label: str   # easy | medium | hard (added by critic)
    expected_answer_shape: str
    source_refs: List[str]
    metadata: Dict[str, Any]


# ── Full question sequence ────────────────────────────────────────────────────

class QuestionSequence(TypedDict):
    sequence_id: str
    case_id: str
    case_name: str
    seed_id: str
    sequence_theme: str
    sequence_index: int
    questions: List[SequenceQuestion]
    coverage_tags: List[str]
    source_refs: List[str]
    metadata: Dict[str, Any]


# ── Answer for one question ───────────────────────────────────────────────────

class QuestionAnswer(TypedDict):
    question_id: str
    model_answer: str       # Because / Unless / But / Therefore format
    strong_answer: str
    common_weak_answer: str
    professor_follow_up_trap: str
    recovery_phrase: str


# ── Full answer sequence ──────────────────────────────────────────────────────

class AnswerSequence(TypedDict):
    sequence_id: str
    answers: List[QuestionAnswer]


# ── Grounding result ──────────────────────────────────────────────────────────

class GroundingResult(TypedDict):
    sequence_id: str
    grounding_status: str   # passed | passed_with_warnings | failed
    claims_checked: List[Dict[str, Any]]
    unsupported_claim_count: int


# ── Compare-and-distinguish prompt ───────────────────────────────────────────

class CompareDistinguishPrompt(TypedDict):
    comparison_id: str
    case_a_id: str
    case_b_id: str
    case_a_name: str
    case_b_name: str
    relationship: str
    question: str
    model_distinction: str
    source_refs: List[str]


# ── Coverage map ──────────────────────────────────────────────────────────────

class CoverageMap(TypedDict):
    overall_coverage_score: float
    coverage_by_question_type: Dict[str, int]
    coverage_by_depth: Dict[str, int]
    missing_or_weak_areas: List[str]
    difficulty_labels_added: bool


# ── Export result ─────────────────────────────────────────────────────────────

class ExportResult(TypedDict):
    export_batch_id: str
    question_sequences_exported: int
    answer_sequences_exported: int
    questions_exported: int  # total Q/A pairs across all sequences (deterministic)


# ── Full agent state ──────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # ── inputs ──────────────────────────────────────────────────────────────
    request: str
    project_id: str
    source_ids: List[str]
    note_id: NotRequired[str]
    requested_sequence_count: int
    target_difficulty: str       # law_1l | day_one_t14 | advanced
    use_voyage: NotRequired[bool]

    # ── ledger identifiers ───────────────────────────────────────────────────
    job_id: NotRequired[str]
    run_id: NotRequired[str]
    user_id: NotRequired[str]

    # ── orchestrator output ──────────────────────────────────────────────────
    job_plan: NotRequired[Dict[str, Any]]

    # ── parallel fan-out accumulators (operator.add reducer) ─────────────────
    source_profiles:    Annotated[List[SourceProfile], operator.add]
    case_rule_objects:  Annotated[List[CaseRuleObject], operator.add]
    seeds:              Annotated[List[Seed], operator.add]
    socratic_sequences: Annotated[List[QuestionSequence], operator.add]
    answer_sequences:   Annotated[List[AnswerSequence], operator.add]
    grounding_results:  Annotated[List[GroundingResult], operator.add]

    # ── sequential pipeline outputs (replace reducer) ────────────────────────
    corpus_analysis:              NotRequired[Dict[str, Any]]   # {cases: [...], doctrine_sections: [...]}
    doctrine_map:                 NotRequired[DoctrineMap]
    compare_distinguish_prompts:  NotRequired[List[CompareDistinguishPrompt]]
    question_type_selection:      NotRequired[QuestionTypeSelection]
    approved_seeds:               NotRequired[List[Seed]]
    seed_attempt_count:           NotRequired[int]

    # final_sequences = socratic_sequences with difficulty_label stamped by critic
    final_sequences:    NotRequired[List[QuestionSequence]]

    coverage_map:       NotRequired[CoverageMap]
    export_result:      NotRequired[ExportResult]

    # ── final output ─────────────────────────────────────────────────────────
    final_output: NotRequired[str]

    # ── budget tracking ──────────────────────────────────────────────────────
    budget: NotRequired[Dict[str, Any]]
