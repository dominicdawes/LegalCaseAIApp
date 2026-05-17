# agents/quiz/state.py
"""
Typed state for the quiz LangGraph agent.

Design principles (thin-state pattern):
  - Parallel fan-out fields use Annotated[List[T], operator.add] so LangGraph
    concatenates branch results instead of last-writer-wins.
  - Batch loop accumulators (accepted_question_ids, used_question_signatures,
    rejected_question_metadata) are plain NotRequired — batch_commit reads the
    existing list and returns the full accumulated list on each iteration.
  - current_batch_* fields carry only the in-flight batch; batch_commit clears
    them before the next iteration starts.
  - Full question text lives in the database; state carries IDs + short
    signatures to keep the context window bounded across many batches.
"""

import operator
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict, NotRequired


# ── Source profile ─────────────────────────────────────────────────────────────

class QuizSourceProfile(TypedDict):
    source_id: str
    filename: str
    doc_type_guess: str           # full_case_opinion | casebook_excerpt | secondary | other
    document_summary: str
    identified_cases: List[str]   # case name strings found in this document
    key_concepts: List[str]
    has_dissent_guess: bool


# ── Case extract ───────────────────────────────────────────────────────────────

class CaseExtract(TypedDict):
    case_id: str          # short slug, e.g. "case_001"
    source_id: str
    case_name: str
    procedural_posture: str
    facts: str
    legally_relevant_facts: List[str]
    issue: str
    holding: str
    rule: str
    reasoning: str
    dicta: str
    dissent: str
    policy: str
    source_refs: List[str]   # chunk_ids supporting the extraction


# ── Question spec (from blueprint planner) ────────────────────────────────────

class QuestionSpec(TypedDict):
    spec_index: int
    question_type: str          # e.g. "RULE_DISCRIMINATION"
    source_ids: List[str]
    case_names: List[str]       # which cases to draw from
    topic: str                  # specific topic/concept for the question
    difficulty: str             # "recall" | "application" | "analysis"
    distractor_types: List[str] # suggested distractor types for the 3 wrong answers


# ── Batch spec ────────────────────────────────────────────────────────────────

class BatchSpec(TypedDict):
    batch_index: int
    question_specs: List[QuestionSpec]


# ── Draft answer (one of four choices A-D) ────────────────────────────────────

class DraftAnswer(TypedDict):
    choice_letter: str      # "A" | "B" | "C" | "D"
    answer_text: str
    is_correct: bool
    distractor_type: str    # "" for the correct answer; filled by false_trap_generator
    feedback: str           # rationale: why correct / why wrong; stored in quiz_answers.feedback


# ── Draft quiz question (in-flight, current batch only) ───────────────────────

class DraftQuizQuestion(TypedDict):
    spec_index: int
    question_type: str
    question_stem: str
    hint: str
    answers: List[DraftAnswer]  # exactly 4 (A-D), shuffled
    source_refs: List[str]      # chunk_ids used
    grounding_verdict: str      # "" | "pass" | "warn" | "fail"
    grounding_notes: str


# ── Batch-level evaluation result ─────────────────────────────────────────────

class BatchEvaluation(TypedDict):
    batch_index: int
    passes: bool
    scores: Dict[str, float]    # source_grounding, single_correct_answer, distractor_quality, ...
    rejection_reasons: List[str]
    revision_instructions: List[str]
    question_verdicts: List[Dict[str, Any]]  # [{spec_index, passes, reason}, ...]


# ── Full agent state ───────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # ── inputs ───────────────────────────────────────────────────────────────
    request: str
    project_id: str
    source_ids: List[str]
    num_questions: int
    batch_size: int
    quiz_mode: str            # "recall" | "application" | "exam-style" | "mixed"
    target_difficulty: str    # "recall" | "application" | "analysis"
    use_voyage: NotRequired[bool]

    # ── ledger identifiers ────────────────────────────────────────────────────
    job_id: NotRequired[str]   # notes.id — parent quiz container; quiz_questions.quiz_id FK
    run_id: NotRequired[str]
    user_id: NotRequired[str]  # auth.users.id — required by quiz_questions.user_id FK

    # ── parallel fan-out accumulator (operator.add reducer) ───────────────────
    source_profiles: Annotated[List[QuizSourceProfile], operator.add]

    # ── sequential pipeline outputs (replace reducer) ─────────────────────────
    case_extracts:      NotRequired[List[CaseExtract]]
    concept_synthesis:  NotRequired[str]    # JSON: {clusters, confusables, traps}
    batch_specs:        NotRequired[List[BatchSpec]]
    num_batches:        NotRequired[int]

    # ── in-flight batch state (replaced each iteration) ──────────────────────
    current_batch_index:  NotRequired[int]
    current_batch_drafts: NotRequired[List[DraftQuizQuestion]]
    current_batch_eval:   NotRequired[Optional[BatchEvaluation]]
    batch_revision_count: NotRequired[int]   # reset to 0 by batch_commit each iteration

    # ── thin accumulators (batch_commit appends full list each iteration) ─────
    accepted_question_ids:     NotRequired[List[str]]   # quiz_questions.id rows
    rejected_question_metadata: NotRequired[List[Dict[str, Any]]]
    used_question_signatures:  NotRequired[List[str]]   # "TYPE:stem[:40]" for dedup
    coverage_summary:          NotRequired[str]         # JSON: {question_type: count, ...}

    # ── post-loop QA ──────────────────────────────────────────────────────────
    critic_report: NotRequired[str]

    # ── final output ─────────────────────────────────────────────────────────
    persisted_question_ids: NotRequired[List[str]]  # final confirmed IDs
    final_output: NotRequired[str]                  # markdown summary

    # ── budget tracking ───────────────────────────────────────────────────────
    budget: NotRequired[Dict[str, Any]]
