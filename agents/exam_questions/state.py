# agents/exam_questions/state.py
"""
Typed state for the exam-questions LangGraph agent.

All fields use TypedDict so LangGraph can merge partial state updates
from parallel Send branches cleanly.
"""

import operator
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict, NotRequired


class SourceProfile(TypedDict):
    source_id: str
    filename: str
    doc_summary: str
    key_concepts: List[str]
    toc: List[Any]


class IssueCluster(TypedDict):
    issue_label: str          # e.g. "Negligence — duty of care"
    source_ids: List[str]     # which docs contain relevant material
    section_hints: List[str]  # suggested section_paths to retrieve
    priority: int             # 1 = high, 2 = medium, 3 = low


class RetrievalBundle(TypedDict):
    issue_label: str
    chunks: List[Dict[str, Any]]   # RetrievedChunk dicts
    table_ids: List[str]           # relevant table chunk IDs


class DraftQuestion(TypedDict):
    question_index: int
    issue_label: str
    fact_pattern: str
    call_of_question: str
    answer_key: str
    chunk_ids_used: List[str]


class VerifiedQuestion(TypedDict):
    question_index: int
    fact_pattern: str
    call_of_question: str
    answer_key: str
    grounding_verdict: str     # "pass" | "warn" | "fail"
    grounding_notes: str
    citations: List[Dict[str, str]]
    revised: bool


class AgentState(TypedDict):
    # ── inputs ──────────────────────────────────────────────────────────────────
    request: str                          # raw user request string
    project_id: str
    source_ids: List[str]
    n_questions: int
    use_voyage: NotRequired[bool]

    # ── ledger identifiers (injected at invocation, read by nodes) ──────────────
    job_id: NotRequired[str]             # agent_jobs.id — used to save artifacts; doubles as notes.id for exam card persistence
    run_id: NotRequired[str]             # agent_runs.id — used for progress tracking
    user_id: NotRequired[str]            # auth.users.id — required by exam_card_writer for DB inserts

    # ── planner output ───────────────────────────────────────────────────────────
    plan: NotRequired[str]                # planner's strategic note

    # ── cross-document concept synthesis (ConceptSynthesizer output) ──────────
    concept_synthesis: NotRequired[str]   # JSON: {throughlines, shared_concepts}

    # ── per-document profiles (populated by parallel SourceProfiler) ──────────
    source_profiles: Annotated[List[SourceProfile], operator.add]

    # ── clustered issues (IssueClusterer output) ─────────────────────────────
    chosen_issues: NotRequired[List[IssueCluster]]

    # ── retrieval results (Retriever parallel fanout) ──────────────────────
    retrieval_bundles: Annotated[List[RetrievalBundle], operator.add]

    # ── drafted questions (QuestionDrafter + AnswerKeyBuilder parallel) ─────
    draft_questions: Annotated[List[DraftQuestion], operator.add]

    # ── grounded questions (Grounder parallel fan-out accumulator) ───────────
    # Separate from verified_questions so Critic can replace verified_questions
    # without fighting the operator.add reducer during the revision loop.
    grounded_questions: Annotated[List[VerifiedQuestion], operator.add]

    # ── verified questions (Critic + optional Reviser + FinalDrafter) ────────
    # Plain field — sequential nodes replace the whole list each pass.
    verified_questions: NotRequired[List[VerifiedQuestion]]

    # ── persisted exam card IDs (exam_card_writer parallel fan-out) ─────────
    persisted_question_ids: Annotated[List[str], operator.add]

    # ── final assembled markdown ─────────────────────────────────────────────
    final_output: NotRequired[str]

    # ── budget tracking ──────────────────────────────────────────────────────
    budget: NotRequired[Dict[str, Any]]   # {input_tokens, output_tokens, cost_usd}

    # ── revision loop counter ────────────────────────────────────────────────
    revision_count: NotRequired[int]
