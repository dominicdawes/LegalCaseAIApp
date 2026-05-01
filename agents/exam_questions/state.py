# agents/exam_questions/state.py
"""
Typed state for the exam-questions LangGraph agent.

All fields use TypedDict so LangGraph can merge partial state updates
from parallel Send branches cleanly.
"""

from typing import Any, Dict, List, Optional
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

    # ── planner output ───────────────────────────────────────────────────────────
    plan: NotRequired[str]                # planner's strategic note

    # ── per-document profiles (populated by parallel SourceProfiler) ──────────
    source_profiles: NotRequired[List[SourceProfile]]

    # ── clustered issues (IssueClusterer output) ─────────────────────────────
    chosen_issues: NotRequired[List[IssueCluster]]

    # ── retrieval results (Retriever parallel fanout) ──────────────────────
    retrieval_bundles: NotRequired[List[RetrievalBundle]]

    # ── drafted questions (QuestionDrafter + AnswerKeyBuilder parallel) ─────
    draft_questions: NotRequired[List[DraftQuestion]]

    # ── verified questions (Grounder + Critic + optional Reviser) ───────────
    verified_questions: NotRequired[List[VerifiedQuestion]]

    # ── final assembled markdown ─────────────────────────────────────────────
    final_output: NotRequired[str]

    # ── budget tracking ──────────────────────────────────────────────────────
    budget: NotRequired[Dict[str, Any]]   # {input_tokens, output_tokens, cost_usd}

    # ── revision loop counter ────────────────────────────────────────────────
    revision_count: NotRequired[int]
