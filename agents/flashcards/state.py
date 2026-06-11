# agents/flashcards/state.py
"""
Typed state for the flashcard LangGraph agent.

Design: thin-state pattern — full card text lives in the DB; state carries
IDs, short signatures, and small metadata only.

Parallel fan-out fields use Annotated[List[T], operator.add] so LangGraph
concatenates branch results instead of last-writer-wins.  This applies to
both the source_profiler fan-out AND the parallel process_batch fan-out.
"""

import operator
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict, NotRequired


# ── Source profile ─────────────────────────────────────────────────────────────

class FlashcardSourceProfile(TypedDict):
    source_id: str
    filename: str
    source_type: str           # case_opinion | casebook_excerpt | class_notes |
                               # attack_outline | doctrine_summary | statute | other
    document_summary: str
    identified_cases: List[str]
    key_concepts: List[str]
    key_rules: List[str]


# ── Blueprint planning ─────────────────────────────────────────────────────────

class FlashcardCardSpec(TypedDict):
    spec_index: int            # global 0-based index across all batches
    card_type: str
    source_ids: List[str]
    case_names: List[str]
    topic: str
    difficulty: str            # "recall" | "application" | "analysis"


class FlashcardBatchSpec(TypedDict):
    batch_index: int
    card_specs: List[FlashcardCardSpec]


# ── Draft card (in-flight, current batch only) ─────────────────────────────────

class DraftFlashcard(TypedDict):
    spec_index: int
    card_type: str
    front_content: str
    back_content: str
    hint: str
    source_refs: List[str]     # chunk_ids used for grounding
    grounding_verdict: str     # "" | "pass" | "warn" | "fail"
    grounding_notes: str


# ── Batch-level evaluation ─────────────────────────────────────────────────────

class FlashcardBatchEvaluation(TypedDict):
    batch_index: int
    passes: bool                            # True if ≥80% of cards pass individually
    scores: Dict[str, float]               # batch-level: uniqueness only
    batch_uniqueness_ok: bool
    rejection_reasons: List[str]
    revision_instructions: List[str]
    # Per-card evaluations: {spec_index, passes, scores:{vagueness,atomic_focus,answer_quality},
    #                        source_grounding_notes, reason}
    card_verdicts: List[Dict[str, Any]]


# ── Parallel batch result (returned by process_batch fan-out) ─────────────────

class BatchResult(TypedDict):
    batch_index: int
    card_ids: List[str]
    coverage: Dict[str, int]   # card_type → count for this batch


# ── Full agent state ───────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # ── inputs ───────────────────────────────────────────────────────────────
    request: str
    project_id: str
    source_ids: List[str]
    num_cards: int
    batch_size: int
    is_essential: NotRequired[bool]
    use_voyage: NotRequired[bool]

    # ── ledger identifiers ────────────────────────────────────────────────────
    job_id: NotRequired[str]
    run_id: NotRequired[str]
    user_id: NotRequired[str]

    # ── parallel fan-out accumulators (operator.add reducer) ──────────────────
    source_profiles: Annotated[List[FlashcardSourceProfile], operator.add]

    # Per-batch parallel fan-out results (process_batch → global_deck_critic)
    accepted_card_ids:      Annotated[List[str], operator.add]
    used_card_signatures:   Annotated[List[str], operator.add]
    rejected_card_metadata: Annotated[List[Dict[str, Any]], operator.add]
    batch_results:          Annotated[List[BatchResult], operator.add]

    # ── sequential pipeline outputs ───────────────────────────────────────────
    concept_inventory: NotRequired[str]
    concept_synthesis: NotRequired[str]
    batch_specs: NotRequired[List[FlashcardBatchSpec]]
    num_batches: NotRequired[int]

    # ── Send payload: injected by card_blueprint_planner_to_batch ─────────────
    current_batch_spec: NotRequired[FlashcardBatchSpec]

    # ── legacy in-flight batch state (kept NotRequired for compat) ───────────
    current_batch_index: NotRequired[int]
    current_batch_drafts: NotRequired[List[DraftFlashcard]]
    current_batch_eval: NotRequired[Optional[FlashcardBatchEvaluation]]
    batch_revision_count: NotRequired[int]

    # ── coverage (built from batch_results post-parallel) ────────────────────
    coverage_summary: NotRequired[str]

    # ── post-loop global QA ───────────────────────────────────────────────────
    global_deck_report: NotRequired[str]

    # ── final output ──────────────────────────────────────────────────────────
    persisted_card_ids: NotRequired[List[str]]
    final_output: NotRequired[str]

    # ── budget tracking ───────────────────────────────────────────────────────
    budget: NotRequired[Dict[str, Any]]
