# agents/flashcards/state.py
"""
Typed state for the flashcard LangGraph agent.

Design: thin-state pattern — full card text lives in the DB; state carries
IDs, short signatures, and small metadata only.  In-flight batch fields are
cleared by batch_commit before the next iteration so the context window stays
bounded regardless of deck size.

Parallel fan-out fields use Annotated[List[T], operator.add] so LangGraph
concatenates branch results instead of last-writer-wins.
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
    identified_cases: List[str]   # case name strings (e.g. "Palsgraf v. Long Island RR")
    key_concepts: List[str]
    key_rules: List[str]


# ── Blueprint planning ─────────────────────────────────────────────────────────

class FlashcardCardSpec(TypedDict):
    spec_index: int            # global 0-based index across all batches
    card_type: str             # e.g. "RULE_RECALL", "FACT_CHANGE_HYPO"
    source_ids: List[str]      # which sources to draw from for this card
    case_names: List[str]      # which cases to reference
    topic: str                 # specific concept / rule / case for this card
    difficulty: str            # "recall" | "application" | "analysis"


class FlashcardBatchSpec(TypedDict):
    batch_index: int
    card_specs: List[FlashcardCardSpec]


# ── Draft card (in-flight, current batch only) ─────────────────────────────────

class DraftFlashcard(TypedDict):
    spec_index: int
    card_type: str
    front_content: str         # the question / prompt side
    back_content: str          # the answer / explanation side
    hint: str                  # optional memory aid (one short sentence)
    source_refs: List[str]     # chunk_ids used for grounding
    grounding_verdict: str     # "" | "pass" | "warn" | "fail"
    grounding_notes: str


# ── Batch-level evaluation ─────────────────────────────────────────────────────

class FlashcardBatchEvaluation(TypedDict):
    batch_index: int
    passes: bool
    scores: Dict[str, float]          # vagueness, uniqueness, source_support,
                                      # atomic_focus, answer_quality
    rejection_reasons: List[str]
    revision_instructions: List[str]
    card_verdicts: List[Dict[str, Any]]   # [{spec_index, passes, reason}, ...]


# ── Full agent state ───────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # ── inputs ───────────────────────────────────────────────────────────────
    request: str
    project_id: str
    source_ids: List[str]
    num_cards: int             # total flashcards to generate (default 10)
    batch_size: int            # cards per batch (default 5, max 10)
    is_essential: NotRequired[bool]    # forwarded to notes row
    use_voyage: NotRequired[bool]

    # ── ledger identifiers ────────────────────────────────────────────────────
    job_id: NotRequired[str]   # notes.id — deck container; individual_cards.deck_id FK
    run_id: NotRequired[str]
    user_id: NotRequired[str]  # auth.users.id — required for individual_cards.user_id FK

    # ── parallel fan-out accumulator (operator.add reducer) ───────────────────
    source_profiles: Annotated[List[FlashcardSourceProfile], operator.add]

    # ── sequential pipeline outputs ───────────────────────────────────────────
    concept_inventory: NotRequired[str]    # JSON: extracted atoms per source
    concept_synthesis: NotRequired[str]    # JSON: cross-doc throughlines
    batch_specs: NotRequired[List[FlashcardBatchSpec]]
    num_batches: NotRequired[int]

    # ── in-flight batch state (replaced each iteration by batch_commit) ───────
    current_batch_index: NotRequired[int]
    current_batch_drafts: NotRequired[List[DraftFlashcard]]
    current_batch_eval: NotRequired[Optional[FlashcardBatchEvaluation]]
    batch_revision_count: NotRequired[int]     # reset to 0 by batch_commit

    # ── thin accumulators (batch_commit appends full list each iteration) ─────
    accepted_card_ids: NotRequired[List[str]]           # individual_cards.id rows
    rejected_card_metadata: NotRequired[List[Dict[str, Any]]]
    used_card_signatures: NotRequired[List[str]]        # "TYPE:front[:40]" for dedup
    coverage_summary: NotRequired[str]                  # JSON: {card_type: count}

    # ── post-loop global QA ───────────────────────────────────────────────────
    global_deck_report: NotRequired[str]

    # ── final output ──────────────────────────────────────────────────────────
    persisted_card_ids: NotRequired[List[str]]
    final_output: NotRequired[str]            # markdown summary written to notes row

    # ── budget tracking ───────────────────────────────────────────────────────
    budget: NotRequired[Dict[str, Any]]       # {input_tokens, output_tokens, cost_usd}
