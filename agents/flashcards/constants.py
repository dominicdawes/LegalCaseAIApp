# agents/flashcards/constants.py
"""
Card-type taxonomy for the flashcard agent.

Two groups:
  BASELINE_TYPES   — cognitive question types shared with exam-question agent
  RECALL_SPECIFIC  — flashcard-native types that exploit spaced-repetition (SR2) best
"""

# ── Full taxonomy ──────────────────────────────────────────────────────────────

BASELINE_TYPES = [
    "PROCEDURAL_POSTURE",
    "CASE_FACTS",
    "LEGALLY_RELEVANT_FACTS",
    "ISSUE_PRECISION",
    "HOLDING_PRECISION",
    "RULE_EXTRACTION",
    "RULE_ELEMENTS",
    "REASONING_CHAIN",
    "HOLDING_VS_DICTA",
    "FACT_CHANGE_HYPO",
    "RULE_BOUNDARY",
    "COUNTERARGUMENT",
    "COMPARE_AND_DISTINGUISH",
    "POLICY_ANALYSIS",
    "EXAM_APPLICATION",
    "EXCEPTION_SPOTTING",
    "BURDEN_OF_PROOF",
    "STANDARD_OF_REVIEW",
]

RECALL_SPECIFIC_TYPES = [
    "RULE_RECALL",
    "ELEMENT_RECALL",
    "DEFINITION_RECALL",
    "CASE_HOLDING_RECALL",
    "PROCEDURAL_POSTURE_RECALL",
    "TRIGGER_FACT_RECALL",
    "CASE_TO_RULE_MAPPING",
    "RULE_TO_CASE_MAPPING",
    "EXCEPTION_RECALL",
    "POLICY_RECALL",
    "TERM_DISTINCTION",
    "MINI_HYPO_RECALL",
    "ATTACK_OUTLINE_NODE_RECALL",
    "BURDEN_RECALL",
    "STANDARD_PHRASE_RECALL",
]

FLASHCARD_CARD_TYPES = BASELINE_TYPES + RECALL_SPECIFIC_TYPES

# ── Cognitive-demand buckets (used by blueprint planner for mix enforcement) ───

# Pure recall — ideal for spaced repetition, high volume
RECALL_TYPES = [
    "RULE_RECALL",
    "ELEMENT_RECALL",
    "DEFINITION_RECALL",
    "CASE_HOLDING_RECALL",
    "PROCEDURAL_POSTURE_RECALL",
    "TRIGGER_FACT_RECALL",
    "BURDEN_RECALL",
    "EXCEPTION_RECALL",
    "STANDARD_PHRASE_RECALL",
    "PROCEDURAL_POSTURE",
]

# Application / analysis — higher cognitive demand, fewer per deck
APPLICATION_TYPES = [
    "FACT_CHANGE_HYPO",
    "RULE_BOUNDARY",
    "COUNTERARGUMENT",
    "COMPARE_AND_DISTINGUISH",
    "MINI_HYPO_RECALL",
    "EXAM_APPLICATION",
    "TERM_DISTINCTION",
    "LEGALLY_RELEVANT_FACTS",
    "ISSUE_PRECISION",
    "REASONING_CHAIN",
]

# Mapping / structural — connects cases to doctrine, good for attack outlines
MAPPING_TYPES = [
    "CASE_TO_RULE_MAPPING",
    "RULE_TO_CASE_MAPPING",
    "ATTACK_OUTLINE_NODE_RECALL",
    "POLICY_RECALL",
    "POLICY_ANALYSIS",
    "HOLDING_VS_DICTA",
    "HOLDING_PRECISION",
    "RULE_EXTRACTION",
    "RULE_ELEMENTS",
    "EXCEPTION_SPOTTING",
    "STANDARD_OF_REVIEW",
    "BURDEN_OF_PROOF",
    "CASE_FACTS",
]

# Default balanced mix: ~40% recall, ~35% application, ~25% mapping
DEFAULT_CARD_MIX = (
    RECALL_TYPES * 4
    + APPLICATION_TYPES * 3
    + MAPPING_TYPES * 2
)

# Source type labels for source_profiler
SOURCE_TYPES = [
    "case_opinion",
    "casebook_excerpt",
    "class_notes",
    "attack_outline",
    "doctrine_summary",
    "statute",
    "secondary",
    "other",
]
