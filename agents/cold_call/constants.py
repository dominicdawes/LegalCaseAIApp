# agents/cold_call/constants.py
"""
Cold-call agent question taxonomy and seed theme constants.

This file is the single source of truth for the question type bank and seed
themes used throughout the cold-call pipeline.  Previously these lists were
hardcoded as local variables inside `question_type_bank_selector` and
`cold_call_seed_generator`, which made them invisible and hard to edit.
Move any additions, removals, or re-categorisations here — nodes.py will
pick them up automatically without further changes.

Layout
──────
QUESTION_TYPES_*   — per-depth buckets that feed question_type_bank_selector.
                     The selector assembles these into a QuestionTypeSelection
                     dict based on run context (multi-case, dissent, difficulty).
QUESTION_TYPES_REQUIRED_ALWAYS
                   — types that must appear in every run regardless of
                     difficulty or case count.
QUESTION_TYPES_CONDITIONAL_MULTI_CASE / _DISSENT / _ADVANCED_*
                   — types unlocked by specific run conditions.
QUESTION_TYPES_OPTIONAL
                   — enrichment types added when difficulty >= day_one_t14.
SEED_THEMES        — high-level framing angles for opening cold-call questions;
                     the seed generator cycles across these to ensure no two
                     threads start from the same conceptual angle.
"""

# ── Core depth buckets ────────────────────────────────────────────────────────

QUESTION_TYPES_EARLY = [
    "PROCEDURAL_POSTURE",       # where the case is in the litigation timeline
    "LEGALLY_RELEVANT_FACTS",   # which facts the court treated as material
    "ISSUE_PRECISION",          # exact legal question the court answered
    "HOLDING_PRECISION",        # narrow holding vs. broader language
]

QUESTION_TYPES_MIDDLE = [
    "RULE_EXTRACTION",          # articulate the rule the case establishes
    "RULE_ELEMENTS",            # break the rule into component elements
    "REASONING_CHAIN",          # how the court moved from facts → holding
    "HOLDING_VS_DICTA",         # distinguish binding holding from obiter dicta
    "COUNTERARGUMENT",          # strongest argument the losing side had
]

QUESTION_TYPES_DEEP = [
    "FACT_CHANGE_HYPO",         # change one fact — does the outcome flip?
    "RULE_BOUNDARY",            # where does the rule stop applying?
    "POLICY_ANALYSIS",          # underlying policy the rule is meant to serve
    "EXAM_APPLICATION",         # apply rule to a novel fact pattern
]

# ── Always-required (present in every run) ────────────────────────────────────

QUESTION_TYPES_REQUIRED_ALWAYS = [
    "PROCEDURAL_POSTURE",
    "LEGALLY_RELEVANT_FACTS",
    "ISSUE_PRECISION",
    "HOLDING_PRECISION",
    "RULE_EXTRACTION",
    "REASONING_CHAIN",
    "FACT_CHANGE_HYPO",
    "RULE_BOUNDARY",
    "COUNTERARGUMENT",
    "POLICY_ANALYSIS",
    "EXAM_APPLICATION",
]

# ── Conditional — unlocked by run context ─────────────────────────────────────

# Added to DEEP when there are multiple cases in scope
QUESTION_TYPES_CONDITIONAL_MULTI_CASE_DEEP = [
    "COMPARE_DISTINGUISH",      # how do the cases relate / diverge doctrinally?
]

# Added to MIDDLE when a dissent exists in any case
QUESTION_TYPES_CONDITIONAL_DISSENT_MIDDLE = [
    "DISSENT_ANALYSIS",         # what did the dissent get right / wrong?
]

# Added to DEEP when target_difficulty == "advanced"
QUESTION_TYPES_CONDITIONAL_ADVANCED_DEEP = [
    "STANDARD_OF_REVIEW",       # what deference standard applies and why?
    "STATUTORY_INTERPRETATION", # how did the court read the statute?
    "BURDEN_OF_PROOF",          # who bears the burden and how heavy is it?
    "ADMINISTRABILITY",         # can courts/agencies actually apply this rule?
]

# Added to MIDDLE when target_difficulty == "advanced"
QUESTION_TYPES_CONDITIONAL_ADVANCED_MIDDLE = [
    "REMEDY_ANALYSIS",          # what relief is available and its limits
    "JURISDICTION_AUTHORITY",   # which court / authority decided and why it matters
]

# ── Optional enrichment (added when difficulty >= day_one_t14) ────────────────

# Added when there are multiple cases in scope
QUESTION_TYPES_OPTIONAL_MULTI_CASE = [
    "ANALOGY_QUESTION",         # draw an analogy to another case or doctrine
    "DISTINGUISHING_QUESTION",  # explain why this case does NOT control
]

# Added when target_difficulty != "law_1l"
QUESTION_TYPES_OPTIONAL_UPPER_LEVEL = [
    "PROFESSOR_TRAP_QUESTION",  # surface a common misreading of the rule
    "RECOVERY_QUESTION",        # redirect after a wrong answer without giving it away
    "FLOODGATES_CONCERN",       # if we rule this way, what comes next?
    "FAIRNESS_EQUITY_PROBE",    # is the outcome fair even if legally correct?
]

# ── Seed themes ───────────────────────────────────────────────────────────────

SEED_THEMES = [
    "CORE_UNDERSTANDING",   # confirm the student grasped the case basics
    "RULE_BOUNDARY",        # probe the edges of where the doctrine applies
    "COMPARE_DISTINGUISH",  # relate to or distinguish from another case
    "POLICY",               # surface the policy rationale behind the rule
    "EXAM_TRANSFER",        # shift to a novel fact pattern for application
    "COUNTERARGUMENT",      # steelman the losing side's best argument
    "HYPO_START",           # open with a hypothetical to reveal the rule's reach
    "POSTURE_FOCUS",        # ground in procedural posture before the substance
]
