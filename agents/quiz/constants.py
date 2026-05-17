# agents/quiz/constants.py
"""
Quiz agent question taxonomy and distractor type constants.

MC_QUESTION_TYPES — the full bank of multiple-choice question types, drawn
    directly from the pedagogical design specification.  quiz_blueprint_planner
    allocates types across batches; question_drafter uses the type tag as its
    primary instruction.

DISTRACTOR_TYPES — taxonomy of wrong-answer misconception categories.
    false_trap_red_herring_generator assigns one of these to each wrong answer;
    the tag is stored in quiz_answers.distractor_type.
"""

# ── Multiple-choice question types ────────────────────────────────────────────

MC_QUESTION_TYPES = [
    # Rule/doctrine discrimination
    "RULE_DISCRIMINATION",          # which rule statement is correct
    "HOLDING_PRECISION",            # which answer overstates or understates the holding
    "HOLDING_VS_DICTA",             # which statement is necessary to the result vs. obiter
    "RULE_EXTRACTION",              # which rule statement is most accurate
    "RULE_ELEMENTS",                # which element is missing or satisfied

    # Fact-level cognition
    "PROCEDURAL_POSTURE",           # which posture best describes the case
    "CASE_FACTS",                   # which fact is legally relevant
    "LEGALLY_RELEVANT_FACTS",       # which fact most affects the outcome
    "ISSUE_PRECISION",              # which is the best issue statement

    # Application
    "FACT_TO_RULE_APPLICATION",     # apply doctrine to a short fact pattern
    "BEST_ANSWER_SELECTION",        # choose the best of several partially correct answers
    "MINI_HYPO_OUTCOME",            # predict result under changed facts
    "EXCEPTION_SPOTTING",           # which facts trigger the exception
    "RULE_BOUNDARY",                # which case falls outside the rule

    # Adversarial / critique
    "COUNTERARGUMENT",              # which argument best supports the losing party
    "MOST_VULNERABLE_ARGUMENT",     # which argument is most vulnerable to challenge
    "TRAP_DETECTION",               # identify overbroad, incomplete, or misleading answer

    # Cross-case / policy
    "COMPARE_DISTINGUISH",          # which distinction best explains different outcomes
    "POLICY_ANALYSIS",              # which policy concern best explains the rule
    "ANALOGY_TOO_STRONG",           # which analogy goes too far

    # Procedure / burden
    "BURDEN_OF_PROOF",              # who must prove this element
    "STANDARD_OF_REVIEW",           # how does the standard of review affect the result
    "PROCEDURAL_POSTURE_IMPACT",    # how posture changes analysis

    # Exam-skill transfer
    "EXAM_APPLICATION",             # which exam answer best applies this case
    "EXAM_ISSUE_SPOTTING",          # which doctrine is most implicated by these facts
]

# ── Distractor types ──────────────────────────────────────────────────────────

DISTRACTOR_TYPES = [
    "OVERBROAD_RULE",               # states a rule broader than the case supports
    "UNDERBROAD_RULE",              # makes the rule too narrow
    "DICTA_AS_HOLDING",             # treats dicta as binding
    "WRONG_ELEMENT",                # focuses on the wrong legal element
    "FACTUALLY_TRUE_LEGALLY_IRRELEVANT",  # true fact, but not legally decisive
    "REVERSED_BURDEN",              # assigns burden to the wrong party
    "POLICY_CONFUSION",             # uses plausible policy but wrong doctrinal result
    "DISSENT_AS_MAJORITY",          # uses dissent's logic as if it were the holding
    "PROCEDURAL_POSTURE_MISS",      # ignores motion posture, appeal, etc.
    "CAUSATION_DUTY_CONFUSION",     # confuses two related doctrinal concepts
    "ANALOGY_TOO_STRONG",           # treats a distinguishable case as controlling
    "EXCEPTION_MISAPPLIED",         # applies exception where trigger facts are missing
]

# ── Quiz modes ────────────────────────────────────────────────────────────────

QUIZ_MODES = [
    "recall",        # basic fact/rule recall; lower cognitive demand
    "application",   # apply doctrine to fact patterns; core mode
    "exam-style",    # multi-layer best-answer; mirrors bar/final exams
    "mixed",         # blend across all cognitive levels (default)
]

# ── Difficulty labels (within a quiz mode) ────────────────────────────────────

DIFFICULTY_LEVELS = [
    "recall",        # surface-level: "what did the court hold?"
    "application",   # mid-level: apply rule to new facts
    "analysis",      # deep: distinguish, policy, best-answer
]

# ── Recall-type question types (lower cognitive demand) ───────────────────────

MC_RECALL_TYPES = [
    "PROCEDURAL_POSTURE",
    "CASE_FACTS",
    "RULE_EXTRACTION",
    "ISSUE_PRECISION",
]

# ── Application-type question types ───────────────────────────────────────────

MC_APPLICATION_TYPES = [
    "FACT_TO_RULE_APPLICATION",
    "MINI_HYPO_OUTCOME",
    "EXCEPTION_SPOTTING",
    "RULE_ELEMENTS",
    "LEGALLY_RELEVANT_FACTS",
    "BEST_ANSWER_SELECTION",
]

# ── Analysis-type question types (higher cognitive demand) ────────────────────

MC_ANALYSIS_TYPES = [
    "RULE_DISCRIMINATION",
    "HOLDING_PRECISION",
    "HOLDING_VS_DICTA",
    "RULE_BOUNDARY",
    "COUNTERARGUMENT",
    "COMPARE_DISTINGUISH",
    "POLICY_ANALYSIS",
    "MOST_VULNERABLE_ARGUMENT",
    "TRAP_DETECTION",
    "STANDARD_OF_REVIEW",
    "BURDEN_OF_PROOF",
    "EXAM_APPLICATION",
    "EXAM_ISSUE_SPOTTING",
]
