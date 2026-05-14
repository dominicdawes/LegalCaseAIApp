# utils/note_processing/exam_processor.py
"""
Validates and coerces VerifiedQuestion dicts from the exam agent pipeline
into DB-ready rows for exam_questions and exam_answers tables.

Analogous to QuizProcessor but operates on structured agent output (typed dicts)
rather than raw LLM JSON strings — so parsing is simpler and fallbacks are
targeted at field-level gaps rather than structural malformation.
"""

import json
import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Strip JSON code-fence artefacts that occasionally leak into LLM text fields
_JSON_FENCE_RE = re.compile(r'^```(?:json)?\s*|\s*```$', re.MULTILINE)
_VALID_VERDICTS = {"pass", "warn", "fail"}


class ExamProcessor:
    """
    Coerces VerifiedQuestion dicts (from agents/exam_questions/state.py) into
    validated, DB-ready rows for exam_questions and exam_answers.

    Each coerced dict carries both the question-row fields AND the answer-row
    fields so the caller can insert them together transactionally:

        {
            # → exam_questions row
            "question_index":   int,
            "issue_label":      str,
            "fact_pattern":     str,
            "call_of_question": str,
            "grounding_verdict": str,   # "pass" | "warn" | "fail"
            "revised":          bool,

            # → exam_answers row
            "answer_key":  str,
            "citations":   list,        # JSON-serialisable list of dicts
        }
    """

    def __init__(self):
        logger.info("ExamProcessor initialised")

    # ── Public API ────────────────────────────────────────────────────────────

    def coerce_questions(self, verified_questions: List[Dict]) -> List[Dict]:
        """
        Validate and normalise a list of VerifiedQuestion dicts.

        Skips questions that are unrecoverable (missing fact_pattern or
        answer_key), deduplicates by fact_pattern prefix, and returns only
        questions that can be safely written to the database.
        """
        if not isinstance(verified_questions, list):
            logger.error("coerce_questions: expected list, got %s", type(verified_questions))
            return []

        coerced: List[Dict] = []
        seen_patterns: set = set()

        for i, q in enumerate(verified_questions):
            result = self.coerce_one(q, fallback_index=i)
            if result is None:
                continue
            # Deduplicate on the first 120 chars of the fact pattern
            dedup_key = result["fact_pattern"][:120].lower().strip()
            if dedup_key in seen_patterns:
                logger.warning("Skipping duplicate exam question at index %d", i)
                continue
            seen_patterns.add(dedup_key)
            coerced.append(result)

        logger.info(
            "ExamProcessor: %d/%d questions coerced successfully",
            len(coerced), len(verified_questions),
        )
        return coerced

    def coerce_one(self, q: Dict, fallback_index: int = 0) -> Optional[Dict]:
        """
        Coerce a single VerifiedQuestion dict.
        Returns None if the question is unrecoverable (missing required fields).
        """
        if not isinstance(q, dict):
            logger.warning("Skipping non-dict question at index %d", fallback_index)
            return None

        # ── required: fact_pattern ────────────────────────────────────────────
        fact_pattern = self._clean_text(q.get("fact_pattern", ""))
        if not fact_pattern:
            logger.warning("Question index %d missing fact_pattern — skipping", fallback_index)
            return None

        # ── required: answer_key ─────────────────────────────────────────────
        answer_key = self._clean_text(q.get("answer_key", ""))
        if not answer_key:
            logger.warning("Question index %d missing answer_key — skipping", fallback_index)
            return None

        # ── required-with-fallback: call_of_question ─────────────────────────
        call_of_question = self._clean_text(q.get("call_of_question", ""))
        if not call_of_question:
            call_of_question = "Discuss all rights and liabilities of the parties."
            logger.warning("Question %d missing call_of_question — using fallback", fallback_index)

        # ── optional fields ───────────────────────────────────────────────────
        issue_label = str(q.get("issue_label", "")).strip() or "General Legal Issue"

        grounding_verdict = str(q.get("grounding_verdict", "pass")).strip().lower()
        if grounding_verdict not in _VALID_VERDICTS:
            grounding_verdict = "warn"

        revised = bool(q.get("revised", False))
        question_index = int(q.get("question_index", fallback_index))

        # ── citations: accept list or JSON string ─────────────────────────────
        raw_cites = q.get("citations", [])
        if isinstance(raw_cites, str):
            try:
                raw_cites = json.loads(raw_cites)
            except (json.JSONDecodeError, ValueError):
                raw_cites = []
        citations = raw_cites if isinstance(raw_cites, list) else []

        return {
            # exam_questions fields
            "question_index":   question_index,
            "issue_label":      issue_label,
            "fact_pattern":     fact_pattern,
            "call_of_question": call_of_question,
            "grounding_verdict": grounding_verdict,
            "revised":          revised,
            # exam_answers fields
            "answer_key":       answer_key,
            "citations":        citations,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_text(text: str) -> str:
        """Strip JSON code-fence artefacts and normalise whitespace."""
        if not isinstance(text, str):
            return ""
        return _JSON_FENCE_RE.sub("", text).strip()
