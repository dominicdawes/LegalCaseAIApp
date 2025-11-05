# utils/note_processing/quiz_processor.py

import json
import logging
import re
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# 1. 🆕 Define a regex to find all substrings that look like a complete
# question object. This is the core of the "Aggressive Coercion" strategy.
# It's non-greedy (.*?) and spans newlines (re.DOTALL).
# 🆕 UPDATED: Now looks for 'questionText' OR 'question_text'
QUESTION_REGEX = re.compile(
    # ⬇️ ADDED '?:' to make this a non-capturing group
    r'\{\s*"(?:questionText|question_text)":.*?"answers":\s*\[.*?\]\s*\}',
    re.DOTALL
)


class QuizProcessor:
    """
    🆕 Handles parsing, validation, and *aggressive coercion* of LLM-generated quiz JSON.
    This processor is designed to salvage valid questions from malformed,
    duplicated, or truncated LLM outputs.
    """

    def __init__(self):
        logger.info("QuizProcessor initialized (Aggressive Coercion Mode)")

    def parse_and_salvage_quiz(self, llm_output: str) -> Dict[str, list]:
        """
        Finds all potential question objects in the raw LLM output,
        parses them individually, and coerces them into a valid format.
        This is robust against malformed JSON, duplicates, and truncation.
        """
        valid_questions = []
        
        # 1. 🆕 Find all potential question strings using the regex
        potential_question_strings = QUESTION_REGEX.findall(llm_output)
        
        if not potential_question_strings:
            logger.warning(f"No potential question objects found in LLM output. Output started with: {llm_output[:500]}...")
            return {"questions": []}

        logger.info(f"Found {len(potential_question_strings)} potential question objects to parse.")
        
        seen_questions = set()  # To prevent duplicates from the LLM

        for i, q_str in enumerate(potential_question_strings):
            try:
                # 2. 🆕 Try to parse the individual string
                q_data = json.loads(q_str)
                
                # 3. 🆕 Coerce and validate the single object
                coerced_q = self._coerce_question(q_data, i)
                
                if coerced_q:
                    # 4. 🆕 Add if valid and not a duplicate
                    # 🆕 UPDATED: Key is now 'question_text'
                    q_text = coerced_q['question_text'].lower().strip()
                    if q_text not in seen_questions:
                        valid_questions.append(coerced_q)
                        seen_questions.add(q_text)
                    else:
                        logger.warning(f"Skipping duplicate question: {q_text[:50]}...")
                        
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse potential question #{i}: {q_str[:150]}...")
            except Exception as e:
                logger.error(f"Error coercing question #{i}: {e}", exc_info=True)

        logger.info(f"Successfully salvaged {len(valid_questions)} valid questions.")
        return {"questions": valid_questions}

    def _coerce_question(self, q_data: dict, index: int) -> dict | None:
        """
        Python version of the 'coerce' logic from handleQuiz.ts.
        Takes a parsed object and returns a valid question dict or None
        in snake_case format.
        Provides fallbacks for missing data.
        """
        if not isinstance(q_data, dict):
            return None
        
        # 1. 🆕 Coerce question_text (snake_case) with fallback to questionText (camelCase)
        question_text = str(q_data.get("question_text", q_data.get("questionText", ""))).strip()
        if not question_text:
            logger.warning(f"Question #{index} is missing 'question_text' or 'questionText'. Skipping.")
            return None # question_text is non-negotiable
        
        # 2. Coerce hint (fallback)
        hint = str(q_data.get("hint", "")).strip()
        if not hint:
            hint = "No hint provided."
            
        # 3. Coerce Answers (most complex part)
        raw_answers = q_data.get("answers")
        if not isinstance(raw_answers, list) or not raw_answers:
            logger.warning(f"Question '{question_text[:50]}...' has no answers list. Skipping.")
            return None # A question with no answers is useless

        coerced_answers = []
        correct_found = False
        for j, ans in enumerate(raw_answers):
            if not isinstance(ans, dict):
                continue
            
            # 🆕 Coerce answer_choice_text (already snake_case)
            ans_text = str(ans.get("answer_choice_text", "")).strip()
            if not ans_text:
                ans_text = f"Salvaged Answer {j + 1}"
            
            # 🆕 Coerce is_correct (snake_case) with fallback to isCorrect (camelCase)
            is_correct = ans.get("is_correct", ans.get("isCorrect"))
            if not isinstance(is_correct, bool):
                is_correct = False
                
            if is_correct:
                correct_found = True
                
            feedback = str(ans.get("feedback", "")).strip()
            if not feedback:
                feedback = "No feedback provided."
            
            # 🆕 Ensure keys in the new dict are snake_case for the DB
            coerced_answers.append({
                "answer_choice_text": ans_text,
                "is_correct": is_correct, # 🆕 RENAMED from isCorrect
                "feedback": feedback
            })

        # 4. 🆕 Ensure at least one answer is correct (just like handleQuiz.ts)
        if not correct_found and coerced_answers:
            logger.warning(f"Question '{question_text[:50]}...' had no correct answer. Assigning first.")
            coerced_answers[0]["is_correct"] = True
        
        # 5. 🆕 Ensure we have 5 answers (pad if necessary)
        while len(coerced_answers) < 5:
            logger.warning(f"Padding answers for question: {question_text[:50]}...")
            coerced_answers.append({
                "answer_choice_text": "Default Padded Answer",
                "is_correct": False, # 🆕 RENAMED from isCorrect
                "feedback": "This is a default answer because the LLM provided too few."
            })

        # 6. 🆕 Return final dict with snake_case keys
        return {
            "question_text": question_text, # 🆕 RENAMED from questionText
            "hint": hint,
            "answers": coerced_answers[:5] # Take only 5
        }

    # ----------------------------------------------------------------------
    # 🛑 DEPRECATED METHODS 🛑
    # Keep these methods to prevent import errors in note_tasks.py,
    # but they will no longer be called by the updated logic.
    # ----------------------------------------------------------------------

    def parse_quiz_content(self, llm_output: str) -> Dict[str, Any]:
        """
        DEPRECATED. Use parse_and_salvage_quiz instead.
        """
        logger.warning("Using DEPRECATED parse_quiz_content. Please update to parse_and_salvage_quiz.")
        # Fallback to old regex logic just in case
        match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found.")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

    def validate_quiz_data(self, quiz_data: Dict[str, Any]) -> bool:
        """
        DEPRECATED. Validation is now part of the coercion process.
        """
        logger.warning("DEPRECATED validate_quiz_data call. Skipping.")
        return True