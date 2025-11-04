# utils/note_processing/quiz_processor.py

import json
import logging
from typing import Dict, List, Any
import re

logger = logging.getLogger(__name__)

class QuizProcessor:
    """
    Handles parsing and validation of LLM-generated quiz JSON.
    """

    def __init__(self):
        logger.info("QuizProcessor initialized")

    def parse_quiz_content(self, llm_output: str) -> Dict[str, Any]:
        """
        Parses the raw LLM JSON string output into a Python dictionary.
        
        Args:
            llm_output: The raw JSON string from the LLM.
            
        Returns:
            A dictionary representing the quiz data.
            
        Raises:
            json.JSONDecodeError: If the string is not valid JSON.
            ValueError: If the top-level structure is not a dictionary.
        """
        try:
            # First, strip potential markdown fences (```json ... ```)
            llm_output = llm_output.strip()
            if llm_output.startswith("```json"):
                llm_output = llm_output[7:]
            if llm_output.startswith("```"):
                llm_output = llm_output[3:]
            if llm_output.endswith("```"):
                llm_output = llm_output[:-3]
            
            llm_output = llm_output.strip()
            
            quiz_data = json.loads(llm_output)
            if not isinstance(quiz_data, dict):
                raise ValueError("Parsed JSON is not a dictionary.")
            return quiz_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from LLM output: {e.doc[e.pos-10:e.pos+10]}", exc_info=True)
            raise ValueError(f"Invalid JSON format from LLM: {e}")
        except Exception as e:
            logger.error(f"Error parsing quiz content: {e}", exc_info=True)
            raise

    def validate_quiz_data(self, quiz_data: Dict[str, Any]) -> bool:
        """
        Validates the structure of the parsed quiz data against the expected schema.
        
        Args:
            quiz_data: The parsed quiz dictionary.
            
        Returns:
            True if valid, raises ValueError otherwise.
        """
        if "questions" not in quiz_data or not isinstance(quiz_data["questions"], list):
            raise ValueError("Invalid quiz data: 'questions' key is missing or not a list.")
            
        if not quiz_data["questions"]:
            raise ValueError("Invalid quiz data: 'questions' list is empty.")

        for i, question in enumerate(quiz_data["questions"]):
            if not isinstance(question, dict):
                raise ValueError(f"Question {i} is not a dictionary.")
            
            if "questionText" not in question or not question["questionText"]:
                raise ValueError(f"Question {i} is missing 'questionText'.")
                
            if "hint" not in question:
                # Hint is allowed to be empty, but key should exist
                logger.warning(f"Question {i} is missing 'hint'.")
            
            if "answers" not in question or not isinstance(question["answers"], list):
                raise ValueError(f"Question {i} is missing 'answers' list.")
            
            if len(question["answers"]) != 5:
                logger.warning(f"Question {i} does not have exactly 5 answers (has {len(question['answers'])}).")
            
            correct_count = 0
            for j, answer in enumerate(question["answers"]):
                if not isinstance(answer, dict):
                    raise ValueError(f"Answer {j} for Question {i} is not a dictionary.")
                if "answerText" not in answer or not answer["answerText"]:
                    raise ValueError(f"Answer {j} for Question {i} is missing 'answerText'.")
                if "isCorrect" not in answer or not isinstance(answer["isCorrect"], bool):
                    raise ValueError(f"Answer {j} for Question {i} is missing or invalid 'isCorrect' flag.")
                if "feedback" not in answer:
                    logger.warning(f"Answer {j} for Question {i} is missing 'feedback'.")
                
                if answer["isCorrect"]:
                    correct_count += 1
            
            if correct_count != 1:
                logger.warning(f"Question {i} does not have exactly one correct answer (has {correct_count}).")

        logger.info("Quiz data validated successfully.")
        return True
