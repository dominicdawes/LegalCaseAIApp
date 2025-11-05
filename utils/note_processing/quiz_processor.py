# utils/note_processing/quiz_processor.py

import json
import logging
import re  # 🆕 Import regex
from typing import Dict, List, Any

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
        
        🆕 Updated with a robust regex extractor to find the JSON blob,
        even if it's surrounded by other text or markdown.
        
        Args:
            llm_output: The raw string from the LLM.
            
        Returns:
            A dictionary representing the quiz data.
            
        Raises:
            ValueError: If no JSON object is found or parsing fails.
        """
        
        # 1. 🆕 Use regex to find the first '{' and the last '}'
        # This is far more robust than stripping markdown fences.
        match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        
        if not match:
            logger.error(f"No JSON object found in LLM output. Output was: {llm_output[:500]}...")
            raise ValueError("No JSON object found in LLM output.")
            
        json_string = match.group(0)
        
        # 2. Try to parse the extracted string
        try:
            quiz_data = json.loads(json_string)
            if not isinstance(quiz_data, dict):
                raise ValueError("Parsed JSON is not a dictionary.")
            return quiz_data
        except json.JSONDecodeError as e:
            # 3. 🆕 If parsing fails, log the *full extracted string* for debugging.
            # This is the "debug print" you need.
            logger.error(f"Failed to decode extracted JSON. Error: {e}")
            
            # Log the area around the error
            snippet_start = max(0, e.pos - 80)
            snippet_end = min(len(e.doc), e.pos + 80)
            snippet = e.doc[snippet_start:snippet_end]
            logger.error(f"Error snippet (char {e.pos}): ...{snippet}...")
            
            # 🆕 Log the *full* failed string for inspection
            logger.debug(f"--- FULL FAILED JSON STRING ---\n{json_string}\n--- END FAILED JSON ---")
            
            raise ValueError(f"Invalid JSON format from LLM even after regex extraction: {e}")
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
            # 🆕 Improved Error Logging: Show what keys *were* found.
            found_keys = list(quiz_data.keys())
            raise ValueError(f"Invalid quiz data: 'questions' key is missing or not a list. Found keys: {found_keys}")
            
        if not quiz_data["questions"]:
            raise ValueError("Invalid quiz data: 'questions' list is empty.")

        for i, question in enumerate(quiz_data["questions"]):
            if not isinstance(question, dict):
                raise ValueError(f"Question {i} is not a dictionary.")
            
            # 🆕 Improved Error Logging: Show what keys *were* found.
            if "questionText" not in question or not question["questionText"]:
                found_keys = list(question.keys())
                raise ValueError(f"Question {i} is missing 'questionText'. Found keys: {found_keys}")
                
            if "hint" not in question:
                logger.warning(f"Question {i} is missing 'hint'.")
            
            if "answers" not in question or not isinstance(question["answers"], list):
                raise ValueError(f"Question {i} is missing 'answers' list.")
            
            if len(question["answers"]) < 4: # Loosen validation slightly, 5 is ideal but 4 is ok
                logger.warning(f"Question {i} does not have 5 answers (has {len(question['answers'])}).")

            correct_count = 0
            for j, answer in enumerate(question["answers"]):
                if not isinstance(answer, dict):
                    raise ValueError(f"Answer {j} for Question {i} is not a dictionary.")
                
                # 🆕 Update: Validate 'answer_choice_text' to match YAML and DB
                if "answer_choice_text" not in answer or not answer["answer_choice_text"]:
                    found_keys = list(answer.keys())
                    raise ValueError(f"Answer {j} for Question {i} is missing 'answer_choice_text'. Found keys: {found_keys}")
                
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

