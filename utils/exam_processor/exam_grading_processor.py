# utils/exam_grading_processor.py

import json
import re
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class ExamGradingProcessor:
    """
    Utility class to robustly clean, parse, and validate the JSON output 
    from the granular grading LLM step.
    
    Handles common LLM formatting errors like:
    - Triple backticks (```json ... ```) or double quotes (''' ... ''') wrappers.
    - Incorrectly escaped characters within the JSON body.
    - Pre/Post-amble text outside the JSON structure.
    """
    
    REQUIRED_KEYS: List[str] = [
        "exam_grade_percentage", 
        "feedback_overview", 
        "feedback_breakdown", 
        "feedback_strength", 
        "feedback_weakness"
    ]

    @staticmethod
    def _clean_text_wrapper(text: str) -> str:
        """
        Removes common LLM code block wrappers (```json, ```, triple quote, ''')
        """
        # 1. Remove Markdown/Code block wrappers
        text = re.sub(r'^\s*```json\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\s*```\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*```\s*$', '', text)
        
        # 2. Remove Python triple-quote wrappers
        text = re.sub(r'^\s*"""\s*', '', text)
        text = re.sub(r'\s*"""\s*$', '', text)
        text = re.sub(r"^\s*'''\s*", '', text)
        text = re.sub(r"\s*'''\s*$", '', text)
        
        return text.strip()

    @staticmethod
    def _extract_json_with_regex(text: str) -> Optional[str]:
        """
        Uses regex to find the first complete JSON object in the text.
        Helpful if the LLM adds text before or after the JSON block.
        """
        # Pattern to find a JSON object starting with '{' and ending with '}'
        # This is a bit greedy but usually captures the main intended output.
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        return None

    @classmethod
    def process_and_validate_json(cls, raw_llm_output: str) -> Dict[str, Any]:
        """
        Performs multi-step cleanup and parsing of the raw LLM string.
        Raises ValueError if final output is not valid or missing required keys.
        """
        
        cleaned_text = cls._clean_text_wrapper(raw_llm_output)
        
        # 1. Attempt standard JSON load
        try:
            data = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON decode failed: {e}. Attempting regex salvage.")
            
            # 2. Attempt regex salvage
            salvaged_text = cls._extract_json_with_regex(raw_llm_output)
            if not salvaged_text:
                raise ValueError("Could not find or salvage JSON object from LLM response.")
            
            # 3. Attempt JSON load on salvaged text
            try:
                data = json.loads(salvaged_text)
            except json.JSONDecodeError as e_final:
                logger.error(f"Final JSON decode failed after salvage: {e_final}")
                raise ValueError("LLM returned unparsable JSON even after cleanup and salvage.")
        
        # 4. Final Validation
        missing_keys = [key for key in cls.REQUIRED_KEYS if key not in data]
        if missing_keys:
            logger.error(f"Parsed JSON is missing required keys: {missing_keys}")
            raise ValueError(f"LLM output is missing required keys: {', '.join(missing_keys)}")
            
        # Ensure percentage is an integer
        if not isinstance(data['exam_grade_percentage'], int):
            try:
                data['exam_grade_percentage'] = int(data['exam_grade_percentage'])
            except (ValueError, TypeError):
                logger.warning("exam_grade_percentage was not an integer; defaulting to 0.")
                data['exam_grade_percentage'] = 0
                
        return data