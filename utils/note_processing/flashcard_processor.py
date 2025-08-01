# utils/note_processing/flashcard_processor.py

import re
import json
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class FlashcardProcessor:
    """
    Utility class to parse LLM-generated flashcard content and structure it 
    for database insertion into public.flashcard_decks and public.individual_cards
    """
    
    def __init__(self):
        pass
    
    def parse_flashcard_content(self, llm_output: str, deck_name: str) -> Tuple[Dict, List[Dict]]:
        """
        Parse LLM output containing flashcards and return structured data.
        
        Args:
            llm_output: Raw text output from LLM containing flashcards
            deck_name: Name for the flashcard deck
            
        Returns:
            Tuple of (deck_data, cards_list)
            - deck_data: Dict with flashcard deck information
            - cards_list: List of dicts with individual card information
        """
        try:
            # Try different parsing strategies based on LLM output format
            if self._is_json_format(llm_output):
                return self._parse_json_flashcards(llm_output, deck_name)
            elif self._is_markdown_format(llm_output):
                return self._parse_markdown_flashcards(llm_output, deck_name)
            else:
                return self._parse_structured_text(llm_output, deck_name)
                
        except Exception as e:
            logger.error(f"Error parsing flashcard content: {e}")
            # Fallback: create a single card with the raw content
            return self._create_fallback_deck(llm_output, deck_name)
    
    def _is_json_format(self, content: str) -> bool:
        """Check if content appears to be JSON formatted"""
        stripped = content.strip()
        return (stripped.startswith('{') and stripped.endswith('}')) or \
               (stripped.startswith('[') and stripped.endswith(']'))
    
    def _is_markdown_format(self, content: str) -> bool:
        """Check if content appears to be markdown formatted"""
        return '**' in content or '#' in content or bool(re.search(r'\d+\.\s', content))
    
    def _parse_json_flashcards(self, content: str, deck_name: str) -> Tuple[Dict, List[Dict]]:
        """Parse JSON formatted flashcards"""
        try:
            data = json.loads(content)
            cards = []
            
            # Handle different JSON structures
            if isinstance(data, list):
                # List of card objects
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        front = item.get('question', item.get('front', item.get('q', '')))
                        back = item.get('answer', item.get('back', item.get('a', '')))
                        cards.append(self._create_card_dict(front, back, i + 1))
            
            elif isinstance(data, dict):
                # Object with cards array or key-value pairs
                if 'cards' in data:
                    for i, card in enumerate(data['cards']):
                        front = card.get('question', card.get('front', ''))
                        back = card.get('answer', card.get('back', ''))
                        cards.append(self._create_card_dict(front, back, i + 1))
                else:
                    # Treat as key-value pairs
                    for i, (key, value) in enumerate(data.items()):
                        cards.append(self._create_card_dict(key, str(value), i + 1))
            
            deck_data = self._create_deck_dict(deck_name, len(cards))
            return deck_data, cards
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse as JSON: {e}")
            return self._parse_structured_text(content, deck_name)
    
    def _parse_markdown_flashcards(self, content: str, deck_name: str) -> Tuple[Dict, List[Dict]]:
        """Parse markdown formatted flashcards"""
        cards = []
        
        # Pattern 1: **Question:** ... **Answer:** ...
        qa_pattern = r'\*\*(?:Question|Q):\*\*\s*(.*?)\s*\*\*(?:Answer|A):\*\*\s*(.*?)(?=\*\*(?:Question|Q):|$)'
        matches = re.findall(qa_pattern, content, re.DOTALL | re.IGNORECASE)
        
        if matches:
            for i, (question, answer) in enumerate(matches):
                cards.append(self._create_card_dict(
                    question.strip(), 
                    answer.strip(), 
                    i + 1
                ))
        else:
            # Pattern 2: Numbered list format
            # 1. Question text
            # Answer: answer text
            numbered_pattern = r'(\d+)\.\s*(.*?)(?:\n(?:Answer|A):\s*(.*?))?(?=\n\d+\.|\Z)'
            matches = re.findall(numbered_pattern, content, re.DOTALL)
            
            for match in matches:
                num, question, answer = match
                if not answer:  # Look for answer on next lines
                    # Try to find answer after this question
                    answer_search = re.search(
                        rf'{re.escape(question)}.*?(?:Answer|A):\s*(.*?)(?=\n\d+\.|\Z)', 
                        content, 
                        re.DOTALL
                    )
                    answer = answer_search.group(1).strip() if answer_search else "No answer provided"
                
                cards.append(self._create_card_dict(
                    question.strip(), 
                    answer.strip(), 
                    int(num)
                ))
        
        # Fallback: Split by double newlines and try to parse Q&A pairs
        if not cards:
            cards = self._parse_structured_text(content, deck_name)[1]
        
        deck_data = self._create_deck_dict(deck_name, len(cards))
        return deck_data, cards
    
    def _parse_structured_text(self, content: str, deck_name: str) -> Tuple[Dict, List[Dict]]:
        """Parse plain text with various Q&A patterns"""
        cards = []
        
        # Split content into potential card blocks
        blocks = re.split(r'\n\s*\n', content.strip())
        
        for i, block in enumerate(blocks):
            if not block.strip():
                continue
            
            # Try different splitting patterns
            qa_separators = [
                r'(?:Answer|A):\s*',
                r'(?:Back|B):\s*',
                r'\n-{2,}\n',  # Dash separator
                r'\|\s*',      # Pipe separator
            ]
            
            parts = None
            for separator in qa_separators:
                parts = re.split(separator, block, maxsplit=1)
                if len(parts) == 2:
                    break
            
            if parts and len(parts) == 2:
                question = re.sub(r'^(?:Question|Q):\s*', '', parts[0].strip())
                answer = parts[1].strip()
                cards.append(self._create_card_dict(question, answer, i + 1))
            else:
                # Single block - treat as question only or split by first sentence
                sentences = re.split(r'[.!?]+', block)
                if len(sentences) >= 2:
                    question = sentences[0].strip() + '.'
                    answer = ' '.join(sentences[1:]).strip()
                    cards.append(self._create_card_dict(question, answer, i + 1))
                else:
                    # Just use the whole block as a concept card
                    cards.append(self._create_card_dict(
                        f"Concept {i + 1}", 
                        block.strip(), 
                        i + 1
                    ))
        
        deck_data = self._create_deck_dict(deck_name, len(cards))
        return deck_data, cards
    
    def _create_card_dict(self, front: str, back: str, order: int) -> Dict:
        """Create a standardized card dictionary"""
        return {
            'front_content': front.strip(),
            'back_content': back.strip(),
            'card_order': order,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'is_active': True
        }
    
    def _create_deck_dict(self, deck_name: str, card_count: int) -> Dict:
        """Create a standardized deck dictionary"""
        return {
            'deck_name': deck_name,
            'description': f'AI-generated flashcard deck with {card_count} cards',
            'card_count': card_count,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'is_active': True
        }
    
    def _create_fallback_deck(self, content: str, deck_name: str) -> Tuple[Dict, List[Dict]]:
        """Create a fallback single-card deck when parsing fails"""
        logger.warning("Using fallback parsing for flashcard content")
        
        card = self._create_card_dict(
            "Generated Content", 
            content[:500] + "..." if len(content) > 500 else content,
            1
        )
        
        deck_data = self._create_deck_dict(deck_name, 1)
        return deck_data, [card]
    
    def validate_flashcard_data(self, deck_data: Dict, cards_list: List[Dict]) -> bool:
        """Validate that flashcard data is properly structured"""
        try:
            # Validate deck data
            required_deck_fields = ['deck_name', 'card_count', 'created_at']
            for field in required_deck_fields:
                if field not in deck_data:
                    logger.error(f"Missing required deck field: {field}")
                    return False
            
            # Validate cards
            required_card_fields = ['front_content', 'back_content', 'card_order']
            for i, card in enumerate(cards_list):
                for field in required_card_fields:
                    if field not in card:
                        logger.error(f"Missing required card field '{field}' in card {i}")
                        return False
                
                if not card['front_content'] or not card['back_content']:
                    logger.warning(f"Card {i} has empty front or back content")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating flashcard data: {e}")
            return False


# Usage example and test functions
def test_flashcard_processor():
    """Test the flashcard processor with sample inputs"""
    processor = FlashcardProcessor()
    
    # Test markdown format
    markdown_sample = """
    **Question:** What is the rule against perpetuities?
    **Answer:** A legal rule that voids certain interests in property that may not vest within a life in being plus 21 years.
    
    **Question:** What is consideration in contract law?
    **Answer:** Something of value that is exchanged between parties to make a contract legally binding.
    """
    
    deck_data, cards = processor.parse_flashcard_content(markdown_sample, "Contract Law Basics")
    print("Deck:", deck_data)
    print("Cards:", cards)
    
    # Test JSON format
    json_sample = """
    {
        "cards": [
            {"question": "What is tort law?", "answer": "A body of law that addresses civil wrongs"},
            {"question": "Define negligence", "answer": "Failure to exercise reasonable care"}
        ]
    }
    """
    
    deck_data, cards = processor.parse_flashcard_content(json_sample, "Tort Law")
    print("JSON Deck:", deck_data)
    print("JSON Cards:", cards)


if __name__ == "__main__":
    test_flashcard_processor()