# utils/llm_clients/stream_normalizer.py
"""
Minimal stream normalizer for handling provider-specific chunk formats.

SINGLE PURPOSE: Convert provider-specific streaming chunks into normalized text.

Usage:
    normalizer = StreamNormalizer()
    text = normalizer.extract_text(raw_chunk, provider="anthropic")
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class StreamNormalizer:
    """
    🎯 SINGLE RESPONSIBILITY: Normalize streaming chunks from different providers
    
    Does NOT handle:
    - LLM client creation (your base clients do this)
    - Citation processing (citation_processor.py does this) 
    - Context building (_build_enhanced_context does this)
    - Performance monitoring (performance_monitor.py does this)
    
    ONLY handles:
    - Extracting text content from provider-specific chunk formats
    - Basic error handling for malformed chunks
    """
    
    def extract_text(self, chunk: Any, provider: str) -> Optional[str]:
        """
        Extract text content from provider-specific chunk format.
        
        Args:
            chunk: Raw chunk from provider's streaming API
            provider: Provider name ("anthropic", "openai", "deepseek", "gemini")
            
        Returns:
            Extracted text string or None if no text found
        """
        if not chunk:
            return None
            
        try:
            provider_lower = provider.lower()
            
            if provider_lower == "anthropic":
                return self._extract_anthropic_text(chunk)
            elif provider_lower == "openai":
                return self._extract_openai_text(chunk)
            elif provider_lower == "deepseek":
                return self._extract_deepseek_text(chunk)
            elif provider_lower == "gemini":
                return self._extract_gemini_text(chunk)
            else:
                logger.warning(f"Unknown provider: {provider}")
                return self._extract_fallback_text(chunk)
                
        except Exception as e:
            logger.warning(f"Failed to extract text from {provider} chunk: {e}")
            return None
    
    def _extract_anthropic_text(self, chunk: Any) -> Optional[str]:
        """Extract text from Anthropic chunk format"""
        # Handle both dict and string chunks
        if isinstance(chunk, str):
            return chunk
        
        if isinstance(chunk, dict):
            # Anthropic streaming format: {"delta": {"text": "content"}}
            if "delta" in chunk and "text" in chunk["delta"]:
                return chunk["delta"]["text"]
            
            # Alternative Anthropic format: {"text": "content"}
            if "text" in chunk:
                return chunk["text"]
        
        return None
    
    def _extract_openai_text(self, chunk: Any) -> Optional[str]:
        """Extract text from OpenAI chunk format"""
        if isinstance(chunk, str):
            return chunk
            
        if isinstance(chunk, dict):
            # OpenAI streaming format: {"choices": [{"delta": {"content": "text"}}]}
            if "choices" in chunk and chunk["choices"]:
                choice = chunk["choices"][0]
                if "delta" in choice and "content" in choice["delta"]:
                    return choice["delta"]["content"]
            
            # Direct delta format: {"delta": {"content": "text"}}
            if "delta" in chunk and "content" in chunk["delta"]:
                return chunk["delta"]["content"]
        
        return None
    
    def _extract_deepseek_text(self, chunk: Any) -> Optional[str]:
        """Extract text from DeepSeek chunk format (OpenAI-compatible)"""
        # DeepSeek uses OpenAI-compatible format
        return self._extract_openai_text(chunk)
    
    def _extract_gemini_text(self, chunk: Any) -> Optional[str]:
        """Extract text from Gemini chunk format"""
        if isinstance(chunk, str):
            return chunk
            
        if isinstance(chunk, dict):
            # Gemini format: {"candidates": [{"content": {"parts": [{"text": "content"}]}}]}
            if "candidates" in chunk:
                for candidate in chunk["candidates"]:
                    if "content" in candidate and "parts" in candidate["content"]:
                        for part in candidate["content"]["parts"]:
                            if "text" in part:
                                return part["text"]
            
            # Direct parts format: {"parts": [{"text": "content"}]}
            if "parts" in chunk:
                for part in chunk["parts"]:
                    if "text" in part:
                        return part["text"]
            
            # Simple text format: {"text": "content"}
            if "text" in chunk:
                return chunk["text"]
        
        return None
    
    def _extract_fallback_text(self, chunk: Any) -> Optional[str]:
        """Fallback extraction for unknown providers"""
        if isinstance(chunk, str):
            return chunk
            
        if isinstance(chunk, dict):
            # Try common text fields
            for field in ["text", "content", "message", "data"]:
                if field in chunk:
                    value = chunk[field]
                    if isinstance(value, str):
                        return value
        
        return None
    
    def is_completion_chunk(self, chunk: Any, provider: str) -> bool:
        """
        Check if chunk indicates stream completion.
        
        Returns True if this chunk signals end of stream.
        """
        if not chunk or not isinstance(chunk, dict):
            return False
            
        provider_lower = provider.lower()
        
        try:
            if provider_lower == "anthropic":
                return chunk.get("type") == "message_stop"
            
            elif provider_lower in ["openai", "deepseek"]:
                if "choices" in chunk and chunk["choices"]:
                    return chunk["choices"][0].get("finish_reason") == "stop"
            
            elif provider_lower == "gemini":
                if "candidates" in chunk and chunk["candidates"]:
                    return chunk["candidates"][0].get("finishReason") == "STOP"
            
        except Exception as e:
            logger.warning(f"Error checking completion for {provider}: {e}")
        
        return False
    
    def extract_metadata(self, chunk: Any, provider: str) -> Dict[str, Any]:
        """
        Extract metadata from chunk (usage info, model info, etc.)
        
        Returns dict with available metadata.
        """
        if not chunk or not isinstance(chunk, dict):
            return {}
            
        metadata = {"provider": provider}
        
        try:
            provider_lower = provider.lower()
            
            if provider_lower == "anthropic":
                if "model" in chunk:
                    metadata["model"] = chunk["model"]
                if "usage" in chunk:
                    metadata["usage"] = chunk["usage"]
            
            elif provider_lower in ["openai", "deepseek"]:
                if "model" in chunk:
                    metadata["model"] = chunk["model"]
                if "usage" in chunk:
                    metadata["usage"] = chunk["usage"]
                if "choices" in chunk and chunk["choices"]:
                    choice = chunk["choices"][0]
                    if "finish_reason" in choice:
                        metadata["finish_reason"] = choice["finish_reason"]
            
            elif provider_lower == "gemini":
                if "model" in chunk:
                    metadata["model"] = chunk["model"]
                if "candidates" in chunk and chunk["candidates"]:
                    candidate = chunk["candidates"][0]
                    if "safetyRatings" in candidate:
                        metadata["safety_ratings"] = candidate["safetyRatings"]
        
        except Exception as e:
            logger.warning(f"Error extracting metadata for {provider}: {e}")
        
        return metadata

# ================================
# USAGE EXAMPLES
# ================================

def example_usage():
    """Example of how to use StreamNormalizer"""
    normalizer = StreamNormalizer()
    
    # Different provider chunks
    anthropic_chunk = {"delta": {"text": "Hello world"}}
    openai_chunk = {"choices": [{"delta": {"content": "Hello"}}]}
    gemini_chunk = {"parts": [{"text": "Hello"}]}
    
    # Extract text uniformly
    text1 = normalizer.extract_text(anthropic_chunk, "anthropic")  # "Hello world"
    text2 = normalizer.extract_text(openai_chunk, "openai")        # "Hello"
    text3 = normalizer.extract_text(gemini_chunk, "gemini")        # "Hello"
    
    print(f"Extracted: {text1}, {text2}, {text3}")
    
    # Check completion
    completion_chunk = {"choices": [{"finish_reason": "stop"}]}
    is_done = normalizer.is_completion_chunk(completion_chunk, "openai")  # True
    
    print(f"Stream complete: {is_done}")

if __name__ == "__main__":
    example_usage()