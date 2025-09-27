# utils/llm_clients/anthropic_client.py

import os
import json
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not set")


class AnthropicClient:
    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", temperature: float = 0.7, max_tokens: int = 1024, streaming: bool = False, callback_manager=None, **kwargs):
        self._client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.streaming = streaming
        self.callback_manager = callback_manager
        # Accept any additional kwargs to be flexible with factory patterns

    def chat(self, prompt: str, system_prompt: str = None) -> str:
        """
        Send a chat message to Claude using the Messages API.
        
        Args:
            prompt: The user's message/prompt (can also be called as positional arg for compatibility)
            system_prompt: Optional system prompt to set context
            
        Returns:
            Claude's response as a string
        """
        # Handle different parameter naming conventions
        if isinstance(prompt, str):
            user_prompt = prompt
        else:
            # Handle case where first arg might be a different parameter name
            user_prompt = str(prompt)
            
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        
        # Prepare the request parameters
        request_params = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages
        }
        
        # Add system prompt if provided
        if system_prompt:
            request_params["system"] = system_prompt
        
        # Use streaming if enabled
        if self.streaming:
            return self._stream_response(**request_params)
        else:
            response = self._client.messages.create(**request_params)
            return response.content[0].text

    def _stream_response(self, **request_params) -> str:
        """Internal method to handle streaming and return complete response."""
        request_params["stream"] = True
        full_response = ""
        
        with self._client.messages.stream(**request_params) as stream:
            for text in stream.text_stream:
                full_response += text
                if self.callback_manager:
                    # Call callback if provided (for logging, etc.)
                    try:
                        self.callback_manager.on_llm_new_token(text)
                    except:
                        pass  # Ignore callback errors
        
        return full_response

    def chat_with_history(self, messages: list, system_prompt: str = None) -> str:
        """
        Send a conversation with message history to Claude.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            system_prompt: Optional system prompt to set context
            
        Returns:
            Claude's response as a string
        """
        # Prepare the request parameters
        request_params = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages
        }
        
        # Add system prompt if provided
        if system_prompt:
            request_params["system"] = system_prompt
        
        response = self._client.messages.create(**request_params)
        
        # Extract the text content from the response
        return response.content[0].text

    async def stream_chat(self, prompt: str, system_prompt: str = None):
        """
        Stream a chat response from Claude.
        
        Args:
            prompt: The user's message/prompt (can be string or dict with context)
            system_prompt: Optional system prompt to set context
            
        Yields:
            Text chunks as they arrive
        """
        # Handle both string prompts and context dicts
        if isinstance(prompt, dict):
            # If prompt is a context dict, extract the actual prompt
            user_prompt = prompt.get('prompt', '') or prompt.get('query', '') or str(prompt)
            # You might also want to extract system_prompt from context if not provided
            if not system_prompt and 'system_prompt' in prompt:
                system_prompt = prompt['system_prompt']
        else:
            user_prompt = prompt
            
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        
        # Prepare the request parameters
        request_params = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
            "stream": True
        }
        
        # Add system prompt if provided
        if system_prompt:
            request_params["system"] = system_prompt
        
        # Use async context manager for streaming
        async with self._client.messages.stream(**request_params) as stream:
            async for text in stream.text_stream:
                yield text