# utils/llm_clients/anthropic_client.py

import os
import json
from anthropic import Anthropic, AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not set")


class AnthropicClient:
    def __init__(self, model_name: str = "claude-3-5-sonnet-20240620", temperature: float = 0.7, max_tokens: int = 4096, streaming: bool = False, callback_manager=None, **kwargs):
        # Use both sync and async clients for flexibility
        self._client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self._async_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.streaming = streaming
        self.callback_manager = callback_manager

    def chat(self, prompt: str, system_prompt: str = None) -> str:
        """
        Send a chat message to Claude using the Messages API.
        """
        user_prompt = str(prompt)
            
        messages = [{"role": "user", "content": user_prompt}]
        
        request_params = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages
        }
        
        if system_prompt:
            request_params["system"] = system_prompt
        
        if self.streaming:
            # This now correctly calls _stream_response without the 'stream' key
            return self._stream_response(**request_params)
        else:
            response = self._client.messages.create(**request_params)
            return response.content[0].text

    def _stream_response(self, **request_params) -> str:
        """Internal method to handle streaming and return complete response."""
        # FIX 1: Removed `request_params["stream"] = True`
        # The .stream() method does not accept a 'stream' argument.
        full_response = ""
        
        with self._client.messages.stream(**request_params) as stream:
            for text in stream.text_stream:
                full_response += text
                if self.callback_manager:
                    try:
                        self.callback_manager.on_llm_new_token(text)
                    except Exception:
                        pass
        
        return full_response

    def chat_with_history(self, messages: list, system_prompt: str = None) -> str:
        """
        Send a conversation with message history to Claude.
        """
        request_params = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages
        }
        
        if system_prompt:
            request_params["system"] = system_prompt
        
        response = self._client.messages.create(**request_params)
        return response.content[0].text

    async def stream_chat(self, prompt: str, system_prompt: str = None):
        """
        Asynchronously stream a chat response from Claude.
        """
        if isinstance(prompt, dict):
            user_prompt = prompt.get('prompt', '') or prompt.get('query', '') or str(prompt)
            if not system_prompt and 'system_prompt' in prompt:
                system_prompt = prompt['system_prompt']
        else:
            user_prompt = str(prompt)
            
        messages = [{"role": "user", "content": user_prompt}]
        
        # FIX 2: Removed `"stream": True` from the dictionary.
        # The call to .stream() handles this implicitly.
        request_params = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
        }
        
        if system_prompt:
            request_params["system"] = system_prompt
        
        # Use the dedicated async client for async operations
        async with self._async_client.messages.stream(**request_params) as stream:
            async for text in stream.text_stream:
                yield text