# utils/llm_clients/openai_client.py

import os
from langchain_openai import ChatOpenAI
import asyncio
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class OpenAIClient:
    def __init__(
            self, 
            model_name: str = "gpt-4o-mini", 
            temperature: float = 0.7, 
            max_output_tokens: int = 4096, 
            streaming: bool = False, 
            callback_manager=None
    ):
        cfg = {
            "o4-mini":      {"supports_temperature": False, "use_max_completion_tokens": True},
            "gpt-4o-mini":  {"supports_temperature": False},
            "gpt-4.1-nano": {"supports_temperature": True,  "min": 0.0, "max": 2.0},
            "gpt-5":        {"supports_temperature": False, "use_max_completion_tokens": True},
        }

        self.model_name = model_name
        self._temperature = temperature
        self.max_output_tokens = max_output_tokens  # Store for streaming usage

        # Newer models (o-series, gpt-5) use max_completion_tokens instead of max_tokens
        info = cfg.get(model_name, {"supports_temperature": True})
        tokens_key = "max_completion_tokens" if info.get("use_max_completion_tokens") else "max_tokens"

        llm_kwargs = {
            "api_key": OPENAI_API_KEY,
            "model": model_name,
            "streaming": streaming,
            tokens_key: max_output_tokens,
        }

        # If temperature is supported, clamp it; otherwise pass None to suppress
        # ChatOpenAI's own default of 0.7 which would be rejected by these models
        if info.get("supports_temperature", True):
            lo, hi = info.get("min", 0.0), info.get("max", 2.0)
            safe_temp = max(lo, min(temperature, hi))
            llm_kwargs["temperature"] = safe_temp
        else:
            llm_kwargs["temperature"] = None

        if streaming and callback_manager:
            llm_kwargs["callback_manager"] = callback_manager

        self._client = ChatOpenAI(**llm_kwargs)

    def chat(self, prompt: str) -> str:
        """Sends a single-turn prompt; returns the entire response as a string."""
        return self._client.predict(prompt)

    def _streaming_client_kwargs(self) -> dict:
        """Build kwargs for a streaming ChatOpenAI client, using the correct token param."""
        cfg = {
            "o4-mini":      {"supports_temperature": False, "use_max_completion_tokens": True},
            "gpt-4o-mini":  {"supports_temperature": False},
            "gpt-4.1-nano": {"supports_temperature": True},
            "gpt-5":        {"supports_temperature": False, "use_max_completion_tokens": True},
        }
        info = cfg.get(self.model_name, {"supports_temperature": True})
        tokens_key = "max_completion_tokens" if info.get("use_max_completion_tokens") else "max_tokens"
        kwargs = {
            "api_key": OPENAI_API_KEY,
            "model": self.model_name,
            "streaming": True,
            tokens_key: self.max_output_tokens,
            "temperature": getattr(self, '_temperature', 0.7) if info.get("supports_temperature", True) else None,
        }
        return kwargs

    async def stream_chat(self, prompt: str, system_prompt: str = None):
        """Stream a chat response from OpenAI"""
        streaming_client = ChatOpenAI(**self._streaming_client_kwargs())
        
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        async for chunk in streaming_client.astream(full_prompt):
            if chunk.content:
                yield chunk.content

    def stream_chat_sync(self, prompt: str, system_prompt: str = None):
        """Stream a chat response from OpenAI (Sync)"""
        streaming_client = ChatOpenAI(**self._streaming_client_kwargs())
        
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        for chunk in streaming_client.stream(full_prompt):
            if chunk.content:
                yield chunk.content
