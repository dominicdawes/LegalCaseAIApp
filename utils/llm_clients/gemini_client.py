# utils/llm_clients/gemini_client.py

import os
import asyncio
import time
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_AI_STUDIO", "").strip()

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY_AI_STUDIO environment variable not set")

# One client per process — holds API key and connection pool
_client = genai.Client(api_key=GEMINI_API_KEY)

# Permissive safety settings for legal/professional content.
# Gemini's default (when safety_settings=None) is BLOCK_MEDIUM_AND_ABOVE, which
# incorrectly flags case law content — labor coercion, dangerous-conditions fact
# patterns, etc. hit MEDIUM probability on harassment/dangerous-content categories.
# BLOCK_ONLY_HIGH still blocks genuinely harmful high-confidence content while
# allowing legal document analysis through.
_PERMISSIVE_SAFETY = [
    types.SafetySetting(
        category=cat,
        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    )
    for cat in (
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    )
]


class GeminiClient:
    """
    Gemini client using the google.genai SDK.
    Compatible with StreamingChatManager interface.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        max_output_tokens: int = 65536,
        top_p: float = 0.95,
        top_k: int = 40,
        streaming: bool = False,
        callback_manager=None,
        enable_safety_filters: bool = False,
        **kwargs
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.streaming = streaming
        self.callback_manager = callback_manager
        self.max_tokens = max_output_tokens

        # enable_safety_filters=True  → BLOCK_MEDIUM_AND_ABOVE (strict)
        # enable_safety_filters=False → BLOCK_ONLY_HIGH (permissive, legal content safe)
        # Never pass None — that defers to Gemini's default (BLOCK_MEDIUM_AND_ABOVE)
        # which incorrectly blocks case law content.
        if enable_safety_filters:
            self._safety_settings = [
                types.SafetySetting(
                    category=cat,
                    threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                )
                for cat in (
                    types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                )
            ]
        else:
            self._safety_settings = _PERMISSIVE_SAFETY

    def _build_config(self, system_prompt: str | None = None) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            safety_settings=self._safety_settings,
            system_instruction=system_prompt,
        )

    def chat(self, prompt: str, system_prompt: str = None) -> str:
        """Send chat message with optional system prompt."""
        try:
            response = _client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self._build_config(system_prompt),
            )
            return response.text or ""
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}")

    async def stream_chat(self, prompt: str, system_prompt: str = None):
        """Stream chat response using the native async SDK — no threading needed."""
        SUB_CHUNK_SIZE = 15
        SUB_CHUNK_DELAY = 0.02

        async for chunk in await _client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=prompt,
            config=self._build_config(system_prompt),
        ):
            text = chunk.text
            if text:
                for i in range(0, len(text), SUB_CHUNK_SIZE):
                    yield text[i : i + SUB_CHUNK_SIZE]
                    await asyncio.sleep(SUB_CHUNK_DELAY)

    def chat_with_retry(self, prompt: str, max_retries: int = 3, system_prompt: str = None) -> str:
        """Chat with automatic retry on provider outage."""
        for attempt in range(max_retries):
            try:
                return self.chat(prompt, system_prompt)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Gemini failed after {max_retries} attempts: {e}")
                wait_time = 2 ** attempt
                print(f"⚠️ Gemini attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

    async def stream_chat_with_retry(self, prompt: str, max_retries: int = 3, system_prompt: str = None):
        """Streaming chat with retry logic."""
        for attempt in range(max_retries):
            try:
                async for chunk in self.stream_chat(prompt, system_prompt):
                    yield chunk
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Gemini streaming failed after {max_retries} attempts: {e}")
                wait_time = 2 ** attempt
                print(f"⚠️ Gemini streaming attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
