# utils/llm_clients/gemini_client.py

import os
import json
import asyncio
import time
from dotenv import load_dotenv

import vertexai
from vertexai.generative_models import (
    GenerativeModel, GenerationConfig, Content, Part, 
    ResponseBlockedError, HarmCategory, HarmBlockThreshold
)
from google.oauth2 import service_account
from google.api_core import exceptions as google_exceptions

load_dotenv()

# Configuration
GOOGLE_PROJECT_ID = os.getenv("GEMINI_PROJECT_ID", "").strip()
GEMINI_LOCATION = os.getenv("GEMINI_LOCATION", "us-central1").strip()
CREDENTIALS_JSON_STR = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "")
CREDENTIALS_FILE_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_FILE", "")

if not GOOGLE_PROJECT_ID:
    raise ValueError("GEMINI_PROJECT_ID environment variable not set")

# Initialize Vertex AI
try:
    if CREDENTIALS_FILE_PATH:
        credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE_PATH)
        vertexai.init(project=GOOGLE_PROJECT_ID, location=GEMINI_LOCATION, credentials=credentials)
    elif CREDENTIALS_JSON_STR:
        info = json.loads(CREDENTIALS_JSON_STR)
        credentials = service_account.Credentials.from_service_account_info(info)
        vertexai.init(project=GOOGLE_PROJECT_ID, location=GEMINI_LOCATION, credentials=credentials)
    else:
        vertexai.init(project=GOOGLE_PROJECT_ID, location=GEMINI_LOCATION)
except Exception as e:
    raise RuntimeError(f"Failed to initialize Vertex AI: {e}")

class GeminiClient:
    """
    Gemini client with streaming support and production-grade error handling.
    Compatible with StreamingChatManager interface.
    """
    
    def __init__(
        self, 
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
        top_p: float = 0.95,
        top_k: int = 40,
        streaming: bool = False,
        callback_manager=None,
        enable_safety_filters: bool = False,  # 🆕 Optional safety
        **kwargs
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.streaming = streaming
        self.callback_manager = callback_manager
        self.max_tokens = max_output_tokens
        
        self.generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k
        )
        
        # 🆕 Optional safety settings (disabled by default for testing)
        self.safety_settings = None
        if enable_safety_filters:
            self.safety_settings = {
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        
        self._model = GenerativeModel(self.model_name)

    def _handle_response(self, response):
        """Extract text from Gemini response with safety checking"""
        if not response.candidates:
            return ""
        
        candidate = response.candidates[0]
        
        # Check safety ratings
        if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
            blocked = [r.category.name for r in candidate.safety_ratings if r.blocked]
            if blocked:
                return "Response blocked due to safety concerns."
        
        if not candidate.content or not candidate.content.parts:
            return ""
        
        first_part = candidate.content.parts[0]
        return first_part.text if hasattr(first_part, 'text') else ""

    def chat(self, prompt: str, system_prompt: str = None) -> str:
        """Send chat message with optional system prompt"""
        try:
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            # Only pass safety_settings if enabled
            kwargs = {
                "generation_config": self.generation_config
            }
            if self.safety_settings:
                kwargs["safety_settings"] = self.safety_settings
            
            response = self._model.generate_content(full_prompt, **kwargs)
            return self._handle_response(response)
            
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}")

    def stream_chat(self, prompt: str, system_prompt: str = None):
        """Stream chat response with optional system prompt"""
        try:
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            # Only pass safety_settings if enabled
            kwargs = {
                "generation_config": self.generation_config,
                "stream": True
            }
            if self.safety_settings:
                kwargs["safety_settings"] = self.safety_settings
            
            response_stream = self._model.generate_content(full_prompt, **kwargs)
            
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            raise RuntimeError(f"Gemini streaming error: {e}")

    # Add retry methods for consistency with other clients
    def chat_with_retry(self, prompt: str, max_retries: int = 3, system_prompt: str = None) -> str:
        """Chat with automatic retry on provider outage"""
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
        """Streaming chat with retry logic"""
        for attempt in range(max_retries):
            try:
                for chunk in self.stream_chat(prompt, system_prompt):
                    yield chunk
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Gemini streaming failed after {max_retries} attempts: {e}")
                
                wait_time = 2 ** attempt
                print(f"⚠️ Gemini streaming attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)