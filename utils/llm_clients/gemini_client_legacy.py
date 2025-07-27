# utils/llm_clients/gemini_client.py

import os
import json
from dotenv import load_dotenv

import vertexai
from vertexai.generative_models import GenerativeModel # Corrected import
from google.oauth2 import service_account
"gemini-2.5-flash-preview-05-20"
load_dotenv()

GOOGLE_PROJECT_ID = os.getenv("GEMINI_PROJECT_ID", "").strip()
GEMINI_LOCATION = os.getenv("GEMINI_LOCATION", "us-central1").strip()
CREDENTIALS_JSON_STR = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "")

if not GOOGLE_PROJECT_ID:
    raise ValueError("GEMINI_PROJECT_ID not set")

if CREDENTIALS_JSON_STR:
    info = json.loads(CREDENTIALS_JSON_STR)
    credentials = service_account.Credentials.from_service_account_info(info)
    vertexai.init(project=GOOGLE_PROJECT_ID, location=GEMINI_LOCATION, credentials=credentials)
else:
    vertexai.init(project=GOOGLE_PROJECT_ID, location=GEMINI_LOCATION)

class GeminiClient:
    def __init__(self, model_name: str = "gemini-2.5-flash-preview-05-20", temperature: float = 0.7, max_output_tokens: int = 512):
        """
        model_name must match one of the “pretrained” Gemini model IDs your project has access to.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def chat(self, prompt: str, system_prompt: str = None) -> str:
        """Send a chat message to Gemini with optional system prompt"""
        try:
            model = GenerativeModel(self.model_name)
            
            # Combine system prompt if provided
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            response = model.generate_content(
                full_prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                }
            )
            
            return response.candidates[0].content.text
            
        except Exception as e:
            raise RuntimeError(f"Gemini API error calling `{self.model_name}`: {e}")

    def stream_chat(self, prompt: str, system_prompt: str = None):
        """Stream a chat response from Gemini"""
        try:
            model = GenerativeModel(self.model_name)
            
            # Combine system prompt if provided
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            # Gemini streaming
            response = model.generate_content(
                full_prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                },
                stream=True  # Enable streaming
            )
            
            # Yield chunks as they arrive
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            raise RuntimeError(f"Gemini streaming error: {e}")