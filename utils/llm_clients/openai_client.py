# utils/llm_clients/openai_client.py

import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class OpenAIClient:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7, streaming: bool = False, callback_manager=None):
        cfg = {
            "o4-mini":      {"supports_temperature": False},
            "gpt-4o-mini":  {"supports_temperature": False},
            "gpt-4.1-nano": {"supports_temperature": True,  "min": 0.0, "max": 2.0},
            # … add others if needed …
        }
        self.model_name = model_name
        llm_kwargs = {
            "api_key": OPENAI_API_KEY,
            "model": model_name,
            "streaming": streaming,
        }
        # If temperature is supported, clamp it
        info = cfg.get(model_name, {"supports_temperature": True})
        if info.get("supports_temperature", False):
            lo, hi = info.get("min", 0.0), info.get("max", 2.0)
            safe_temp = max(lo, min(temperature, hi))
            llm_kwargs["temperature"] = safe_temp

        if streaming and callback_manager:
            llm_kwargs["callback_manager"] = callback_manager

        self._client = ChatOpenAI(**llm_kwargs)  # uses langchain_openai under the hood

    def chat(self, prompt: str) -> str:
        """
        Sends a single‐turn prompt; returns the entire response as a string.
        """
        return self._client.predict(prompt)
