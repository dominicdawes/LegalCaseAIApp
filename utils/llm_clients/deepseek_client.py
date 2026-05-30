# utils/llm_clients/deepseek_client.py
import os
from typing import Optional, Dict, Any
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Remove the SSL_CERT_FILE variable from the environment if it exists.
os.environ.pop("SSL_CERT_FILE", None)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com/v1")  # Default to deepseek.com

if not DEEPSEEK_API_KEY:
    raise ValueError("DeepSeek API key not found in environment")

class DeepSeekClient:
    # DeepSeek V4 model names (April 2026+).
    # deepseek-chat and deepseek-reasoner are now deprecated aliases for the
    # non-thinking and thinking modes of deepseek-v4-flash (deprecated July 24 2026).
    # Thinking is controlled by the `thinking` extra_body parameter, NOT model selection.
    MODEL_MAPPINGS = {
        # Current V4 models (recommended)
        "deepseek-v4-flash":     "deepseek-v4-flash",
        "deepseek-v4-pro":       "deepseek-v4-pro",
        # Deprecated aliases (will be removed July 24 2026)
        "deepseek-reasoner":     "deepseek-reasoner",   # thinking=enabled alias of v4-flash
        "deepseek-chat":         "deepseek-chat",       # thinking=disabled alias of v4-flash
        "deepseek-coder":        "deepseek-coder",
        # Legacy V3 / older
        "deepseek-v3":           "deepseek-chat",
        "deepseek-v2.5":         "deepseek-chat",
        "deepseek-llm-67b":      "deepseek-llm-67b-chat",
        "deepseek-3.1":          "deepseek-llm-67b-chat",
        "deepseek-3.2":          "deepseek-llm-67b-chat",
        "deepseek-3.2-speciale": "deepseek-reasoner",
    }

    # Legacy model names that imply thinking=disabled (non-thinking aliases)
    _NON_THINKING_ALIASES = {"deepseek-chat", "deepseek-v3", "deepseek-v2.5"}

    def __init__(
        self,
        model_name: str = "deepseek-v4-flash",
        temperature: float = 0.7,
        streaming: bool = False,
        max_output_tokens: int = 4096,
        base_url: Optional[str] = None,
        **kwargs
    ):
        # Map model name (pass-through for V4; aliases for legacy names)
        self.actual_model_name = self.MODEL_MAPPINGS.get(model_name, model_name)
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        # Handle base URL (without /v1 suffix)
        self.base_url = base_url or "https://api.deepseek.com"

        # ── Thinking parameter (DeepSeek V4 API, April 2026+) ─────────────────
        # V4 models default to thinking=enabled.  Must be passed via extra_body
        # because it is a DeepSeek-specific extension not in the OpenAI spec.
        # Callers pass `thinking=True/False`; we translate to the required dict form.
        thinking_enabled = kwargs.pop("thinking", True)

        # Legacy non-thinking aliases should default to thinking=disabled
        if model_name in self._NON_THINKING_ALIASES:
            thinking_enabled = False

        thinking_param = {"type": "enabled"} if thinking_enabled else {"type": "disabled"}
        self._raw_thinking_param = thinking_param  # saved for raw_chat_completion

        # Inject into model_kwargs → extra_body so LangChain forwards it correctly
        model_kwargs = kwargs.pop("model_kwargs", {})
        model_kwargs["extra_body"] = {
            **model_kwargs.get("extra_body", {}),
            "thinking": thinking_param,
        }

        # Create clients
        common_params = {
            "model": self.actual_model_name,
            "temperature": temperature,
            "api_key": DEEPSEEK_API_KEY,
            "base_url": self.base_url,
            "max_tokens": max_output_tokens,
            "model_kwargs": model_kwargs,
            **kwargs
        }

        self._client = ChatOpenAI(**common_params, streaming=False)
        self._streaming_client = ChatOpenAI(**common_params, streaming=True)
        self._raw_client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=self.base_url
        )

    def chat(self, prompt: str, system_prompt: str = None) -> str:
        messages = self._build_messages(prompt, system_prompt)
        return self._client.invoke(messages).content
    
    def raw_chat_completion(self, messages: list, **kwargs) -> Dict[str, Any]:
        """Direct API call — passes thinking via extra_body for DeepSeek V4."""
        extra_body = kwargs.pop("extra_body", {})
        extra_body["thinking"] = self._raw_thinking_param
        response = self._raw_client.chat.completions.create(
            model=self.actual_model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            extra_body=extra_body,
            **kwargs
        )
        return response.dict()
    
    def list_available_models(self):
        """
        List all available models from the DeepSeek API
        """
        try:
            models = self._raw_client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []

    def stream_chat_sync(self, prompt: str, system_prompt: str = None):
        """Legacy sync version"""
        messages = self._build_messages(prompt, system_prompt)
        for chunk in self._streaming_client.stream(messages):
            if chunk.content:
                yield chunk.content
    
    async def stream_chat(self, prompt: str, system_prompt: str = None):
        """Async streaming method"""
        messages = self._build_messages(prompt, system_prompt)
        async for chunk in self._streaming_client.astream(messages):
            if chunk.content:
                yield chunk.content
    
    def _build_messages(self, prompt: str, system_prompt: str = None):
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        return messages