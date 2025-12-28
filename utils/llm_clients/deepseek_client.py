# utils/llm_clients/deepseek_client.py
import os
from typing import Optional, Dict, Any
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
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
    # Model name mappings for different DeepSeek models
    MODEL_MAPPINGS = {
        # Latest models (recommended)
        "deepseek-chat": "deepseek-chat",  # Default chat model
        "deepseek-coder": "deepseek-coder",  # Latest coding model
        
        # Reasoning models (newer)
        "deepseek-reasoner": "deepseek-reasoner",  # Latest reasoning model
        
        # Legacy/older models (if still supported by API)
        "deepseek-3.1": "deepseek-llm-67b-chat",  # Legacy naming for 3.1
        "deepseek-3.2": "deepseek-llm-67b-chat",  # May need to check exact name
        "deepseek-3.2-speciale": "deepseek-reasoner",  # Special reasoning variant
        
        # Alternative names to try
        "deepseek-v3": "deepseek-chat",
        "deepseek-v2.5": "deepseek-chat",
        "deepseek-llm-67b": "deepseek-llm-67b-chat",
    }
    
    def __init__(
        self,
        model_name: str = "deepseek-chat", # <-- default to: deepseek-chat 
        temperature: float = 0.7,
        streaming: bool = False,
        max_output_tokens: int = 4096,
        base_url: Optional[str] = None,
        **kwargs
    ):
        # Map model name to actual API model name
        self.actual_model_name = self.MODEL_MAPPINGS.get(model_name, model_name)
        self.model_name = model_name  # Keep original for reference
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        
        # Use provided base_url or default
        self.base_url = base_url or DEEPSEEK_BASE_URL
        
        # Additional parameters
        self.kwargs = kwargs
        
        # Create both regular and streaming clients
        common_params = {
            "model": self.actual_model_name,
            "temperature": temperature,
            "api_key": DEEPSEEK_API_KEY,
            "base_url": self.base_url,
            "max_tokens": max_output_tokens,
            **kwargs
        }
        
        self._client = ChatOpenAI(
            **common_params,
            streaming=False
        )
        
        self._streaming_client = ChatOpenAI(
            **common_params,
            streaming=True
        )
        
        # Also create raw OpenAI client for direct API calls
        self._raw_client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=self.base_url
        )

    def chat(self, prompt: str, system_prompt: str = None) -> str:
        messages = self._build_messages(prompt, system_prompt)
        return self._client.invoke(messages).content
    
    def raw_chat_completion(self, messages: list, **kwargs) -> Dict[str, Any]:
        """
        Direct API call for more control over parameters
        """
        response = self._raw_client.chat.completions.create(
            model=self.actual_model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
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