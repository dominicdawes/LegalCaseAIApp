# utils/llm_clients/deepseek_client.py
import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Remove the SSL_CERT_FILE variable from the environment if it exists.
os.environ.pop("SSL_CERT_FILE", None)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")  # e.g. "https://api.deepseek.ai/v1"

if not DEEPSEEK_API_KEY or not DEEPSEEK_BASE_URL:
    raise ValueError("DeepSeek credentials or base URL not found in environment")

class DeepSeekClient:
    def __init__(
        self,
        model_name: str = "deepseek-chat",
        temperature: float = 0.7,
        streaming: bool = False,
        callback_manager=None,
    ):
        
        # make model_name available on the client
        self.model_name = model_name
        
        llm_kwargs = {
            "model_name": model_name,
            "temperature": temperature,
            "streaming": streaming,
            "openai_api_key": DEEPSEEK_API_KEY,
            "openai_api_base": DEEPSEEK_BASE_URL,
        }
        self._client = ChatOpenAI(**llm_kwargs)

    def chat(self, prompt: str) -> str:
        return self._client.predict(prompt)
    
    def stream_chat(self, prompt: str, system_prompt: str = None):
        """Stream a chat response from DeepSeek"""
        # Create streaming version
        streaming_client = ChatOpenAI(
            model_name=self.model_name,
            temperature=0.7,
            streaming=True,
            openai_api_key=DEEPSEEK_API_KEY,
            openai_api_base=DEEPSEEK_BASE_URL,
        )
        
        # Format prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        # Stream response
        for chunk in streaming_client.stream(full_prompt):
            if chunk.content:
                yield chunk.content