# utils/llm_clients/deepseek_client.py
import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
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
    ):
        self.model_name = model_name
        self.temperature = temperature
        
        # Create both regular and streaming clients
        self._client = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            streaming=False  # Non-streaming client
        )
        
        self._streaming_client = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            streaming=True  # Dedicated streaming client
        )

    def chat(self, prompt: str, system_prompt: str = None) -> str:
        messages = self._build_messages(prompt, system_prompt)
        return self._client.invoke(messages).content
    
    def stream_chat(self, prompt: str, system_prompt: str = None):
        messages = self._build_messages(prompt, system_prompt)
        for chunk in self._streaming_client.stream(messages):
            if chunk.content:
                yield chunk.content

    def _build_messages(self, prompt: str, system_prompt: str = None):
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        return messages