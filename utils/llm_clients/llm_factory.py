# Lagacy Factory/Dispatcher (fetches clients from /llm_clients)

'''
`utils/llm_factory.py`: reads provider + model name + temperature and 
returns an instance of whichever client class you need.
'''

from typing import Any
from utils.llm_clients.openai_client import OpenAIClient
from utils.llm_clients.deepseek_client import DeepSeekClient
from utils.llm_clients.anthropic_client import AnthropicClient
from utils.llm_clients.gemini_client import GeminiClient
# from utils.llm_clients.qwen_client import QWENClient
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

class LLMFactory:
    """
    🎯 SIMPLE FACTORY: Just route provider names to client classes
    
    Does NOT handle:
    - Citation processing (citation_processor.py does this)
    - Stream normalization (stream_normalizer.py does this) 
    - Context building (your _build_enhanced_context does this)
    - Performance monitoring (performance_monitor.py does this)
    
    ONLY handles:
    - Creating the right client class for each provider
    - Consistent parameter passing
    - Error handling for unknown providers
    """
    
    @staticmethod
    def get_client_for(
        provider: str, 
        model_name: str, 
        temperature: float = 0.7, 
        streaming: bool = True,
        **kwargs
    ) -> Any:
        """
        Create LLM client for specified provider.
        
        Args:
            provider: Provider name ("anthropic", "openai", "deepseek", "gemini")
            model_name: Model identifier 
            temperature: Temperature setting (0.0-2.0)
            streaming: Enable streaming mode
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Configured client instance with .stream_chat() method
            
        Raises:
            ValueError: If provider is not supported
        """
        provider_key = provider.lower().strip()
        
        try:
            if provider_key == "anthropic":
                from .anthropic_client import AnthropicClient
                return AnthropicClient(
                    model_name=model_name,
                    temperature=temperature,
                    streaming=streaming,
                    **kwargs
                )
            
            elif provider_key == "openai":
                from .openai_client import OpenAIClient
                return OpenAIClient(
                    model_name=model_name,
                    temperature=temperature,
                    streaming=streaming,
                    **kwargs
                )
            
            elif provider_key == "deepseek":
                from .deepseek_client import DeepSeekClient
                return DeepSeekClient(
                    model_name=model_name,
                    temperature=temperature,
                    streaming=streaming,
                    **kwargs
                )
            
            elif provider_key == "gemini":
                from .gemini_client import GeminiClient
                return GeminiClient(
                    model_name=model_name,
                    temperature=temperature,
                    streaming=streaming,
                    **kwargs
                )
            
            else:
                available_providers = ["anthropic", "openai", "deepseek", "gemini"]
                raise ValueError(
                    f"Unsupported provider: '{provider}'. "
                    f"Available providers: {', '.join(available_providers)}"
                )
                
        except ImportError as e:
            logger.error(f"Failed to import {provider} client: {e}")
            raise ValueError(f"Provider '{provider}' client not available") from e
        
        except Exception as e:
            logger.error(f"Failed to create {provider} client: {e}")
            raise ValueError(f"Failed to create {provider} client: {str(e)}") from e
    
    @staticmethod
    def get_available_providers() -> list[str]:
        """Get list of supported providers"""
        return ["anthropic", "openai", "deepseek", "gemini"]
    
    @staticmethod
    def is_provider_supported(provider: str) -> bool:
        """Check if provider is supported"""
        return provider.lower().strip() in LLMFactory.get_available_providers()
    

# class LLMFactory:
#     """
#     Given a provider name + model + temperature, return an object with .chat(prompt) -> str
#     """
#     @staticmethod
#     def get_client(provider: str, model_name: str, temperature: float = 0.7, streaming: bool = False, callback_manager: Any = None):
#         provider = provider.lower()
#         if provider == "openai":
#             return OpenAIClient(model_name=model_name, temperature=temperature, streaming=streaming, callback_manager=callback_manager)
#         elif provider == "deepseek":
#             return DeepSeekClient(model_name=model_name, temperature=temperature, streaming=streaming, callback_manager=callback_manager)
#         elif provider == "anthropic":
#             return AnthropicClient(model_name=model_name, temperature=temperature)
#         elif provider == "gemini":
#             return GeminiClient(model_name=model_name, temperature=temperature)
#         elif provider == "qwen":
#             return QWENClient(model_name=model_name, temperature=temperature)
#         else:
#             raise ValueError(f"Unknown provider '{provider}'. Valid options: openai, deepseek, anthropic, gemini, sonnet, qwen.")
