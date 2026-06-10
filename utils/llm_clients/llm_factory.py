# Lagacy Factory/Dispatcher (fetches clients from /llm_clients)

'''
`utils/llm_factory.py`: reads provider + model name + temperature and
returns an instance of whichever client class you need.
'''

import asyncio
from typing import Any, List, Optional, Tuple
# Provider clients are imported lazily inside get_client_for() so that a missing
# credential for one provider (e.g. no GEMINI_PROJECT_ID) doesn't prevent the
# module from loading when a different provider is active.
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

# ── Capacity-error fallback chains per worker tier ────────────────────────────
# On a 503 / capacity error the caller retries in order until one succeeds.
# Ordered cheapest-reliable-first so cost stays low under pressure.
WORKER_FALLBACK_CHAINS: dict = {
    "worker_low":  [("deepseek", "deepseek-v4-flash"), ("openai", "gpt-4o-mini")],
    "worker_mid":  [("deepseek", "deepseek-v4-pro"),   ("openai", "gpt-4o")],
    "orchestrator":[("deepseek", "deepseek-v4-pro"),   ("openai", "o4-mini")],
}

# Fallback chain used by the verification tool (always worker_low equivalent)
VERIFY_FALLBACK_CHAIN: List[Tuple[str, str]] = WORKER_FALLBACK_CHAINS["worker_low"]


def _is_capacity_error(exc: Exception) -> bool:
    """Return True for 503 / capacity-overload errors from any provider."""
    err = str(exc).lower()
    return (
        "503" in err
        or "unavailable" in err
        or "high demand" in err
        or "overloaded" in err
        or "capacity" in err
    )

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
        max_output_tokens: int = 4056,
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
                    max_output_tokens=max_output_tokens,
                    **kwargs
                )
            
            elif provider_key == "openai":
                from .openai_client import OpenAIClient
                return OpenAIClient(
                    model_name=model_name,
                    temperature=temperature,
                    streaming=streaming,
                    max_output_tokens=max_output_tokens,
                    **kwargs
                )
            
            elif provider_key == "deepseek":
                from .deepseek_client import DeepSeekClient
                return DeepSeekClient(
                    model_name=model_name,
                    temperature=temperature,
                    streaming=streaming,
                    max_output_tokens=max_output_tokens,
                    **kwargs
                )
            
            elif provider_key == "gemini":
                from .gemini_client import GeminiClient
                return GeminiClient(
                    model_name=model_name,
                    temperature=temperature,
                    streaming=streaming,
                    max_output_tokens=max_output_tokens,
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
    def get_langchain_model(tier: str = "mid"):
        """
        Return a LangChain BaseChatModel for LangGraph nodes.

        tier options:
            "cheap"    → claude-haiku-4-5-20251001  (fast, low-cost; grounding, routing)
            "mid"      → claude-sonnet-4-6           (default; most nodes)
            "flagship" → claude-opus-4-7             (drafting + answer-key nodes)
        """
        from langchain_anthropic import ChatAnthropic

        model_map = {
            "cheap": "claude-haiku-4-5-20251001",
            "mid": "claude-sonnet-4-6",
            "flagship": "claude-opus-4-7",
        }
        model_name = model_map.get(tier, "claude-sonnet-4-6")
        return ChatAnthropic(model=model_name, max_tokens=4096)

    @staticmethod
    async def async_call_with_fallback(
        provider: str,
        model: str,
        prompt: str,
        system: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        fallback_chain: Optional[List[Tuple[str, str]]] = None,
        **client_kwargs,
    ) -> str:
        """
        Call the primary (provider, model); on a 503/capacity error try each
        (provider, model) pair in fallback_chain in order.

        The `thinking` kwarg is automatically stripped for non-DeepSeek providers
        so DeepSeek-specific kwargs don't cause errors on fallback targets.

        Returns the first successful response string.
        Raises the last exception if all options are exhausted.
        """
        chain = [(provider, model)] + (fallback_chain or [])
        last_exc: Exception = RuntimeError("No LLM providers configured")

        for i, (attempt_provider, attempt_model) in enumerate(chain):
            try:
                kwargs = {**client_kwargs}
                if attempt_provider != "deepseek":
                    kwargs.pop("thinking", None)

                client = LLMFactory.get_client_for(
                    attempt_provider, attempt_model,
                    temperature=temperature, streaming=False,
                    max_output_tokens=max_tokens,
                    **kwargs,
                )
                if hasattr(client, "achat"):
                    return await client.achat(prompt, system_prompt=system or None)
                parts: list = []
                async for chunk in client.stream_chat(prompt, system_prompt=system or None):
                    parts.append(chunk)
                return "".join(parts)

            except Exception as exc:
                if _is_capacity_error(exc):
                    if i < len(chain) - 1:
                        logger.warning(
                            "⚡ Capacity error on %s/%s — falling back to %s/%s",
                            attempt_provider, attempt_model,
                            chain[i + 1][0], chain[i + 1][1],
                        )
                    last_exc = exc
                    continue
                raise  # non-capacity errors propagate immediately

        raise last_exc

    @staticmethod
    def get_available_providers() -> list[str]:
        """Get list of supported providers"""
        return ["anthropic", "openai", "deepseek", "gemini"]
    
    @staticmethod
    def is_provider_supported(provider: str) -> bool:
        """Check if provider is supported"""
        return provider.lower().strip() in LLMFactory.get_available_providers()
    

