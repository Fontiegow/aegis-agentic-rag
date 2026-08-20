from core.config import settings
from .base import BaseLLMProvider
from .ollama import OllamaProvider
from .vllm import OpenAICompatibleProvider


def get_llm_provider(provider_name: str) -> BaseLLMProvider:
    provider = provider_name.lower()
    if provider == "ollama":
        return OllamaProvider(base_url=settings.OLLAMA_BASE_URL)
    elif provider == "vllm":
        return OpenAICompatibleProvider(base_url=settings.VLLM_BASE_URL)
    elif provider == "openrouter":
        return OpenAICompatibleProvider(
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY
        )
    else:
        raise ValueError(f"Unsupported LLM provider: '{provider_name}'")