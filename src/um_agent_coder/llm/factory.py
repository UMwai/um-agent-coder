from typing import Dict, Any
from .base import LLM
from .providers.openai import OpenAILLM
from .providers.anthropic import AnthropicLLM
from .providers.google import GoogleLLM


class LLMFactory:
    """Factory for creating LLM providers."""
    
    PROVIDERS = {
        "openai": OpenAILLM,
        "anthropic": AnthropicLLM,
        "google": GoogleLLM,
    }
    
    @classmethod
    def create(cls, provider: str, config: Dict[str, Any]) -> LLM:
        """
        Create an LLM provider instance.
        
        Args:
            provider: Provider name (openai, anthropic, google)
            config: Provider-specific configuration
            
        Returns:
            LLM instance
        """
        if provider not in cls.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")
        
        provider_class = cls.PROVIDERS[provider]
        return provider_class(**config)
    
    @classmethod
    def get_available_providers(cls) -> list:
        """Get list of available providers."""
        return list(cls.PROVIDERS.keys())