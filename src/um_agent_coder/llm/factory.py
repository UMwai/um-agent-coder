import os
from typing import Dict, Any, Union
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
    def create(cls, provider: str, config: Union[Dict[str, Any], Any]) -> LLM:
        """
        Create an LLM provider instance.
        
        Args:
            provider: Provider name (openai, anthropic, google)
            config: Configuration object or dictionary
            
        Returns:
            LLM instance
        """
        if provider not in cls.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Available: {', '.join(cls.PROVIDERS.keys())}")
        
        # Extract provider config
        if hasattr(config, 'get'):
            # Config object with get method
            provider_config = config.get(f"llm.{provider}", {})
            if isinstance(provider_config, dict):
                # Handle environment variable substitution
                api_key = provider_config.get("api_key", "")
                if api_key.startswith("${") and api_key.endswith("}"):
                    env_var = api_key[2:-1]
                    api_key = os.environ.get(env_var, "")
                    if not api_key:
                        # Try without provider prefix
                        api_key = os.environ.get(f"{provider.upper()}_API_KEY", "")
                    provider_config["api_key"] = api_key
                
                # Set default model if not specified
                if "model" not in provider_config:
                    default_models = {
                        "openai": "gpt-4",
                        "anthropic": "claude-3-opus-20240229",
                        "google": "gemini-pro"
                    }
                    provider_config["model"] = default_models.get(provider, "default")
            else:
                provider_config = {}
        elif isinstance(config, dict):
            # Direct dictionary config
            provider_config = config.get(provider, {})
        else:
            provider_config = {}
        
        # Validate API key
        if not provider_config.get("api_key"):
            # Try environment variable
            env_key = f"{provider.upper()}_API_KEY"
            api_key = os.environ.get(env_key)
            if api_key:
                provider_config["api_key"] = api_key
            else:
                raise ValueError(f"No API key found for {provider}. Set {env_key} environment variable or add to config.")
        
        provider_class = cls.PROVIDERS[provider]
        return provider_class(**provider_config)
    
    @classmethod
    def get_available_providers(cls) -> list:
        """Get list of available providers."""
        return list(cls.PROVIDERS.keys())