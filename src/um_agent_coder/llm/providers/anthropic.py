import json
import os
from typing import Any, Optional

import requests

from um_agent_coder.llm.base import LLM
from um_agent_coder.models import ModelRegistry


class AnthropicLLM(LLM):
    """
    LLM provider for Anthropic Claude models.
    """

    API_URL = "https://api.anthropic.com/v1/messages"

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3.5-sonnet-20241022",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_registry = ModelRegistry()

        if not self.api_key:
            raise ValueError("Anthropic API key not provided")

        # Use a persistent session for connection pooling
        self.session = requests.Session()

    def chat(self, prompt: str, messages: Optional[list[dict[str, str]]] = None) -> str:
        """
        Send a chat request to Anthropic API.

        Args:
            prompt: The user prompt
            messages: Optional conversation history

        Returns:
            The model's response
        """
        if messages is None:
            messages = []

        # Convert to Anthropic format
        anthropic_messages = []
        for msg in messages:
            anthropic_messages.append(
                {
                    "role": msg["role"] if msg["role"] != "system" else "user",
                    "content": msg["content"],
                }
            )

        # Add the new user message
        anthropic_messages.append({"role": "user", "content": prompt})

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            # Use session for connection pooling
            response = self.session.post(self.API_URL, headers=headers, json=data)
            response.raise_for_status()

            result = response.json()
            return result["content"][0]["text"]

        except requests.exceptions.RequestException as e:
            return f"Error calling Anthropic API: {str(e)}"
        except (KeyError, IndexError) as e:
            return f"Error parsing Anthropic response: {str(e)}"

    def stream_chat(self, prompt: str, messages: Optional[list[dict[str, str]]] = None):
        """
        Stream a chat response from Anthropic API.
        """
        if messages is None:
            messages = []

        anthropic_messages = []
        for msg in messages:
            anthropic_messages.append(
                {
                    "role": msg["role"] if msg["role"] != "system" else "user",
                    "content": msg["content"],
                }
            )

        anthropic_messages.append({"role": "user", "content": prompt})

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }

        try:
            # Use session for connection pooling
            response = self.session.post(self.API_URL, headers=headers, json=data, stream=True)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        try:
                            chunk = json.loads(line[6:])
                            if chunk.get("type") == "content_block_delta":
                                yield chunk["delta"].get("text", "")
                        except Exception:
                            continue

        except requests.exceptions.RequestException as e:
            yield f"Error calling Anthropic API: {str(e)}"

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        Note: This is a rough estimate.
        """
        # Rough estimate: ~4 characters per token
        return len(text) // 4

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model."""
        model_info = self.model_registry.get(self.model)
        if model_info:
            return {
                "name": model_info.name,
                "context_window": model_info.context_window,
                "cost_per_1k_input": model_info.cost_per_1k_input,
                "cost_per_1k_output": model_info.cost_per_1k_output,
                "capabilities": model_info.capabilities,
            }
        return {"name": self.model, "info": "Model not in registry"}
