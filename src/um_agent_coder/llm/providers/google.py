import json
import os
from typing import Any, Optional

import requests

from um_agent_coder.llm.base import LLM
from um_agent_coder.models import ModelRegistry


class GoogleLLM(LLM):
    """
    LLM provider for Google Gemini models.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_registry = ModelRegistry()

        if not self.api_key:
            raise ValueError("Google API key not provided")

        self.api_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        )
        # Use a persistent session for connection pooling
        self.session = requests.Session()

    def _build_contents(
        self, prompt: str, messages: Optional[list[dict[str, str]]] = None
    ) -> list[dict[str, Any]]:
        """Helper to build contents array for Gemini API."""
        if messages is None:
            messages = []

        contents = []
        for msg in messages:
            contents.append(
                {
                    "role": "model" if msg["role"] == "assistant" else "user",
                    "parts": [{"text": msg["content"]}],
                }
            )

        contents.append({"role": "user", "parts": [{"text": prompt}]})
        return contents

    def chat(self, prompt: str, messages: Optional[list[dict[str, str]]] = None) -> str:
        """
        Send a chat request to Google Gemini API.

        Args:
            prompt: The user prompt
            messages: Optional conversation history

        Returns:
            The model's response
        """
        contents = self._build_contents(prompt, messages)

        headers = {"Content-Type": "application/json"}

        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }

        params = {"key": self.api_key}

        try:
            # Use session for connection pooling
            response = self.session.post(self.api_url, headers=headers, json=data, params=params)
            response.raise_for_status()

            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]

        except requests.exceptions.RequestException as e:
            return f"Error calling Google API: {str(e)}"
        except (KeyError, IndexError) as e:
            return f"Error parsing Google response: {str(e)}"

    def stream_chat(self, prompt: str, messages: Optional[list[dict[str, str]]] = None):
        """
        Stream a chat response from Google Gemini API.
        Uses SSE (Server-Sent Events) for streaming.
        """
        contents = self._build_contents(prompt, messages)

        headers = {"Content-Type": "application/json"}

        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }

        # Use streamGenerateContent endpoint
        stream_url = self.api_url.replace(":generateContent", ":streamGenerateContent")

        params = {"key": self.api_key, "alt": "sse"}

        try:
            with self.session.post(
                stream_url, headers=headers, json=data, params=params, stream=True
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode("utf-8")
                        if decoded_line.startswith("data: "):
                            json_str = decoded_line[6:]
                            try:
                                chunk = json.loads(json_str)
                                if "candidates" in chunk:
                                    parts = (
                                        chunk["candidates"][0].get("content", {}).get("parts", [])
                                    )
                                    for part in parts:
                                        if "text" in part:
                                            yield part["text"]
                            except json.JSONDecodeError:
                                continue
        except requests.exceptions.RequestException as e:
            yield f"Error calling Google API: {str(e)}"

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
