"""Provider-agnostic LLM router for the daemon.

Routes generate() calls to Gemini, OpenAI, or Anthropic based on model name.
Falls back to Gemini when other providers aren't configured.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class LLMRouter:
    """Routes LLM calls to the appropriate provider.

    Wraps Gemini (Code Assist), OpenAI, and Anthropic behind a unified
    generate() interface matching GeminiCodeAssistClient.generate() return shape.
    """

    def __init__(
        self,
        gemini_client=None,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-5.4",
        anthropic_api_key: Optional[str] = None,
        anthropic_model: str = "claude-sonnet-4-6-20250627",
    ):
        self._gemini = gemini_client
        self._openai_key = openai_api_key
        self._openai_model = openai_model
        self._anthropic_key = anthropic_api_key
        self._anthropic_model = anthropic_model
        self._http: Optional[httpx.AsyncClient] = None

        providers = ["gemini" if gemini_client else None]
        if openai_api_key:
            providers.append("openai")
        if anthropic_api_key:
            providers.append("anthropic")
        logger.info("LLMRouter initialized: providers=%s", [p for p in providers if p])

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
        return self._http

    async def close(self):
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    @property
    def available_providers(self) -> list[str]:
        providers = []
        if self._gemini:
            providers.append("gemini")
        if self._openai_key:
            providers.append("openai")
        if self._anthropic_key:
            providers.append("anthropic")
        return providers

    def resolve_provider(self, model: str) -> str:
        """Determine which provider to use based on model name."""
        m = model.lower()
        if any(k in m for k in ("gpt", "o1", "o3", "o4", "codex")):
            if self._openai_key:
                return "openai"
            logger.warning(
                "Model %s looks like OpenAI but no API key set, falling back to gemini", model
            )
            return "gemini"
        if any(k in m for k in ("claude", "sonnet", "opus", "haiku")):
            if self._anthropic_key:
                return "anthropic"
            logger.warning(
                "Model %s looks like Anthropic but no API key set, falling back to gemini", model
            )
            return "gemini"
        return "gemini"

    async def generate(
        self,
        prompt: str,
        model: str = "",
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 65536,
        timeout: float = 300.0,
        provider: Optional[str] = None,
    ) -> dict:
        """Generate a response, routing to the right provider.

        Returns the same shape as GeminiCodeAssistClient.generate():
            {"text": str, "usage": dict, "model": str}
        """
        if provider is None:
            provider = self.resolve_provider(model) if model else "gemini"

        if provider == "openai":
            return await self._generate_openai(
                prompt,
                model or self._openai_model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        elif provider == "anthropic":
            return await self._generate_anthropic(
                prompt,
                model or self._anthropic_model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        else:
            if not self._gemini:
                # Lazy retry — mirrors get_gemini_client() pattern used by REST routes
                try:
                    from um_agent_coder.daemon.app import get_gemini_client

                    self._gemini = get_gemini_client()
                except Exception as exc:
                    logger.warning("Gemini lazy re-init failed: %s", exc)
            if not self._gemini:
                raise RuntimeError(
                    "Gemini client not available and no fallback provider configured"
                )
            return await self._generate_gemini_with_fallback(
                prompt,
                model or "gemini-3-flash-preview",
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

    async def _generate_gemini_with_fallback(
        self,
        prompt: str,
        model: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 65536,
        timeout: float = 300.0,
    ) -> dict:
        """Call Gemini with retry + automatic fallback to Flash on rate limit."""
        from um_agent_coder.daemon.gemini_client import RateLimitError

        FLASH = "gemini-3-flash-preview"
        for attempt, use_model in enumerate(
            [model, model, FLASH] if model != FLASH else [FLASH, FLASH]
        ):
            try:
                return await self._gemini.generate(
                    prompt,
                    use_model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
            except RateLimitError:
                if use_model != FLASH and attempt < 2:
                    wait = 2 ** attempt
                    logger.warning(
                        "Rate limited on %s, retrying in %ds (attempt %d)",
                        use_model, wait, attempt + 1,
                    )
                    await asyncio.sleep(wait)
                elif use_model == model and model != FLASH:
                    logger.warning(
                        "Rate limited on %s, falling back to %s", model, FLASH
                    )
                else:
                    raise

    async def _generate_openai(
        self,
        prompt: str,
        model: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 65536,
        timeout: float = 300.0,
    ) -> dict:
        """Call OpenAI Chat Completions API."""
        http = await self._get_http()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = await http.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self._openai_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})
        return {
            "text": choice["message"]["content"],
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "finish_reason": choice.get("finish_reason", ""),
            },
            "model": data.get("model", model),
        }

    async def _generate_anthropic(
        self,
        prompt: str,
        model: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 65536,
        timeout: float = 300.0,
    ) -> dict:
        """Call Anthropic Messages API."""
        http = await self._get_http()
        body: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": min(max_tokens, 16384),
            "temperature": temperature,
        }
        if system_prompt:
            body["system"] = system_prompt

        resp = await http.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self._anthropic_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")

        usage = data.get("usage", {})
        return {
            "text": text,
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                "finish_reason": data.get("stop_reason", ""),
            },
            "model": data.get("model", model),
        }
