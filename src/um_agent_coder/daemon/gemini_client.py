"""Async client for Google's Generative Language API (generativelanguage.googleapis.com).

Uses OAuth credentials from the user's Google AI Ultra subscription or a
GOOGLE_API_KEY to call the production Gemini API directly — NOT the
Code Assist proxy (cloudcode-pa) which has CLI-tier rate limits.

Supports streaming, auto token refresh, and multi-turn conversations.
Drop-in replacement for the previous GeminiCodeAssistClient.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import AsyncIterator, Optional

import httpx

logger = logging.getLogger(__name__)

# --- Constants ---
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"

# OAuth scopes for Generative Language API (Google AI Ultra)
GEMINI_OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/generative-language",
    "https://www.googleapis.com/auth/generative-language.retriever",
]

# Default credential file path (Gemini CLI stores creds here)
DEFAULT_CREDS_PATH = Path.home() / ".gemini" / "oauth_creds.json"


def _parse_sse_text(line: str) -> tuple[str, dict]:
    """Parse an SSE data line, returning (text_chunk, raw_data)."""
    if not line.startswith("data: "):
        return "", {}
    try:
        data = json.loads(line[6:])
    except json.JSONDecodeError:
        return "", {}
    candidates = data.get("candidates", [])
    if not candidates:
        return "", data
    text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    return text, data


def _extract_usage(data: dict) -> dict:
    """Extract token usage and finish reason from the last SSE chunk."""
    meta = data.get("usageMetadata", {})
    candidates = data.get("candidates", [])
    finish_reason = candidates[0].get("finishReason", "") if candidates else ""
    return {
        "prompt_tokens": meta.get("promptTokenCount", 0),
        "completion_tokens": meta.get("candidatesTokenCount", 0),
        "total_tokens": meta.get("totalTokenCount", 0),
        "finish_reason": finish_reason,
    }


class GeminiCodeAssistClient:
    """Async client for the Generative Language API.

    Supports two auth modes:
    1. API key (GOOGLE_API_KEY) — simplest, works with AI Ultra
    2. OAuth refresh token — from Gemini CLI or manual setup

    Maintains the same interface as the previous Code Assist proxy client
    so all callers (LLMRouter, routes, etc.) work without changes.
    """

    def __init__(
        self,
        refresh_token: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        creds_path: Optional[Path] = None,
    ):
        # Check for API key first (simplest path)
        self._api_key = os.environ.get("GOOGLE_API_KEY", "")

        if self._api_key:
            logger.info("Gemini client using API key auth (generativelanguage API)")
            self._refresh_token = ""
            self._client_id = ""
            self._client_secret = ""
        else:
            # OAuth flow
            if client_id and client_secret:
                self._client_id = client_id
                self._client_secret = client_secret
            else:
                self._client_id, self._client_secret = _load_cli_oauth_app()

            # Load refresh token
            if refresh_token:
                self._refresh_token = refresh_token
            else:
                self._refresh_token = _load_refresh_token(creds_path)

            logger.info("Gemini client using OAuth auth (generativelanguage API)")

        # Token state (OAuth only)
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0.0

        # Compatibility: tier/project for callers that check these
        self._project: Optional[str] = None
        self._tier: Optional[str] = None

        # Shared async HTTP client
        self._http: Optional[httpx.AsyncClient] = None

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
        return self._http

    async def close(self):
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    # --- OAuth token management ---

    async def refresh_access_token(self) -> str:
        """Refresh the OAuth access token. Caches until expiry."""
        if self._api_key:
            return ""  # Not needed for API key auth

        if self._access_token and time.time() < self._token_expiry - 60:
            return self._access_token

        http = await self._get_http()
        resp = await http.post(
            OAUTH_TOKEN_URL,
            data={
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "refresh_token": self._refresh_token,
                "grant_type": "refresh_token",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        self._access_token = data["access_token"]
        self._token_expiry = time.time() + data.get("expires_in", 3600)
        logger.debug("Refreshed Gemini access token (expires in %ds)", data.get("expires_in", 0))
        return self._access_token

    async def _auth_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            # API key auth — passed as query param, not header
            return headers
        token = await self.refresh_access_token()
        headers["Authorization"] = f"Bearer {token}"
        return headers

    def _auth_params(self) -> dict[str, str]:
        """Query params for auth (API key mode)."""
        if self._api_key:
            return {"key": self._api_key}
        return {}

    # --- Project loading (compatibility) ---

    async def load_project(self) -> str:
        """Verify credentials work by listing models. Sets tier info."""
        if self._project:
            return self._project

        http = await self._get_http()
        headers = await self._auth_headers()
        params = self._auth_params()

        resp = await http.get(
            f"{GEMINI_API_BASE}/models",
            headers=headers,
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()
        model_count = len(data.get("models", []))
        self._project = "generative-language"
        self._tier = "ultra" if self._api_key or self._refresh_token else "unknown"
        logger.info(
            "Gemini API verified: %d models available, auth=%s",
            model_count,
            "api_key" if self._api_key else "oauth",
        )
        return self._project

    @property
    def tier(self) -> Optional[str]:
        return self._tier

    @property
    def authenticated(self) -> bool:
        return bool(self._api_key or self._refresh_token)

    # --- Generation ---

    def _build_url(self, model: str, method: str = "generateContent") -> str:
        return f"{GEMINI_API_BASE}/models/{model}:{method}"

    def _build_payload(
        self,
        contents: list[dict],
        *,
        temperature: float = 0.7,
        max_tokens: int = 65536,
    ) -> dict:
        return {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

    def _prompt_to_contents(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> list[dict]:
        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": system_prompt}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        return contents

    async def generate(
        self,
        prompt: str,
        model: str = "gemini-3-flash-preview",
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 65536,
        timeout: float = 300.0,
    ) -> dict:
        """Generate a response via the streaming endpoint. Returns full text + usage.

        Returns:
            {"text": str, "usage": {"prompt_tokens": int, ...}, "model": str}
        """
        contents = self._prompt_to_contents(prompt, system_prompt)
        payload = self._build_payload(
            contents, temperature=temperature, max_tokens=max_tokens
        )

        headers = await self._auth_headers()
        params = {**self._auth_params(), "alt": "sse"}
        url = self._build_url(model, "streamGenerateContent")

        http = await self._get_http()
        full_text = ""
        usage = {}
        last_data = {}

        async with http.stream(
            "POST", url, headers=headers, json=payload, params=params, timeout=timeout,
        ) as resp:
            if resp.status_code == 429:
                raise RateLimitError(f"Rate limited on model {model}")
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                text, data = _parse_sse_text(line)
                if text:
                    full_text += text
                if data:
                    last_data = data

        if last_data:
            usage = _extract_usage(last_data)

        return {"text": full_text, "usage": usage, "model": model}

    async def generate_stream(
        self,
        prompt: str,
        model: str = "gemini-3-flash-preview",
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 65536,
        timeout: float = 300.0,
    ) -> AsyncIterator[str]:
        """Yield text chunks as an async generator."""
        contents = self._prompt_to_contents(prompt, system_prompt)
        payload = self._build_payload(
            contents, temperature=temperature, max_tokens=max_tokens
        )

        headers = await self._auth_headers()
        params = {**self._auth_params(), "alt": "sse"}
        url = self._build_url(model, "streamGenerateContent")

        http = await self._get_http()
        async with http.stream(
            "POST", url, headers=headers, json=payload, params=params, timeout=timeout,
        ) as resp:
            if resp.status_code == 429:
                raise RateLimitError(f"Rate limited on model {model}")
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                text, _ = _parse_sse_text(line)
                if text:
                    yield text

    async def generate_multi_turn(
        self,
        contents: list[dict],
        model: str = "gemini-3-flash-preview",
        *,
        temperature: float = 0.7,
        max_tokens: int = 65536,
        timeout: float = 300.0,
    ) -> dict:
        """Generate a response from a pre-built contents list (multi-turn).

        Args:
            contents: List of {"role": "user"|"model", "parts": [{"text": ...}]}

        Returns:
            {"text": str, "usage": {"prompt_tokens": int, ...}, "model": str}
        """
        payload = self._build_payload(
            contents, temperature=temperature, max_tokens=max_tokens
        )

        headers = await self._auth_headers()
        params = {**self._auth_params(), "alt": "sse"}
        url = self._build_url(model, "streamGenerateContent")

        http = await self._get_http()
        full_text = ""
        usage = {}
        last_data = {}

        async with http.stream(
            "POST", url, headers=headers, json=payload, params=params, timeout=timeout,
        ) as resp:
            if resp.status_code == 429:
                raise RateLimitError(f"Rate limited on model {model}")
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                text, data = _parse_sse_text(line)
                if text:
                    full_text += text
                if data:
                    last_data = data

        if last_data:
            usage = _extract_usage(last_data)

        return {"text": full_text, "usage": usage, "model": model}


class RateLimitError(Exception):
    """Raised when the Gemini API returns 429."""

    pass


# --- Credential helpers ---


def _load_cli_oauth_app() -> tuple[str, str]:
    """Load the Gemini CLI's OAuth client ID and secret."""
    client_id = os.environ.get("GEMINI_CLI_CLIENT_ID", "")
    client_secret = os.environ.get("GEMINI_CLI_CLIENT_SECRET", "")
    if client_id and client_secret:
        return client_id, client_secret

    creds_path = DEFAULT_CREDS_PATH
    if creds_path.exists():
        try:
            data = json.loads(creds_path.read_text())
            cid = data.get("client_id", "")
            cs = data.get("client_secret", "")
            if cid and cs:
                return cid, cs
        except (json.JSONDecodeError, OSError):
            pass

    raise RuntimeError(
        "Gemini CLI OAuth app credentials not found. Set GEMINI_CLI_CLIENT_ID "
        "and GEMINI_CLI_CLIENT_SECRET, or set GOOGLE_API_KEY for API key auth."
    )


def _load_refresh_token(creds_path: Optional[Path] = None) -> str:
    """Load the OAuth refresh token from file or env var."""
    path = creds_path or DEFAULT_CREDS_PATH
    env_creds = os.environ.get("GEMINI_OAUTH_CREDS")

    if path.exists():
        creds = json.loads(path.read_text())
        return creds["refresh_token"]
    elif env_creds:
        creds = json.loads(env_creds)
        return creds["refresh_token"]

    raise FileNotFoundError(
        f"Gemini OAuth credentials not found at {path}. "
        "Set GEMINI_OAUTH_CREDS env var, set GOOGLE_API_KEY, "
        "or run `npx @google/gemini-cli auth` first."
    )
