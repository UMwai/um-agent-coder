"""Async client for Google's Code Assist API (cloudcode-pa.googleapis.com).

Uses Gemini CLI OAuth credentials with your AI Ultra subscription.
Optimized for throughput with:
- 4-model round-robin (~80+ req/min across independent rate limits)
- Async semaphore to prevent burst overload
- Multi-turn session support to reduce request count
- Automatic retry with model fallback on 429s
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import AsyncIterator, Optional

import httpx

logger = logging.getLogger(__name__)

# --- Constants ---
CODE_ASSIST_BASE = "https://cloudcode-pa.googleapis.com"
CODE_ASSIST_API_VERSION = "v1internal"
CODE_ASSIST_ENDPOINT = f"{CODE_ASSIST_BASE}/{CODE_ASSIST_API_VERSION}"

OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
GEMINI_CLI_USER_AGENT = "GeminiCLI/0.32.1/{model} (linux; x64)"

DEFAULT_CREDS_PATH = Path.home() / ".gemini" / "oauth_creds.json"

# Available models on Code Assist — round-robin across independent rate limits
# Each model has independent quota; combined throughput ~80+ req/min
ROUND_ROBIN_MODELS = [
    "gemini-3-flash-preview",         # ~30 req/min, fastest
    "gemini-3-pro-preview",           # ~18 req/min
    "gemini-3.1-pro-preview",         # ~18 req/min
    "gemini-3.1-flash-lite-preview",  # cheapest, high-volume
]

# Max concurrent requests to Code Assist (prevents burst overload)
# With 4 models: flash ~30 + pro ~18 + 3.1-pro ~18 + 3.1-flash-lite ~30 = ~96/min
MAX_CONCURRENT = 4


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
        "and GEMINI_CLI_CLIENT_SECRET environment variables."
    )


def _parse_sse_text(line: str) -> tuple[str, dict]:
    """Parse an SSE data line, returning (text_chunk, raw_data)."""
    if not line.startswith("data: "):
        return "", {}
    try:
        data = json.loads(line[6:])
    except json.JSONDecodeError:
        return "", {}
    resp = data.get("response", data)
    candidates = resp.get("candidates", [])
    if not candidates:
        return "", data
    text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    return text, data


def _extract_usage(data: dict) -> dict:
    """Extract token usage and finish reason from the last SSE chunk."""
    resp = data.get("response", data)
    meta = resp.get("usageMetadata", {})
    candidates = resp.get("candidates", [])
    finish_reason = candidates[0].get("finishReason", "") if candidates else ""
    return {
        "prompt_tokens": meta.get("promptTokenCount", 0),
        "completion_tokens": meta.get("candidatesTokenCount", 0),
        "total_tokens": meta.get("totalTokenCount", 0),
        "finish_reason": finish_reason,
    }


class GeminiCodeAssistClient:
    """Async client for the Code Assist API with rate-aware optimizations.

    Features:
    - 4-model round-robin for ~80+ req/min throughput
    - Async semaphore prevents burst overload
    - Per-model cooldown tracking on 429s
    - Multi-turn session support
    """

    def __init__(
        self,
        refresh_token: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        creds_path: Optional[Path] = None,
    ):
        if client_id and client_secret:
            self._client_id = client_id
            self._client_secret = client_secret
        else:
            self._client_id, self._client_secret = _load_cli_oauth_app()

        # Load refresh token
        if refresh_token:
            self._refresh_token = refresh_token
        else:
            path = creds_path or DEFAULT_CREDS_PATH
            env_creds = os.environ.get("GEMINI_OAUTH_CREDS")
            if path.exists():
                creds = json.loads(path.read_text())
                self._refresh_token = creds["refresh_token"]
            elif env_creds:
                creds = json.loads(env_creds)
                self._refresh_token = creds["refresh_token"]
            else:
                raise FileNotFoundError(
                    f"Gemini OAuth credentials not found at {path}. "
                    "Set GEMINI_OAUTH_CREDS env var or run `npx @google/gemini-cli auth` first."
                )

        # Token state
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0.0

        # Project state
        self._project: Optional[str] = None
        self._tier: Optional[str] = None

        # HTTP client
        self._http: Optional[httpx.AsyncClient] = None

        # Rate management
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self._model_index = 0
        self._model_cooldowns: dict[str, float] = {}  # model -> earliest_retry_time

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
        return self._http

    async def close(self):
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    # --- Round-robin model selection ---

    def _next_model(self, preferred: Optional[str] = None) -> str:
        """Pick the next available model, respecting cooldowns."""
        now = time.time()

        # If preferred model isn't on cooldown, use it
        if preferred and self._model_cooldowns.get(preferred, 0) <= now:
            return preferred

        # Round-robin through models, skip those on cooldown
        for _ in range(len(ROUND_ROBIN_MODELS)):
            model = ROUND_ROBIN_MODELS[self._model_index % len(ROUND_ROBIN_MODELS)]
            self._model_index += 1
            if self._model_cooldowns.get(model, 0) <= now:
                return model

        # All on cooldown — use the one with earliest expiry
        return min(self._model_cooldowns, key=self._model_cooldowns.get)

    def _cooldown_model(self, model: str, seconds: float = 10.0):
        """Put a model on cooldown after a 429."""
        self._model_cooldowns[model] = time.time() + seconds
        logger.debug("Model %s on cooldown for %.0fs", model, seconds)

    # --- OAuth token management ---

    async def refresh_access_token(self) -> str:
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
        logger.debug("Refreshed access token (expires in %ds)", data.get("expires_in", 0))
        return self._access_token

    async def _auth_headers(self, model: str = "") -> dict[str, str]:
        token = await self.refresh_access_token()
        ua = GEMINI_CLI_USER_AGENT.format(model=model or "unknown")
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": ua,
        }

    # --- Project loading ---

    async def load_project(self) -> str:
        if self._project:
            return self._project

        http = await self._get_http()
        headers = await self._auth_headers()
        resp = await http.post(
            f"{CODE_ASSIST_ENDPOINT}:loadCodeAssist",
            headers=headers,
            json={},
        )
        resp.raise_for_status()
        data = resp.json()
        self._project = data.get("cloudaicompanionProject", "")
        tier_data = data.get("currentTier", {})
        self._tier = (
            tier_data.get("id", "unknown") if isinstance(tier_data, dict) else str(tier_data)
        )
        logger.info(
            "Code Assist: project=%s tier=%s models=%s",
            self._project,
            self._tier,
            ROUND_ROBIN_MODELS,
        )
        return self._project

    @property
    def tier(self) -> Optional[str]:
        return self._tier

    @property
    def authenticated(self) -> bool:
        return self._refresh_token is not None

    # --- Payload building ---

    def _build_payload(
        self,
        contents: list[dict],
        model: str,
        project: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 65536,
    ) -> dict:
        return {
            "model": model,
            "project": project,
            "user_prompt_id": str(uuid.uuid4()),
            "request": {
                "contents": contents,
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            },
        }

    # --- Core streaming call (with semaphore + retry) ---

    async def _stream_call(
        self,
        contents: list[dict],
        model: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 65536,
        timeout: float = 300.0,
        retries: int = 2,
    ) -> tuple[str, dict]:
        """Execute a streaming call with rate-aware retry.

        Returns (full_text, usage_dict).
        """
        project = await self.load_project()
        last_error = None

        for attempt in range(retries + 1):
            use_model = self._next_model(model) if attempt > 0 else model
            if self._model_cooldowns.get(use_model, 0) > time.time():
                wait = self._model_cooldowns[use_model] - time.time()
                if wait > 0:
                    await asyncio.sleep(min(wait, 5.0))
                use_model = self._next_model()

            async with self._semaphore:
                try:
                    headers = await self._auth_headers(use_model)
                    payload = self._build_payload(
                        contents, use_model, project,
                        temperature=temperature, max_tokens=max_tokens,
                    )

                    http = await self._get_http()
                    full_text = ""
                    usage = {}
                    last_data = {}

                    async with http.stream(
                        "POST",
                        f"{CODE_ASSIST_ENDPOINT}:streamGenerateContent",
                        headers=headers,
                        json=payload,
                        params={"alt": "sse"},
                        timeout=timeout,
                    ) as resp:
                        if resp.status_code == 429:
                            self._cooldown_model(use_model, 15.0)
                            raise RateLimitError(f"Rate limited on {use_model}")
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            text, data = _parse_sse_text(line)
                            if text:
                                full_text += text
                            if data:
                                last_data = data

                    if last_data:
                        usage = _extract_usage(last_data)

                    return full_text, usage

                except RateLimitError as e:
                    last_error = e
                    if attempt < retries:
                        fallback = self._next_model()
                        logger.warning(
                            "429 on %s (attempt %d/%d), retrying with %s",
                            use_model, attempt + 1, retries + 1, fallback,
                        )
                        await asyncio.sleep(1.0 * (attempt + 1))
                    else:
                        raise

        raise last_error  # Should not reach here

    # --- Public API ---

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
        """Generate a response. Returns {"text": str, "usage": dict, "model": str}."""
        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": system_prompt}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        text, usage = await self._stream_call(
            contents, model,
            temperature=temperature, max_tokens=max_tokens, timeout=timeout,
        )
        return {"text": text, "usage": usage, "model": model}

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
        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": system_prompt}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        project = await self.load_project()
        model_to_use = self._next_model(model)

        async with self._semaphore:
            headers = await self._auth_headers(model_to_use)
            payload = self._build_payload(
                contents, model_to_use, project,
                temperature=temperature, max_tokens=max_tokens,
            )

            http = await self._get_http()
            async with http.stream(
                "POST",
                f"{CODE_ASSIST_ENDPOINT}:streamGenerateContent",
                headers=headers,
                json=payload,
                params={"alt": "sse"},
                timeout=timeout,
            ) as resp:
                if resp.status_code == 429:
                    self._cooldown_model(model_to_use, 15.0)
                    raise RateLimitError(f"Rate limited on {model_to_use}")
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
        """Generate from a pre-built contents list (multi-turn).

        This is the most token-efficient method — pass the full conversation
        history and the model continues from where it left off. Reduces
        request count by keeping context in a single call chain.

        Args:
            contents: List of {"role": "user"|"model", "parts": [{"text": ...}]}
        """
        text, usage = await self._stream_call(
            contents, model,
            temperature=temperature, max_tokens=max_tokens, timeout=timeout,
        )
        return {"text": text, "usage": usage, "model": model}


class RateLimitError(Exception):
    """Raised when the Code Assist API returns 429."""
    pass
