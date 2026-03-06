"""Async client for Google's Code Assist API (cloudcode-pa.googleapis.com).

Uses the Gemini CLI's OAuth credentials to call the API directly,
bypassing the CLI subprocess. Supports streaming, auto token refresh,
and round-robin model selection.
"""

from __future__ import annotations

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

# User-Agent the CLI sends — influences rate limit tier assignment
GEMINI_CLI_USER_AGENT = "GeminiCLI/0.32.1/{model} (linux; x64)"

# Default credential file path
DEFAULT_CREDS_PATH = Path.home() / ".gemini" / "oauth_creds.json"


def _load_cli_oauth_app() -> tuple[str, str]:
    """Load the Gemini CLI's OAuth client ID and secret.

    These are the CLI's registered OAuth application credentials (public,
    embedded in the open-source Gemini CLI). They identify the application,
    not the user. Loaded from env vars or from the user's oauth_creds.json.
    """
    client_id = os.environ.get("GEMINI_CLI_CLIENT_ID", "")
    client_secret = os.environ.get("GEMINI_CLI_CLIENT_SECRET", "")
    if client_id and client_secret:
        return client_id, client_secret

    # Fall back to reading from the credential file which may contain them
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
    text = (
        candidates[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
    )
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
    """Async client for the Code Assist API.

    Handles OAuth token refresh, project loading, and streaming generation.
    Thread-safe for concurrent use from FastAPI handlers.
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

        # Project state (from loadCodeAssist)
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
        # expires_in is typically 3600s
        self._token_expiry = time.time() + data.get("expires_in", 3600)
        logger.debug("Refreshed Gemini access token (expires in %ds)", data.get("expires_in", 0))
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
        """Call loadCodeAssist to get the project ID. Caches result."""
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
        self._tier = tier_data.get("id", "unknown") if isinstance(tier_data, dict) else str(tier_data)
        logger.info("Loaded Code Assist project=%s tier=%s", self._project, self._tier)
        return self._project

    @property
    def tier(self) -> Optional[str]:
        return self._tier

    @property
    def authenticated(self) -> bool:
        return self._refresh_token is not None

    # --- Generation ---

    def _build_payload(
        self,
        prompt: str,
        model: str,
        project: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> dict:
        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": system_prompt}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

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

    async def generate(
        self,
        prompt: str,
        model: str = "gemini-3-flash-preview",
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        timeout: float = 300.0,
    ) -> dict:
        """Generate a response via the streaming endpoint. Returns full text + usage.

        Returns:
            {"text": str, "usage": {"prompt_tokens": int, ...}, "model": str}
        """
        project = await self.load_project()
        headers = await self._auth_headers(model)
        payload = self._build_payload(
            prompt, model, project,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
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
        max_tokens: int = 8192,
        timeout: float = 300.0,
    ) -> AsyncIterator[str]:
        """Yield text chunks as an async generator."""
        project = await self.load_project()
        headers = await self._auth_headers(model)
        payload = self._build_payload(
            prompt, model, project,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
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
        max_tokens: int = 8192,
        timeout: float = 300.0,
    ) -> dict:
        """Generate a response from a pre-built contents list (multi-turn).

        Args:
            contents: List of {"role": "user"|"model", "parts": [{"text": ...}]}
                      dicts representing the full conversation history.

        Returns:
            {"text": str, "usage": {"prompt_tokens": int, ...}, "model": str}
        """
        project = await self.load_project()
        headers = await self._auth_headers(model)

        payload = {
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
    """Raised when the Code Assist API returns 429."""
    pass
