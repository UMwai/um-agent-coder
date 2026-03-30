"""Async Gemini client using the official google-generativeai SDK.

Auth priority:
1. GOOGLE_API_KEY — works with AI Ultra plan, no GCP costs
2. OAuth via cloud-platform scope — uses Vertex AI endpoint as fallback

This matches the original Cloud Run deployment approach (commit c8cd561).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from functools import partial
from pathlib import Path
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)

DEFAULT_CREDS_PATH = Path.home() / ".gemini" / "oauth_creds.json"


def _configure_genai() -> str:
    """Configure the google.generativeai SDK. Returns auth type string."""
    import google.generativeai as genai

    # Option 1: API key (preferred — works with AI Ultra, no GCP costs)
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if api_key:
        genai.configure(api_key=api_key)
        logger.info("Gemini configured with API key")
        return "api_key"

    # Option 2: OAuth with cloud-platform scope (Gemini CLI creds)
    creds_data = _load_oauth_creds()
    if creds_data and "refresh_token" in creds_data:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        client_id = creds_data.get("client_id") or os.environ.get("GEMINI_CLI_CLIENT_ID", "")
        client_secret = (
            creds_data.get("client_secret") or os.environ.get("GEMINI_CLI_CLIENT_SECRET", "")
        )

        if client_id and client_secret:
            # Use only cloud-platform scope (registered for Gemini CLI OAuth app)
            credentials = Credentials(
                token=None,
                refresh_token=creds_data["refresh_token"],
                token_uri=creds_data.get("token_uri", "https://oauth2.googleapis.com/token"),
                client_id=client_id,
                client_secret=client_secret,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            credentials.refresh(Request())
            genai.configure(credentials=credentials)
            logger.info("Gemini configured with OAuth (cloud-platform scope)")
            return "oauth"

    raise RuntimeError(
        "No Gemini credentials found. Set GOOGLE_API_KEY (recommended for AI Ultra) "
        "or provide OAuth creds via GEMINI_OAUTH_CREDS."
    )


def _load_oauth_creds() -> Optional[dict]:
    """Load OAuth credentials from available sources."""
    # Env var (JSON string) — used in K8s/Cloud Run
    env_creds = os.environ.get("GEMINI_OAUTH_CREDS")
    if env_creds:
        try:
            return json.loads(env_creds)
        except json.JSONDecodeError:
            pass

    # Gemini CLI credential file
    if DEFAULT_CREDS_PATH.exists():
        try:
            return json.loads(DEFAULT_CREDS_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    # GOOGLE_APPLICATION_CREDENTIALS
    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gac and Path(gac).exists():
        with open(gac) as f:
            return json.load(f)

    # Mounted secret
    for secret_path in ["/secrets/gemini-oauth/credentials.json"]:
        if Path(secret_path).exists():
            with open(secret_path) as f:
                return json.load(f)

    return None


class GeminiCodeAssistClient:
    """Async wrapper around google.generativeai SDK.

    Same interface as the previous Code Assist proxy client — drop-in
    replacement for all callers (LLMRouter, routes, etc.).
    """

    def __init__(
        self,
        refresh_token: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        creds_path: Optional[Path] = None,
    ):
        if refresh_token:
            cid = client_id or os.environ.get("GEMINI_CLI_CLIENT_ID", "")
            cs = client_secret or os.environ.get("GEMINI_CLI_CLIENT_SECRET", "")
            os.environ["GEMINI_OAUTH_CREDS"] = json.dumps({
                "refresh_token": refresh_token,
                "client_id": cid,
                "client_secret": cs,
            })

        self._auth_type = _configure_genai()
        self._project: Optional[str] = None
        self._tier: Optional[str] = None

    async def _run_sync(self, func, *args, **kwargs):
        """Run a sync SDK call in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(func, *args, **kwargs))

    async def close(self):
        pass

    async def load_project(self) -> str:
        """Verify credentials work."""
        if self._project:
            return self._project

        import google.generativeai as genai

        try:
            models = await self._run_sync(lambda: list(genai.list_models()))
            self._project = "generative-language"
            self._tier = self._auth_type
            logger.info("Gemini API verified: %d models, auth=%s", len(models), self._auth_type)
        except Exception as e:
            logger.warning("Gemini model listing failed: %s", e)
            self._project = "generative-language"
            self._tier = self._auth_type

        return self._project

    @property
    def tier(self) -> Optional[str]:
        return self._tier

    @property
    def authenticated(self) -> bool:
        return self._auth_type is not None

    async def refresh_access_token(self) -> str:
        return ""

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
        import google.generativeai as genai

        gen_model = genai.GenerativeModel(
            model,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        contents = _build_contents(prompt, system_prompt)

        try:
            response = await self._run_sync(
                lambda: gen_model.generate_content(
                    contents, request_options={"timeout": timeout}
                )
            )
        except Exception as e:
            if _is_rate_limit(e):
                raise RateLimitError(f"Rate limited on model {model}") from e
            raise

        return {"text": response.text, "usage": _extract_usage(response), "model": model}

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
        import google.generativeai as genai

        gen_model = genai.GenerativeModel(
            model,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        contents = _build_contents(prompt, system_prompt)

        try:
            response = await self._run_sync(
                lambda: gen_model.generate_content(
                    contents, stream=True, request_options={"timeout": timeout}
                )
            )
        except Exception as e:
            if _is_rate_limit(e):
                raise RateLimitError(f"Rate limited on model {model}") from e
            raise

        for chunk in response:
            if chunk.text:
                yield chunk.text

    async def generate_multi_turn(
        self,
        contents: list[dict],
        model: str = "gemini-3-flash-preview",
        *,
        temperature: float = 0.7,
        max_tokens: int = 65536,
        timeout: float = 300.0,
    ) -> dict:
        """Generate from a pre-built contents list (multi-turn)."""
        import google.generativeai as genai

        gen_model = genai.GenerativeModel(
            model,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )

        try:
            response = await self._run_sync(
                lambda: gen_model.generate_content(
                    contents, request_options={"timeout": timeout}
                )
            )
        except Exception as e:
            if _is_rate_limit(e):
                raise RateLimitError(f"Rate limited on model {model}") from e
            raise

        return {"text": response.text, "usage": _extract_usage(response), "model": model}


class RateLimitError(Exception):
    """Raised when the Gemini API returns 429."""
    pass


def _build_contents(prompt: str, system_prompt: Optional[str] = None) -> list[dict]:
    contents = []
    if system_prompt:
        contents.append({"role": "user", "parts": [{"text": system_prompt}]})
        contents.append({"role": "model", "parts": [{"text": "Understood."}]})
    contents.append({"role": "user", "parts": [{"text": prompt}]})
    return contents


def _extract_usage(response) -> dict:
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        um = response.usage_metadata
        return {
            "prompt_tokens": getattr(um, "prompt_token_count", 0),
            "completion_tokens": getattr(um, "candidates_token_count", 0),
            "total_tokens": getattr(um, "total_token_count", 0),
            "finish_reason": "",
        }
    return {}


def _is_rate_limit(e: Exception) -> bool:
    err_str = str(e).lower()
    return "429" in err_str or "resource exhausted" in err_str or "rate" in err_str
