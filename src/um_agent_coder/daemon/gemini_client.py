"""Async Gemini client using the official google-generativeai SDK.

Authenticates via GOOGLE_API_KEY (preferred) or OAuth refresh token
(from Gemini CLI / AI Ultra subscription). Calls the production
generativelanguage.googleapis.com endpoint — NOT the Code Assist proxy.

This is the same auth approach used in the original Cloud Run deployment
(commit c8cd561), wrapped in an async interface for FastAPI compatibility.
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

# Default credential file path (Gemini CLI stores creds here)
DEFAULT_CREDS_PATH = Path.home() / ".gemini" / "oauth_creds.json"


def _configure_genai():
    """Configure the google.generativeai SDK with available credentials.

    Priority:
    1. GOOGLE_API_KEY env var (simplest, works with AI Ultra)
    2. OAuth refresh token (from Gemini CLI or GEMINI_OAUTH_CREDS env var)
    3. GOOGLE_APPLICATION_CREDENTIALS file path
    """
    import google.generativeai as genai

    # Option 1: API key
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if api_key:
        genai.configure(api_key=api_key)
        logger.info("Gemini configured with API key (generativelanguage API)")
        return "api_key"

    # Option 2: OAuth from env var (GEMINI_OAUTH_CREDS — JSON string)
    creds_data = None
    env_creds = os.environ.get("GEMINI_OAUTH_CREDS")
    if env_creds:
        creds_data = json.loads(env_creds)

    # Option 3: OAuth from Gemini CLI credential file
    if not creds_data and DEFAULT_CREDS_PATH.exists():
        try:
            creds_data = json.loads(DEFAULT_CREDS_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    # Option 4: OAuth from GOOGLE_APPLICATION_CREDENTIALS
    if not creds_data:
        gac_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if gac_path and Path(gac_path).exists():
            with open(gac_path) as f:
                creds_data = json.load(f)

    # Option 5: Mounted secret (Cloud Run / GKE)
    if not creds_data:
        secret_path = "/secrets/gemini-oauth/credentials.json"
        if Path(secret_path).exists():
            with open(secret_path) as f:
                creds_data = json.load(f)

    if creds_data and "refresh_token" in creds_data:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        # Need client_id/secret — check creds_data first, then env vars
        client_id = creds_data.get("client_id") or os.environ.get("GEMINI_CLI_CLIENT_ID", "")
        client_secret = (
            creds_data.get("client_secret") or os.environ.get("GEMINI_CLI_CLIENT_SECRET", "")
        )

        if not client_id or not client_secret:
            raise RuntimeError(
                "OAuth credentials found but missing client_id/client_secret. "
                "Set GEMINI_CLI_CLIENT_ID and GEMINI_CLI_CLIENT_SECRET."
            )

        credentials = Credentials(
            token=None,
            refresh_token=creds_data["refresh_token"],
            token_uri=creds_data.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=client_id,
            client_secret=client_secret,
            scopes=[
                "https://www.googleapis.com/auth/generative-language",
                "https://www.googleapis.com/auth/generative-language.retriever",
                "https://www.googleapis.com/auth/cloud-platform",
            ],
        )
        credentials.refresh(Request())
        genai.configure(credentials=credentials)
        logger.info("Gemini configured with OAuth credentials (generativelanguage API)")
        return "oauth"

    raise RuntimeError(
        "No Gemini credentials found. Set GOOGLE_API_KEY, GEMINI_OAUTH_CREDS, "
        "or GOOGLE_APPLICATION_CREDENTIALS."
    )


class GeminiCodeAssistClient:
    """Async wrapper around google.generativeai SDK.

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
        # If a specific refresh_token is passed, inject it into env
        # so _configure_genai picks it up
        if refresh_token:
            client_id = client_id or os.environ.get("GEMINI_CLI_CLIENT_ID", "")
            client_secret = client_secret or os.environ.get("GEMINI_CLI_CLIENT_SECRET", "")
            creds = json.dumps({
                "refresh_token": refresh_token,
                "client_id": client_id,
                "client_secret": client_secret,
            })
            os.environ["GEMINI_OAUTH_CREDS"] = creds

        self._auth_type = _configure_genai()
        self._project: Optional[str] = None
        self._tier: Optional[str] = None
        self._loop = None

    def _get_model(self, model: str):
        """Create a GenerativeModel instance."""
        import google.generativeai as genai
        return genai.GenerativeModel(model)

    async def _run_sync(self, func, *args, **kwargs):
        """Run a sync SDK call in a thread pool to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(func, *args, **kwargs))

    async def close(self):
        """No-op — SDK manages its own connections."""
        pass

    # --- Project loading (compatibility) ---

    async def load_project(self) -> str:
        """Verify credentials work by listing models."""
        if self._project:
            return self._project

        import google.generativeai as genai

        try:
            models = await self._run_sync(lambda: list(genai.list_models()))
            model_count = len(models)
            self._project = "generative-language"
            self._tier = "ultra" if self._auth_type in ("api_key", "oauth") else "unknown"
            logger.info(
                "Gemini API verified: %d models available, auth=%s",
                model_count,
                self._auth_type,
            )
        except Exception as e:
            logger.warning("Gemini model listing failed (auth may still work): %s", e)
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
        """Compatibility — SDK handles token refresh internally."""
        return ""

    # --- Generation ---

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

        def _call():
            resp = gen_model.generate_content(
                contents,
                request_options={"timeout": timeout},
            )
            return resp

        try:
            response = await self._run_sync(_call)
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "resource exhausted" in err_str or "rate" in err_str:
                raise RateLimitError(f"Rate limited on model {model}") from e
            raise

        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", 0),
                "completion_tokens": getattr(um, "candidates_token_count", 0),
                "total_tokens": getattr(um, "total_token_count", 0),
                "finish_reason": "",
            }

        return {"text": response.text, "usage": usage, "model": model}

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

        def _stream():
            return gen_model.generate_content(
                contents,
                stream=True,
                request_options={"timeout": timeout},
            )

        try:
            response = await self._run_sync(_stream)
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "resource exhausted" in err_str:
                raise RateLimitError(f"Rate limited on model {model}") from e
            raise

        # Iterate chunks in thread pool
        loop = asyncio.get_event_loop()
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
        """Generate from a pre-built contents list (multi-turn).

        Args:
            contents: List of {"role": "user"|"model", "parts": [{"text": ...}]}

        Returns:
            {"text": str, "usage": dict, "model": str}
        """
        import google.generativeai as genai

        gen_model = genai.GenerativeModel(
            model,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )

        def _call():
            return gen_model.generate_content(
                contents,
                request_options={"timeout": timeout},
            )

        try:
            response = await self._run_sync(_call)
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "resource exhausted" in err_str or "rate" in err_str:
                raise RateLimitError(f"Rate limited on model {model}") from e
            raise

        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", 0),
                "completion_tokens": getattr(um, "candidates_token_count", 0),
                "total_tokens": getattr(um, "total_token_count", 0),
                "finish_reason": "",
            }

        return {"text": response.text, "usage": usage, "model": model}


class RateLimitError(Exception):
    """Raised when the Gemini API returns 429."""

    pass


# --- Helpers ---


def _build_contents(prompt: str, system_prompt: Optional[str] = None) -> list[dict]:
    """Build contents array for the SDK."""
    contents = []
    if system_prompt:
        contents.append({"role": "user", "parts": [{"text": system_prompt}]})
        contents.append({"role": "model", "parts": [{"text": "Understood."}]})
    contents.append({"role": "user", "parts": [{"text": prompt}]})
    return contents
