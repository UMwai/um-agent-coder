"""Gemini API client using OAuth credentials for the daemon worker."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _load_oauth_creds():
    """Load OAuth credentials from file path or Secret Manager."""
    # Option 1: File path via env var
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and Path(creds_path).exists():
        with open(creds_path) as f:
            return json.load(f)

    # Option 2: Mounted secret (Cloud Run)
    secret_path = "/secrets/gemini-oauth/credentials.json"
    if Path(secret_path).exists():
        with open(secret_path) as f:
            return json.load(f)

    # Option 3: Inline JSON from env var
    creds_json = os.environ.get("GEMINI_OAUTH_CREDS_JSON")
    if creds_json:
        return json.loads(creds_json)

    return None


def create_gemini_client(model: str = "gemini-2.5-pro"):
    """Create a Gemini GenerativeModel using API key or OAuth credentials."""
    import google.generativeai as genai

    # Prefer API key (simpler, works with AI Ultra plan)
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        logger.info("Gemini configured with API key")
        return genai.GenerativeModel(model)

    # Fall back to OAuth
    creds_data = _load_oauth_creds()
    if creds_data:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request

        credentials = Credentials(
            token=None,
            refresh_token=creds_data["refresh_token"],
            token_uri=creds_data.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=creds_data["client_id"],
            client_secret=creds_data["client_secret"],
            scopes=[
                "https://www.googleapis.com/auth/generative-language",
                "https://www.googleapis.com/auth/generative-language.retriever",
                "https://www.googleapis.com/auth/cloud-platform",
            ],
        )
        credentials.refresh(Request())
        genai.configure(credentials=credentials)
        logger.info("Gemini configured with OAuth credentials")
        return genai.GenerativeModel(model)

    raise RuntimeError(
        "No Gemini credentials found. Set GEMINI_API_KEY, GEMINI_OAUTH_CREDS_JSON, "
        "or GOOGLE_APPLICATION_CREDENTIALS"
    )


def gemini_chat(prompt: str, model: str = "gemini-2.5-pro") -> str:
    """Simple chat interface to Gemini. Creates a new client per call."""
    client = create_gemini_client(model=model)
    response = client.generate_content(prompt)
    return response.text
