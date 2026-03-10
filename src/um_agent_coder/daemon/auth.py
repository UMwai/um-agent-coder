"""Authentication and webhook signature verification."""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_settings():
    """Lazy import to avoid circular deps."""
    from um_agent_coder.daemon.app import get_settings as _get

    return _get()


async def verify_api_key(
    api_key: Optional[str] = Security(_api_key_header),
) -> Optional[str]:
    """Verify the API key if one is configured. Returns the key or None if auth is disabled."""
    settings = get_settings()
    if not settings.api_key:
        return None  # Auth disabled
    if not api_key or api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


def verify_github_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify GitHub webhook HMAC-SHA256 signature."""
    if not signature.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)


def verify_slack_signature(payload: bytes, timestamp: str, signature: str, secret: str) -> bool:
    """Verify Slack request signing secret."""
    # Reject requests older than 5 minutes
    if abs(time.time() - int(timestamp)) > 300:
        return False
    sig_basestring = f"v0:{timestamp}:{payload.decode()}"
    computed = (
        "v0=" + hmac.new(secret.encode(), sig_basestring.encode(), hashlib.sha256).hexdigest()
    )
    return hmac.compare_digest(computed, signature)


def verify_discord_signature(
    payload: bytes, timestamp: str, signature: str, public_key: str
) -> bool:
    """Verify Discord Ed25519 signature. Requires PyNaCl (optional)."""
    try:
        from nacl.signing import VerifyKey

        vk = VerifyKey(bytes.fromhex(public_key))
        vk.verify(timestamp.encode() + payload, bytes.fromhex(signature))
        return True
    except ImportError:
        # PyNaCl not installed - skip verification with a warning
        import logging

        logging.getLogger(__name__).warning(
            "PyNaCl not installed. Discord signature verification skipped."
        )
        return True
    except Exception:
        return False
