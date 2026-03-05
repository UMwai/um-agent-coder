"""Query API endpoints - proxy queries through CLI subscriptions (Codex/Gemini)."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
import uuid
from enum import Enum
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from um_agent_coder.daemon.auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/query", tags=["query"])


class Provider(str, Enum):
    codex = "codex"
    gemini = "gemini"


class QueryRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=100_000)
    provider: Provider
    model: Optional[str] = None
    timeout: int = Field(default=300, ge=10, le=1800)


class QueryResponse(BaseModel):
    id: str
    provider: str
    model: str
    response: str
    duration_ms: int


class ProviderStatus(BaseModel):
    name: str
    available: bool
    default_model: str
    authenticated: bool


class ProvidersResponse(BaseModel):
    providers: list[ProviderStatus]


def get_settings():
    from um_agent_coder.daemon.app import get_settings as _get
    return _get()


def _check_cli_available(cli_name: str) -> bool:
    return shutil.which(cli_name) is not None


def _check_codex_auth() -> bool:
    """Check if codex has valid credentials."""
    from pathlib import Path
    auth_file = Path.home() / ".codex" / "auth.json"
    if not auth_file.exists():
        return False
    try:
        data = json.loads(auth_file.read_text())
        return bool(data.get("tokens", {}).get("refresh_token"))
    except (json.JSONDecodeError, OSError):
        return False


def _check_gemini_auth() -> bool:
    """Check if gemini has valid credentials."""
    from pathlib import Path
    creds_file = Path.home() / ".gemini" / "oauth_creds.json"
    if not creds_file.exists():
        return False
    try:
        data = json.loads(creds_file.read_text())
        return bool(data.get("refresh_token"))
    except (json.JSONDecodeError, OSError):
        return False


async def _run_codex(prompt: str, model: str, timeout: int) -> str:
    """Execute query via codex CLI."""
    cmd = ["codex", "exec", "--json"]
    cmd.extend(["--model", model])
    cmd.extend(["--sandbox", "read-only"])
    cmd.append(prompt)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise HTTPException(status_code=504, detail="Codex request timed out")

    if proc.returncode != 0:
        err = stderr.decode().strip()
        logger.error("Codex failed (rc=%d): %s", proc.returncode, err)
        raise HTTPException(status_code=502, detail=f"Codex error: {err}")

    output = stdout.decode().strip()
    # Parse JSONL output - last message has the response
    for line in reversed(output.split("\n")):
        try:
            data = json.loads(line)
            if data.get("type") == "message" and data.get("content"):
                return data["content"]
            if data.get("response"):
                return data["response"]
        except json.JSONDecodeError:
            continue

    return output or "Codex completed with no output"


async def _run_gemini(prompt: str, model: str, timeout: int) -> str:
    """Execute query via gemini CLI."""
    cmd = ["gemini", prompt, "-m", model]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise HTTPException(status_code=504, detail="Gemini request timed out")

    if proc.returncode != 0:
        err = stderr.decode().strip()
        logger.error("Gemini failed (rc=%d): %s", proc.returncode, err)
        raise HTTPException(status_code=502, detail=f"Gemini error: {err}")

    return stdout.decode().strip() or "Gemini completed with no output"


@router.post("", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    _key: Optional[str] = Depends(verify_api_key),
):
    """Submit a query to be processed via CLI subscription billing."""
    settings = get_settings()
    query_id = f"q-{uuid.uuid4().hex[:8]}"

    if req.provider == Provider.codex:
        model = req.model or settings.codex_model
        if not _check_cli_available("codex"):
            raise HTTPException(status_code=503, detail="Codex CLI not installed")
        start = time.monotonic()
        response = await _run_codex(req.prompt, model, req.timeout)
        duration_ms = int((time.monotonic() - start) * 1000)

    elif req.provider == Provider.gemini:
        model = req.model or settings.gemini_model
        if not _check_cli_available("gemini"):
            raise HTTPException(status_code=503, detail="Gemini CLI not installed")
        start = time.monotonic()
        response = await _run_gemini(req.prompt, model, req.timeout)
        duration_ms = int((time.monotonic() - start) * 1000)

    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {req.provider}")

    logger.info("Query %s completed via %s/%s in %dms", query_id, req.provider, model, duration_ms)

    return QueryResponse(
        id=query_id,
        provider=req.provider.value,
        model=model,
        response=response,
        duration_ms=duration_ms,
    )


@router.get("/providers", response_model=ProvidersResponse)
async def list_providers(
    _key: Optional[str] = Depends(verify_api_key),
):
    """List available query providers and their status."""
    settings = get_settings()
    providers = [
        ProviderStatus(
            name="codex",
            available=_check_cli_available("codex"),
            default_model=settings.codex_model,
            authenticated=_check_codex_auth(),
        ),
        ProviderStatus(
            name="gemini",
            available=_check_cli_available("gemini"),
            default_model=settings.gemini_model,
            authenticated=_check_gemini_auth(),
        ),
    ]
    return ProvidersResponse(providers=providers)
