"""Query API endpoints — direct Code Assist API proxy (Gemini 3 models)."""

from __future__ import annotations

import itertools
import logging
import time
import uuid
from enum import Enum
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from um_agent_coder.daemon.auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/query", tags=["query"])


# --- Models ---

GEMINI_MODELS = {
    "flash": "gemini-3-flash-preview",
    "pro": "gemini-3-pro-preview",
    "pro-3.1": "gemini-3.1-pro-preview",
}


class GeminiModel(str, Enum):
    flash = "flash"
    pro = "pro"
    pro_3_1 = "pro-3.1"
    auto = "auto"


class QueryRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=100_000)
    model: GeminiModel = GeminiModel.auto
    system_prompt: Optional[str] = Field(default=None, max_length=50_000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=8192, ge=1, le=65536)
    timeout: int = Field(default=300, ge=10, le=1800)


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class QueryResponse(BaseModel):
    id: str
    model: str
    response: str
    duration_ms: int
    usage: UsageInfo = UsageInfo()


class ModelStatus(BaseModel):
    name: str
    model_id: str
    available: bool


class ModelsResponse(BaseModel):
    authenticated: bool
    tier: Optional[str] = None
    models: list[ModelStatus]


# --- Round-robin model selector ---

_round_robin: Optional[itertools.cycle] = None


def _get_round_robin_models() -> list[str]:
    """Get the auto model list from settings."""
    from um_agent_coder.daemon.app import get_settings
    settings = get_settings()
    return [m.strip() for m in settings.gemini_auto_models.split(",") if m.strip()]


def _next_auto_model() -> str:
    """Return the next model in round-robin order."""
    global _round_robin
    if _round_robin is None:
        models = _get_round_robin_models()
        _round_robin = itertools.cycle(models)
    return next(_round_robin)


def _resolve_model(model: GeminiModel) -> str:
    """Resolve a model enum to the actual model name string."""
    if model == GeminiModel.auto:
        return _next_auto_model()
    return GEMINI_MODELS[model.value]


# --- Client accessor ---

def _get_client():
    from um_agent_coder.daemon.app import get_gemini_client
    return get_gemini_client()


# --- Endpoints ---

@router.post("", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    _key: Optional[str] = Depends(verify_api_key),
):
    """Submit a query to Gemini via the Code Assist API."""
    client = _get_client()
    model_name = _resolve_model(req.model)
    query_id = f"q-{uuid.uuid4().hex[:8]}"

    start = time.monotonic()
    try:
        result = await client.generate(
            prompt=req.prompt,
            model=model_name,
            system_prompt=req.system_prompt,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            timeout=float(req.timeout),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        from um_agent_coder.daemon.gemini_client import RateLimitError
        if isinstance(e, RateLimitError):
            raise HTTPException(status_code=429, detail=str(e))
        logger.error("Gemini API error: %s", e)
        raise HTTPException(status_code=502, detail=f"Gemini API error: {e}")

    duration_ms = int((time.monotonic() - start) * 1000)
    logger.info("Query %s completed via %s in %dms", query_id, model_name, duration_ms)

    usage = result.get("usage", {})
    return QueryResponse(
        id=query_id,
        model=result.get("model", model_name),
        response=result.get("text", ""),
        duration_ms=duration_ms,
        usage=UsageInfo(**usage) if usage else UsageInfo(),
    )


@router.get("/models", response_model=ModelsResponse)
async def list_models(
    _key: Optional[str] = Depends(verify_api_key),
):
    """List available Gemini models and auth status."""
    client = _get_client()
    auto_models = _get_round_robin_models()

    models = []
    for short_name, model_id in GEMINI_MODELS.items():
        models.append(
            ModelStatus(
                name=short_name,
                model_id=model_id,
                available=model_id in auto_models,
            )
        )

    return ModelsResponse(
        authenticated=client.authenticated,
        tier=client.tier,
        models=models,
    )
