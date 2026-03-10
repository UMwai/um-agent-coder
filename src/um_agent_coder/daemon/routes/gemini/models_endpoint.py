"""GET /api/gemini/models — list configured + available Gemini models.

Returns the current model registry from config and optionally probes the
Code Assist API to verify which models are actually reachable.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query

from um_agent_coder.daemon.auth import verify_api_key

router = APIRouter()
logger = logging.getLogger(__name__)


def _get_client():
    from um_agent_coder.daemon.app import get_gemini_client

    return get_gemini_client()


def _get_settings():
    from um_agent_coder.daemon.app import get_settings

    return get_settings()


async def _probe_model(client, model_name: str) -> dict:
    """Send a tiny request to verify a model is reachable."""
    try:
        result = await client.generate(
            prompt="Reply with just the word 'ok'.",
            model=model_name,
            max_tokens=8,
            temperature=0.0,
            timeout=15.0,
        )
        tokens = result.get("usage", {}).get("total_tokens", 0)
        return {"model": model_name, "status": "available", "tokens": tokens}
    except Exception as e:
        return {"model": model_name, "status": "error", "error": str(e)[:200]}


@router.get("/models")
async def list_models(
    probe: bool = Query(
        False, description="Probe each model with a tiny request to verify availability"
    ),
    _key: Optional[str] = Depends(verify_api_key),
):
    """List configured Gemini models and their roles.

    Set ?probe=true to verify each model is actually reachable (adds ~5s).
    """
    settings = _get_settings()

    configured = {
        "flash": settings.gemini_model_flash,
        "pro": settings.gemini_model_pro,
        "pro_latest": settings.gemini_model_pro_latest,
    }

    roles = {
        "generation_default": settings.gemini_model,
        "generation_iterate": settings.gemini_iterate_generation_model,
        "eval_general": settings.gemini_eval_model,
        "eval_accuracy": settings.gemini_accuracy_eval_model,
        "eval_iterate": settings.gemini_iterate_eval_models,
        "auto_model_pool": settings.gemini_auto_models,
    }

    response = {
        "configured_models": configured,
        "roles": roles,
        "env_var_reference": {
            "UM_DAEMON_GEMINI_MODEL_FLASH": "Flash tier model name",
            "UM_DAEMON_GEMINI_MODEL_PRO": "Pro tier model name",
            "UM_DAEMON_GEMINI_MODEL_PRO_LATEST": "Pro latest tier model name",
            "UM_DAEMON_GEMINI_EVAL_MODEL": "General eval model (fast)",
            "UM_DAEMON_GEMINI_ACCURACY_EVAL_MODEL": "Accuracy/fulfillment eval model",
            "UM_DAEMON_GEMINI_ITERATE_GENERATION_MODEL": "Iteration runner generation model",
            "UM_DAEMON_GEMINI_ITERATE_EVAL_MODELS": "Iteration runner eval models (comma-separated)",
        },
    }

    if probe:
        client = _get_client()
        unique_models = list(set(configured.values()))
        probes = await asyncio.gather(*[_probe_model(client, m) for m in unique_models])
        response["probe_results"] = probes

    return response
