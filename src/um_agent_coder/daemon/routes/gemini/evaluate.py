"""POST /api/gemini/evaluate — Standalone evaluation of any prompt+response pair."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends

from um_agent_coder.daemon.auth import verify_api_key

from ._evaluator import evaluate_response
from .models import (
    GEMINI_MODEL_MAP,
    EvalInfo,
    EvaluateRequest,
    EvaluateResponse,
    GeminiModelTier,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_client():
    from um_agent_coder.daemon.app import get_gemini_client

    return get_gemini_client()


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_text(
    req: EvaluateRequest,
    _key: Optional[str] = Depends(verify_api_key),
):
    """Evaluate an existing prompt+response pair without generating anything new."""
    client = _get_client()
    query_id = f"ev-{uuid.uuid4().hex[:8]}"
    start = time.monotonic()

    # Resolve eval model
    if req.model == GeminiModelTier.auto:
        eval_model = "gemini-3-flash-preview"
    else:
        eval_model = GEMINI_MODEL_MAP[req.model.value]

    logger.info(
        "Evaluate %s: prompt=%d chars, response=%d chars, context=%d chars, model=%s",
        query_id,
        len(req.prompt),
        len(req.response),
        len(req.eval_context) if req.eval_context else 0,
        eval_model,
    )

    eval_result = await evaluate_response(
        client,
        req.prompt,
        req.response,
        model=eval_model,
        eval_context=req.eval_context,
    )

    duration_ms = int((time.monotonic() - start) * 1000)
    logger.info("Evaluate %s: score=%.2f in %dms", query_id, eval_result.score, duration_ms)

    return EvaluateResponse(
        id=query_id,
        eval_model=eval_model,
        duration_ms=duration_ms,
        evaluation=EvalInfo(
            score=eval_result.score,
            accuracy=eval_result.accuracy,
            completeness=eval_result.completeness,
            clarity=eval_result.clarity,
            actionability=eval_result.actionability,
            issues=eval_result.issues,
        ),
    )
