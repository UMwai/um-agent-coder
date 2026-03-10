"""POST /api/gemini/enhance — Enhanced query with prompt pipeline + self-eval."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from um_agent_coder.daemon.auth import verify_api_key

from ._evaluator import build_retry_prompt, evaluate_response
from ._pipeline import enhance_prompt
from ._router import score_complexity, select_model
from .models import (
    GEMINI_MODEL_MAP,
    EnhancementInfo,
    EnhanceRequest,
    EnhanceResponse,
    EvalInfo,
    GeminiModelTier,
    UsageInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_client():
    from um_agent_coder.daemon.app import get_gemini_client

    return get_gemini_client()


def _get_settings():
    from um_agent_coder.daemon.app import get_settings

    return get_settings()


def _resolve_model(model: GeminiModelTier, prompt: str, settings) -> str:
    """Resolve model tier to full model name."""
    if model == GeminiModelTier.auto:
        return select_model(prompt, threshold=settings.gemini_complexity_threshold)
    return GEMINI_MODEL_MAP[model.value]


def _resolve_eval_model(
    req_eval_model: Optional[GeminiModelTier],
    generation_model: str,
    settings,
) -> str:
    """Resolve which model to use for self-evaluation.

    Priority: request-level override > config default > Flash.
    'auto' means match the generation model.
    """
    # Request-level override
    if req_eval_model is not None:
        if req_eval_model == GeminiModelTier.auto:
            return generation_model
        return GEMINI_MODEL_MAP[req_eval_model.value]

    # Config default
    config_val = settings.gemini_self_eval_model
    if config_val == "auto":
        return generation_model
    # Config can be a tier name or full model name
    if config_val in GEMINI_MODEL_MAP:
        return GEMINI_MODEL_MAP[config_val]
    return config_val  # assume full model name


@router.post("/enhance", response_model=EnhanceResponse)
async def enhance_query(
    req: EnhanceRequest,
    _key: Optional[str] = Depends(verify_api_key),
):
    """Enhanced query with prompt pipeline and optional self-eval."""
    client = _get_client()
    settings = _get_settings()
    query_id = f"ge-{uuid.uuid4().hex[:8]}"
    start = time.monotonic()

    # Model selection
    model_name = _resolve_model(req.model, req.prompt, settings)
    complexity = score_complexity(req.prompt)

    # Prompt enhancement
    enhancement_info = None
    prompt_to_send = req.prompt

    if req.enable_enhancement and settings.gemini_enhance_enabled:
        result = enhance_prompt(
            req.prompt,
            domain_hint=req.domain_hint,
        )
        prompt_to_send = result.enhanced
        enhancement_info = EnhancementInfo(
            original_prompt=req.prompt,
            enhanced_prompt=result.enhanced,
            stages_applied=result.stages_applied,
            model_selected=model_name,
            complexity_score=complexity,
        )

    # Generate response
    try:
        gen_result = await client.generate(
            prompt=prompt_to_send,
            model=model_name,
            system_prompt=req.system_prompt,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
    except Exception as e:
        from um_agent_coder.daemon.gemini_client import RateLimitError

        if isinstance(e, RateLimitError):
            raise HTTPException(status_code=429, detail=str(e))
        logger.error("Gemini API error: %s", e)
        raise HTTPException(status_code=502, detail=f"Gemini API error: {e}")

    response_text = gen_result.get("text", "")
    usage = gen_result.get("usage", {})

    # Self-evaluation + retry loop
    eval_info = None
    if req.enable_self_eval and settings.gemini_self_eval_enabled:
        eval_model_name = _resolve_eval_model(req.eval_model, model_name, settings)
        logger.info("Query %s: using eval model %s", query_id, eval_model_name)

        eval_result = await evaluate_response(
            client,
            req.prompt,
            response_text,
            model=eval_model_name,
            eval_context=req.eval_context,
        )
        retry_count = 0

        while (
            eval_result.score < settings.gemini_self_eval_threshold
            and retry_count < settings.gemini_max_retries
        ):
            retry_count += 1
            logger.info(
                "Query %s: eval score %.2f < %.2f, retrying (%d/%d)",
                query_id,
                eval_result.score,
                settings.gemini_self_eval_threshold,
                retry_count,
                settings.gemini_max_retries,
            )

            improved_prompt = build_retry_prompt(prompt_to_send, response_text, eval_result)
            try:
                gen_result = await client.generate(
                    prompt=improved_prompt,
                    model=model_name,
                    system_prompt=req.system_prompt,
                    temperature=max(req.temperature - 0.1 * retry_count, 0.1),
                    max_tokens=req.max_tokens,
                )
                response_text = gen_result.get("text", "")
                retry_usage = gen_result.get("usage", {})
                # Accumulate usage
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    usage[k] = usage.get(k, 0) + retry_usage.get(k, 0)
            except Exception:
                break

            eval_result = await evaluate_response(
                client,
                req.prompt,
                response_text,
                model=eval_model_name,
                eval_context=req.eval_context,
            )

        eval_result.retry_count = retry_count
        eval_info = EvalInfo(
            score=eval_result.score,
            accuracy=eval_result.accuracy,
            completeness=eval_result.completeness,
            clarity=eval_result.clarity,
            actionability=eval_result.actionability,
            issues=eval_result.issues,
            retry_count=retry_count,
        )

    duration_ms = int((time.monotonic() - start) * 1000)
    logger.info("Enhanced query %s completed in %dms (model=%s)", query_id, duration_ms, model_name)

    return EnhanceResponse(
        id=query_id,
        model=model_name,
        response=response_text,
        duration_ms=duration_ms,
        usage=UsageInfo(**usage) if usage else UsageInfo(),
        enhancement=enhancement_info,
        evaluation=eval_info,
    )
