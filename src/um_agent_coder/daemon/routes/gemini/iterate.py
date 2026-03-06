"""Iteration runner — generate → evaluate → strategize → retry loop.

POST /iterate   — Start an iteration run (background task)
GET  /iterate/{id}  — Get iteration status + results + step history
GET  /iterations    — List iteration runs
DELETE /iterate/{id} — Cancel a running iteration
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException

from um_agent_coder.daemon.auth import verify_api_key

from ._evaluator import EvalResult, evaluate_accuracy, evaluate_response
from ._pipeline import enhance_prompt
from ._router import select_model
from ._strategies import FixStrategy, build_strategic_retry_prompt, select_strategies
from .models import (
    GEMINI_MODEL_MAP,
    AccuracyCheckInfo,
    EvalInfo,
    GeminiModelTier,
    IterateRequest,
    IterateResponse,
    IterationStatus,
    IterationStepInfo,
    IterationSummaryResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Track running iteration tasks for cancellation
_iteration_tasks: dict[str, asyncio.Task] = {}


def _get_db():
    from um_agent_coder.daemon.app import get_db
    return get_db()


def _get_client():
    from um_agent_coder.daemon.app import get_gemini_client
    return get_gemini_client()


def _get_settings():
    from um_agent_coder.daemon.app import get_settings
    return get_settings()


def _resolve_model(model: GeminiModelTier, prompt: str, settings) -> str:
    if model == GeminiModelTier.auto:
        return select_model(prompt, threshold=settings.gemini_complexity_threshold)
    return GEMINI_MODEL_MAP[model.value]


def _eval_result_to_info(er: EvalResult) -> EvalInfo:
    return EvalInfo(
        score=er.score,
        accuracy=er.accuracy,
        completeness=er.completeness,
        clarity=er.clarity,
        actionability=er.actionability,
        issues=er.issues,
        retry_count=er.retry_count,
        accuracy_checks=[
            AccuracyCheckInfo(
                check=c.check, status=c.status,
                severity=c.severity, detail=c.detail,
            )
            for c in er.accuracy_checks
        ],
    )


async def _multi_model_evaluate(
    client,
    prompt: str,
    response: str,
    eval_models: List[str],
    eval_context: Optional[str] = None,
) -> EvalResult:
    """Run evaluation across multiple models and take conservative (min) scores.

    For each dimension, takes the minimum score across all models.
    Deduplicates issues across models.
    """
    if len(eval_models) == 1:
        return await evaluate_response(
            client, prompt, response,
            model=eval_models[0], eval_context=eval_context,
        )

    # Run all evals concurrently
    tasks = [
        evaluate_response(
            client, prompt, response,
            model=model, eval_context=eval_context,
        )
        for model in eval_models
    ]
    results: List[EvalResult] = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions
    valid_results = [r for r in results if isinstance(r, EvalResult)]
    if not valid_results:
        logger.warning("All eval models failed, returning default")
        return EvalResult(score=0.7)

    # Take minimum per dimension (conservative — Flash leniency doesn't override Pro strictness)
    accuracy = min(r.accuracy for r in valid_results)
    completeness = min(r.completeness for r in valid_results)
    clarity = min(r.clarity for r in valid_results)
    actionability = min(r.actionability for r in valid_results)
    overall = (accuracy + completeness + clarity + actionability) / 4

    # Deduplicate issues
    all_issues: list[str] = []
    seen: set[str] = set()
    for r in valid_results:
        for issue in r.issues:
            normalized = issue.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                all_issues.append(issue)

    return EvalResult(
        score=overall,
        accuracy=accuracy,
        completeness=completeness,
        clarity=clarity,
        actionability=actionability,
        issues=all_issues,
    )


async def _run_iteration(iteration_id: str, req: IterateRequest):
    """Background task: generate → evaluate → strategize → retry loop."""
    db = _get_db()
    client = _get_client()
    settings = _get_settings()
    overall_start = time.monotonic()

    # Resolve generation model
    gen_model = _resolve_model(req.model, req.prompt, settings)

    # Resolve eval models
    if req.eval_models:
        eval_models = req.eval_models
    else:
        eval_models = [
            m.strip()
            for m in settings.gemini_iterate_eval_models.split(",")
            if m.strip()
        ]

    best_response: Optional[str] = None
    best_score = 0.0
    best_iteration = 0
    total_tokens = 0
    current_prompt = req.prompt
    current_system = req.system_prompt or ""
    current_temp = req.temperature

    try:
        for step_num in range(1, req.max_iterations + 1):
            step_start = time.monotonic()

            # --- Generate ---
            if step_num == 1 and req.use_multi_turn:
                # Step 1: multi-turn for completeness
                prompt_to_send = current_prompt
                if req.enable_enhancement and settings.gemini_enhance_enabled:
                    enhanced = enhance_prompt(prompt_to_send, domain_hint=req.domain_hint)
                    prompt_to_send = enhanced.enhanced

                # Build multi-turn contents with system prompt injected
                contents = []
                if current_system:
                    contents.append({"role": "user", "parts": [{"text": current_system}]})
                    contents.append({"role": "model", "parts": [{"text": "Understood. I'll follow these instructions."}]})
                contents.append({"role": "user", "parts": [{"text": prompt_to_send}]})

                gen_result = await client.generate_multi_turn(
                    contents=contents,
                    model=gen_model,
                    temperature=max(0.0, min(2.0, current_temp)),
                    max_tokens=req.max_tokens,
                    timeout=300.0,
                )
            else:
                # Steps 2+: single-shot with strategic prompt
                prompt_to_send = current_prompt

                gen_result = await client.generate(
                    prompt=prompt_to_send,
                    model=gen_model,
                    system_prompt=current_system or None,
                    temperature=max(0.0, min(2.0, current_temp)),
                    max_tokens=req.max_tokens,
                    timeout=300.0,
                )

            gen_duration_ms = int((time.monotonic() - step_start) * 1000)
            response_text = gen_result.get("text", "")
            usage = gen_result.get("usage", {})
            gen_tokens = usage.get("total_tokens", 0)
            finish_reason = usage.get("finish_reason", "")
            total_tokens += gen_tokens

            # --- Evaluate (accuracy-first cascade) ---
            eval_start = time.monotonic()
            accuracy_passed = True

            if req.eval_context:
                # Phase 1: Accuracy check with pass/fail checklist
                accuracy_result = await evaluate_accuracy(
                    client, req.prompt, response_text,
                    eval_context=req.eval_context,
                )

                if accuracy_result.accuracy < 0.7:
                    # Accuracy failed — skip full eval, use accuracy result only
                    accuracy_passed = False
                    eval_result = accuracy_result
                    logger.info(
                        "Iteration %s step %d: accuracy=%.3f < 0.7, skipping full eval "
                        "(%d checks, %d failed)",
                        iteration_id, step_num, accuracy_result.accuracy,
                        len(accuracy_result.accuracy_checks),
                        sum(1 for c in accuracy_result.accuracy_checks if c.status == "fail"),
                    )
                else:
                    # Accuracy passed — run full eval for other dimensions
                    full_result = await _multi_model_evaluate(
                        client, req.prompt, response_text,
                        eval_models=eval_models,
                        eval_context=req.eval_context,
                    )
                    # Merge: use accuracy from checklist, other dims from full eval
                    eval_result = EvalResult(
                        score=(accuracy_result.accuracy + full_result.completeness
                               + full_result.clarity + full_result.actionability) / 4,
                        accuracy=accuracy_result.accuracy,
                        completeness=full_result.completeness,
                        clarity=full_result.clarity,
                        actionability=full_result.actionability,
                        issues=accuracy_result.issues + full_result.issues,
                        accuracy_checks=accuracy_result.accuracy_checks,
                    )
            else:
                # No eval_context — use standard multi-model eval
                eval_result = await _multi_model_evaluate(
                    client, req.prompt, response_text,
                    eval_models=eval_models,
                    eval_context=None,
                )

            eval_duration_ms = int((time.monotonic() - eval_start) * 1000)

            # Track best
            if eval_result.score > best_score:
                best_score = eval_result.score
                best_response = response_text
                best_iteration = step_num

            # --- Select strategies ---
            strategies = select_strategies(
                eval_result,
                eval_context=req.eval_context,
                threshold=0.7,
            )
            strategy_names = [s.name for s in strategies]

            # --- Store step ---
            eval_scores = {
                "score": eval_result.score,
                "accuracy": eval_result.accuracy,
                "completeness": eval_result.completeness,
                "clarity": eval_result.clarity,
                "actionability": eval_result.actionability,
                "issues": eval_result.issues,
                "accuracy_checks": [
                    {"check": c.check, "status": c.status,
                     "severity": c.severity, "detail": c.detail}
                    for c in eval_result.accuracy_checks
                ],
                "accuracy_passed": accuracy_passed,
            }
            await db.add_gemini_iteration_step(
                iteration_id=iteration_id,
                step_number=step_num,
                prompt_sent=prompt_to_send[:50000],  # truncate for DB
                response=response_text,
                generation_model=gen_model,
                generation_duration_ms=gen_duration_ms,
                generation_tokens=gen_tokens,
                eval_scores=eval_scores,
                eval_models=eval_models,
                eval_duration_ms=eval_duration_ms,
                strategies_applied=strategy_names,
                finish_reason=finish_reason,
            )

            # Update iteration progress
            duration_ms = int((time.monotonic() - overall_start) * 1000)
            await db.update_gemini_iteration(
                iteration_id,
                best_response=best_response,
                best_score=best_score,
                best_iteration=best_iteration,
                total_iterations=step_num,
                total_tokens=total_tokens,
            )

            logger.info(
                "Iteration %s step %d: score=%.3f (best=%.3f@%d) strategies=%s",
                iteration_id, step_num, eval_result.score,
                best_score, best_iteration, strategy_names,
            )

            # --- Check threshold ---
            if eval_result.score >= req.score_threshold:
                now = datetime.now(timezone.utc).isoformat()
                duration_ms = int((time.monotonic() - overall_start) * 1000)
                await db.update_gemini_iteration(
                    iteration_id,
                    status="threshold_met",
                    best_response=best_response,
                    best_score=best_score,
                    best_iteration=best_iteration,
                    total_iterations=step_num,
                    total_tokens=total_tokens,
                    completed_at=now,
                )
                logger.info(
                    "Iteration %s threshold met at step %d (%.3f >= %.3f)",
                    iteration_id, step_num, eval_result.score, req.score_threshold,
                )
                return

            # --- Build strategic retry prompt for next step ---
            if strategies:
                retry_prompt, system_addendum, temp_delta = build_strategic_retry_prompt(
                    original_prompt=req.prompt,
                    previous_response=response_text,
                    eval_result=eval_result,
                    strategies=strategies,
                    eval_context=req.eval_context,
                )
                current_prompt = retry_prompt
                if system_addendum:
                    current_system = (
                        (current_system + " " + system_addendum).strip()
                        if current_system
                        else system_addendum
                    )
                current_temp = max(0.0, min(2.0, current_temp + temp_delta))
            else:
                # No specific strategy — use generic retry
                current_prompt = (
                    f"{req.prompt}\n\n"
                    f"[Previous attempt scored {eval_result.score:.2f}. "
                    f"Please improve the response.]"
                )

        # Max iterations reached — finalize with best
        now = datetime.now(timezone.utc).isoformat()
        duration_ms = int((time.monotonic() - overall_start) * 1000)
        await db.update_gemini_iteration(
            iteration_id,
            status="max_iterations_reached",
            best_response=best_response,
            best_score=best_score,
            best_iteration=best_iteration,
            total_iterations=req.max_iterations,
            total_tokens=total_tokens,
            completed_at=now,
        )
        logger.info(
            "Iteration %s max iterations reached. Best score=%.3f at step %d",
            iteration_id, best_score, best_iteration,
        )

    except asyncio.CancelledError:
        now = datetime.now(timezone.utc).isoformat()
        await db.update_gemini_iteration(
            iteration_id,
            status="cancelled",
            best_response=best_response,
            best_score=best_score,
            best_iteration=best_iteration,
            total_tokens=total_tokens,
            completed_at=now,
        )
        logger.info("Iteration %s cancelled", iteration_id)
    except Exception as e:
        logger.exception("Iteration %s failed: %s", iteration_id, e)
        now = datetime.now(timezone.utc).isoformat()
        await db.update_gemini_iteration(
            iteration_id,
            status="failed",
            error=str(e),
            best_response=best_response,
            best_score=best_score,
            best_iteration=best_iteration,
            total_tokens=total_tokens,
            completed_at=now,
        )


# --- Helper: build response from DB ---

async def _build_iterate_response(iteration_id: str) -> IterateResponse:
    db = _get_db()
    iteration = await db.get_gemini_iteration(iteration_id)
    if not iteration:
        raise HTTPException(status_code=404, detail="Iteration not found")

    steps_rows = await db.get_gemini_iteration_steps(iteration_id)
    steps = []
    for s in steps_rows:
        eval_scores = s.get("eval_scores") or {}
        evaluation = EvalInfo(
            score=eval_scores.get("score", 0.0),
            accuracy=eval_scores.get("accuracy", 0.0),
            completeness=eval_scores.get("completeness", 0.0),
            clarity=eval_scores.get("clarity", 0.0),
            actionability=eval_scores.get("actionability", 0.0),
            issues=eval_scores.get("issues", []),
        ) if eval_scores else None

        steps.append(IterationStepInfo(
            step=s["step_number"],
            prompt_sent=s.get("prompt_sent") or "",
            response=s.get("response") or "",
            generation_model=s.get("generation_model") or "",
            generation_duration_ms=s.get("generation_duration_ms", 0),
            generation_tokens=s.get("generation_tokens", 0),
            evaluation=evaluation,
            eval_models_used=s.get("eval_models") or [],
            strategies_applied=s.get("strategies_applied") or [],
            finish_reason=s.get("finish_reason") or "",
        ))

    # Calculate duration
    duration_ms = 0
    if iteration.get("started_at") and iteration.get("completed_at"):
        try:
            started = datetime.fromisoformat(iteration["started_at"])
            completed = datetime.fromisoformat(iteration["completed_at"])
            duration_ms = int((completed - started).total_seconds() * 1000)
        except (ValueError, TypeError):
            pass

    return IterateResponse(
        id=iteration["id"],
        status=iteration["status"],
        original_prompt=iteration["original_prompt"],
        best_response=iteration.get("best_response"),
        best_score=iteration.get("best_score", 0.0),
        best_iteration=iteration.get("best_iteration", 0),
        total_iterations=iteration.get("total_iterations", 0),
        total_tokens=iteration.get("total_tokens", 0),
        duration_ms=duration_ms,
        config=iteration.get("config"),
        steps=steps,
        error=iteration.get("error"),
        created_at=iteration.get("created_at"),
    )


# --- Endpoints ---

@router.post("/iterate", response_model=IterateResponse)
async def start_iteration(
    req: IterateRequest,
    _key: Optional[str] = Depends(verify_api_key),
):
    """Start an iteration loop: generate → evaluate → strategize → retry."""
    db = _get_db()
    iteration_id = f"gi-{uuid.uuid4().hex[:12]}"

    config = {
        "model": req.model.value,
        "eval_models": req.eval_models,
        "max_iterations": req.max_iterations,
        "score_threshold": req.score_threshold,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "enable_enhancement": req.enable_enhancement,
        "use_multi_turn": req.use_multi_turn,
        "domain_hint": req.domain_hint,
    }

    await db.create_gemini_iteration(
        iteration_id=iteration_id,
        original_prompt=req.prompt,
        system_prompt=req.system_prompt,
        eval_context=req.eval_context,
        config=config,
    )

    # Launch background task
    task = asyncio.create_task(_run_iteration(iteration_id, req))
    _iteration_tasks[iteration_id] = task

    def _cleanup(t):
        _iteration_tasks.pop(iteration_id, None)
    task.add_done_callback(_cleanup)

    return await _build_iterate_response(iteration_id)


@router.get("/iterate/{iteration_id}", response_model=IterateResponse)
async def get_iteration(
    iteration_id: str,
    _key: Optional[str] = Depends(verify_api_key),
):
    """Get iteration status, results, and step history."""
    return await _build_iterate_response(iteration_id)


@router.get("/iterations", response_model=List[IterationSummaryResponse])
async def list_iterations(
    limit: int = 50,
    status: Optional[str] = None,
    _key: Optional[str] = Depends(verify_api_key),
):
    """List iteration runs."""
    db = _get_db()
    iterations = await db.list_gemini_iterations(limit=limit, status=status)

    result = []
    for it in iterations:
        duration_ms = 0
        if it.get("started_at") and it.get("completed_at"):
            try:
                started = datetime.fromisoformat(it["started_at"])
                completed = datetime.fromisoformat(it["completed_at"])
                duration_ms = int((completed - started).total_seconds() * 1000)
            except (ValueError, TypeError):
                pass

        result.append(IterationSummaryResponse(
            id=it["id"],
            status=it["status"],
            best_score=it.get("best_score", 0.0),
            total_iterations=it.get("total_iterations", 0),
            total_tokens=it.get("total_tokens", 0),
            duration_ms=duration_ms,
            created_at=it.get("created_at"),
        ))
    return result


@router.delete("/iterate/{iteration_id}")
async def cancel_iteration(
    iteration_id: str,
    _key: Optional[str] = Depends(verify_api_key),
):
    """Cancel a running iteration."""
    db = _get_db()
    iteration = await db.get_gemini_iteration(iteration_id)
    if not iteration:
        raise HTTPException(status_code=404, detail="Iteration not found")

    if iteration["status"] != "running":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel iteration in '{iteration['status']}' state",
        )

    # Cancel the background task
    task = _iteration_tasks.get(iteration_id)
    if task and not task.done():
        task.cancel()

    now = datetime.now(timezone.utc).isoformat()
    await db.update_gemini_iteration(iteration_id, status="cancelled", completed_at=now)
    return {"status": "cancelled", "iteration_id": iteration_id}
